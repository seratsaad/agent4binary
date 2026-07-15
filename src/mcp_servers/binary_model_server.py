#!/usr/bin/env python3
"""
MCP Server for SB2 Binary Modeling.

The SB2 forward-model and detection layer. The science lives in src/physics.py
(the El-Badry et al. 2018b implementation); this server is a thin wrapper that
loads observed spectra and returns COMPACT results.

Forward model: compose_sb2 builds the El-Badry Eq. 2 composite from
physics.compose_binary -- the primary and an isochrone-tied secondary summed in
physical flux (R1^2 f1 + R2^2 f2) and re-normalized; the flux ratio is fixed by
the mass ratio q via the isochrone (q -> Teff2, logg2, R2, R1).

Detection: chi2_single_vs_binary defaults to the production self-consistent fit
(model="dr19_sc", physics.dr19_sc_single_vs_binary): RAW flux+ivar in, normalized
by physics.continuum_normalize, OUR SC-trained 5-label net, OUR un-normalization
continuum -- no survey continuum, no binspec net. model="dr19" is the survey-
normalized comparison; model="binspec" is the vendored oracle. All report
Delta-chi2, the El-Badry Eq. B1 f_imp, the Table B1 thresholds, and a verdict.

All heavy arrays flow through data/cache/<spec_id>.npz (written by the APOGEE
Data server's load_spectrum) or the raw npz dirs. Tool return values are COMPACT:
scalars, short lists, file paths. Composite spectra are cached, never returned.

APOGEE log-linear grid: a one-pixel shift is a constant velocity
    dv_pix = CDELT1 * ln(10) * c = 6e-6 * 2.302585 * 299792.458 ~= 4.145 km/s/pix
"""

import os
import sys

import numpy as np
from scipy.signal import find_peaks
from mcp.server.fastmcp import FastMCP

# Import the shared science module. physics.py lives in src/, one level up from
# src/mcp_servers/, so add src/ to the path before importing.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import physics  # noqa: E402

mcp = FastMCP("Binary Model", log_level="WARNING")

C_KMS = 299792.458
CDELT1 = 6e-6
DV_PIX = CDELT1 * np.log(10.0) * C_KMS  # ~4.145 km/s per pixel

_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
CACHE_DIR = os.path.join(_ROOT, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Reference thresholds (NOT the active gate). The verdict is decided by the
# El-Badry et al. 2018b Table B1 SLIDING scale in physics (passes_table_b1):
# Delta-chi2 >= 3000 accepts any f_imp, lower Delta-chi2 needs higher f_imp. These
# single-pair values are kept only as a coarse human reference.
DELTA_CHI2_THRESHOLD = 300.0   # minimum Delta-chi2 to consider a binary
F_IMP_FLOOR = 0.10             # minimum improvement fraction to accept


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _cache_path(spec_id: str) -> str:
    sid = spec_id.strip()
    if sid.startswith("aspcapStar-dr17-"):
        sid = sid[len("aspcapStar-dr17-"):]
    if sid.endswith(".fits"):
        sid = sid[:-5]
    if not sid.startswith("2M"):
        sid = "2M" + sid
    return os.path.join(CACHE_DIR, f"{sid}.npz")


def _load_cached(spec_id: str):
    """Return (wl, flux, error) from the on-disk cache; raises if missing.

    error may be absent in older caches (composites); then it is returned None.
    """
    path = _cache_path(spec_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no cache for {spec_id} (call apogee_data.load_spectrum first)"
        )
    data = np.load(path)
    err = np.asarray(data["error"], float) if "error" in data.files else None
    return data["wl"], np.asarray(data["flux"], float), err


# Raw-spectrum search dirs for the SELF-CONSISTENT detector. An observed spectrum
# enters the SC path as RAW flux + ivar (npz {wl, flux_raw, ivar}, written by
# src/download_dr19_raw.py) and is normalized by physics.continuum_normalize -- NO
# survey continuum. We resolve a spec_id (sdss_id or a *_raw.npz cache) against
# these dirs in order; the first hit wins.
_RAW_DIRS = [
    os.path.join(CACHE_DIR),                               # raw caches
    os.path.join(_ROOT, "data", "dr19_raw_sb2"),          # SB2 test set
    os.path.join(_ROOT, "data", "dr19_raw_controls"),     # held-out controls
    os.path.join(_ROOT, "data", "dr19_raw"),              # training pool
]


def _load_raw(spec_id: str):
    """Return (flux_raw, ivar) for the SC detector, raises if missing.

    Looks for an npz holding {flux_raw, ivar} keyed by sdss_id (the Stage 1 raw
    products) or a <id>_raw.npz cache. NO survey continuum is read here; the
    caller passes these straight into physics.continuum_normalize.
    """
    sid = spec_id.strip()
    # Strip common decorations so a 2MASS-styled id still resolves by its number.
    cands = [sid, sid + "_raw"]
    if sid.startswith("2M"):
        cands.append(sid[2:])
    for d in _RAW_DIRS:
        for c in cands:
            p = os.path.join(d, c + ".npz")
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                data = np.load(p)
                if "flux_raw" in data.files and "ivar" in data.files:
                    return (np.asarray(data["flux_raw"], float),
                            np.asarray(data["ivar"], float))
    raise FileNotFoundError(
        f"no RAW spectrum (flux_raw+ivar) for {spec_id}; the SC detector needs "
        f"a raw npz from src/download_dr19_raw.py")


def _continuum_subtract(flux: np.ndarray) -> np.ndarray:
    """
    Pseudo-continuum subtraction for the CCF placeholder: replace
    non-finite/zero gaps, subtract a smooth median-filtered baseline so we
    cross-correlate line residuals only.
    """
    f = np.array(flux, dtype=float)
    bad = ~np.isfinite(f) | (f == 0)
    if bad.all():
        return np.zeros_like(f)
    med = np.median(f[~bad])
    f[bad] = med
    win = 101
    kernel = np.ones(win) / win
    base = np.convolve(f, kernel, mode="same")
    resid = f - base
    resid[bad] = 0.0
    return resid


def _ccf(res_a: np.ndarray, res_b: np.ndarray, max_lag_pix: int):
    """Normalized cross-correlation of two residual arrays over +/- max_lag_pix."""
    a = res_a - res_a.mean()
    b = res_b - res_b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        denom = 1.0
    lags = np.arange(-max_lag_pix, max_lag_pix + 1)
    ccf = np.empty(lags.size)
    n = a.size
    for i, lag in enumerate(lags):
        if lag < 0:
            ov = np.sum(a[:n + lag] * b[-lag:])
        elif lag > 0:
            ov = np.sum(a[lag:] * b[:n - lag])
        else:
            ov = np.sum(a * b)
        ccf[i] = ov
    return lags, ccf / denom


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@mcp.tool()
def ccf_rv_scan(
    spec_id: str,
    template_id: str = None,
    rv_min: float = -500.0,
    rv_max: float = 500.0,
    dv: float = 5.0,
) -> dict:
    """
    Cross-correlate an observed spectrum against a template to find RV peaks.

    Placeholder detector: the decisive SB2 test is chi2_single_vs_binary; the CCF
    only finds large-Delta-RV pairs and is kept for diagnostics.

    Continuum-subtracts both spectra and computes a CCF over the velocity range.
    One dominant peak => single-lined; two well-separated peaks => SB2 candidate.
    Returns only compact peak structure (positions/heights), never arrays.
    Requires the spectra to be cached (apogee_data.load_spectrum first).

    Args:
        spec_id: Observed spectrum 2MASS id.
        template_id: Template 2MASS id. If None, a smoothed self-template is used
                     (auto-correlation of line residuals).
        rv_min: Minimum RV of the scan (km/s).
        rv_max: Maximum RV of the scan (km/s).
        dv: Reporting velocity step (km/s); the underlying grid is per-pixel
            (~4.145 km/s) and peaks are mapped through it.

    Returns:
        {dv_per_pixel_kms, n_peaks, peak_rvs_kms, peak_heights, peak_separation_kms,
         bimodality_score, bimodal, sb2_candidate}
    """
    try:
        _, flux, _ = _load_cached(spec_id)
    except FileNotFoundError as e:
        return {"error": str(e)}

    res_obs = _continuum_subtract(flux)

    if template_id is None:
        kernel = np.ones(5) / 5.0
        res_tmpl = np.convolve(res_obs, kernel, mode="same")
    else:
        try:
            _, tflux, _ = _load_cached(template_id)
        except FileNotFoundError as e:
            return {"error": str(e)}
        res_tmpl = _continuum_subtract(tflux)

    max_lag_pix = int(np.ceil(max(abs(rv_min), abs(rv_max)) / DV_PIX))
    max_lag_pix = min(max_lag_pix, res_obs.size // 2 - 1)
    lags, ccf = _ccf(res_obs, res_tmpl, max_lag_pix)
    rv_axis = lags * DV_PIX

    in_range = (rv_axis >= rv_min) & (rv_axis <= rv_max)
    rv_axis = rv_axis[in_range]
    ccf = ccf[in_range]

    if ccf.size == 0 or not np.any(np.isfinite(ccf)):
        return {"error": "empty or invalid CCF"}

    ccf = np.nan_to_num(ccf, nan=0.0)
    cmax = float(ccf.max())
    min_sep_pix = max(1, int(round(2.0 * dv / DV_PIX)))
    height_thresh = 0.30 * cmax if cmax > 0 else 0.0
    peak_idx, props = find_peaks(ccf, height=height_thresh, distance=min_sep_pix)

    if peak_idx.size == 0:
        peak_idx = np.array([int(np.argmax(ccf))])
        heights = np.array([cmax])
    else:
        heights = props["peak_heights"]

    order = np.argsort(heights)[::-1][:5]
    peak_idx = peak_idx[order]
    heights = heights[order]
    rvs = rv_axis[peak_idx]

    rel_heights = (heights / cmax) if cmax > 0 else heights
    n_peaks = int(peak_idx.size)

    if n_peaks >= 2:
        sorted_h = np.sort(rel_heights)[::-1]
        bimodality_score = float(sorted_h[1])
        rv_sorted = rvs[np.argsort(rel_heights)[::-1]]
        peak_sep = float(abs(rv_sorted[0] - rv_sorted[1]))
    else:
        bimodality_score = 0.0
        peak_sep = 0.0

    bimodal = bool(n_peaks >= 2 and bimodality_score >= 0.4)
    sb2_candidate = bool(bimodal and peak_sep >= 4.0 * dv)

    return {
        "dv_per_pixel_kms": round(float(DV_PIX), 4),
        "n_peaks": n_peaks,
        "peak_rvs_kms": [round(float(x), 2) for x in rvs],
        "peak_heights": [round(float(x), 4) for x in rel_heights],
        "peak_separation_kms": round(peak_sep, 2),
        "bimodality_score": round(bimodality_score, 4),
        "bimodal": bimodal,
        "sb2_candidate": sb2_candidate,
    }


@mcp.tool()
def compose_sb2(
    teff1: float,
    logg1: float,
    feh: float,
    q: float,
    rv1_kms: float,
    rv2_kms: float,
    age_gyr: float = 5.0,
) -> dict:
    """
    Build a physical SB2 composite (El-Badry Eq. 2) and cache it.

    Uses physics.compose_binary: the primary is the Payne spectrum at
    (Teff1, logg1, [Fe/H]); the secondary's (Teff2, logg2, R2) and the primary
    radius R1 come from the isochrone given the mass ratio q. Each component is
    un-normalized by its trained-flux-net pseudo-continuum (ported from binspec),
    Doppler-shifted by its RV, summed in flux as R1^2 f1 + R2^2 f2, and
    re-normalized. The composite is
    written to data/cache/composite_<...>.npz; the array is not returned.

    The OLD signature (spec1_id, spec2_id, free flux_ratio) and the
    normalized-space blend are REMOVED. The flux ratio is now tied to q through
    the isochrone, as in the paper.

    Args:
        teff1: Primary effective temperature (K).
        logg1: Primary surface gravity (dex).
        feh: Metallicity [Fe/H] (shared by both components).
        q: Mass ratio m2/m1 in (0, 1].
        rv1_kms: Primary radial velocity (km/s).
        rv2_kms: Secondary radial velocity (km/s).
        age_gyr: System age in Gyr (default 5.0).

    Returns:
        {cache_path, npix, teff2, logg2, h_band_flux_ratio, R1, R2, q,
         rv1, rv2} or {error}.
    """
    if not (0 < q <= 1.0):
        return {"error": "q must be in (0, 1]"}
    try:
        teff2, logg2, R2, R1 = physics.secondary_from_q(
            teff1, logg1, feh, q, age_gyr)
        composite = physics.compose_binary(
            teff1, logg1, feh, q, rv1_kms, rv2_kms, age_gyr)
    except ValueError as e:
        return {"error": str(e)}

    # The isochrone H-band flux ratio: emitting-area ratio times the trained-flux-
    # net surface-brightness ratio (mean over the H band). This is the physical
    # ratio the composite uses, reported for transparency.
    pc1 = physics.pseudo_continuum(teff1, logg1, feh)
    pc2 = physics.pseudo_continuum(teff2, logg2, feh)
    h_band_flux_ratio = float((R2 ** 2 * pc2.mean()) / (R1 ** 2 * pc1.mean()))

    name = (f"composite_{teff1:.0f}_{logg1:.2f}_{feh:.2f}"
            f"_q{q:.2f}_rv{rv1_kms:.0f}_{rv2_kms:.0f}.npz")
    cache_path = os.path.join(CACHE_DIR, name)
    np.savez_compressed(
        cache_path,
        wl=physics.WAVELENGTH,
        flux=composite.astype(np.float32),
        teff1=teff1, logg1=logg1, feh=feh, q=q,
        teff2=teff2, logg2=logg2, R1=R1, R2=R2,
        rv1=rv1_kms, rv2=rv2_kms,
    )
    return {
        "cache_path": cache_path,
        "npix": int(composite.size),
        "teff2": round(teff2, 1),
        "logg2": round(logg2, 3),
        "h_band_flux_ratio": round(h_band_flux_ratio, 5),
        "R1": round(R1, 4),
        "R2": round(R2, 4),
        "q": float(q),
        "rv1": float(rv1_kms),
        "rv2": float(rv2_kms),
    }


def _pack(r: dict, model: str) -> dict:
    """Compact, transport-rounded result dict shared by all detector backends."""
    return {
        "model": model,
        "chi2_single": round(r["chi2_single"], 2),
        "chi2_binary": round(r["chi2_binary"], 2),
        "delta_chi2": round(r["delta_chi2"], 2),
        "f_imp": round(r["f_imp"], 4),
        "min_fimp_required": r["min_fimp_required"],
        "prefers_binary": r["prefers_binary"],
        "best_q": round(r["best_q"], 3),
        "best_rv1": round(r["best_rv1"], 2),
        "best_rv2": round(r["best_rv2"], 2),
        "best_teff1": round(r["best_teff1"], 1),
        "teff_single": round(r["teff_single"], 1),
        "logg_single": round(r["logg_single"], 3),
        "feh_single": round(r["feh_single"], 3),
    }


# The OPEN detector (the paper's headline ~63%): OUR curated five-label net used as
# the single-star line model THROUGH El-Badry's public fitting layer. We monkeypatch
# binspec's normalized-spectrum forward pass to call our net; binspec's trained LINE
# weights are never loaded into this path (oracle weights stay in model="binspec").
# Net path via AB_MLP_NET (default: the curated net).
_MLP_PATCHED = False


def _patch_binspec_mlp():
    global _MLP_PATCHED
    if _MLP_PATCHED:
        return
    import torch
    pt = os.environ.get("AB_MLP_NET",
                        os.path.join(_ROOT, "models", "payne_dr19_curated.pt"))
    c = torch.load(pt, map_location="cpu", weights_only=False)
    s = c["state_dict"]
    W0 = s["net.0.weight"].numpy().astype(np.float64); b0 = s["net.0.bias"].numpy().astype(np.float64)
    W1 = s["net.2.weight"].numpy().astype(np.float64); b1 = s["net.2.bias"].numpy().astype(np.float64)
    W2 = s["net.4.weight"].numpy().astype(np.float64); b2 = s["net.4.bias"].numpy().astype(np.float64)
    lmin = np.asarray(c["label_min"], np.float64); lmax = np.asarray(c["label_max"], np.float64)
    span = lmax - lmin
    lr = lambda z: np.where(z > 0, z, 0.01 * z)
    def fwd(labels):
        lab = np.clip(np.asarray(labels, np.float64), lmin, lmax)
        xs = (lab - lmin) / span - 0.5
        h = lr(W0 @ xs + b0); h = lr(W1 @ h + b1); return W2 @ h + b2
    import vendor.binspec.spectral_model as sm
    orig = sm.get_spectrum_from_neural_net
    def patched(labels, NN_coeffs, normalized=False):
        if normalized:
            return fwd(labels)
        return orig(labels, NN_coeffs, normalized)
    sm.get_spectrum_from_neural_net = patched
    _MLP_PATCHED = True


@mcp.tool()
def chi2_single_vs_binary(
    spec_id: str,
    teff1: float = None,
    logg1: float = None,
    feh: float = None,
    mgh: float = None,
    vmacro: float = None,
    age_gyr: float = 5.0,
    model: str = "dr19_sc",
) -> dict:
    """
    Decide single vs binary for an observed spectrum (the decisive SB2 detector).

    DEFAULT (model="dr19_sc"): runs OUR SELF-CONSISTENT single-vs-binary fit
    (physics.dr19_sc_single_vs_binary). The observed spectrum is loaded as RAW
    flux + ivar and normalized by physics.continuum_normalize (a per-chip sigma-
    clipped Chebyshev continuum from the RAW flux ITSELF) -- NO survey continuum,
    NO binspec net. The single-star spectral model is the net WE trained on that
    SAME self-consistent normalization (payne_dr19_sc.pt); the binary composite
    sums two components in flux with OUR OWN Teff-keyed un-normalization continuum.
    The q=1 / equal-RV limit equals the single model exactly. Steps mirror the
    dr19 path below, with the SC normalization + SC net + SC continuum.

    model="dr19" (survey-normalized, kept for comparison): runs OUR DR19 5-label
    single-vs-binary fit (physics.dr19_single_vs_binary). The single-star spectral
    model is the net trained on the SURVEY-normalized DR19 dwarfs (Teff, logg,
    [Fe/H], [Mg/H], v_macro -> flux[8575]) on OUR 8575-pixel grid. Steps:
      1. Normalize flux + error with the shared continuum routine + mask bad
         pixels, so model and data share one normalization.
      2. Single fit: OUR five-label net via scipy least_squares, seeded by the
         catalog labels (teff1/logg1/feh/mgh/vmacro) if given.
      3. Binary fit: El-Badry Eq. 2 composite with OUR net, grid q + RV scan,
         primary five-label refine, binary-nests-single floor.
      4. Statistics: Delta-chi2; the EXACT El-Badry Eq. B1 f_imp; the Table B1
         sliding thresholds; Delta-chi2 >= 0.

    model="binspec" (the ORACLE, kept for head-to-head comparison): runs the
    binspec-faithful fit (physics.binspec_single_vs_binary) with El-Badry's
    VENDORED trained nets on binspec's 7214-pixel grid. Same statistic / verdict
    shape, so the two are drop-in comparable.

    prefers_binary uses the El-Badry et al. 2018b Table B1 SLIDING scale: Delta-
    chi2 >= 3000 accepts any f_imp; lower Delta-chi2 needs a higher minimum f_imp;
    below 300 it is a non-detection. min_fimp_required is the bin minimum.

    Args:
        spec_id: Observed spectrum 2MASS id (load_spectrum first).
        teff1: Optional primary Teff seed (K); used by the dr19 single fit.
        logg1: Optional primary logg seed (dex); used by the dr19 single fit.
        feh: Optional [Fe/H] seed; used by the dr19 single fit.
        mgh: Optional [Mg/H] seed; used by the dr19 single fit.
        vmacro: Optional v_macro seed; used by the dr19 single fit.
        age_gyr: Accepted for interface compatibility; the detectors use a fixed
                 representative main-sequence age internally.
        model: "dr19_sc" (OUR self-consistent net, default; loads RAW flux+ivar),
               "binspec_mlp" (THE OPEN DETECTOR, paper headline ~63%: OUR curated
               net through El-Badry's public fitting layer; AB_MLP_NET selects the
               net), "dr19" (survey-normalized net), or "binspec" (the oracle,
               ~76% on the enlarged benchmark).

    Returns:
        {chi2_single, chi2_binary, delta_chi2, f_imp, min_fimp_required,
         prefers_binary, best_q, best_rv1, best_rv2, best_teff1, teff_single,
         logg_single, feh_single, model} or {error}.
    """
    # Common single-fit seed (catalog labels) for the OUR-net paths.
    seed = None
    if teff1 is not None and logg1 is not None and feh is not None:
        seed = (teff1, logg1, feh,
                0.0 if mgh is None else mgh,
                5.0 if vmacro is None else vmacro)

    if model == "dr19_sc":
        # DEFAULT, SELF-CONSISTENT: load RAW flux + ivar (NO survey continuum) and
        # let physics.continuum_normalize do the normalization inside the detector.
        try:
            flux_raw, ivar = _load_raw(spec_id)
        except FileNotFoundError as e:
            return {"error": str(e)}
        try:
            r = physics.dr19_sc_single_vs_binary(flux_raw, ivar, seed=seed)
        except RuntimeError as e:
            return {"error": str(e)}
        return _pack(r, model)

    if model == "binspec_mlp":
        # THE OPEN DETECTOR (paper headline, ~63%): OUR curated net as the single-star
        # line model through El-Badry's public fitting layer. Loads RAW flux + ivar
        # (like dr19_sc) and runs the binspec-faithful fit with our net patched in.
        try:
            flux_raw, ivar = _load_raw(spec_id)
        except FileNotFoundError as e:
            return {"error": str(e)}
        _patch_binspec_mlp()
        err = np.where(ivar > 0, 1.0 / np.sqrt(np.maximum(ivar, 1e-30)), np.inf)
        try:
            r = physics.binspec_single_vs_binary(physics.WAVELENGTH, flux_raw, err)
        except RuntimeError as e:
            return {"error": str(e)}
        return _pack(r, model)

    # The survey-cache paths (dr19, binspec) read the normalized {wl, flux, error}.
    try:
        wl, flux, err = _load_cached(spec_id)
    except FileNotFoundError as e:
        return {"error": str(e)}
    if err is None:
        return {"error": f"no error array cached for {spec_id}"}

    wl = np.asarray(wl, float)
    flux = np.asarray(flux, float)
    err = np.asarray(err, float)

    if model == "binspec":
        # The ORACLE: binspec vendored nets on the 7214-pixel grid (kept for the
        # head-to-head comparison; not the default detection path anymore).
        r = physics.binspec_single_vs_binary(wl, flux, err)
    else:
        # model="dr19": OUR survey-normalized 5-label net on the 8575-pixel grid.
        try:
            r = physics.dr19_single_vs_binary(wl, flux, err, seed=seed)
        except RuntimeError as e:
            return {"error": str(e)}
    return _pack(r, model)


if __name__ == "__main__":
    mcp.run()
