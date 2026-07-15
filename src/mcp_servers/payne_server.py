#!/usr/bin/env python3
"""
MCP Server for the Payne single-star spectral emulator.

A small MLP maps 3 stellar labels (Teff, logg, [Fe/H]=[M/H]) -> 8575 normalized
rest-frame flux pixels. Broadening (vsini/vmacro) and RV are handled separately
by broadening_server.py and doppler_server.py; this emulator is rest-frame only.

Heavy 8575-pixel arrays are NEVER returned. predict_spectrum caches the array to
data/cache/payne_<teff>_<logg>_<feh>.npz and returns a compact summary.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from astropy.io import fits
from scipy.optimize import minimize
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Payne", log_level="WARNING")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
ASPCAP_DIR = os.path.join(_ROOT, "data", "aspcap")
CACHE_DIR = os.path.join(_ROOT, "data", "cache")
MODEL_PATH = os.path.join(_ROOT, "models", "payne.pt")
os.makedirs(CACHE_DIR, exist_ok=True)

NPIX = 8575
_FNAME_PREFIX = "aspcapStar-dr17-"


class Payne(nn.Module):
    def __init__(self, n_label=3, n_hidden=300, n_pix=NPIX):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_label, n_hidden), nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(),
            nn.Linear(n_hidden, n_pix),
        )

    def forward(self, x):
        return self.net(x)


# ---- load model once (CPU is fine for inference / optimization) ----
_CKPT = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
_NET = Payne(_CKPT["n_label"], _CKPT["n_hidden"], _CKPT["n_pix"])
_NET.load_state_dict(_CKPT["state_dict"])
_NET.eval()
_LMIN = np.asarray(_CKPT["label_min"], dtype=np.float32)
_LMAX = np.asarray(_CKPT["label_max"], dtype=np.float32)


def _normalize_id(spec_id: str) -> str:
    sid = spec_id.strip()
    if sid.startswith(_FNAME_PREFIX):
        sid = sid[len(_FNAME_PREFIX):]
    if sid.endswith(".fits"):
        sid = sid[: -len(".fits")]
    if not sid.startswith("2M"):
        sid = "2M" + sid
    return sid


def _norm_labels(arr):
    return (arr - _LMIN) / (_LMAX - _LMIN) - 0.5


def _predict(teff, logg, feh):
    lab = np.array([[teff, logg, feh]], dtype=np.float32)
    x = torch.from_numpy(_norm_labels(lab))
    with torch.no_grad():
        return _NET(x).numpy().ravel()


def _in_range(teff, logg, feh) -> bool:
    lab = np.array([teff, logg, feh], dtype=np.float32)
    return bool(np.all(lab >= _LMIN) and np.all(lab <= _LMAX))


def _wl_grid(header):
    crval1 = float(header["CRVAL1"])
    cdelt1 = float(header["CDELT1"])
    naxis1 = int(header["NAXIS1"])
    return 10.0 ** (crval1 + cdelt1 * np.arange(naxis1))


def _load_observed(spec_id: str):
    """Return (flux, error) from data/cache/<spec_id>.npz, reading FITS if needed."""
    sid = _normalize_id(spec_id)
    cache = os.path.join(CACHE_DIR, f"{sid}.npz")
    if not os.path.exists(cache):
        fpath = os.path.join(ASPCAP_DIR, f"{_FNAME_PREFIX}{sid}.fits")
        if not os.path.exists(fpath):
            return None, None
        with fits.open(fpath) as h:
            flux = np.asarray(h[1].data, dtype=float)
            error = np.asarray(h[2].data, dtype=float)
            try:
                model = np.asarray(h[3].data, dtype=float)
            except Exception:
                model = np.full_like(flux, np.nan)
            wl = _wl_grid(h[1].header)
        np.savez_compressed(cache, wl=wl, flux=flux, error=error, model=model)
    d = np.load(cache)
    return np.asarray(d["flux"], float), np.asarray(d["error"], float)


@mcp.tool()
def predict_spectrum(teff: float, logg: float, feh: float) -> dict:
    """
    Predict a rest-frame normalized APOGEE spectrum from stellar labels.

    Runs the trained Payne MLP on (Teff, logg, [Fe/H]) to produce 8575 normalized
    flux pixels on the standard log-linear grid (15100.8-16999.8 A). The array is
    NOT returned; it is cached to data/cache/payne_<teff>_<logg>_<feh>.npz. Apply
    broadening_server / doppler_server afterwards for vsini/vmacro/RV.

    Args:
        teff: Effective temperature (K), ~3000-8000.
        logg: Surface gravity (dex), ~0-5.
        feh: Metallicity [Fe/H] (= ASPCAP [M/H]), ~ -2..+0.5.

    Returns:
        {cache_path, npix, label_in_range}
    """
    flux = _predict(teff, logg, feh)
    cache_path = os.path.join(
        CACHE_DIR, f"payne_{teff:.1f}_{logg:.3f}_{feh:.3f}.npz")
    np.savez_compressed(cache_path, flux=flux.astype(np.float32),
                        labels=np.array([teff, logg, feh], np.float32))
    return {
        "cache_path": cache_path,
        "npix": int(flux.size),
        "label_in_range": _in_range(teff, logg, feh),
    }


@mcp.tool()
def fit_labels(spec_id: str) -> dict:
    """
    Fit (Teff, logg, [Fe/H]) of an observed APOGEE spectrum with the Payne model.

    Loads the observed flux/error from data/cache/<spec_id>.npz (reading the FITS
    and caching it first if absent), then optimizes the 3 labels to best match the
    model (scipy Nelder-Mead on the inverse-variance chi2 over good pixels).

    Args:
        spec_id: 2MASS id (with or without the 2M prefix).

    Returns:
        {teff, logg, feh, chi2, reduced_chi2}  (or {error: ...})
    """
    flux, error = _load_observed(spec_id)
    if flux is None:
        return {"error": f"spectrum not found: {spec_id}"}

    good = np.isfinite(flux) & np.isfinite(error) & (error > 0) & (flux > 0.1)
    if np.count_nonzero(good) < 100:
        return {"error": "too few good pixels"}
    f = flux[good]
    ivar = 1.0 / (error[good] ** 2)

    def chi2_of(p):
        m = _predict(p[0], p[1], p[2])[good]
        r = (f - m)
        return float(np.sum(r * r * ivar))

    # start from the middle of the label box
    x0 = (_LMIN + _LMAX) / 2.0
    bounds = [(float(_LMIN[i]), float(_LMAX[i])) for i in range(3)]
    res = minimize(chi2_of, x0, method="Nelder-Mead",
                   options={"xatol": 1e-2, "fatol": 1e-1, "maxiter": 600})
    # clip into the trained box
    p = np.clip(res.x, _LMIN, _LMAX)
    chi2 = chi2_of(p)
    ndof = int(np.count_nonzero(good)) - 3
    return {
        "teff": round(float(p[0]), 1),
        "logg": round(float(p[1]), 3),
        "feh": round(float(p[2]), 3),
        "chi2": round(float(chi2), 2),
        "reduced_chi2": round(float(chi2 / max(ndof, 1)), 3),
    }


if __name__ == "__main__":
    mcp.run()
