#!/usr/bin/env python3
"""
MCP Server for APOGEE Data Access

Loads DR17 aspcapStar FITS spectra and exposes them to agents WITHOUT ever
returning the full 8575-pixel arrays. Heavy arrays are cached on disk as .npz
in data/cache/, keyed by 2MASS spectrum id; tool return values are compact
(scalars, short lists, file paths).

FITS layout: HDU1 = flux, HDU2 = error, HDU3 = ASPCAP best-fit model.
Wavelength grid (log-linear) from the HDU1 header:
    wl = 10**(CRVAL1 + CDELT1*arange(NAXIS1))    # 15100.8 - 16999.8 A
"""

import glob
import os
import sys

import numpy as np
from astropy.io import fits
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("APOGEE Data", log_level="WARNING")

# Resolve project paths relative to this file (src/mcp_servers/ -> project root).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
ASPCAP_DIR = os.path.join(_ROOT, "data", "aspcap")
CACHE_DIR = os.path.join(_ROOT, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_FNAME_PREFIX = "aspcapStar-dr17-"

# RAW DR19 product locations, searched in order for load_dr19_spectrum. Each holds
# per-star npz files keyed by sdss_id with {wl, flux_raw, ivar} (written by
# src/download_dr19_raw.py), plus a sibling Parquet table that packs the same dir.
# The Parquet is read via src.data_io.read_spectrum_parquet when the npz is absent.
_DR19_RAW_DIRS = [
    os.path.join(_ROOT, "data", "dr19_raw_sb2"),       # SB2 test set
    os.path.join(_ROOT, "data", "dr19_raw_controls"),  # held-out controls
    os.path.join(_ROOT, "data", "dr19_raw"),           # training pool
]
_DR19_RAW_PARQUETS = [
    os.path.join(_ROOT, "data", "dr19_raw_sb2.parquet"),
    os.path.join(_ROOT, "data", "dr19_raw_controls.parquet"),
    os.path.join(_ROOT, "data", "dr19_raw.parquet"),
]


def _normalize_id(spec_id: str) -> str:
    """Return the canonical 2MASS id (with the leading 2M) from any input form."""
    sid = spec_id.strip()
    if sid.startswith(_FNAME_PREFIX):
        sid = sid[len(_FNAME_PREFIX):]
    if sid.endswith(".fits"):
        sid = sid[: -len(".fits")]
    if not sid.startswith("2M"):
        sid = "2M" + sid
    return sid


def _fits_path(spec_id: str) -> str:
    sid = _normalize_id(spec_id)
    return os.path.join(ASPCAP_DIR, f"{_FNAME_PREFIX}{sid}.fits")


def _cache_path(spec_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{_normalize_id(spec_id)}.npz")


def _wl_grid(header) -> np.ndarray:
    crval1 = float(header["CRVAL1"])
    cdelt1 = float(header["CDELT1"])
    naxis1 = int(header["NAXIS1"])
    return 10.0 ** (crval1 + cdelt1 * np.arange(naxis1))


@mcp.tool()
def list_spectra(limit: int = 20) -> dict:
    """
    List available APOGEE spectrum ids.

    Scans data/aspcap/ for aspcapStar-dr17-*.fits files and returns the parsed
    2MASS ids. Use these ids with load_spectrum / get_window and with the
    binary_model tools.

    Args:
        limit: Maximum number of ids to return (the full count is always given).

    Returns:
        {spec_ids: [...], returned: int, total_count: int}
    """
    files = sorted(glob.glob(os.path.join(ASPCAP_DIR, f"{_FNAME_PREFIX}*.fits")))
    ids = [
        os.path.basename(f)[len(_FNAME_PREFIX):-len(".fits")]
        for f in files
    ]
    limit = max(1, int(limit))
    return {
        "spec_ids": ids[:limit],
        "returned": min(limit, len(ids)),
        "total_count": len(ids),
    }


@mcp.tool()
def load_spectrum(spec_id: str) -> dict:
    """
    Load an APOGEE spectrum and cache its arrays to disk.

    Reads flux (HDU1), error (HDU2) and the best-fit model (HDU3), builds the
    log-linear wavelength grid, and caches wl/flux/error/model to
    data/cache/<spec_id>.npz. The full arrays are NOT returned; only a compact
    summary is. Other tools read the cache by spec_id.

    Args:
        spec_id: 2MASS id (with or without the leading 2M prefix).

    Returns:
        {spec_id, npix, wl_min, wl_max, median_snr, cache_path}
    """
    path = _fits_path(spec_id)
    if not os.path.exists(path):
        return {"error": f"spectrum not found: {spec_id}", "path": path}

    with fits.open(path) as hdul:
        flux = np.asarray(hdul[1].data, dtype=float)
        error = np.asarray(hdul[2].data, dtype=float)
        try:
            model = np.asarray(hdul[3].data, dtype=float)
        except Exception:
            model = np.full_like(flux, np.nan)
        wl = _wl_grid(hdul[1].header)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = flux / error
    good = np.isfinite(snr) & (error > 0) & (flux != 0)
    median_snr = float(np.median(snr[good])) if np.any(good) else float("nan")

    sid = _normalize_id(spec_id)
    cache_path = _cache_path(sid)
    np.savez_compressed(cache_path, wl=wl, flux=flux, error=error, model=model)

    return {
        "spec_id": sid,
        "npix": int(flux.size),
        "wl_min": round(float(wl.min()), 3),
        "wl_max": round(float(wl.max()), 3),
        "median_snr": round(median_snr, 3) if np.isfinite(median_snr) else None,
        "cache_path": cache_path,
    }


def _dr19_cache_path(sdss_id) -> str:
    """Cache path for a RAW DR19 spectrum, keyed by the bare integer sdss_id.

    The binary_model server's SC detector (_load_raw) searches data/cache/ first,
    so writing the raw cache here makes chi2_single_vs_binary(model="dr19_sc")
    resolve the same star by sdss_id without any further argument plumbing.
    """
    return os.path.join(CACHE_DIR, f"{int(sdss_id)}.npz")


def _read_dr19_raw(sdss_id):
    """Return (wl, flux_raw, ivar) for a DR19 star, raising if not found anywhere.

    Resolution order: per-star npz in each raw dir, then the packed Parquet tables
    via src.data_io.read_spectrum_parquet. The RAW flux (flux_raw) and ivar are the
    inputs physics.dr19_sc_single_vs_binary expects -- NO survey continuum.
    """
    sid = int(sdss_id)
    # 1. per-star npz {wl, flux_raw, ivar}
    for d in _DR19_RAW_DIRS:
        p = os.path.join(d, f"{sid}.npz")
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            data = np.load(p)
            if "flux_raw" in data.files and "ivar" in data.files:
                wl = (np.asarray(data["wl"], float) if "wl" in data.files
                      else None)
                return wl, np.asarray(data["flux_raw"], float), \
                    np.asarray(data["ivar"], float)
    # 2. packed Parquet tables (the combined dr19_raw.parquet etc.)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    try:
        from data_io import read_spectrum_parquet
    except Exception as e:  # pragma: no cover - import guard
        raise FileNotFoundError(
            f"no raw npz for sdss_id {sid} and data_io unavailable ({e})")
    for pq_path in _DR19_RAW_PARQUETS:
        if os.path.exists(pq_path):
            try:
                wl, flux, ivar = read_spectrum_parquet(pq_path, sid)
                return wl, flux, ivar
            except KeyError:
                continue
    raise FileNotFoundError(
        f"no RAW DR19 spectrum for sdss_id {sid} in {_DR19_RAW_DIRS} or "
        f"{_DR19_RAW_PARQUETS}")


@mcp.tool()
def load_dr19_spectrum(sdss_id: int) -> dict:
    """
    Load a RAW DR19 APOGEE spectrum by sdss_id and cache flux_raw + ivar.

    This is the loader for the PRODUCTION self-consistent detector
    (physics.dr19_sc_single_vs_binary), which needs RAW flux + ivar (NO survey
    continuum). It resolves the star from data/dr19_raw_sb2/<sdss_id>.npz,
    data/dr19_raw_controls/, data/dr19_raw/, or the packed Parquet tables
    (data/dr19_raw*.parquet via src.data_io.read_spectrum_parquet), then writes the
    raw arrays to data/cache/<sdss_id>.npz as {wl, flux_raw, ivar}.

    The binary_model server's chi2_single_vs_binary(spec_id, model="dr19_sc")
    searches data/cache/ first, so after this call the SAME sdss_id (as a string)
    runs the detector end-to-end over MCP. The full 8575-pixel arrays are NOT
    returned; only a compact summary is.

    Args:
        sdss_id: SDSS-V sdss_id (integer) of the DR19 star.

    Returns:
        {sdss_id, npix, median_snr, n_good, cache_path, source} or {error}.
    """
    try:
        wl, flux_raw, ivar = _read_dr19_raw(sdss_id)
    except FileNotFoundError as e:
        return {"error": str(e)}

    # Median S/N from the RAW flux and ivar (sigma = 1/sqrt(ivar)); pixels with
    # ivar<=0 or non-finite flux are excluded. This is a quick load-time quality cue.
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = 1.0 / np.sqrt(ivar)
        snr = flux_raw * np.sqrt(ivar)
    good = np.isfinite(snr) & (ivar > 0) & np.isfinite(flux_raw)
    median_snr = float(np.median(snr[good])) if np.any(good) else float("nan")

    # If the raw npz had no wavelength grid, fall back to the global APOGEE grid.
    if wl is None or np.size(wl) != np.size(flux_raw):
        wl = 10.0 ** (4.179 + 6e-6 * np.arange(np.size(flux_raw)))

    cache_path = _dr19_cache_path(sdss_id)
    # Write flux_raw + ivar so binary_model._load_raw resolves this star, and keep
    # wl for completeness. NOTE: the SC detector does NOT consume any continuum here.
    np.savez_compressed(cache_path, wl=np.asarray(wl, float),
                        flux_raw=flux_raw, ivar=ivar)

    return {
        "sdss_id": int(sdss_id),
        "npix": int(flux_raw.size),
        "median_snr": round(median_snr, 3) if np.isfinite(median_snr) else None,
        "n_good": int(np.count_nonzero(good)),
        "cache_path": cache_path,
        "source": "dr19_raw",
    }


@mcp.tool()
def get_window(
    spec_id: str,
    wl_lo: float,
    wl_hi: float,
    max_points: int = 16,
) -> dict:
    """
    Return a SMALL downsampled flux excerpt over a wavelength window.

    Lets the agent eyeball spectral features (line depths, asymmetries) without
    flooding context. The window is downsampled to at most max_points samples.
    Requires load_spectrum to have been called first (uses the disk cache).

    Args:
        spec_id: 2MASS id.
        wl_lo: Lower wavelength bound (Angstroms).
        wl_hi: Upper wavelength bound (Angstroms).
        max_points: Maximum number of samples returned (kept small, <=~20).

    Returns:
        {spec_id, wl_lo, wl_hi, n_in_window, wl: [...], flux: [...]}
    """
    cache_path = _cache_path(spec_id)
    if not os.path.exists(cache_path):
        loaded = load_spectrum(spec_id)
        if "error" in loaded:
            return loaded
    data = np.load(cache_path)
    wl = data["wl"]
    flux = data["flux"]

    if wl_hi < wl_lo:
        wl_lo, wl_hi = wl_hi, wl_lo
    mask = (wl >= wl_lo) & (wl <= wl_hi)
    n_in = int(np.count_nonzero(mask))
    if n_in == 0:
        return {
            "spec_id": _normalize_id(spec_id),
            "wl_lo": float(wl_lo),
            "wl_hi": float(wl_hi),
            "n_in_window": 0,
            "wl": [],
            "flux": [],
        }

    wl_w = wl[mask]
    flux_w = flux[mask]
    max_points = max(2, int(max_points))
    if n_in > max_points:
        idx = np.linspace(0, n_in - 1, max_points).round().astype(int)
        wl_w = wl_w[idx]
        flux_w = flux_w[idx]

    return {
        "spec_id": _normalize_id(spec_id),
        "wl_lo": float(wl_lo),
        "wl_hi": float(wl_hi),
        "n_in_window": n_in,
        "wl": [round(float(x), 3) for x in wl_w],
        "flux": [round(float(x), 5) for x in flux_w],
    }


if __name__ == "__main__":
    mcp.run()
