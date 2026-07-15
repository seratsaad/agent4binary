#!/usr/bin/env python3
"""
MCP Server for Gaia DR3 BP/RP (XP) low-resolution spectra.

Retrieves XP spectra via the Gaia archive datalink service. XP spectra are VERY
low resolution (R ~ 50) but they constrain the overall SED and therefore the
flux ratio / temperature contrast of binary components — useful as an independent
cross-check on a spectroscopic SB2 decomposition.

CRITICAL: the full flux/wavelength arrays are NEVER returned to the caller (that
would flood the LLM context). They are cached to data/cache/xp_<source_id>.npz and
only a compact summary (a handful of sampled flux points + the cache path) is
returned. All tools degrade gracefully on network errors.
"""

import os
import warnings
from mcp.server.fastmcp import FastMCP

warnings.filterwarnings("ignore")

mcp = FastMCP("Gaia XP", log_level="WARNING")

# data/cache relative to project root (this file is src/mcp_servers/...).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")


def _get_gaia():
    try:
        from astroquery.gaia import Gaia
        return Gaia, None
    except Exception as e:  # pragma: no cover
        return None, f"astroquery.gaia unavailable: {e}"


def _extract_xp_arrays(table):
    """
    From a datalink XP_SAMPLED astropy Table, pull wavelength (nm) and flux arrays.
    Returns (wavelength_nm, flux, flux_error) as numpy arrays, or (None, None, None).
    """
    import numpy as np

    cols = {c.lower(): c for c in table.colnames}

    def find(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    wcol = find("wavelength", "lambda")
    fcol = find("flux")
    ecol = find("flux_error", "flux_err")
    if wcol is None or fcol is None:
        return None, None, None

    wl = np.asarray(table[wcol], dtype=float)
    fl = np.asarray(table[fcol], dtype=float)
    fe = np.asarray(table[ecol], dtype=float) if ecol else np.full_like(fl, np.nan)

    # Gaia XP_SAMPLED wavelengths are in nm (~336-1020). Leave as-is.
    return wl, fl, fe


@mcp.tool()
def get_xp_spectrum(source_id: int, sampled: bool = True) -> dict:
    """
    Retrieve the Gaia DR3 BP/RP (XP) spectrum for a source via the datalink service.

    With sampled=True (default) retrieves the XP_SAMPLED product: flux (W m^-2 nm^-1)
    vs wavelength (nm) on a fixed grid (~343 points, ~336-1020 nm). With sampled=False
    retrieves XP_CONTINUOUS (the Hermite continuous-representation coefficients).

    XP is VERY low resolution (R ~ 50) — it does not resolve individual lines, but it
    constrains the broadband SED / flux ratio of binary components, a useful independent
    check on a spectroscopic SB2 decomposition.

    The full arrays are cached to data/cache/xp_<source_id>.npz; only a compact summary
    is returned (never the full arrays). Sources lacking an XP spectrum return
    {"available": false, "source_id": ...}.

    Args:
        source_id: Gaia DR3 source_id (integer).
        sampled: True for XP_SAMPLED (flux vs wavelength), False for XP_CONTINUOUS coeffs.

    Returns:
        {source_id, available, retrieval_type, n_points, wl_min_nm, wl_max_nm,
         flux_samples (<=8 (wavelength_nm, flux) points), cache_path} or
        {"available": false, "source_id": ...} / {"error": "..."}.
    """
    import numpy as np

    try:
        sid = int(source_id)
    except Exception:
        return {"error": "source_id must be an integer"}

    Gaia, err = _get_gaia()
    if err:
        return {"error": err}

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"xp_{sid}.npz")
    retrieval_type = "XP_SAMPLED" if sampled else "XP_CONTINUOUS"

    try:
        results = Gaia.load_data(
            ids=[sid],
            data_release="Gaia DR3",
            retrieval_type=retrieval_type,
            data_structure="INDIVIDUAL",
            format="votable",
            verbose=False,
        )
    except Exception as e:
        return {"error": f"datalink retrieval failed: {e}"}

    if not results:
        return {"available": False, "source_id": sid}

    # results is a dict {key: [astropy Table-like products]}. Find the matching one.
    table = None
    for key, val in results.items():
        if retrieval_type.split("_")[1].lower() in key.lower() or "xp" in key.lower():
            prods = val if isinstance(val, (list, tuple)) else [val]
            for p in prods:
                tbl = getattr(p, "to_table", None)
                table = tbl() if callable(tbl) else p
                break
        if table is not None:
            break
    if table is None:
        # fall back to first product
        first = next(iter(results.values()))
        prods = first if isinstance(first, (list, tuple)) else [first]
        p = prods[0]
        tbl = getattr(p, "to_table", None)
        table = tbl() if callable(tbl) else p

    if table is None or len(table) == 0:
        return {"available": False, "source_id": sid}

    if sampled:
        wl, fl, fe = _extract_xp_arrays(table)
        if wl is None:
            return {"error": f"could not parse XP_SAMPLED columns; got {list(table.colnames)}"}

        np.savez_compressed(cache_path, source_id=sid, wavelength_nm=wl, flux=fl,
                            flux_error=fe, retrieval_type=retrieval_type)

        n = len(wl)
        # <=8 evenly spaced sample points across the band.
        idx = np.linspace(0, n - 1, min(8, n)).astype(int)
        samples = [
            {"wavelength_nm": round(float(wl[i]), 2),
             "flux": (None if np.isnan(fl[i]) else float(f"{fl[i]:.4e}"))}
            for i in idx
        ]
        return {
            "source_id": sid,
            "available": True,
            "retrieval_type": retrieval_type,
            "n_points": int(n),
            "wl_min_nm": round(float(np.nanmin(wl)), 2),
            "wl_max_nm": round(float(np.nanmax(wl)), 2),
            "flux_samples": samples,
            "cache_path": cache_path,
        }
    else:
        # XP_CONTINUOUS: cache whatever numeric columns exist, summarize counts.
        saved = {"source_id": sid, "retrieval_type": retrieval_type}
        n_coeff = None
        for c in table.colnames:
            try:
                arr = np.asarray(table[c])
                saved[c] = arr
                if "coefficient" in c.lower() and n_coeff is None:
                    n_coeff = int(np.asarray(arr).reshape(len(table), -1).shape[-1])
            except Exception:
                pass
        np.savez_compressed(cache_path, **{k: v for k, v in saved.items()
                                           if not isinstance(v, str)},
                            source_id=sid)
        return {
            "source_id": sid,
            "available": True,
            "retrieval_type": retrieval_type,
            "columns": list(table.colnames)[:20],
            "n_coefficients": n_coeff,
            "cache_path": cache_path,
            "note": "XP_CONTINUOUS coefficients cached; use XP_SAMPLED for flux-vs-wavelength.",
        }


if __name__ == "__main__":
    mcp.run()
