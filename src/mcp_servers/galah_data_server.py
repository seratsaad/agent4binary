#!/usr/bin/env python3
"""
MCP Server for GALAH Data Access (survey-portability demonstration).

This mirrors apogee_data_server.py for the GALAH/HERMES optical survey. It is the
worked example behind the paper's claim that the MCP + Skill stack re-hosts a new
survey by swapping only two of the nine servers: this data server and the
single-star spectral model server. The survey-agnostic servers (binary_model with
the El-Badry Eq. 2 composite + f_imp + Table B1, isochrone, doppler, broadening,
gaia_*) and the Skill's operating decisions are reused unchanged.

GALAH DR3 (Buder et al. 2021) is taken with HERMES on the AAT: four optical arms
(blue 4713-4903, green 5648-5873, red 6478-6737, IR 7585-7887 A) at R ~ 28000,
delivered as reduced 1D spectra through DataCentral. Like apogee_data, tool
returns are COMPACT (scalars, short lists, cache paths); heavy flux arrays are
cached on disk and never returned inline.

Real DR3 ingestion (DataCentral FITS -> {wl, flux, ivar}) is a drop-in for
_load_spectrum below; the demonstration path serves a payne-zero-synthesized
green-arm spectrum so the same downstream servers and Skill can be exercised on a
GALAH-range input end to end.
"""
import os
import numpy as np

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("GALAH Data", log_level="WARNING")
except Exception:  # mcp not installed in this environment; the tools still import
    mcp = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
CACHE_DIR = os.path.join(_ROOT, "data", "galah_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# GALAH / HERMES four arms (Angstrom) and nominal resolving power.
GALAH_ARMS = {
    "blue":  (4713.0, 4903.0),
    "green": (5648.0, 5873.0),
    "red":   (6478.0, 6737.0),
    "ir":    (7585.0, 7887.0),
}
GALAH_R = 28000.0


def _cache_path(source_id, arm):
    return os.path.join(CACHE_DIR, "galah_%s_%s.npz" % (str(source_id), arm))


def _load_spectrum(source_id, arm):
    """Return (wl, flux, ivar) for a GALAH source/arm.

    Drop-in point for real DR3 ingestion (DataCentral FITS). The demonstration
    path reads a cached npz (e.g. the payne-zero green-arm spectrum) so the rest
    of the stack can run on a GALAH-range input.
    """
    p = _cache_path(source_id, arm)
    if not os.path.exists(p):
        raise FileNotFoundError(
            "no cached GALAH spectrum for %s/%s; wire DR3 ingestion or cache a "
            "spectrum at %s" % (source_id, arm, p))
    z = np.load(p)
    return z["wl"], z["flux"], z["ivar"]


def _survey_info_impl():
    return {
        "survey": "GALAH DR3",
        "instrument": "HERMES / AAT (2dF)",
        "resolving_power": GALAH_R,
        "arms": {k: {"wl_start_A": v[0], "wl_end_A": v[1]}
                 for k, v in GALAH_ARMS.items()},
        "reference": "Buder et al. 2021; binary method Traven et al. 2020",
    }


def _load_impl(source_id, arm):
    if arm not in GALAH_ARMS:
        return {"error": "unknown arm %r; choose from %s"
                % (arm, list(GALAH_ARMS))}
    try:
        wl, flux, ivar = _load_spectrum(source_id, arm)
    except FileNotFoundError as e:
        return {"error": str(e)}
    good = np.isfinite(flux) & (ivar > 0)
    return {
        "source_id": str(source_id), "arm": arm,
        "n_pixels": int(len(wl)),
        "wl_start_A": float(np.nanmin(wl)), "wl_end_A": float(np.nanmax(wl)),
        "resolving_power": GALAH_R,
        "median_snr": float(np.nanmedian(np.sqrt(ivar[good]) * flux[good]))
                      if good.any() else 0.0,
        "cache_path": _cache_path(source_id, arm),
    }


if mcp is not None:
    @mcp.tool()
    def survey_info() -> dict:
        """GALAH/HERMES instrument description: arms, wavelength ranges, R."""
        return _survey_info_impl()

    @mcp.tool()
    def load_spectrum(source_id: str, arm: str = "green") -> dict:
        """Load a GALAH spectrum (compact metadata + a cache path to {wl,flux,ivar};
        heavy arrays are never returned inline, exactly as apogee_data does)."""
        return _load_impl(source_id, arm)

    if __name__ == "__main__":
        mcp.run()
