#!/usr/bin/env python3
"""
MCP Server for MIST Isochrones (binary-star disentangling support).

In El-Badry et al. 2018b the binary flux ratio is fixed by stellar physics: given
a primary and a mass ratio q = m2/m1, an isochrone gives each component's
(Teff, logg, R), and the H-band flux ratio follows from R^2 times surface flux.
This server exposes that mapping so the forward model can convert q <-> flux ratio
and derive the secondary's (Teff, logg) labels for the Payne.

q -> secondary map: physics.secondary_from_q, which is OUR OWN MIST v1.2
interpolation (src/isochrone_mist.py), the same map the production forward model
uses. The ported binspec radius net (models/binspec_NN_radius.npz) and flux net
still back star_from_mass and the H-band surface flux below. Primary mass is read
from the raw MIST single-star table (data/binspec_MIST_single_stars.npz).

The H-band flux ratio reuses the ported binspec flux net (physics._flux_net_predict
-> emergent surface flux), NOT a blackbody, so the q <-> flux-ratio bookkeeping
here is consistent with physics.compose_binary.
"""

import os
import sys

import numpy as np
from mcp.server.fastmcp import FastMCP

# Import the shared science module. physics owns the q -> secondary map (OUR MIST
# interpolation), the ported binspec radius / flux nets, and the MIST table; this
# server is a thin wrapper, so its outputs match the spectral forward model.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import physics  # noqa: E402

mcp = FastMCP("Isochrone", log_level="WARNING")

# APOGEE H-band central wavelength (microns), kept for reporting only.
H_BAND_UM = 1.6


def _hband_surface_flux(teff, logg, feh):
    """Mean emergent H-band SURFACE flux from the ported binspec flux net.

    Reuses physics._flux_net_predict (the R^2 surface-flux path already in the
    spectral model) and averages it over the binspec H-band grid. This is the
    surface brightness that, weighted by R^2, sets the binary flux ratio. Using
    the trained flux net (not a blackbody) keeps this consistent with
    physics.compose_binary, per the project requirement.
    """
    flux = physics._flux_net_predict(teff, logg, feh)
    return float(np.mean(flux))


@mcp.tool()
def star_from_mass(initial_mass: float, age_gyr: float, feh: float) -> dict:
    """
    Interpolate the MIST single-star table for one star.

    Reads (Teff, logg, R) at the requested mass off the raw MIST single-star
    table that backs physics.primary_mass: snap [Fe/H] and age to the nearest
    table nodes, then interpolate the mass track. Radius comes from the ported
    binspec MIST radius network at the interpolated (Teff, logg, [Fe/H]).

    Args:
        initial_mass: Initial stellar mass in Msun.
        age_gyr: Age in Gyr.
        feh: Metallicity [Fe/H].

    Returns:
        {teff, logg, radius_rsun} or {error}.
    """
    # Snap [Fe/H] to the nearest MIST node, then snap age to one that exists
    # there (the table is sparse in the feh x age plane).
    feh_nodes = physics._MIST_FEH_NODES
    feh_n = feh_nodes[np.argmin(np.abs(feh_nodes - feh))]
    feh_sel = np.abs(physics._MIST_FEH - feh_n) < 1e-9
    ages_here = np.unique(physics._MIST_AGE[feh_sel])
    age_n = ages_here[np.argmin(np.abs(ages_here - age_gyr))]
    sel = feh_sel & (np.abs(physics._MIST_AGE - age_n) < 1e-9)

    mass = physics._MIST_MASS[sel]
    teff = physics._MIST_TEFF[sel]
    logg = physics._MIST_LOGG[sel]
    order = np.argsort(mass)
    mass, teff, logg = mass[order], teff[order], logg[order]
    if initial_mass < mass.min() or initial_mass > mass.max():
        return {"error": "mass out of MIST range (%.2f..%.2f Msun) at feh=%.2f age=%.1f Gyr"
                % (mass.min(), mass.max(), feh_n, age_n)}

    t = float(np.interp(initial_mass, mass, teff))
    lg = float(np.interp(initial_mass, mass, logg))
    # Radius from the ported binspec MIST radius network (same one physics uses).
    r = physics.get_radius_NN(t, lg, feh_n)
    return {
        "teff": round(t, 1),
        "logg": round(lg, 4),
        "radius_rsun": round(r, 4),
    }


@mcp.tool()
def secondary_labels(
    teff1: float,
    logg1: float,
    feh: float,
    q: float,
    age_gyr: float = 5.0,
) -> dict:
    """
    Derive the secondary's stellar labels from the primary and mass ratio.

    Calls physics.secondary_from_q (OUR MIST v1.2 interpolation), mapping
    (Teff1, logg1, [Fe/H], q) -> (Teff2, logg2, R2, R1) for an equal-age,
    equal-[Fe/H] main-sequence pair. The primary and secondary masses are read
    from the raw MIST single-star table for reporting. This is the same map the
    spectral forward model uses, so the secondary labels feed the Payne
    self-consistently.

    Args:
        teff1: Primary effective temperature (K).
        logg1: Primary surface gravity (dex).
        feh: Metallicity [Fe/H] (shared by both components).
        q: Mass ratio m2/m1 (0 < q <= 1).
        age_gyr: System age in Gyr (default 5.0). Selects the MIST isochrone
            (snapped to the nearest log-age node) and the primary-mass lookup.

    Returns:
        {teff2, logg2, radius1_rsun, radius2_rsun, mass1_msun, mass2_msun}
        or {error}.
    """
    if not (0 < q <= 1.0):
        return {"error": "q must be in (0, 1]"}
    # Secondary labels and both radii from OUR MIST v1.2 interpolation.
    teff2, logg2, r2, r1 = physics.secondary_from_q(teff1, logg1, feh, q, age_gyr)
    # Masses for reporting, from the raw MIST single-star table.
    m1 = physics.primary_mass(teff1, logg1, feh, age_gyr)
    m2 = q * m1
    return {
        "teff2": round(float(teff2), 1),
        "logg2": round(float(logg2), 4),
        "radius1_rsun": round(float(r1), 4),
        "radius2_rsun": round(float(r2), 4),
        "mass1_msun": round(float(m1), 4),
        "mass2_msun": round(float(m2), 4),
    }


@mcp.tool()
def h_band_flux_ratio(
    teff1: float,
    logg1: float,
    radius1_rsun: float,
    teff2: float,
    logg2: float,
    radius2_rsun: float,
    feh: float = 0.0,
) -> dict:
    """
    Compute the secondary/primary flux ratio f2/f1 in the APOGEE H band.

    f2/f1 = (R2/R1)^2 * SB(Teff2, logg2) / SB(Teff1, logg1), where SB is the
    emergent H-band SURFACE flux from the ported binspec flux net (the R^2
    surface-flux path already in physics), NOT a blackbody. This keeps the
    q <-> flux-ratio bookkeeping consistent with physics.compose_binary.

    Args:
        teff1: Primary Teff (K).
        logg1: Primary logg (dex).
        radius1_rsun: Primary radius (Rsun).
        teff2: Secondary Teff (K).
        logg2: Secondary logg (dex).
        radius2_rsun: Secondary radius (Rsun).
        feh: Metallicity [Fe/H], shared by both components (default 0.0).

    Returns:
        {flux_ratio, lambda_um} or {error}.
    """
    if min(teff1, teff2, radius1_rsun, radius2_rsun) <= 0:
        return {"error": "teff and radius must be positive"}
    area_ratio = (radius2_rsun / radius1_rsun) ** 2
    sb_ratio = (_hband_surface_flux(teff2, logg2, feh)
                / _hband_surface_flux(teff1, logg1, feh))
    fr = float(area_ratio * sb_ratio)
    return {"flux_ratio": round(fr, 5), "lambda_um": H_BAND_UM}


@mcp.tool()
def flux_ratio_from_q(
    teff1: float,
    logg1: float,
    feh: float,
    q: float,
    age_gyr: float = 5.0,
) -> dict:
    """
    Convenience: secondary labels + H-band flux ratio in one call.

    Runs physics.secondary_from_q for the secondary labels and both radii, then
    forms the H-band flux ratio from the ported binspec surface flux weighted by
    R^2. Identical physics to the spectral forward model.

    Args:
        teff1: Primary Teff (K).
        logg1: Primary logg (dex).
        feh: Metallicity [Fe/H].
        q: Mass ratio m2/m1 (0 < q <= 1).
        age_gyr: System age in Gyr (default 5.0).

    Returns:
        {q, teff2, logg2, flux_ratio} or {error}.
    """
    if not (0 < q <= 1.0):
        return {"error": "q must be in (0, 1]"}
    teff2, logg2, r2, r1 = physics.secondary_from_q(teff1, logg1, feh, q, age_gyr)
    fr = h_band_flux_ratio(teff1, logg1, r1, teff2, logg2, r2, feh)
    if "error" in fr:
        return fr
    return {
        "q": q,
        "teff2": round(float(teff2), 1),
        "logg2": round(float(logg2), 4),
        "flux_ratio": fr["flux_ratio"],
    }


if __name__ == "__main__":
    mcp.run()
