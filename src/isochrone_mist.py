#!/usr/bin/env python3
"""
OUR own MIST v1.2 isochrone interpolation for the binary forward model.

This module REPLACES the ported binspec isochrone neural networks
(NN_Teff2_logg2 + NN_radius) in the PRODUCTION DR19 detection path. It reads
genuine MIST v1.2 isochrones we downloaded from the MIST website and builds a
direct interpolation, so the production path depends on NO binspec neural net for
the q -> secondary map or for stellar radii. The binspec oracle path in
physics.py keeps its vendored nets; this module is the reproduction, not the
reference.

What MIST gives us and how we use it
------------------------------------
A MIST isochrone is a single-age, single-composition locus: for one
log10_isochrone_age_yr and one [Fe/H], it lists, along increasing initial_mass,
the stellar log_Teff, log_g and log_L of a coeval, equal-composition population.
A main-sequence (MS) binary is exactly such a coeval, equal-[Fe/H] pair (the
El-Badry et al. 2018b assumption set: the two stars formed together, so they
share age and composition and differ only in mass). So, given the primary's
(Teff1, logg1) and the system [Fe/H] and age:

  1. locate the primary's initial mass m1 on the (age, [Fe/H]) isochrone by
     matching its (Teff1, logg1),
  2. set the secondary mass m2 = q * m1 (q = m2/m1, the mass ratio),
  3. read the secondary's (Teff2, logg2, L2) at m2 on the SAME isochrone (same
     age, same [Fe/H]) -- this is the coeval / equal-composition constraint,
  4. take both radii from the MIST luminosity-temperature relation,
     R = sqrt(L/Lsun) * (Teffsun/Teff)^2 (Stefan-Boltzmann), which is internally
     consistent with the MIST log_g it also reports.

El-Badry et al. 2018b assumptions: coeval (both components on the SAME isochrone),
equal composition (both share the system [Fe/H]), main sequence (we keep phase-0
rows over the dwarf logg range, the regime the detector operates in).

Source data
-----------
MIST v1.2 (vvcrit0.4), UBVRIplus isochrone+photometry set, from
https://waps.cfa.harvard.edu/MIST/ (redirects to https://mist.science/). We
extracted four per-[Fe/H] .iso.cmd files ([Fe/H] = -0.50, 0.00, +0.25, +0.50;
afe = +0.0) into data/mist_v1.2/. The UBVRIplus set lists the SAME isochrone rows
as the theoretical "basic" set but ALSO carries synthetic photometry, including the
2MASS J/H/Ks absolute magnitudes, which is what the binary composite needs for the
H-band flux ratio (the 2MASS H carries the real bolometric correction, so a cool
secondary's molecular H-band absorption is included -- a blackbody/Planck weight
cannot capture this).

The columns read from a .iso.cmd data row (0-based index): 1 log10_isochrone_age_yr,
2 initial_mass (Msun), 4 log_Teff, 5 log_g, 6 log_L, 14 2MASS_H (absolute mag),
33 phase (0 = main sequence). At import we parse these into a compact MS grid
(models/mist_ms_grid.npz), so the .cmd files are not needed at runtime (see
data/mist_v1.2/PROVENANCE.md). The detailed format is 107 isochrone blocks
(log age 5.0 .. 10.3) of 34 columns.

Older grids (built before the H-band fix) carry only the theoretical columns from
the "basic" .iso set and lack the 2MASS H magnitude. _build_grid below reads the
.cmd files; if the grid on disk is missing the abs_H column, it is rebuilt.
"""

import os

import numpy as np

# --------------------------------------------------------------------------- #
# Paths. This file lives in src/, so the project root is one level up.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

# The four extracted MIST v1.2 .iso.cmd files (one per [Fe/H]) and their [Fe/H].
# These are the UBVRIplus (isochrone+photometry) set: same rows as the theoretical
# "basic" .iso, plus 2MASS J/H/Ks. The H magnitude is what the binary composite
# weights each component by (see physics.compose_binary5_sc).
_MIST_DIR = os.path.join(_ROOT, "data", "mist_v1.2")
_MIST_ISO_FILES = {
    -0.50: "MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd",
    0.00: "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd",
    0.25: "MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd",
    0.50: "MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd",
}

# The compact MS grid we build once and reload thereafter. It holds, for every
# (feh, log_age, MS row), the columns we need. Storing only MS rows keeps it tiny.
_GRID_PATH = os.path.join(_ROOT, "models", "mist_ms_grid.npz")

# 0-based column indices inside a MIST .iso.cmd data row (see module docstring).
# The UBVRIplus .cmd layout is: 0 EEP, 1 log10_isochrone_age_yr, 2 initial_mass,
# 3 star_mass, 4 log_Teff, 5 log_g, 6 log_L, 7 [Fe/H]_init, 8 [Fe/H], 9..13 Bessell
# UBVRI, 14 2MASS_J, 15 2MASS_H, 16 2MASS_Ks, ... , 33 phase (34 columns total).
_COL_AGE = 1
_COL_MASS = 2
_COL_LOGT = 4
_COL_LOGG = 5
_COL_LOGL = 6
_COL_ABSH = 15            # 2MASS_H absolute magnitude
_COL_PHASE = 33

# MIST main-sequence phase code. phase == 0 is the main sequence in MIST v1.2
# (phase -1 is pre-MS; >=2 are subgiant / RGB / later). We keep phase 0 only.
_MS_PHASE = 0.0

# Dwarf logg window we keep when building the grid. The detector clamps primaries
# to logg in [3.5, 5.0]; we keep a slightly wider MS window so interpolation at
# the edges is interior, not extrapolated.
_LOGG_LO = 3.4
_LOGG_HI = 5.4

# Solar effective temperature (K), the IAU nominal value, used in the
# Stefan-Boltzmann radius relation R = sqrt(L/Lsun) * (Teffsun/Teff)^2.
_TEFF_SUN = 5772.0


# --------------------------------------------------------------------------- #
# Build the compact MS grid from the raw .iso files (runs once if missing).
# --------------------------------------------------------------------------- #
def _parse_iso_file(path):
    """Parse one MIST .iso.cmd file -> (age, mass, logT, logg, logL, absH, phase).

    Every non-comment, non-blank line is a stellar point that already carries its
    own log10_isochrone_age_yr in column 1, so we do not need to track the
    per-isochrone block boundaries: we just read all data rows and slice the
    columns we want. absH is the 2MASS H absolute magnitude. Returns seven 1-D
    float arrays of equal length.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            # MIST comments start with '#'; skip them and blank lines.
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            rows.append(line.split())
    a = np.array(rows, dtype=np.float64)
    return (a[:, _COL_AGE], a[:, _COL_MASS], a[:, _COL_LOGT],
            a[:, _COL_LOGG], a[:, _COL_LOGL], a[:, _COL_ABSH],
            a[:, _COL_PHASE])


def _build_grid():
    """Parse the four .iso files into the compact MS grid and save it.

    Keeps only main-sequence rows (phase == 0) inside the dwarf logg window, and
    stores, per row: feh, log_age, initial_mass, log_Teff, log_g, log_L, abs_H
    (2MASS H absolute magnitude). The result is a flat table (a few tens of
    thousands of rows) that we reload at import; the raw .cmd files are not read
    again at runtime.
    """
    feh_all, age_all, mass_all = [], [], []
    logt_all, logg_all, logl_all, absh_all = [], [], [], []
    for feh, fname in _MIST_ISO_FILES.items():
        path = os.path.join(_MIST_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "MIST .iso.cmd file missing: %s. Download "
                "MIST_v1.2_vvcrit0.4_UBVRIplus.txz from the MIST website and "
                "extract the four feh .iso.cmd files into data/mist_v1.2/." % path)
        age, mass, logt, logg, logl, absh, phase = _parse_iso_file(path)
        # Keep main-sequence dwarf rows only.
        keep = (phase == _MS_PHASE) & (logg >= _LOGG_LO) & (logg <= _LOGG_HI)
        n = int(keep.sum())
        feh_all.append(np.full(n, float(feh)))
        age_all.append(age[keep])
        mass_all.append(mass[keep])
        logt_all.append(logt[keep])
        logg_all.append(logg[keep])
        logl_all.append(logl[keep])
        absh_all.append(absh[keep])
    grid = {
        "feh": np.concatenate(feh_all),
        "log_age": np.concatenate(age_all),     # log10(age / yr)
        "mass": np.concatenate(mass_all),        # initial_mass, Msun
        "log_teff": np.concatenate(logt_all),    # log10(Teff / K)
        "log_g": np.concatenate(logg_all),       # cgs dex
        "log_l": np.concatenate(logl_all),       # log10(L / Lsun)
        "abs_H": np.concatenate(absh_all),       # 2MASS H absolute magnitude
    }
    os.makedirs(os.path.dirname(_GRID_PATH), exist_ok=True)
    np.savez_compressed(_GRID_PATH, **grid)
    return grid


# Load the compact grid, building it on first run. A grid built before the
# H-band fix lacks the abs_H column; rebuild from the .cmd files in that case
# (raises a clear FileNotFoundError if they are not present locally).
if os.path.exists(_GRID_PATH):
    _G = np.load(_GRID_PATH)
    _GRID = {k: _G[k].astype(np.float64) for k in _G.files}
    _G.close()
    if "abs_H" not in _GRID:
        _GRID = _build_grid()
else:
    _GRID = _build_grid()

_FEH = _GRID["feh"]
_LOGAGE = _GRID["log_age"]
_MASS = _GRID["mass"]
_LOGT = _GRID["log_teff"]
_LOGG = _GRID["log_g"]
_LOGL = _GRID["log_l"]
_ABSH = _GRID["abs_H"]

# The discrete nodes present in the grid, used to snap requests to a real track.
_FEH_NODES = np.unique(_FEH)
_LOGAGE_NODES = np.unique(np.round(_LOGAGE, 6))

# The [Fe/H] span we built (for clamping a request into the grid we actually have).
_FEH_MIN = float(_FEH_NODES.min())
_FEH_MAX = float(_FEH_NODES.max())


def _snap_feh(feh):
    """Snap a requested [Fe/H] to the nearest grid node we built."""
    return float(_FEH_NODES[int(np.argmin(np.abs(_FEH_NODES - feh)))])


def _snap_logage(feh_node, age_gyr):
    """Snap a requested age (Gyr) to the nearest log-age that EXISTS at feh_node."""
    log_age = np.log10(max(age_gyr, 1e-3) * 1e9)  # Gyr -> log10(yr)
    sel = np.abs(_FEH - feh_node) < 1e-9
    ages_here = np.unique(np.round(_LOGAGE[sel], 6))
    return float(ages_here[int(np.argmin(np.abs(ages_here - log_age)))])


def _track(feh_node, logage_node):
    """Return one isochrone track (mass-sorted) at (feh_node, logage_node).

    Picks the rows on exactly that (feh, log_age) node, sorts by initial_mass
    (the MS is monotonic in mass), and returns the mass / Teff / logg / L / abs_H
    arrays. abs_H is the 2MASS H absolute magnitude (carries the real bolometric
    correction). The MS is single-valued in mass, so np.interp on these is well
    defined.
    """
    sel = (np.abs(_FEH - feh_node) < 1e-9) & (np.abs(_LOGAGE - logage_node) < 1e-6)
    m = _MASS[sel]
    t = 10.0 ** _LOGT[sel]              # Teff, K
    g = _LOGG[sel]                       # logg, cgs dex
    L = 10.0 ** _LOGL[sel]               # L / Lsun
    h = _ABSH[sel]                       # 2MASS H absolute magnitude
    order = np.argsort(m)
    return m[order], t[order], g[order], L[order], h[order]


def _radius_from_LT(L_lsun, teff_k):
    """Stefan-Boltzmann radius: R/Rsun = sqrt(L/Lsun) * (Teffsun/Teff)^2.

    L = 4 pi R^2 sigma Teff^4, so for two stars in solar units the constants
    cancel and R/Rsun = sqrt(L/Lsun) / (Teff/Teffsun)^2. This is exactly the
    radius implied by the MIST log_L and log_Teff the isochrone reports, so it is
    internally consistent with the MIST track (no extra calibration).
    """
    return float(np.sqrt(L_lsun) * (_TEFF_SUN / teff_k) ** 2)


def _mass_from_teff_logg(m, t, g, teff, logg):
    """Find the initial mass on a track whose (Teff, logg) matches (teff, logg).

    The MS is monotonic in mass, so we densely resample the track in mass and pick
    the mass whose (Teff, logg) is closest to the requested primary in a SCALED
    metric (Teff scaled by 500 K, logg by 0.3 dex, so the two labels weigh
    comparably). This mirrors the primary-mass recovery the project already used
    for the raw MIST table, but here on OUR own isochrone track.
    """
    mm = np.linspace(m.min(), m.max(), 4000)
    tt = np.interp(mm, m, t)
    gg = np.interp(mm, m, g)
    dt = (tt - teff) / 500.0
    dg = (gg - logg) / 0.3
    d2 = dt * dt + dg * dg
    return float(mm[int(np.argmin(d2))])


# --------------------------------------------------------------------------- #
# Public API used by the production path (physics.secondary_from_q /
# physics.radius wiring). These names are the OUR-own replacements for the
# binspec NN_Teff2_logg2 + NN_radius forward passes.
# --------------------------------------------------------------------------- #
def radius_ours(teff, logg, feh, age_gyr=4.0):
    """OUR MIST radius (Rsun) for a single MS star at (Teff, logg, [Fe/H]).

    Snaps [Fe/H] and age to the nearest grid nodes (default 4.0 Gyr, the same
    representative MS age as secondary_from_q_ours; on the lower MS the radius is
    nearly age-independent), finds the mass that matches (Teff, logg), reads the
    track's log_L and log_Teff at that mass, and returns the Stefan-Boltzmann
    radius. Replaces the binspec NN_radius forward pass in the production path.
    """
    feh_n = _snap_feh(feh)
    logage_n = _snap_logage(feh_n, age_gyr)
    m, t, g, L, _h = _track(feh_n, logage_n)
    m_match = _mass_from_teff_logg(m, t, g, teff, logg)
    teff_m = float(np.interp(m_match, m, t))
    L_m = float(np.interp(m_match, m, L))
    return _radius_from_LT(L_m, teff_m)


def secondary_from_q_ours(teff1, logg1, feh, q, age_gyr=4.0):
    """OUR MIST q -> secondary map: (Teff1, logg1, [Fe/H], q, age) -> labels + radii.

    The default age is 4.0 Gyr, a representative MS age for the APOGEE dwarf
    sample. The binary signal is driven by the flux ratio R2^2 / R1^2; at 4 Gyr
    OUR (R2 / R1)^2 over q tracks the binspec NN this module replaces and the SC
    detector reproduces the binspec-NN completeness on the 25 SB2 / 25 control
    verify set. Older isochrones swell R1 and suppress the flux ratio; younger ones
    raise contamination. A solar primary recovers m1 ~ 1.0 Msun at 4 Gyr.

    Coeval, equal-composition MS pair (El-Badry et al. 2018b):
      1. snap [Fe/H] and age to the nearest MIST grid nodes,
      2. find the primary mass m1 on that single isochrone from (Teff1, logg1),
      3. m2 = q * m1 (the mass ratio definition),
      4. read the secondary (Teff2, logg2, L2) at m2 on the SAME isochrone (same
         age, same [Fe/H]) -- the coeval / equal-composition constraint,
      5. R1, R2 from the MIST L-Teff relation (Stefan-Boltzmann).
    The secondary is the lower-mass star, so it is forced no hotter and no less
    compact than the primary (the MS ordering), matching the binspec convention.

    The q = 1 / equal-mass limit is EXACT: we return the REQUESTED primary labels
    and R1 == R2, so the binary at q = 1 reduces to the single star at the same
    labels (the identity the detector's binary-nests-single floor relies on).

    Returns (teff2, logg2, R2_rsun, R1_rsun), the SAME tuple order as the binspec
    physics.secondary_from_q it replaces.
    """
    q = float(np.clip(q, 1e-3, 1.0))
    feh_n = _snap_feh(feh)
    logage_n = _snap_logage(feh_n, age_gyr)
    m, t, g, L, _h = _track(feh_n, logage_n)

    # The primary's mass and its OWN radius (from the matched track point, so the
    # q = 1 limit is self-consistent with the secondary read off the same track).
    m1 = _mass_from_teff_logg(m, t, g, teff1, logg1)
    teff1_track = float(np.interp(m1, m, t))
    L1 = float(np.interp(m1, m, L))
    R1 = _radius_from_LT(L1, teff1_track)

    # q = 1 identity: the equal-mass binary IS the single star. Return the
    # REQUESTED primary labels (not the track-snapped ones) and R2 == R1 so the
    # composite at q = 1 equals the single-star model at the requested labels.
    if q >= 0.995:
        return float(teff1), float(logg1), R1, R1

    # Secondary mass and its labels / radius, read off the SAME isochrone.
    m2 = float(np.clip(q * m1, m.min(), m.max()))
    teff2 = float(np.interp(m2, m, t))
    logg2 = float(np.interp(m2, m, g))
    L2 = float(np.interp(m2, m, L))
    R2 = _radius_from_LT(L2, teff2)

    # Enforce the MS ordering (lower-mass secondary is cooler and more compact),
    # the same guard binspec's force_lower_teff applies, for stability.
    if teff2 > teff1:
        teff2 = float(teff1)
    if logg2 < logg1:
        logg2 = float(logg1)
    return teff2, logg2, R2, R1


def primary_mass_ours(teff1, logg1, feh, age_gyr=4.0):
    """OUR MIST primary mass (Msun) from (Teff1, logg1, [Fe/H], age).

    Same matching as secondary_from_q_ours step 2, exposed for reporting / the q
    floor. Reads the mass off OUR isochrone track (no binspec net, no separate
    MIST single-star table).
    """
    feh_n = _snap_feh(feh)
    logage_n = _snap_logage(feh_n, age_gyr)
    m, t, g, _L, _h = _track(feh_n, logage_n)
    return _mass_from_teff_logg(m, t, g, teff1, logg1)


def mh_from_q_ours(teff1, logg1, feh, q, age_gyr=4.0):
    """OUR MIST 2MASS H absolute magnitudes (M_H1, M_H2) for the binary pair.

    Same coeval, equal-[Fe/H] MS construction as secondary_from_q_ours: locate the
    primary mass m1 from (Teff1, logg1) on the (age, [Fe/H]) isochrone, set
    m2 = q * m1, and read each component's 2MASS H absolute magnitude off the SAME
    isochrone. M_H carries the real bolometric correction, so a cool secondary's
    H-band molecular absorption is included -- the H-band LUMINOSITY ratio of the
    pair is 10^(-0.4 (M_H2 - M_H1)), which is exactly the per-component flux weight
    the binary composite needs (it folds in BOTH the emitting area and the true
    surface brightness, replacing the separate R^2 x continuum-level weighting).

    The q = 1 limit is EXACT: at q >= 0.995 we return M_H2 == M_H1 (the primary's
    own value), so the H-band weight ratio is 1 and the binary reduces to the
    single star (the identity the detector relies on).

    Returns (M_H1, M_H2) in magnitudes (smaller = brighter).
    """
    q = float(np.clip(q, 1e-3, 1.0))
    feh_n = _snap_feh(feh)
    logage_n = _snap_logage(feh_n, age_gyr)
    m, t, g, _L, h = _track(feh_n, logage_n)

    m1 = _mass_from_teff_logg(m, t, g, teff1, logg1)
    MH1 = float(np.interp(m1, m, h))

    # q = 1 identity: equal-mass pair -> equal H magnitude -> weight ratio 1.
    if q >= 0.995:
        return MH1, MH1

    m2 = float(np.clip(q * m1, m.min(), m.max()))
    MH2 = float(np.interp(m2, m, h))
    # The secondary is the lower-mass star, so it cannot be BRIGHTER (smaller M_H)
    # than the primary; clamp for stability at the MS turn, mirroring the Teff/logg
    # MS-ordering guard in secondary_from_q_ours.
    if MH2 < MH1:
        MH2 = MH1
    return MH1, MH2
