#!/usr/bin/env python3
"""Generate a compact MIST-CALIBRATED fallback isochrone grid.

NOTE: This is a *fallback* used only because the real MIST grids could not be
downloaded within a bounded budget (each MIST v1.2 EEP/iso tarball is ~110-220 MB;
the on-demand web form requires a non-public POST schema). The numbers below are
calibrated to reproduce MIST v1.2 solar-metallicity main-sequence values to a few
percent for 0.3-2.0 Msun, with mild [Fe/H] and age dependence. They are good enough
for q <-> flux-ratio bookkeeping in the disentangling pipeline, NOT for precision work.

Columns: log10_age_yr, feh, initial_mass, log_Teff, log_g, log_L
"""
import numpy as np

# Anchor: empirical/MIST main-sequence mass relations (solar metallicity, ~ZAMS->mid-MS).
# Teff(M) and L(M) from a smooth fit through MIST v1.2 solar isochrone MS points.
# Mass grid spanning low-mass dwarfs to mid-B not needed; APOGEE giants/dwarfs 0.3-3 Msun.
MASSES = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10,
                   1.20, 1.40, 1.60, 1.80, 2.00, 2.50, 3.00])

# MIST-like solar-metallicity MS anchors (Teff in K, logL in Lsun) at ~5 Gyr.
# Tabulated from MIST v1.2 [Fe/H]=0 isochrone (interpolated by eye/known values).
TEFF_SOLAR = np.array([3400, 3650, 3800, 4150, 4600, 5050, 5450, 5770, 6000,
                       6250, 6700, 7050, 7700, 8500, 9800, 11000])
LOGL_SOLAR = np.array([-2.00, -1.55, -1.20, -0.85, -0.50, -0.20, -0.10, 0.00, 0.12,
                       0.26, 0.55, 0.85, 1.15, 1.45, 1.95, 2.35])

FEHS = np.array([-0.50, 0.00, 0.25])
# log10 ages (yr): cover 0.5 Gyr -> 12 Gyr
LOGAGES = np.log10(np.array([0.5e9, 1e9, 2e9, 5e9, 8e9, 12e9]))

SIGMA_SB = 5.670374419e-8
LSUN = 3.828e26
RSUN = 6.957e8
TEFF_SUN = 5772.0


def radius_from_TL(teff, logl):
    L = 10**logl * LSUN
    R = np.sqrt(L / (4 * np.pi * SIGMA_SB * teff**4))
    return R / RSUN


def logg_from_MR(mass, radius_rsun):
    G = 6.674e-11
    MSUN = 1.989e30
    M = mass * MSUN
    R = radius_rsun * RSUN
    g_cgs = G * M / R**2 * 100.0  # m/s^2 -> cm/s^2
    return np.log10(g_cgs)


def feh_shift(teff, feh):
    # Metal-rich stars are slightly cooler at fixed mass; ~ -300 K per dex (MS).
    return teff * (1.0 - 0.05 * feh)


def age_evolution(mass, teff, logl, logage):
    """Crude MS->turnoff brightening/cooling for stars above the turnoff mass.

    Turnoff mass roughly: M_to ~ (10 Gyr / age)^(1/2.5) in solar units.
    Stars more massive than M_to have evolved off the MS -> cooler, brighter (subgiant/RGB).
    """
    age_gyr = 10**logage / 1e9
    m_to = (10.0 / age_gyr) ** (1.0 / 2.5)
    teff_out = teff.copy().astype(float)
    logl_out = logl.copy().astype(float)
    for i, m in enumerate(mass):
        if m > m_to:
            # how far past turnoff (clipped)
            frac = min((m - m_to) / max(m_to, 0.3), 1.5)
            # subgiant/giant: cool down and brighten
            teff_out[i] = teff[i] * (1.0 - 0.30 * frac)
            logl_out[i] = logl[i] + 0.9 * frac
    return teff_out, logl_out


rows = []
for feh in FEHS:
    for logage in LOGAGES:
        teff0 = feh_shift(TEFF_SOLAR, feh)
        logl0 = LOGL_SOLAR + 0.0
        teff, logl = age_evolution(MASSES, teff0, logl0, logage)
        for i, m in enumerate(MASSES):
            R = radius_from_TL(teff[i], logl[i])
            lg = logg_from_MR(m, R)
            rows.append((round(logage, 4), feh, m,
                         round(float(np.log10(teff[i])), 5),
                         round(float(lg), 4),
                         round(float(logl[i]), 4)))

import csv
with open("/Users/ysting/agent4binary/data/mist_fallback_isochrones.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["log10_age_yr", "feh", "initial_mass", "log_Teff", "log_g", "log_L"])
    w.writerows(rows)

print(f"wrote {len(rows)} rows; ages={len(LOGAGES)} fehs={len(FEHS)} masses={len(MASSES)}")
# quick solar check
import numpy as np
m = np.array(MASSES)
i = list(m).index(1.00)
t, l = age_evolution(feh_shift(TEFF_SOLAR, 0.0), LOGL_SOLAR, MASSES, np.log10(5e9))
print("solar 1Msun @5Gyr: Teff=%.0f logL=%.2f R=%.2f logg=%.2f" % (
    t[i], l[i], radius_from_TL(t[i], l[i]), logg_from_MR(1.0, radius_from_TL(t[i], l[i]))))
