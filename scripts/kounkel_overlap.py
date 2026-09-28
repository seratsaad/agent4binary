#!/usr/bin/env python3
"""Overlap of the released DR19 SB2 catalog with Kounkel et al. (2021).

Kounkel et al. (2021, AJ 162 184; VizieR J/AJ/162/184) searched the APOGEE
DR16/17 spectra, so the denominator is the catalog stars that APOGEE DR17
observed, found by a CDS XMatch of the catalog positions against the DR17 allStar
table (VizieR III/286) at 3 arcsec. Kounkel's table1 (SB2 and higher) is matched
positionally at the same radius. Catalog positions are the verified Gaia DR3
positions of dr19_sb2_catalog_open.csv; rows without one are skipped.

Run under a python with astroquery (e.g. ~/miniforge3/bin/python). Writes
resources/census/kounkel_overlap.csv (one row per catalog star with a position)
and prints the overlap. The Kounkel table is read from the cached copy in
resources/ab17_work/kounkel2021_table1.dat.
"""
import os
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.xmatch import XMatch

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAD = 3.0

cat = pd.read_csv(os.path.join(R, "resources/census/dr19_sb2_catalog_open.csv"), low_memory=False,
                  dtype={"sdss_id": str})
cat["sdss_id"] = cat.sdss_id.str.split(".").str[0]
pos = cat[cat.ra.notna() & cat.dec.notna()][["sdss_id", "ra", "dec", "best_q", "best_rv1", "best_rv2"]].copy()

# DR17 footprint (XMatch in chunks)
foot = set()
for i in range(0, len(pos), 20000):
    t = Table.from_pandas(pos.iloc[i:i + 20000][["sdss_id", "ra", "dec"]].rename(columns={"ra": "RA", "dec": "DEC"}))
    xm = XMatch.query(cat1=t, cat2="vizier:III/286/catalog", max_distance=RAD * u.arcsec,
                      colRA1="RA", colDec1="DEC").to_pandas()
    foot |= set(xm.sdss_id.astype(str))
pos["in_dr17"] = pos.sdss_id.isin(foot)

# Kounkel table1 (ID|RAdeg|DEdeg|SBn)
rows = []
for ln in open(os.path.join(R, "resources/ab17_work/kounkel2021_table1.dat")):
    p = ln.split("|")
    try:
        rows.append((p[0].strip(), float(p[1]), float(p[2]), int(p[3])))
    except (IndexError, ValueError):
        continue
K = pd.DataFrame(rows, columns=["kid", "kra", "kdec", "ksbn"])
c1 = SkyCoord(pos.ra.values * u.deg, pos.dec.values * u.deg)
c2 = SkyCoord(K.kra.values * u.deg, K.kdec.values * u.deg)
idx, sep, _ = c1.match_to_catalog_sky(c2)
ok = sep.arcsec <= RAD
pos["in_kounkel21"] = ok
pos["kounkel_sbn"] = np.where(ok, K.ksbn.values[idx], np.nan)
pos["sep_kms"] = (pos.best_rv1 - pos.best_rv2).abs()
pos.drop(columns=["best_q", "best_rv1", "best_rv2"]).to_csv(
    os.path.join(R, "resources/census/kounkel_overlap.csv"), index=False)

f = pos[pos.in_dr17]
print("Kounkel table1 rows %d (SB2 %d)" % (len(K), (K.ksbn == 2).sum()))
print("catalog with position %d | in DR17 footprint %d (%.1f%%)" % (len(pos), len(f), 100 * len(f) / len(pos)))
print("footprint stars also in Kounkel %d = %.1f%%" % (f.in_kounkel21.sum(), 100 * f.in_kounkel21.mean()))
print("  all-sky matches %d, of which outside footprint %d" % (pos.in_kounkel21.sum(), (pos.in_kounkel21 & ~pos.in_dr17).sum()))
for lo, hi in [(0, 10), (10, 30), (30, 1e9)]:
    g = f[(f.sep_kms >= lo) & (f.sep_kms < hi)]
    print("  coadd separation %g-%g km/s: %.1f%% of %d" % (lo, hi, 100 * g.in_kounkel21.mean(), len(g)))
