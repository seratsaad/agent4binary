#!/usr/bin/env python3
"""Visit-fit confirmation rate of catalog SB2 that are also in published SB2 catalogs.

A catalog SB2 that an independent survey also found double-lined is very likely a
real binary, so its confirmation rate by the corrected multi-epoch fit measures
how often that fit can confirm a real SB2 from the visits available. Uses the
verified Gaia DR3 ids (stage2_gaia_coords.csv) for an exact match to Kovalev et
al. (2022, 2024) and the verified positions for a 2 arcsec match to Kounkel et
al. (2021), with the catalogs pulled live from VizieR as in
crossmatch_external_sb2.py. Writes resources/census/stage2_external_sb2.csv.
"""
import csv, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crossmatch_external_sb2 import tap, rows, sep_arcsec, VIZIER

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = lambda p: os.path.join(R, "resources", "census", p)
F = lambda x: float(x) if x not in (None, "", "nan") else np.nan
ser = lambda s: np.array([float(x) for x in str(s).split(";") if x not in ("", "nan")])

fit = {}
for f in glob.glob(C(os.environ.get("A4B_S2_SHARDS", "stage2_fixed_shards") + "/shard_*.csv")):
    for r in csv.DictReader(open(f)):
        fit[r["sdss_id"].split(".")[0]] = r
gaia = {r["sdss_id"]: r for r in csv.DictReader(open(C("stage2_gaia_coords.csv")))}
cat = {r["sdss_id"].split(".")[0]: r for r in csv.DictReader(open(C("dr19_sb2_catalog_open.csv")))}
ids = [s for s in fit if s in cat and s in gaia and gaia[s].get("gaia_dr3_source_id")]

k22 = {r["gid"].split(".")[0] for r in rows(tap(VIZIER,
    'SELECT "GaiaEDR3" AS gid FROM "J/MNRAS/517/356/tablec1" WHERE "GaiaEDR3" IS NOT NULL'))}
k24 = {r["gid"].split(".")[0] for r in rows(tap(VIZIER,
    'SELECT "GaiaDR3" AS gid FROM "J/MNRAS/527/521/tableb1" WHERE "GaiaDR3" IS NOT NULL'))}
kou = [r for r in rows(tap(VIZIER, 'SELECT "ID", "RAJ2000", "DEJ2000", "SBn" FROM "J/AJ/162/184/table1"'))
       if r["RAJ2000"].strip()]
print("Kovalev+22 %d, Kovalev+24 %d, Kounkel+21 %d" % (len(k22), len(k24), len(kou)))

cell, grid = 0.05, {}
for s in ids:
    ra, de = F(gaia[s]["ra"]), F(gaia[s]["dec"])
    grid.setdefault((int(ra / cell), int(de / cell)), []).append((s, ra, de))
kmatch = {}
for r in kou:
    ra, de = float(r["RAJ2000"]), float(r["DEJ2000"])
    for i in (int(ra / cell) - 1, int(ra / cell), int(ra / cell) + 1):
        for j in (int(de / cell) - 1, int(de / cell), int(de / cell) + 1):
            for s, sra, sde in grid.get((i, j), ()):
                d = sep_arcsec(ra, de, sra, sde)
                if d <= 2.0 and (s not in kmatch or d < kmatch[s][0]):
                    kmatch[s] = (d, r["SBn"])

conf = lambda s: fit[s]["prefers_binary"] == "True"
nv = lambda s: int(F(fit[s]["n_visits"])) if np.isfinite(F(fit[s]["n_visits"])) else 0
with open(C("stage2_external_sb2.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["sdss_id", "gaia_dr3_source_id", "in_kovalev22", "in_kovalev24", "in_kounkel21",
                                    "kounkel_sbn", "n_visits", "prefers_binary"])
    for s in ids:
        g = gaia[s]["gaia_dr3_source_id"]
        if g in k22 or g in k24 or s in kmatch:
            w.writerow([s, g, int(g in k22), int(g in k24), int(s in kmatch),
                        kmatch[s][1] if s in kmatch else "", nv(s), fit[s]["prefers_binary"]])

for lab, sel in (("Kovalev+22/24 (LAMOST-MRS)", lambda s: gaia[s]["gaia_dr3_source_id"] in k22 | k24),
                 ("Kounkel+21 (APOGEE)", lambda s: s in kmatch),
                 ("any published SB2", lambda s: gaia[s]["gaia_dr3_source_id"] in k22 | k24 or s in kmatch)):
    for nmin in (2, 3):
        a = [s for s in ids if sel(s) and nv(s) >= nmin]
        b = [s for s in ids if not sel(s) and nv(s) >= nmin]
        print("%-28s >=%d epochs: n=%4d confirmed %.1f%% | rest n=%5d confirmed %.1f%%"
              % (lab, nmin, len(a), 100 * np.mean([conf(s) for s in a]), len(b), 100 * np.mean([conf(s) for s in b])))

# numbers quoted in Section 4.4
c3 = [s for s in ids if conf(s) and nv(s) >= 3]
qd = np.array([F(fit[s]["q_dyn"]) for s in c3 if fit[s].get("q_dyn_at_bound") != "True"])
print("median q_dyn (confirmed, >=3 epochs, off the bounds) %.2f (n=%d) | catalog q of same %.2f"
      % (np.nanmedian(qd), np.isfinite(qd).sum(), np.median([F(cat[s]["best_q"]) for s in c3])))
cc = [s for s in fit if s in cat and conf(s) and nv(s) >= 2]
sep = np.concatenate([np.abs(ser(fit[s]["v1_per_visit"]) - ser(fit[s]["v2_per_visit"])) for s in cc])
print("per-visit |v1-v2| of confirmed (all visits): median %.1f, 90th %.1f km/s (n=%d visits, %d systems)"
      % (np.median(sep), np.percentile(sep, 90), sep.size, len(cc)))
