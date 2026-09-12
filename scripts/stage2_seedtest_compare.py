#!/usr/bin/env python3
"""Compare the seed-test refits (job 54131762, physics.py with the untied seeds)
against the first corrected rerun, per test group.

Groups (resources/census/stage2_seedtest_ids.txt, in this order):
  published SB2 whose first-rerun fit fell back to single (Delta chi2 = 0)
  200 other catalog SB2 with Delta chi2 = 0
  100 confirmed catalog SB2
  100 unconfirmed catalog SB2 with Delta chi2 > 0
The confirmed controls must stay confirmed. The unconfirmed controls with
Delta chi2 > 0 show whether the fix also matters away from the collapsed fits,
which decides between a full rerun and a rerun of the collapsed fits only.
"""
import csv, glob, os
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = lambda p: os.path.join(R, "resources", "census", p)
F = lambda x: float(x) if x not in (None, "", "nan") else np.nan

old = {r["sdss_id"]: r for r in csv.DictReader(open(C("stage2_catalog_full.csv")))}
ext = {r["sdss_id"] for r in csv.DictReader(open(C("stage2_external_sb2.csv")))}
new = {}
for f in glob.glob(C("stage2_seedtest_shards/shard_*.csv")):
    for r in csv.DictReader(open(f)):
        new[r["sdss_id"].split(".")[0]] = r


def group(s):
    o = old[s]
    if F(o["delta_chi2"]) == 0:
        return "zero, published SB2" if s in ext else "zero, other"
    return "confirmed control" if o["prefers_binary"] == "True" else "unconfirmed, dchi2 > 0"


rows = {}
for s in (l.strip() for l in open(C("stage2_seedtest_ids.txt")) if l.strip()):
    if s in new:
        rows.setdefault(group(s), []).append(s)
print("finished %d of %d" % (len(new), sum(1 for l in open(C("stage2_seedtest_ids.txt")) if l.strip())))
for g in ("zero, published SB2", "zero, other", "confirmed control", "unconfirmed, dchi2 > 0"):
    ss = rows.get(g, [])
    if not ss:
        continue
    ob = np.array([old[s]["prefers_binary"] == "True" for s in ss])
    nb = np.array([new[s]["prefers_binary"] == "True" for s in ss])
    od = np.array([F(old[s]["delta_chi2"]) for s in ss])
    nd = np.array([F(new[s]["delta_chi2"]) for s in ss])
    print("%-24s n=%3d  confirmed %5.1f%% -> %5.1f%%  | lost %d gained %d | dchi2 up %d, down >1%% %d, still 0 %d"
          % (g, len(ss), 100 * ob.mean(), 100 * nb.mean(), (ob & ~nb).sum(), (~ob & nb).sum(),
             (nd > od * 1.01 + 1).sum(), (nd < od * 0.99 - 1).sum(), (nd == 0).sum()))
# velocity agreement for the confirmed controls: the new seeds must not move a good solution
cc = rows.get("confirmed control", [])
if cc:
    dv = []
    for s in cc:
        a = [float(x) for x in old[s]["v1_per_visit"].split(";") if x]
        b = [float(x) for x in new[s]["v1_per_visit"].split(";") if x]
        if len(a) == len(b):
            dv.append(np.max(np.abs(np.array(a) - np.array(b))))
    dv = np.array(dv)
    print("confirmed controls: max per-visit v1 change median %.2f km/s, >5 km/s for %d of %d"
          % (np.median(dv), (dv > 5).sum(), len(dv)))
