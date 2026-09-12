#!/usr/bin/env python3
"""Stage-2 multi-epoch census shard: for each candidate (sigma_v>1 OR coadd dv>10),
download mwmVisit spectra, run the joint multi-epoch single-vs-binary fit, and record
per-visit velocities + q_dyn. Sharded strided over resources/census/stage2_candidates.txt.
Aggregate with a simple pandas concat afterward."""
import os, sys, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
_R = os.path.expanduser("~/agent4binary"); sys.path.insert(0, os.path.join(_R, "src"))
import numpy as np, pandas as pd
import physics, download_dr19_visits as dv

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--n-shards", type=int, default=200)
ap.add_argument("--max-visits", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()

man = pd.read_csv(os.path.join(_R, "resources/dr19_census272k_manifest.csv"))
seed = {int(r.sdss_id): (r.teff, r.logg, r.fe_h, r.mg_h, r.v_macro) for r in man.itertuples()}
IDS = os.environ.get("A4B_S2_IDS", "resources/census/stage2_candidates.txt")
ids = [int(x) for x in open(os.path.join(_R, IDS)).read().split()]
mine = ids[a.shard::a.n_shards]
dv.fetch_many(mine, label="s2-%d" % a.shard)   # stage this shard's visit spectra

FIELDS = ["sdss_id", "n_visits", "delta_chi2", "f_imp", "prefers_binary", "q_spec", "q_dyn",
          "q_dyn_at_bound", "v_single", "gamma", "v1_range", "v1_per_visit", "v2_per_visit",
          "visit_index_per_visit", "mjd_per_visit", "rv1_untied_per_visit",
          "rv2_untied_per_visit", "error"]
# Write each star as it finishes and skip stars already in the file, so a shard
# that hits its time limit can be resubmitted without losing work.
import csv
done = set()
if os.path.exists(a.out):
    done = {int(float(r["sdss_id"])) for r in csv.DictReader(open(a.out)) if r.get("sdss_id")}
_fh = open(a.out, "a", newline="")
_w = csv.DictWriter(_fh, fieldnames=FIELDS, extrasaction="ignore")
if not done:
    _w.writeheader()
n_new = 0
def _emit(row):
    global n_new
    _w.writerow(row); _fh.flush(); n_new += 1
for sid in mine:
    if sid in done:
        continue
    try:
        r = physics.dr19_visit_single_vs_binary(sid, seed=seed.get(sid), max_visits=a.max_visits,
                                                input_frame="rest")
        v1 = list(np.round(np.asarray(r.get("v1_per_visit", []) or [], float), 2))
        v2 = list(np.round(np.asarray(r.get("v2_per_visit", []) or [], float), 2))
        _emit(dict(sdss_id=sid, n_visits=int(r.get("n_visits_used", 0)),
            delta_chi2=float(r.get("delta_chi2", np.nan)), f_imp=float(r.get("f_imp", np.nan)),
            prefers_binary=bool(r.get("prefers_binary", False)), q_spec=float(r.get("q", np.nan)),
            # a Wilson-plot slope needs at least three epochs
            q_dyn=(r.get("q_dyn", np.nan) if int(r.get("n_visits_used", 0)) >= 3 else np.nan),
            q_dyn_at_bound=bool(int(r.get("n_visits_used", 0)) >= 3 and
                                min(abs(float(r.get("q_dyn", np.nan)) - 0.1), abs(float(r.get("q_dyn", np.nan)) - 1.5)) < 0.005),
            v_single=r.get("v_single", np.nan), gamma=r.get("gamma", np.nan),
            v1_range=(max(v1) - min(v1)) if len(v1) >= 2 else 0.0,
            v1_per_visit=";".join(map(str, v1)), v2_per_visit=";".join(map(str, v2)),
            visit_index_per_visit=";".join(map(str, r.get("visit_index_per_visit", []) or [])),
            mjd_per_visit=";".join("%.5f" % m for m in (r.get("mjd_per_visit", []) or [])),
            rv1_untied_per_visit=";".join("%.2f" % v for v in (r.get("rv1_untied_per_visit", []) or [])),
            rv2_untied_per_visit=";".join("%.2f" % v for v in (r.get("rv2_untied_per_visit", []) or [])),
            error=""))
    except Exception as e:
        _emit(dict(sdss_id=sid, n_visits=0, error="%s:%s" % (type(e).__name__, str(e)[:50])))
_fh.close()
print("shard %d: %d new rows (%d already done) -> %s" % (a.shard, n_new, len(done), a.out))
