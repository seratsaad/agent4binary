#!/usr/bin/env python3
"""Build the deep stage-2 table (resources/census/stage2_deep.csv) from shards of
scripts/stage2_census_shard.py run with --max-visits 16.

Keeps rows that
  - fitted without error,
  - used n_visits >= 8 visits,
  - are in the released catalog (resources/census/dr19_sb2_catalog_open.csv),
  - prefer the binary model (prefers_binary == "True"),
and asserts that each kept row carries one epoch (mjd_per_visit) per velocity.
All shard columns are written unchanged, sorted by sdss_id. Several shard
directories can be given; the n_visits >= 8 cut also selects the deep stars out
of a merged run that contains stars with fewer visits. A star present in more
than one shard keeps the row with the most visits.

  python scripts/build_stage2_deep.py --shards resources/census/stage2_deep_shards
"""
import argparse, csv, glob, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda p: p if os.path.isabs(p) else os.path.join(_ROOT, p)

ap = argparse.ArgumentParser()
ap.add_argument("--shards", nargs="+", required=True,
                help="shard directories; every *.csv inside is read")
ap.add_argument("--catalog", default="resources/census/dr19_sb2_catalog_open.csv")
ap.add_argument("--min-visits", type=int, default=8)
ap.add_argument("--out", default="resources/census/stage2_deep.csv")
a = ap.parse_args()

sid_of = lambda s: int(float(s))
nval = lambda s: len([x for x in str(s or "").split(";") if x not in ("", "nan")])

files = []
for d in a.shards:
    fs = sorted(glob.glob(os.path.join(R(d), "*.csv")))
    if not fs:
        sys.exit("no *.csv in %s" % R(d))
    files += fs

cols, rows = [], []
for f in files:
    with open(f, newline="") as fh:
        rd = csv.DictReader(fh)
        for c in rd.fieldnames or []:
            if c not in cols:
                cols.append(c)
        rows += [r for r in rd if r.get("sdss_id")]
need = ["sdss_id", "n_visits", "prefers_binary", "v1_per_visit", "v2_per_visit",
        "mjd_per_visit", "error"]
miss = [c for c in need if c not in cols]
if miss:
    sys.exit("shards lack columns %s (written before the epoch fix?)" % miss)

cat = {}
with open(R(a.catalog), newline="") as fh:
    for r in csv.DictReader(fh):
        cat[sid_of(r["sdss_id"])] = float(r["best_q"]) if r.get("best_q") not in (None, "", "nan") else float("nan")

drop = dict(error=0, few_visits=0, not_in_catalog=0, not_binary=0)
best = {}
for r in rows:
    if (r.get("error") or "").strip():
        drop["error"] += 1; continue
    nv = int(float(r["n_visits"] or 0))
    if nv < a.min_visits:
        drop["few_visits"] += 1; continue
    sid = sid_of(r["sdss_id"])
    if sid not in cat:
        drop["not_in_catalog"] += 1; continue
    if r["prefers_binary"] != "True":
        drop["not_binary"] += 1; continue
    n1, n2, nt = nval(r["v1_per_visit"]), nval(r["v2_per_visit"]), nval(r["mjd_per_visit"])
    assert nt == n1 and n2 == n1, ("sdss_id %d: %d v1, %d v2, %d mjd entries"
                                   % (sid, n1, n2, nt))
    if sid not in best or nv >= int(float(best[sid]["n_visits"])):
        best[sid] = r
n_pass = sum(1 for r in rows) - sum(drop.values())
kept = [best[s] for s in sorted(best)]

out = R(a.out)
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(kept)

twins = sum(1 for s in best if cat[s] > 0.95)
print("shard files        : %d" % len(files))
print("rows read          : %d" % len(rows))
print("dropped            : %s" % ", ".join("%s %d" % kv for kv in drop.items()))
print("duplicate stars    : %d" % (n_pass - len(kept)))
print("kept               : %d" % len(kept))
print("twins (best_q>0.95): %d" % twins)
print("wrote %s" % out)
