#!/usr/bin/env python3
"""Manifest for the v_macro re-fit: every released SB2 plus 10,000 random stars
the open census searched without flagging, in 16 shards of the census format."""
import csv, glob, os, random
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cat = {r["sdss_id"].split(".")[0] for r in csv.DictReader(open(f"{R}/resources/census/dr19_sb2_catalog_open.csv"))}
rows, fields = {}, None
for f in sorted(glob.glob(f"{R}/resources/census_shards272k/manifest_shard_*.csv")):
    rd = csv.DictReader(open(f)); fields = fields or rd.fieldnames
    for r in rd:
        rows.setdefault(r["sdss_id"].split(".")[0], r)
fitted = set()
for f in glob.glob(f"{R}/resources/census/catalog_shard272k_open_*.csv"):
    for r in csv.DictReader(open(f)):
        if not r.get("error"):
            fitted.add(r["sdss_id"].split(".")[0])
flag = [rows[s] for s in sorted(cat) if s in rows]
pool = sorted(s for s in fitted if s not in cat and s in rows)
random.seed(20260911)
ctl = [rows[s] for s in random.sample(pool, 10000)]
out = flag + ctl
d = f"{R}/resources/census_vmacro_shards"; os.makedirs(d, exist_ok=True)
for k in range(16):
    with open(f"{d}/manifest_shard_{k}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(out[k::16])
print("flagged %d (catalog %d) | unflagged pool %d -> sampled %d | total %d" % (len(flag), len(cat), len(pool), len(ctl), len(out)))
