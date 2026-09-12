#!/usr/bin/env python3
"""Concatenate per-shard CSV outputs into one table, checking that no shard is
missing and no key is repeated. Used for the eccentricity chain:

  orbit_deep_shard.py  resources/census/orbit_deep_shards/shard_*.csv -> orbit_deep_posteriors.csv
      python scripts/concat_shards.py --shards resources/census/orbit_deep_shards \
          --expect 100 --key sdss_id --check-deep --out resources/census/orbit_deep_posteriors.csv
  ecc_real_marginal.py resources/census/ecc_real_shards/shard_*.csv -> ecc_marginal_real.csv
      python scripts/concat_shards.py --shards resources/census/ecc_real_shards \
          --expect 40 --key sdss_id --out resources/census/ecc_marginal_real.csv
  ecc_val_suite.py     resources/census/ecc_val_shards/shard_*.csv -> ecc_validation.csv
      python scripts/concat_shards.py --shards resources/census/ecc_val_shards \
          --expect 16 --key da_true,rep --rows 16 --out resources/census/ecc_validation.csv

Values are copied as text, so nothing is reformatted.
"""
import argparse, glob, os, re, sys
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
R = lambda p: p if os.path.isabs(p) else os.path.join(_ROOT, p)

ap = argparse.ArgumentParser()
ap.add_argument("--shards", required=True, help="directory holding the shard files")
ap.add_argument("--pattern", default="shard_*.csv")
ap.add_argument("--out", required=True)
ap.add_argument("--expect", type=int, default=None,
                help="number of shards (files shard_0..shard_{N-1} must all be present)")
ap.add_argument("--key", default="sdss_id", help="comma-separated unique key ('' to skip)")
ap.add_argument("--rows", type=int, default=None, help="required total number of rows")
ap.add_argument("--check-deep", action="store_true",
                help="require the sdss_id set to equal the deep table's (A4B_DEEP_TABLE)")
a = ap.parse_args()

files = glob.glob(os.path.join(R(a.shards), a.pattern))
num = lambda f: int(re.findall(r"(\d+)", os.path.basename(f))[-1])
files = sorted(files, key=num)
if not files:
    sys.exit("no %s in %s" % (a.pattern, R(a.shards)))
if a.expect is not None:
    have = {num(f) for f in files}
    missing = sorted(set(range(a.expect)) - have)
    if missing:
        sys.exit("%d of %d shards missing, e.g. %s" % (len(missing), a.expect, missing[:10]))

parts = []
for f in files:
    try:
        parts.append(pd.read_csv(f, dtype=str, keep_default_na=False))
    except pd.errors.EmptyDataError:
        print("empty shard file: %s" % f)
d = pd.concat(parts, ignore_index=True, sort=False).fillna("")
if a.key:
    key = a.key.split(",")
    dup = d.duplicated(key, keep=False)
    if dup.any():
        sys.exit("%d rows share a key %s, e.g.\n%s" % (dup.sum(), key, d[dup][key].head()))
    d = d.sort_values(key, key=lambda s: pd.to_numeric(s, errors="coerce")).reset_index(drop=True)
if a.rows is not None and len(d) != a.rows:
    sys.exit("expected %d rows, found %d" % (a.rows, len(d)))
if a.check_deep:
    sys.path.insert(0, _HERE)
    import deep_table as DT
    deep_ids = {int(s) for s in DT.load_deep().sdss_id}
    got = {int(float(s)) for s in d.sdss_id}
    if got != deep_ids:
        sys.exit("sdss_id set differs from the deep table: %d missing, %d extra"
                 % (len(deep_ids - got), len(got - deep_ids)))
d.to_csv(R(a.out), index=False)
print("%d shard files -> %s (%d rows)" % (len(files), R(a.out), len(d)))
if "status" in d.columns:
    print(d.status.value_counts().to_string())
