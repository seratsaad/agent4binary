#!/usr/bin/env python3
"""Split a census manifest into K shards (round-robin) for a Slurm array census.
Writes <dir>/manifest_shard_{0..K-1}.csv. Usage:
    python scripts/census_shard.py resources/dr19_census_full_manifest.csv resources/census_shards 4
"""
import os, sys
import pandas as pd

src, outdir, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(outdir, exist_ok=True)
df = pd.read_csv(src)
for k in range(K):
    df.iloc[k::K].to_csv(os.path.join(outdir, "manifest_shard_%d.csv" % k), index=False)
    print("shard %d: %d rows" % (k, len(df.iloc[k::K])))
print("wrote %d shards of %d total to %s" % (K, len(df), outdir))
