#!/usr/bin/env python3
"""Merge census catalog shards -> resources/census/dr19_full_sb2_catalog.csv."""
import glob, os, sys
import pandas as pd
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
shards = sorted(glob.glob(os.path.join(_ROOT, "resources/census/catalog_shard_*.csv")))
df = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
df = df.drop_duplicates("sdss_id")
out = os.path.join(_ROOT, "resources/census/dr19_full_sb2_catalog.csv")
df.to_csv(out, index=False)
print("merged %d shards -> %s (%d rows)" % (len(shards), out, len(df)))
