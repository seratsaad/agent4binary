#!/usr/bin/env python3
"""Download DR19 mwmStar COADDs for the census manifest into data/dr19_census_raw.
Run on the Pitzer LOGIN node (compute nodes have no internet). Reuses the proven
threaded downloader (download_dr19_raw.fetch_many).

Usage: python scripts/download_census.py [--workers 16]
"""
import argparse
import os
import sys

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import download_dr19_raw as dlr

MANIFEST = os.path.join(_ROOT, "resources", "dr19_census_manifest.csv")
OUTDIR = os.path.join(_ROOT, "data", "dr19_census_raw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--outdir", default=OUTDIR)
    a = ap.parse_args()
    dlr.WORKERS = a.workers
    outdir, manifest = a.outdir, a.manifest
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(manifest)
    rows = [(int(s), {}) for s in df["sdss_id"]]
    print("downloading %d census coadds -> %s (workers=%d)"
          % (len(rows), outdir, a.workers), flush=True)
    present = dlr.fetch_many(rows, out_dir=outdir, label="census")
    print("present on disk: %d / %d" % (len(present), len(rows)), flush=True)


if __name__ == "__main__":
    main()
