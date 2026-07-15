#!/usr/bin/env python3
# =========================================================================== #
# run_coherence.py  (AB-2)
#
# For every RAW coadd in a directory, run the SAME single-star fit the production
# SC detector uses (continuum_normalize -> mask_bad_pixels -> fit_single5_sc with
# the real-data net), then compute the velocity-coherence statistic (src/coherence)
# on the single-fit residual. Output one CSV row per star:
#     id, coherence, v_peak, coh_over_base, chi2_single_red
#
# The net is whatever physics loads (AGENT4BINARY_SC_MODEL; default payne_dr19_sc.pt
# = the real-data net). Seeds: a --sample CSV (sdss_id + 5 labels) if given, else a
# solar-dwarf default -- matching the detector.
#
#   python src/run_coherence.py --raw-dir data/dr19_raw_sb2 \
#       --sample resources/fair_tests/binspec_overlap/sample.csv --out .../coh_sb2.csv
#   python src/run_coherence.py --raw-dir data/dr19_raw_controls --out .../coh_controls.csv
# =========================================================================== #
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import glob
import sys
import warnings
from multiprocessing import cpu_count, get_context

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_SEED = (5500.0, 4.5, 0.0, 0.0, 5.0)
_SEEDS = {}


def _one(args):
    npz_path, seed = args
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import physics
        import coherence
    name = os.path.basename(npz_path).replace(".npz", "")
    try:
        d = np.load(npz_path)
        flux = np.asarray(d["flux_raw"], float)
        ivar = np.asarray(d["ivar"], float)
        sc_flux, sc_err = physics.continuum_normalize(
            flux, ivar, return_error=True, use_mask=physics.SC_USE_MASK)
        sc_flux = np.where(np.isfinite(sc_flux), sc_flux, 1.0)
        sc_err_finite = np.where(np.isfinite(sc_err), sc_err, 1e6)
        obs = sc_flux
        obs_err = physics.mask_bad_pixels(obs, sc_err_finite, snr_cap=physics.SC_SNR_CAP)
        obs_err = np.where(np.isfinite(obs_err), obs_err, 1e6)
        t0, g0, h0, m0, v0 = seed
        t0 = float(np.clip(t0, physics._LMINSC[0], physics._LMAXSC[0]))
        g0 = float(np.clip(g0, physics._LMINSC[1], physics._LMAXSC[1]))
        h0 = float(np.clip(h0, physics._LMINSC[2], physics._LMAXSC[2]))
        m0 = float(np.clip(m0, physics._LMINSC[3], physics._LMAXSC[3]))
        v0 = float(np.clip(v0, physics._LMINSC[4], physics._LMAXSC[4]))
        p_s, model_s, chi2_s = physics.fit_single5_sc(obs, obs_err, t0, g0, h0, m0, v0)
        ng = int(np.sum(obs_err < 1e5))
        r = coherence.coherence_score(obs, obs_err, model_s)
        return (name, r["coherence"], r["v_peak"], r["coh_over_base"],
                float(chi2_s / max(ng, 1)))
    except Exception as e:                            # never kill the pool
        return (name, -1.0, 0.0, 0.0, -1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--sample", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    a = ap.parse_args()

    seeds = {}
    if a.sample and os.path.exists(a.sample):
        s = pd.read_csv(a.sample)
        for _, r in s.iterrows():
            try:
                seeds[int(r["sdss_id"])] = (float(r["teff"]), float(r["logg"]),
                                            float(r["fe_h"]), float(r["mg_h"]),
                                            float(r["v_macro"]))
            except Exception:
                pass

    files = sorted(glob.glob(os.path.join(a.raw_dir, "*.npz")))
    tasks = []
    for f in files:
        sid = int(os.path.basename(f)[:-4]) if os.path.basename(f)[:-4].isdigit() else -1
        tasks.append((f, seeds.get(sid, DEFAULT_SEED)))
    print("coherence on %d spectra in %s (workers=%d, seeds=%d)"
          % (len(tasks), a.raw_dir, a.workers, len(seeds)), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ctx = get_context("spawn")
    rows = []
    with ctx.Pool(processes=a.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_one, tasks, chunksize=1), 1):
            rows.append(row)
            if i % 50 == 0:
                print("  %d/%d" % (i, len(tasks)), flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "coherence", "v_peak", "coh_over_base", "chi2_single_red"])
        w.writerows(rows)
    coh = np.array([r[1] for r in rows])
    print("wrote %s  median coherence=%.2f" % (a.out, np.median(coh[coh >= 0])), flush=True)


if __name__ == "__main__":
    main()
