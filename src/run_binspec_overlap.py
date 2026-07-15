#!/usr/bin/env python3
# =========================================================================== #
# run_binspec_overlap.py
#
# DECISIVE CONSISTENCY TEST: compare our Stage-1 SC detector against binspec on
# the same DR19 coadds.
#
# For a RANDOM ~250-star sample of the El-Badry 2018b SB2 / DR19 overlap
# (resources/elbadry2018b_binaries.csv type=="SB2", mapped apogee_id<->sdss_id
# via resources/dr19_sb2_manifest.csv), run BOTH single-vs-binary detectors on
# the SAME DR19 mwmStar COADD, star-for-star:
#
#   - the binspec ORACLE  (physics.binspec_single_vs_binary): El-Badry's OWN
#     vendored flux net + Eq. B1 / Table B1 decision, on binspec's 7214 grid.
#     We feed it the RAW coadd flux + err=1/sqrt(ivar); binspec's own
#     get_apogee_continuum self-normalizes inside ingest_to_binspec_grid.
#   - OUR Stage-1 detector (physics.dr19_sc_single_vs_binary): the production
#     self-consistent combined-spectrum path on the RAW (flux_raw, ivar).
#
# Current known result: binspec >> ours. The signal is present in the coadd, so
# fix SC detector sensitivity before rerunning the census or moving to Stage 2.
#
# PARALLELISM. One task = one star = (run binspec + run SC). Tasks run across
# PROCESSES under 'spawn' with torch / BLAS capped to ONE thread per worker
# BEFORE importing physics (torch's intra-op pool deadlocks under 'spawn' when
# many workers each spin one up). Results are written INCREMENTALLY, one row per
# star, so the run is resumable: a star already in results.csv is skipped.
#
# CLI:
#   python src/run_binspec_overlap.py \
#       [--sample resources/fair_tests/binspec_overlap/sample.csv] \
#       [--raw-dir data/dr19_raw_sb2] \
#       [--out resources/fair_tests/binspec_overlap/results.csv] \
#       [--workers N] [--limit N]
# =========================================================================== #

import os

# Cap BLAS / OpenMP to one thread PER PROCESS BEFORE numpy / torch import (we
# parallelize across processes; per-worker multithreaded BLAS only oversubscribes
# and torch's intra-op pool deadlocks under 'spawn').
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
import time
import warnings
from multiprocessing import cpu_count, get_context

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_FAIR_TEST_DIR = os.path.join(_ROOT, "resources/fair_tests/binspec_overlap")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# One row per star: identity + El-Badry truth (for binning) + both verdicts.
FIELDS = [
    "sdss_id", "apogee_id",
    "q", "q_dyn", "gamma",                 # El-Badry truth (q_dyn empty => combined-only)
    "teff_seed", "logg_seed", "feh_seed",  # catalog 5-label seed (3 shown)
    # binspec ORACLE
    "bs_prefers_binary", "bs_delta_chi2", "bs_f_imp", "bs_min_fimp",
    "bs_best_q", "bs_best_rv1", "bs_best_rv2", "bs_teff_single",
    # OUR Stage-1 SC detector
    "sc_prefers_binary", "sc_delta_chi2", "sc_f_imp", "sc_min_fimp",
    "sc_best_q", "sc_best_rv1", "sc_best_rv2", "sc_teff_single",
    "error",
]

DEFAULT_SEED = (5500.0, 4.5, 0.0, 0.0, 5.0)


def _clean_seed(seed):
    if seed is None:
        return DEFAULT_SEED
    out = []
    for i in range(5):
        try:
            v = float(seed[i])
        except (TypeError, ValueError, IndexError):
            v = float("nan")
        out.append(DEFAULT_SEED[i] if not np.isfinite(v) else v)
    return tuple(out)


def _fit_one(task):
    """One star: load the RAW coadd, run the binspec oracle AND our SC detector.

    Imports physics INSIDE the child (loads the SC + binspec nets once per
    address space) and caps torch to one thread BEFORE that import. Never raises:
    a bad star records an `error` and still gets a row.
    """
    (sdss_id, apogee_id, q, q_dyn, gamma, seed, raw_dir) = task
    base = {k: "" for k in FIELDS}
    base["sdss_id"] = sdss_id
    base["apogee_id"] = apogee_id
    base["q"] = q
    base["q_dyn"] = "" if (q_dyn is None or (isinstance(q_dyn, float) and not np.isfinite(q_dyn))) else q_dyn
    base["gamma"] = "" if (gamma is None or (isinstance(gamma, float) and not np.isfinite(gamma))) else gamma
    s = _clean_seed(seed)
    base["teff_seed"], base["logg_seed"], base["feh_seed"] = s[0], s[1], s[2]
    try:
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import physics

            npz_path = os.path.join(raw_dir, "%d.npz" % int(sdss_id))
            if not (os.path.exists(npz_path) and os.path.getsize(npz_path) > 1000):
                base["error"] = "no_coadd_file"
                return base
            d = np.load(npz_path)
            wl = np.asarray(d["wl"], float)
            flux = np.asarray(d["flux_raw"], float)
            ivar = np.asarray(d["ivar"], float)

            # binspec ORACLE: feed RAW flux + err=1/sqrt(ivar); binspec self-
            # normalizes inside ingest_to_binspec_grid (get_apogee_continuum).
            with np.errstate(divide="ignore"):
                err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), 0.0)
            rb = physics.binspec_single_vs_binary(wl, flux, err)

            # OUR Stage-1 SC detector on the RAW (flux_raw, ivar).
            rs = physics.dr19_sc_single_vs_binary(flux, ivar, seed=s)

        base.update({
            "bs_prefers_binary": bool(rb["prefers_binary"]),
            "bs_delta_chi2": float(rb["delta_chi2"]),
            "bs_f_imp": float(rb["f_imp"]),
            "bs_min_fimp": ("" if rb.get("min_fimp_required") is None
                            else float(rb["min_fimp_required"])),
            "bs_best_q": float(rb["best_q"]),
            "bs_best_rv1": float(rb["best_rv1"]),
            "bs_best_rv2": float(rb["best_rv2"]),
            "bs_teff_single": float(rb["teff_single"]),
            "sc_prefers_binary": bool(rs["prefers_binary"]),
            "sc_delta_chi2": float(rs["delta_chi2"]),
            "sc_f_imp": float(rs["f_imp"]),
            "sc_min_fimp": ("" if rs.get("min_fimp_required") is None
                            else float(rs["min_fimp_required"])),
            "sc_best_q": float(rs["best_q"]),
            "sc_best_rv1": float(rs["best_rv1"]),
            "sc_best_rv2": float(rs["best_rv2"]),
            "sc_teff_single": float(rs["teff_single"]),
        })
        return base
    except Exception as exc:  # one bad star must not kill the batch
        base["error"] = "%s: %s" % (type(exc).__name__, exc)
        return base


def _load_done(out_path):
    """sdss_ids already in results.csv (resume checkpoint)."""
    if not os.path.exists(out_path):
        return set()
    try:
        df = pd.read_csv(out_path)
        return set(int(x) for x in df["sdss_id"].tolist())
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=os.path.join(_FAIR_TEST_DIR, "sample.csv"))
    ap.add_argument("--raw-dir", default=os.path.join(_ROOT, "data/dr19_raw_sb2"))
    ap.add_argument("--out", default=os.path.join(_FAIR_TEST_DIR, "results.csv"))
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    samp = pd.read_csv(args.sample)
    if args.limit > 0:
        samp = samp.iloc[:args.limit].copy()
    done = _load_done(args.out)
    print("sample: %d stars; already done: %d; workers: %d"
          % (len(samp), len(done), args.workers), flush=True)

    tasks = []
    for _, r in samp.iterrows():
        sid = int(r["sdss_id"])
        if sid in done:
            continue
        seed = (r.get("teff"), r.get("logg"), r.get("fe_h"),
                r.get("mg_h"), r.get("v_macro"))
        tasks.append((sid, str(r["apogee_id"]), float(r["q"]),
                      r.get("q_dyn"), r.get("gamma"), seed, args.raw_dir))
    if not tasks:
        print("nothing to do (all done).", flush=True)
        return

    # Open results.csv in append mode; write the header only if new/empty.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_file = (not os.path.exists(args.out)) or os.path.getsize(args.out) == 0
    fout = open(args.out, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
        fout.flush()

    ctx = get_context("spawn")
    t0 = time.time()
    n = 0
    with ctx.Pool(processes=args.workers) as pool:
        for row in pool.imap_unordered(_fit_one, tasks, chunksize=1):
            w.writerow(row)
            fout.flush()
            n += 1
            tag = "ERR" if row["error"] else (
                "bs=%s sc=%s" % (str(row["bs_prefers_binary"])[:1],
                                 str(row["sc_prefers_binary"])[:1]))
            print("[%d/%d] %d  %s  (%.0fs)"
                  % (n, len(tasks), row["sdss_id"], tag, time.time() - t0),
                  flush=True)
    fout.close()
    print("done: %d new rows in %.0fs" % (n, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
