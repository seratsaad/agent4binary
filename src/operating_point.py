#!/usr/bin/env python3
# =========================================================================== #
# operating_point.py
#
# The HONEST detector metric: completeness (SB2 recovery) AND purity (control
# false-positive rate) TOGETHER, plus the operating curve (recovery vs FPR as the
# f_imp floor is swept) and the recovery at binspec-matched purity. Raw recovery
# alone is misleading -- it can be bought with a high false-positive rate (see
# METHODS.md sec 2.5). This is the number to quote.
#
# Two modes:
#   --run-controls : run BOTH detectors (OUR SC + binspec oracle) on every spectrum
#                    in a directory (default data/dr19_raw_controls) -> a results CSV
#                    with sc/bs prefers + delta + f_imp per star.
#   --roc          : given an SB2 results CSV (from run_binspec_overlap.py) and a
#                    controls results CSV (from --run-controls), print the operating
#                    curve and the SC recovery at binspec-matched FPR.
#
# Examples:
#   python src/operating_point.py --run-controls --out resources/fair_tests/controls.csv
#   python src/operating_point.py --roc resources/fair_tests/binspec_overlap/results.csv \
#                                       resources/fair_tests/controls.csv
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

# El-Badry 2018b Table B1 sliding scale (delta_chi2 lower bound, min f_imp).
TABLE_B1 = [(3000, 0.0), (2500, 0.05), (2000, 0.075), (1500, 0.10), (1000, 0.125),
            (750, 0.15), (600, 0.175), (450, 0.20), (300, 0.225)]


def _passes(delta, fimp, floor, waive=1.0e5):
    for lo, mf in TABLE_B1:
        if delta >= lo:
            return fimp >= (mf if delta >= waive else max(mf, floor))
    return False


def _rate(delta, fimp, floor):
    return float(np.mean([_passes(d, f, floor) for d, f in zip(delta, fimp)]))


def _one(npz_path):
    """Run OUR SC detector + the binspec oracle on one raw coadd. Never raises."""
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import physics
    d = np.load(npz_path)
    wl = np.asarray(d["wl"], float)
    flux = np.asarray(d["flux_raw"], float)
    ivar = np.asarray(d["ivar"], float)
    with np.errstate(divide="ignore"):
        err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), 0.0)
    name = os.path.basename(npz_path).replace(".npz", "")
    try:
        rs = physics.dr19_sc_single_vs_binary(flux, ivar)
    except Exception:
        rs = {"prefers_binary": False, "delta_chi2": -1.0, "f_imp": -1.0}
    try:
        rb = physics.binspec_single_vs_binary(wl, flux, err)
    except Exception:
        rb = {"prefers_binary": False, "delta_chi2": -1.0, "f_imp": -1.0}
    return (name, int(bool(rs["prefers_binary"])), float(rs["delta_chi2"]),
            float(rs["f_imp"]), int(bool(rb["prefers_binary"])),
            float(rb["delta_chi2"]), float(rb["f_imp"]))


def run_dir(raw_dir, out_csv, workers):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
    print("running SC + binspec on %d spectra in %s (workers=%d)"
          % (len(files), raw_dir, workers), flush=True)
    ctx = get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        rows = pool.map(_one, files)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "sc_pref", "sc_delta", "sc_fimp",
                    "bs_pref", "bs_delta", "bs_fimp"])
        w.writerows(rows)
    sc = sum(r[1] for r in rows)
    bs = sum(r[4] for r in rows)
    print("wrote %s  |  SC flagged=%d/%d (%.1f%%)  binspec flagged=%d/%d (%.1f%%)"
          % (out_csv, sc, len(rows), 100 * sc / max(len(rows), 1),
             bs, len(rows), 100 * bs / max(len(rows), 1)), flush=True)


def roc(sb2_csv, ctl_csv):
    sb2 = pd.read_csv(sb2_csv)
    ctl = pd.read_csv(ctl_csv)
    # SB2 results come from run_binspec_overlap.py (sc_delta_chi2/sc_f_imp);
    # controls come from --run-controls (sc_delta/sc_fimp). Accept either schema.
    sd = pd.to_numeric(sb2.get("sc_delta_chi2", sb2.get("sc_delta")), errors="coerce").values
    sf = pd.to_numeric(sb2.get("sc_f_imp", sb2.get("sc_fimp")), errors="coerce").values
    cd = pd.to_numeric(ctl.get("sc_delta", ctl.get("sc_delta_chi2")), errors="coerce").values
    cf = pd.to_numeric(ctl.get("sc_fimp", ctl.get("sc_f_imp")), errors="coerce").values
    bs_rec = pd.Series(sb2.get("bs_prefers_binary", sb2.get("bs_pref"))).astype(str)\
        .str.lower().isin(["true", "1"]).mean()
    bs_fpr = pd.Series(ctl.get("bs_pref", ctl.get("bs_prefers_binary"))).astype(str)\
        .str.lower().isin(["true", "1"]).mean()
    print("binspec: recovery=%.1f%%  FPR=%.1f%%" % (100 * bs_rec, 100 * bs_fpr))
    print("\nf_imp_floor   SC_recovery   SC_FPR")
    for fl in [0.06, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30]:
        print("   %.2f         %5.1f%%       %5.1f%%"
              % (fl, 100 * _rate(sd, sf, fl), 100 * _rate(cd, cf, fl)))
    for fl in np.arange(0.06, 0.7, 0.005):
        if _rate(cd, cf, fl) <= bs_fpr:
            print("\n@binspec-matched FPR (%.1f%%): floor=%.3f -> SC recovery=%.1f%% "
                  "(binspec %.1f%%)" % (100 * bs_fpr, fl, 100 * _rate(sd, sf, fl),
                                        100 * bs_rec))
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-controls", action="store_true")
    ap.add_argument("--raw-dir", default=os.path.join(_HERE, "..", "data/dr19_raw_controls"))
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "resources/fair_tests/controls.csv"))
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--roc", nargs=2, metavar=("SB2_CSV", "CONTROLS_CSV"))
    args = ap.parse_args()
    if args.run_controls:
        run_dir(args.raw_dir, args.out, args.workers)
    if args.roc:
        roc(args.roc[0], args.roc[1])
    if not args.run_controls and not args.roc:
        ap.print_help()


if __name__ == "__main__":
    main()
