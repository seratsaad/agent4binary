#!/usr/bin/env python3
# =========================================================================== #
# analyze_coherence.py  (AB-2)
#
# Build the honest operating point for the velocity-coherence statistic and compare
# it to the real-data Delta-chi2/f_imp detector and to binspec. Inputs:
#   coherence CSVs  (run_coherence.py): <set>: id, coherence, v_peak, ...
#   detector CSVs   (run_binspec_overlap / operating_point): sc_delta(_chi2),
#                   sc_f(_)imp, bs_pref(ers_binary) per star -- for the baseline + the
#                   binspec-matched FPR target.
#
# Reports SB2 recovery at the binspec-matched FPR for: (a) coherence alone, (b) the
# real-data Table-B1 detector (baseline ~49%), (c) coherence OR detector (union), and
# (d) coherence AND a mild Delta-chi2 floor (cull). Writes benchmarks/ab2_coherence_purity.{json,md}.
# =========================================================================== #
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

TABLE_B1 = [(3000, 0.0), (2500, 0.05), (2000, 0.075), (1500, 0.10), (1000, 0.125),
            (750, 0.15), (600, 0.175), (450, 0.20), (300, 0.225)]


def _passes_b1(delta, fimp, floor, waive=1.0e5):
    for lo, mf in TABLE_B1:
        if delta >= lo:
            return fimp >= (mf if delta >= waive else max(mf, floor))
    return False


def _truthy(s):
    return pd.Series(s).astype(str).str.lower().isin(["true", "1"]).values


def _col(df, *names):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce").values
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coh-sb2", required=True)
    ap.add_argument("--coh-controls", required=True)
    ap.add_argument("--det-sb2", required=True)        # real_sb2.csv (detector stats)
    ap.add_argument("--det-controls", required=True)   # real_controls.csv
    ap.add_argument("--out", default=os.path.join(_ROOT, "benchmarks", "ab2_coherence_purity"))
    a = ap.parse_args()

    cs = pd.read_csv(a.coh_sb2).set_index("id")
    cc = pd.read_csv(a.coh_controls).set_index("id")
    ds = pd.read_csv(a.det_sb2)
    dc = pd.read_csv(a.det_controls)
    ds.index = ds[("sdss_id" if "sdss_id" in ds.columns else "id")].astype(int)
    dc.index = dc[("id" if "id" in dc.columns else "sdss_id")].astype(int)

    # align on the intersection of ids
    sb2_ids = [i for i in cs.index if i in ds.index]
    ctl_ids = [i for i in cc.index if i in dc.index]
    coh_sb2 = cs.loc[sb2_ids, "coherence"].values
    coh_ctl = cc.loc[ctl_ids, "coherence"].values
    sd = _col(ds.loc[sb2_ids], "sc_delta_chi2", "sc_delta"); sf = _col(ds.loc[sb2_ids], "sc_f_imp", "sc_fimp")
    cd = _col(dc.loc[ctl_ids], "sc_delta", "sc_delta_chi2"); cf = _col(dc.loc[ctl_ids], "sc_fimp", "sc_f_imp")
    bs_rec = float(np.mean(_truthy(ds.loc[sb2_ids].get("bs_prefers_binary", ds.loc[sb2_ids].get("bs_pref")))))
    bs_fpr = float(np.mean(_truthy(dc.loc[ctl_ids].get("bs_pref", dc.loc[ctl_ids].get("bs_prefers_binary")))))

    n_sb2, n_ctl = len(sb2_ids), len(ctl_ids)
    print("n_sb2=%d n_ctl=%d   binspec: recovery=%.1f%% FPR=%.1f%%"
          % (n_sb2, n_ctl, 100 * bs_rec, 100 * bs_fpr))

    def at_matched(rec_fn, fpr_fn, grid):
        """Sweep threshold; return (recovery, fpr, thr) at the tightest point whose FPR <= bs_fpr."""
        best = None
        for thr in grid:
            fpr = fpr_fn(thr)
            if fpr <= bs_fpr + 1e-9:
                best = (rec_fn(thr), fpr, thr)
                break
        return best

    # (a) coherence alone
    cgrid = np.unique(np.concatenate([np.linspace(0, 50, 501), coh_ctl]))
    cgrid = np.sort(cgrid)
    a_best = at_matched(lambda t: np.mean(coh_sb2 >= t),
                        lambda t: np.mean(coh_ctl >= t), cgrid)
    # (b) detector Table-B1 baseline (sweep f_imp floor)
    fgrid = np.arange(0.06, 0.7, 0.005)
    b_best = at_matched(lambda fl: np.mean([_passes_b1(d, f, fl) for d, f in zip(sd, sf)]),
                        lambda fl: np.mean([_passes_b1(d, f, fl) for d, f in zip(cd, cf)]), fgrid)
    # (c) union: detector(at production floor 0.14) OR coherence>=t
    det_sb2_14 = np.array([_passes_b1(d, f, 0.14) for d, f in zip(sd, sf)])
    det_ctl_14 = np.array([_passes_b1(d, f, 0.14) for d, f in zip(cd, cf)])
    c_best = at_matched(lambda t: np.mean(det_sb2_14 | (coh_sb2 >= t)),
                        lambda t: np.mean(det_ctl_14 | (coh_ctl >= t)), cgrid)
    # (d) cull: coherence>=t AND a mild delta floor (>=300, the B1 minimum)
    d_best = at_matched(lambda t: np.mean((coh_sb2 >= t) & (sd >= 300)),
                        lambda t: np.mean((coh_ctl >= t) & (cd >= 300)), cgrid)

    def fmt(b):
        return ("recovery=%.1f%% @ FPR=%.1f%% (thr=%.3f)" % (100 * b[0], 100 * b[1], b[2])) if b else "n/a"

    res = {
        "n_sb2": n_sb2, "n_controls": n_ctl,
        "binspec_recovery": bs_rec, "binspec_fpr": bs_fpr,
        "coherence_alone": ({"recovery": b[0], "fpr": b[1], "thr": b[2]} if (b := a_best) else None),
        "detector_baseline": ({"recovery": b[0], "fpr": b[1], "floor": b[2]} if (b := b_best) else None),
        "union_det_or_coh": ({"recovery": b[0], "fpr": b[1], "thr": b[2]} if (b := c_best) else None),
        "coh_and_delta300": ({"recovery": b[0], "fpr": b[1], "thr": b[2]} if (b := d_best) else None),
        "coherence_curve": [{"thr": float(t),
                             "recovery": float(np.mean(coh_sb2 >= t)),
                             "fpr": float(np.mean(coh_ctl >= t))}
                            for t in (3, 4, 5, 6, 7, 8, 10, 12, 15)],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump(res, f, indent=2)

    lines = ["# AB-2 velocity-coherence operating point (real-data net, %d SB2 + %d controls)" % (n_sb2, n_ctl), "",
             "Bar: binspec recovery %.1f%% @ FPR %.1f%%. Baseline: real-data Delta-chi2/f_imp." % (100 * bs_rec, 100 * bs_fpr), "",
             "| method | SB2 recovery @ binspec-matched FPR |", "| --- | --- |",
             "| real-data detector (Table B1) | %s |" % fmt(b_best),
             "| coherence alone | %s |" % fmt(a_best),
             "| detector OR coherence | %s |" % fmt(c_best),
             "| coherence AND delta>=300 | %s |" % fmt(d_best),
             "| binspec (reference) | recovery=%.1f%% @ FPR=%.1f%% |" % (100 * bs_rec, 100 * bs_fpr), "",
             "coherence curve (thr: recovery / FPR):"]
    for c in res["coherence_curve"]:
        lines.append("  %.0f: %.1f%% / %.1f%%" % (c["thr"], 100 * c["recovery"], 100 * c["fpr"]))
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("\nwrote %s.{json,md}" % os.path.relpath(a.out, _ROOT), flush=True)


if __name__ == "__main__":
    main()
