#!/usr/bin/env python3
"""Per-realization twin minus non-twin dAlpha of the injection null, from
ecc_null_full.csv, with the estimator of scripts/ecc_solve.py (injection_das).
Writes resources/census/ecc_null_dalpha.csv (alpha, rep, da, n_t, n_n), which
paper2/figures/fig_ecc_validation.py plots as the zero-input point.

  python scripts/ecc_null_dalpha.py [--null FILE] [--out FILE]
"""
import os, sys, argparse
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import ecc_solve as S

R = lambda p: p if os.path.isabs(p) else os.path.join(_ROOT, p)
ap = argparse.ArgumentParser()
ap.add_argument("--null", default="resources/census/ecc_null_full.csv")
ap.add_argument("--out", default="resources/census/ecc_null_dalpha.csv")
a = ap.parse_args()

das = S.injection_das(R(a.null))
if das is None or not len(das):
    sys.exit("no usable realizations in %s" % R(a.null))
das.to_csv(R(a.out), index=False)
for al, (m, s, n) in sorted(S.bias_from_das(das).items()):
    print("alpha=%.2f : %+.3f +/- %.3f (sd, %d reps)" % (al, m, s, n))
bm, bse = S.pooled_bias(S.bias_from_das(das))
print("pooled bias %+.3f +/- %.3f ; all reps mean %+.3f sd %.3f (n=%d)"
      % (bm, bse, das.da.mean(), das.da.std(ddof=1), len(das)))
print("wrote", R(a.out))
