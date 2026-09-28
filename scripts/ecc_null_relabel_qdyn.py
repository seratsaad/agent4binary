#!/usr/bin/env python3
"""Rescore the injection null with twins defined by the dynamical mass ratio.

The null (ecc_null_full.csv) puts the same index into every mock system, each
built from its own real system's cadence, K1 and q, so the twin label enters
only when the realization is scored. Here the label is min(q_dyn, 1/q_dyn) >
0.95 from the deep table (as in ecc_relabel_qdyn.py), systems without a usable
q_dyn are left out, and each realization is scored with ecc_solve.injection_das.
Writes resources/census/ecc_null_full_qdyn.csv and ecc_null_dalpha_qdyn.csv.
"""
import os, sys
import numpy as np, pandas as pd
_H = os.path.dirname(os.path.abspath(__file__)); _R = os.path.dirname(_H)
sys.path.insert(0, _H)
import ecc_solve as S

C = lambda p: os.path.join(_R, "resources/census", p)
n = pd.read_csv(C("ecc_null_full.csv"))
deep = pd.read_csv(C("stage2_deep.csv"), usecols=["sdss_id", "q_dyn", "q_dyn_at_bound"], low_memory=False)
ok = deep.q_dyn.notna() & (deep.q_dyn_at_bound.astype(str).str.lower() != "true") & (deep.q_dyn > 0.1) & (deep.q_dyn < 1.5)
deep = deep[ok].copy()
deep["twin_q"] = (np.minimum(deep.q_dyn, 1 / deep.q_dyn) > 0.95).astype(int)
n = n.merge(deep[["sdss_id", "twin_q"]], on="sdss_id", how="inner")
n["twin"] = n.twin_q
n.drop(columns="twin_q").to_csv(C("ecc_null_full_qdyn.csv"), index=False)
das = S.injection_das(C("ecc_null_full_qdyn.csv"))
das.to_csv(C("ecc_null_dalpha_qdyn.csv"), index=False)
print("systems per realization: twins %d, non-twins %d" % (das.n_t.iloc[0], das.n_n.iloc[0]))
for al, (m, s, k) in sorted(S.bias_from_das(das).items()):
    print("alpha=%.2f : %+.3f +/- %.3f (sd, %d reps)" % (al, m, s, k))
bm, bse = S.pooled_bias(S.bias_from_das(das))
print("pooled null bias %+.3f +/- %.3f ; all reps sd %.3f (n=%d)" % (bm, bse, das.da.std(ddof=1), len(das)))
