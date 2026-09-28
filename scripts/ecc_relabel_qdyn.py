#!/usr/bin/env python3
"""Eccentricity comparison with twins defined by the dynamical mass ratio.

The published comparison (ecc_final_table.py) labels a system a twin when its
combined-spectrum q (best_q) exceeds 0.95. That q is a parameter of the
composite model, not a measured mass ratio (Spearman 0.27 against q_dyn), so
here the label comes from q_dyn of the deep refit (stage2_deep.csv, the same
visits the orbit likelihoods use). q_dyn above 1 means the fitted secondary
moves less than the primary, so we use q_sym = min(q_dyn, 1/q_dyn); values at
the fit bounds (0.1, 1.5) are left out. Everything else (sample cuts, matching,
bootstrap, likelihood grid) is ecc_final_table.variant unchanged.

Local only; reuses the per-system likelihoods in ecc_marginal_real.csv. Does not
rerun the null or validation mocks, so no null-bias or slope correction is
applied here. Prints a table and writes resources/census/ecc_variants_qdyn.csv.
"""
import os, sys
import numpy as np, pandas as pd
_H = os.path.dirname(os.path.abspath(__file__)); _R = os.path.dirname(_H)
sys.path.insert(0, _H)
import ecc_final_table as T
import deep_table as DT

d = pd.read_csv(os.path.join(_R, "resources/census/ecc_marginal_real.csv"))
d = d[(d.P50 >= 6) & (d.P50 <= 400)].copy()
deep = pd.read_csv(os.path.join(_R, "resources/census/stage2_deep.csv"), low_memory=False)
d = d.merge(deep[["sdss_id", "q_dyn", "q_dyn_at_bound"]], on="sdss_id", how="left")
dd = DT.load_deep().set_index("sdss_id"); meth = DT.table_method(d)
d["baseline"] = [np.ptp(DT.parse(dd.loc[s, "mjd_per_visit"])) for s in d.sdss_id]
d["dvmax"] = [DT.observed_dv(dd.loc[s], meth) for s in d.sdss_id]
d["twin_spec"] = d.twin
ok = d.q_dyn.notna() & (d.q_dyn_at_bound != True) & (d.q_dyn > 0.1) & (d.q_dyn < 1.5)
d = d[ok].copy()
d["q_sym"] = np.minimum(d.q_dyn, 1.0 / d.q_dyn)

m = d[d.K1 > 12]
print("K1>12 with usable q_dyn: %d (of 309)" % len(m))
print("overlap of labels (K1>12): spec-twin & dyn-twin %d, spec-only %d, dyn-only %d"
      % (((m.twin_spec) & (m.q_sym > 0.95)).sum(), ((m.twin_spec) & ~(m.q_sym > 0.95)).sum(),
         (~(m.twin_spec) & (m.q_sym > 0.95)).sum()))
V = []
for thr in (0.95, 0.90):
    for sub, cols, lab in [(m, ["K1", "n_ep"], "K1 + epochs"),
                           (m, ["logP", "n_ep"], "period + epochs"),
                           (d[d.dvmax > 25], ["n_ep", "baseline", "dvmax"], "observed only")]:
        s = sub.copy(); s["twin"] = s.q_sym > thr
        v = T.variant(s, cols, "q_dyn>%.2f, %s" % (thr, lab)); V.append(v)
s = m.copy(); s["twin"] = s.twin_spec
V.append(T.variant(s, ["K1", "n_ep"], "best_q>0.95 (published), same stars"))
print("%-40s %7s %7s %5s %5s" % ("variant", "dAlpha", "boot", "N_t", "N_n"))
for v in V:
    print("%-40s %+7.3f %7.3f %5d %5d" % (v["label"], v["da"], v["boot"], v["n_t"], v["n_n"]))
pd.DataFrame(V).to_csv(os.path.join(_R, "resources/census/ecc_variants_qdyn.csv"), index=False)
