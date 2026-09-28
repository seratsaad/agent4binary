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

Local only; reuses the per-system likelihoods in ecc_marginal_real.csv, and also
writes the relabelled copy ecc_marginal_real_qdyn.csv (usable q_dyn rows only).
The null is rescored by ecc_null_relabel_qdyn.py, the validation is
ecc_val_suite.py --twin-label qdyn, and ecc_headline.py combines the three.
Prints the six matching variants of ecc_final_table.py for q_sym > 0.95 and > 0.90
and writes resources/census/ecc_variants_qdyn.csv and ecc_variants_qdyn090.csv.
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
def variants(thr):
    """The six matching variants of ecc_final_table.py, with twins = q_sym > thr."""
    lab = lambda s: s.assign(twin=s.q_sym > thr)
    cat = pd.read_csv(os.path.join(_R, "resources/census/dr19_sb2_catalog_full.csv"))[["sdss_id", "prefers_binary"]]
    mm = m.merge(cat, on="sdss_id", how="left")
    return [T.variant(lab(m), ["K1", "n_ep"], "K1 + epochs (fiducial)"),
            T.variant(lab(m), ["logP", "n_ep"], "period + epochs"),
            T.variant(lab(m), ["logP", "n_ep", "K1"], "period + epochs + K1"),
            T.variant(lab(d[(d.K1 > 12) & (d.P50 > 15)]), ["logP", "n_ep"], "period + epochs, P>15d"),
            T.variant(lab(d[d.dvmax > 25]), ["n_ep", "baseline", "dvmax"], "observed only (no fitted vars)"),
            T.variant(lab(mm[mm.prefers_binary == True]), ["K1", "n_ep"], "confident detections only")]

for thr, name in ((0.95, "ecc_variants_qdyn.csv"), (0.90, "ecc_variants_qdyn090.csv")):
    V = variants(thr)
    print("twins q_sym > %.2f" % thr)
    print("%-34s %7s %7s %5s %5s" % ("variant", "dAlpha", "boot", "N_t", "N_n"))
    for v in V:
        print("%-34s %+7.3f %7.3f %5d %5d" % (v["label"], v["da"], v["boot"], v["n_t"], v["n_n"]))
    pd.DataFrame(V).to_csv(os.path.join(_R, "resources/census", name), index=False)

out = d.drop(columns=["q_dyn", "q_dyn_at_bound", "baseline", "dvmax", "twin_spec", "q_sym"]).assign(twin=d.q_sym > 0.95)
out.to_csv(os.path.join(_R, "resources/census/ecc_marginal_real_qdyn.csv"), index=False)
