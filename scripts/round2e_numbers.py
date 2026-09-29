#!/usr/bin/env python3
"""Remaining multi-epoch numbers quoted in the paper and reply, recomputed after the
refit of the 141 stars whose fitted visits included one with no pipeline velocity
(stage2_framefix_shards). Complements round2b_numbers.py."""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = lambda p: os.path.join(R, "resources", "census", p)
s = pd.read_csv(os.environ.get("S2", C("stage2_catalog_full.csv")), low_memory=False)
cat = pd.read_csv(os.environ.get("CAT", C("dr19_sb2_catalog_open.csv")), low_memory=False)
cat["sdss_id"] = cat.sdss_id.astype(str).str.split(".").str[0].astype(np.int64)
ext = pd.read_csv(C("stage2_external_sb2.csv")); ext["sdss_id"] = ext.sdss_id.astype(str).str.split(".").str[0].astype(np.int64)
pub = set(ext[(ext[["in_kovalev22", "in_kovalev24", "in_kounkel21"]].fillna(0) > 0).any(axis=1)].sdss_id)
m = s.merge(cat[["sdss_id", "best_q", "best_rv1", "best_rv2"]], on="sdss_id", how="left")
b = m[(m.in_sb2_catalog == True) & m.error.isna() & (m.n_visits >= 2)]
cf = b.prefers_binary == True
print("refit catalog >=2 ep %d, confirmed %d (%.1f%%)" % (len(b), cf.sum(), 100 * cf.mean()))
pb = b.sdss_id.isin(pub)
print("published %d, confirmed %.1f%%, rest %.1f%%" % (pb.sum(), 100 * b[pb].prefers_binary.eq(True).mean(), 100 * b[~pb].prefers_binary.eq(True).mean()))
sep = (b.best_rv1 - b.best_rv2).abs()
print("coadd sep <15: %.1f%%, 15-30: %.1f%%" % (100 * cf[sep < 15].mean(), 100 * cf[(sep >= 15) & (sep < 30)].mean()))
tw = b.best_q > 0.95
print("twins %.1f%% vs non-twins %.1f%%" % (100 * cf[tw].mean(), 100 * cf[~tw].mean()))
print("catalog >=3 ep %d" % (b.n_visits >= 3).sum())
c3 = b[cf & (b.n_visits >= 3)]
sep_v = []
for v1, v2 in zip(c3.v1_per_visit, c3.v2_per_visit):
    a = np.abs(np.array(str(v1).split(";"), float) - np.array(str(v2).split(";"), float)); sep_v.append(np.median(a))
conf_all = b[cf]
sv = [np.median(np.abs(np.array(str(x).split(";"), float) - np.array(str(y).split(";"), float))) for x, y in zip(conf_all.v1_per_visit, conf_all.v2_per_visit)]
print("median per-visit separation of confirmed: %.1f km/s" % np.median(sv))
o = b[b.v1_span_gt20 == True]
n = o.mjd_per_visit.astype(str).apply(lambda x: len(set(np.floor(np.array(x.split(";"), float)))))
print("v1_span_gt20 %d, >=4 nights %d; published %.1f%%" % (len(o), (n >= 4).sum(), 100 * o.sdss_id.isin(pub).mean()))
spl = []
for v1, v2 in zip(o.v1_per_visit, o.v2_per_visit):
    spl.append(np.ptp(np.abs(np.array(str(v1).split(";"), float) - np.array(str(v2).split(";"), float))))
spl = np.array(spl); cs = spl < 5
print("constant split %d, of which q>0.9 %d, published %d" % (cs.sum(), (o.best_q.values[cs] > 0.9).sum(), o.sdss_id.isin(pub).values[cs].sum()))
q = s.q_dyn.dropna()
print("q_dyn %d, flagged %d (%.1f%%), exactly 0.5: %d, 1: %d" % (len(q), (s.q_dyn_at_bound == True).sum(), 100 * (s.q_dyn_at_bound == True).sum() / len(q), (abs(q - 0.5) < 1e-3).sum(), (abs(q - 1) < 1e-3).sum()))
x = s[s.q_dyn.notna() & (s.q_dyn_at_bound != True)]
print("unflagged q_dyn %d, >1: %.1f%%, rho(q_dyn, visit q_spec) %.2f" % (len(x), 100 * (x.q_dyn > 1).mean(), spearmanr(x.q_dyn, x.q_spec)[0]))
cc = b[cf & (b.n_visits >= 3) & (b.v1_range >= 1)]
d = cc.dropna(subset=["dv_rad_max_astra"])
print("Spearman(v1_range, pipeline change) confirmed>=3: %.2f; coadd q<0.7: %.2f" % (spearmanr(d.v1_range, d.dv_rad_max_astra)[0], spearmanr(d[d.best_q < 0.7].v1_range, d[d.best_q < 0.7].dv_rad_max_astra)[0]))
dd = b[cf & (b.n_visits >= 3)].dropna(subset=["dv_rad_max_astra"])
print("  (all confirmed>=3, incl. still: %.2f; q<0.7: %.2f)" % (spearmanr(dd.v1_range, dd.dv_rad_max_astra)[0], spearmanr(dd[dd.best_q < 0.7].v1_range, dd[dd.best_q < 0.7].dv_rad_max_astra)[0]))
un = m[(m.prefers_binary == False) & m.error.isna() & (m.refit == True) & (m.v_rad_std_astra < 1)]
dv = un.v_single - un.v_rad_median_astra
print("single-star frame check: n=%d median %+.3f robust %.3f" % (len(dv.dropna()), dv.median(), 1.4826 * np.median(np.abs(dv - dv.median()).dropna())))
tw2 = b[(b.n_visits == 2)]
print("two-visit rows %d, confirmed %d, unconfirmed twins q>=0.95 %d" % (len(tw2), (tw2.prefers_binary == True).sum(), ((tw2.prefers_binary == False) & (tw2.best_q >= 0.95)).sum()))
print("velocity-confirmed x FP: about %.0f" % (0.4 * (b.n_visits >= 3).sum() * 6 / 83))
