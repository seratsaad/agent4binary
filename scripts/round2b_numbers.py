#!/usr/bin/env python3
"""Numbers for the second round-2 pass (internal referee audit, 2026-09-28).

Recomputes, from the released tables, every value the paper and reply quote in
the passages revised in this pass: the visit-contradicted coadd flag, the
metal-poor sign test, q_dyn after blanking stars with no primary motion, the
catalog-property medians, the Gaia enrichment on the fair random sample, and the
orbit-ready breakdown. Prints one line per quantity.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = lambda p: os.path.join(R, "resources", "census", p)


def ids(s):
    return s.astype(str).str.split(".").str[0].astype(np.int64)


cat = pd.read_csv(C("dr19_sb2_catalog_open.csv"), low_memory=False)
cat["sdss_id"] = ids(cat.sdss_id)
s2 = pd.read_csv(C("stage2_catalog_full.csv"), low_memory=False)
ext = pd.read_csv(C("stage2_external_sb2.csv"))
ext["sdss_id"] = ids(ext.sdss_id)
pub = set(ext[(ext[["in_kovalev22", "in_kovalev24", "in_kounkel21"]].fillna(0) > 0).any(axis=1)].sdss_id)
kou = set(ext[ext.in_kounkel21.fillna(0) > 0].sdss_id)
kov = set(ext[(ext[["in_kovalev22", "in_kovalev24"]].fillna(0) > 0).any(axis=1)].sdss_id)

m = cat.merge(s2[["sdss_id", "n_visits", "prefers_binary", "v_rad_std_astra", "v1_range",
                  "q_dyn", "q_dyn_at_bound", "q_wilson", "v1_span_gt20", "v1_per_visit",
                  "v2_per_visit"]].rename(columns={"prefers_binary": "conf"}),
              on="sdss_id", how="left")
m["pub"] = m.sdss_id.isin(pub)
m["sep"] = (m.best_rv1 - m.best_rv2).abs()
m["far"] = np.where(m.best_rv1.abs() > m.best_rv2.abs(), m.best_rv1, m.best_rv2)

print("== coadd flag (blocker 3)")
b = m[m.n_visits >= 2]
flag = (b.conf == False) & (b.v_rad_std_astra < 1) & (b.sep > 15)
P = b.pub.sum()
print("multi-visit catalog %d, flagged %d (%.1f%%), published flagged %d/%d (%.1f%%), metal-poor flagged %d"
      % (len(b), flag.sum(), 100 * flag.mean(), b[flag].pub.sum(), P, 100 * b[flag].pub.sum() / P,
         b[flag].feh_metalpoor.sum()))
for X in (10, 20):
    f = (b.conf == False) & (b.v_rad_std_astra < 1) & (b.sep > X)
    print("  threshold %d km/s: flagged %d, published %.1f%%" % (X, f.sum(), 100 * b[f].pub.sum() / P))
tw = b[(b.best_q >= 0.95) & (b.conf == False) & ((b.best_rv1 + b.best_rv2).abs() > 2)]
print("referee twins (unconfirmed, |v1+v2|>2): all %d flagged %d | two-visit %d flagged %d"
      % (len(tw), flag[tw.index].sum(), (tw.n_visits == 2).sum(), (flag[tw.index] & (tw.n_visits == 2)).sum()))
print("referee stars flagged:", bool(flag[b.sdss_id.isin([61585132, 76096409])].all()))
for lo, hi in [(0, 15), (15, 30), (30, 1e9)]:
    p = b[b.pub & (b.sep >= lo) & (b.sep < hi)]
    print("  published SB2 visit-fit preference at coadd sep %g-%g: %.1f%% (n=%d)" % (lo, hi, 100 * p.conf.mean(), len(p)))
mp, nm = m[m.feh_metalpoor == 1], m[m.feh_metalpoor == 0]
print("far component positive: metal-poor %.1f%% (n=%d), rest %.1f%% (n=%d)"
      % (100 * (mp.far > 0).mean(), len(mp), 100 * (nm.far > 0).mean(), len(nm)))
print("metal-poor far component 30-50 km/s: %.1f%%" % (100 * ((mp.far > 30) & (mp.far < 50)).mean()))

print("== visit fit (blocker 2)")
c3 = m[(m.conf == True) & (m.n_visits >= 3)]
still = c3.v1_range < 1
print("confirmed >=3 ep %d, primary change < 1 km/s %d (%.1f%%), published among them %.1f%% vs moving %.1f%%"
      % (len(c3), still.sum(), 100 * still.mean(), 100 * c3[still].pub.mean(), 100 * c3[~still].pub.mean()))
b2 = m[m.n_visits >= 2]
st = b2.v_rad_std_astra < 1
print("two-component preference: pipeline-steady %.1f%% (n=%d), moving %.1f%% (n=%d)"
      % (100 * b2[st].conf.mean(), st.sum(), 100 * b2[~st].conf.mean(), (~st).sum()))
mot = c3[c3.v1_range > 10]
print("confirmed >=3 ep with change > 10 km/s: %d" % len(mot))
unc3 = m[(m.conf == False) & (m.n_visits >= 3)]
print("median primary change: confirmed %.2f, unconfirmed %.2f km/s" % (c3.v1_range.median(), unc3.v1_range.median()))
for k in ("prefers" ,):
    pass
print("preference by epochs:", {n: round(100 * b2[b2.n_visits.clip(upper=5) == n].conf.mean(), 1) for n in (2, 3, 4, 5)})

print("== q_dyn after blanking stars with no primary motion")
qd = c3[~still]
inr = qd[qd.q_dyn.notna() & (qd.q_dyn_at_bound != True)]
print("q_dyn released %d (was %d), inside range %d, at bound %.1f%%"
      % (qd.q_dyn.notna().sum(), c3.q_dyn.notna().sum(), len(inr),
         100 * (qd.q_dyn_at_bound == True).sum() / qd.q_dyn.notna().sum()))
print("median q_dyn %.2f, median coadd q %.2f, Spearman %.2f"
      % (inr.q_dyn.median(), inr.best_q.median(), spearmanr(inr.q_dyn, inr.best_q)[0]))
w = inr[(inr.q_wilson > 0.05) & (inr.q_wilson < 3)]
print("Wilson (0.05<q_W<3): n=%d, rho(q_W,q_dyn)=%.2f, rho(q_W,coadd q)=%.2f"
      % (len(w), spearmanr(w.q_wilson, w.q_dyn)[0], spearmanr(w.q_wilson, w.best_q)[0]))
old = c3[c3.q_dyn.notna() & (c3.q_dyn_at_bound != True)]
print("  (old, all confirmed >=3 ep: inside range %d, median %.2f)" % (len(old), old.q_dyn.median()))

print("== orbit-ready")
o = m[m.v1_span_gt20 == True]
split = []
for v1, v2 in zip(o.v1_per_visit, o.v2_per_visit):
    a = np.array(str(v1).split(";"), float) - np.array(str(v2).split(";"), float)
    split.append(np.ptp(np.abs(a)))
o = o.assign(dsplit=split)
cs = o.dsplit < 5
print("orbit-ready %d, constant split %d, strict %d; Kounkel %d, Kovalev %d (also Kounkel %d), neither %d"
      % (len(o), cs.sum(), len(o) - cs.sum(), o.sdss_id.isin(kou).sum(), o.sdss_id.isin(kov).sum(),
         (o.sdss_id.isin(kov) & o.sdss_id.isin(kou)).sum(), (~o.sdss_id.isin(pub)).sum()))

print("== catalog properties")
print("N catalog %d; ratio to 2,645: %.1f" % (len(cat), len(cat) / 2645))
print("median f_imp %.2f, median coadd q %.2f" % (cat.f_imp.median(), cat.best_q.median()))
print("median |v1-v2| %.1f km/s, 90th pct %.1f; twins q>0.95 median sep %.1f"
      % (m.sep.median(), m.sep.quantile(0.9), m[m.best_q > 0.95].sep.median()))
full = pd.read_csv(os.path.join(R, "resources", "dr19_census272k_manifest.csv"), usecols=["sdss_id", "fe_h", "teff"])
print("median pipeline [Fe/H]: flagged %.2f, full sample %.2f" % (cat.feh_seed.median(), full.fe_h.median()))
ct = pd.read_csv(C("sb2_component_teff_open.csv"))
print("median Teff primary %.0f, secondary %.0f (n=%d)" % (ct.teff1.median(), ct.teff2.median(), len(ct)))
print("teff_floor flag %d; primaries with pipeline teff < 4200: %d" % (cat.teff_floor.sum(), (cat.teff_seed < 4200).sum()))

print("== Gaia fair sample")
g = pd.read_csv(C("ruwe_fair_sample.csv"))
g = g[g.error.isna()] if "error" in g else g
fl, un = g[g.prefers_binary == True], g[g.prefers_binary == False]
for name, x in (("flagged", fl), ("unflagged", un)):
    r = x[x.RUWE.notna()]
    print("%s: n=%d, RUWE>1.4 %.1f%% (n with RUWE %d), NSS %.1f%%"
          % (name, len(x), 100 * (r.RUWE > 1.4).mean(), len(r), 100 * (x.NSS.fillna(0) > 0).mean()))
q = g[(g.RUWE < 1.1) & (g.NSS.fillna(0) == 0)]
p = q.prefers_binary.mean()
print("quiet (RUWE<1.1, no NSS) flag rate %.2f%% +- %.2f (n=%d)" % (100 * p, 100 * np.sqrt(p * (1 - p) / len(q)), len(q)))
