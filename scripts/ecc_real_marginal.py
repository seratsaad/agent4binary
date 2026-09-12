#!/usr/bin/env python3
"""Re-measure the twin against non-twin eccentricity difference on the real
sample with the marginalized estimator (scripts/ecc_marginal.py), replacing the
rejection-sampling chain that an injection test showed to be insensitive.

--method twovel (default): two-velocity likelihood
    (ecc_marginal.system_loglike_on_egrid_twovel) on the UNTIED per-visit pairs
    rv1_untied_per_visit / rv2_untied_per_visit. The pair at each visit is
    unordered and its label assignment is summed over; q = q_dyn clipped to
    [0.1, 1.5], or q_spec when q_dyn is missing or at a bound (deep_table.q_twovel).
    Visits with a missing or non-finite pair are dropped; at least 8 usable
    visits. See docs/ecc_twovelocity_method.md.

--method v1: the previous primary-velocity-only likelihood, with
    v1_alt = gam + (gam - v2) q as the alternative. Kept for reproducibility.
    Note: v2_per_visit is the momentum tie of v1 with the same gamma and q, so
    this v1_alt equals v1 wherever q was not clipped, and the label sum then
    does nothing.

Output columns: sdss_id, twin, P50, K1, n_ep, loglike (as before), plus method.
P50 and K1 are copied from orbit_deep_posteriors.csv, which should come from the
same method (a mismatch is reported).
"""
import os, sys, argparse
import numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_ROOT, "src"))
import ecc_marginal as EM
EGRID = EM.default_egrid()
NMIN = 8

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n-shards", type=int, default=1)
ap.add_argument("--draws", type=int, default=60_000)   # per e-grid point
ap.add_argument("--sigma", type=float, default=1.5)
ap.add_argument("--method", choices=["twovel", "v1"], default="twovel")
ap.add_argument("--out", required=True)
a = ap.parse_args()

import deep_table as DT
# Epochs come from each row's mjd_per_visit (same order as v1_per_visit and the
# untied lists); load_deep checks the column exists and that every row has one
# epoch per velocity. A4B_DEEP_TABLE overrides the default resources/census/stage2_deep.csv.
deep = DT.load_deep()
cat = pd.read_csv(os.path.join(_ROOT, "resources/census/dr19_sb2_catalog_open.csv"),
                  usecols=["sdss_id", "best_q"])
orb = pd.read_csv(os.path.join(_ROOT, "resources/census/orbit_deep_posteriors.csv"))
orb_method = sorted(set(orb["method"].dropna())) if "method" in orb.columns else ["v1"]
if orb_method != [a.method]:
    print("WARNING: orbit_deep_posteriors.csv was made with method %s, this run uses %s"
          % (",".join(orb_method), a.method), flush=True)
orb = orb[["sdss_id", "status", "P50", "K1"]]
d = deep.merge(cat, on="sdss_id", how="left").merge(orb, on="sdss_id", how="left")
d = d[d.status.eq("ok") & d.best_q.notna()]
d = d[(d.P50 >= 6) & (d.P50 < 400)]

def parse(s):
    return np.array([float(x) for x in str(s).split(";") if x not in ("", "nan")])

rows = []
ids = d.sdss_id.tolist()
for idx, sid in enumerate(ids):
    if idx % a.n_shards != a.shard:
        continue
    r = d[d.sdss_id == sid].iloc[0]
    rng = np.random.default_rng(int(sid) % (2**31))
    if a.method == "twovel":
        t, pa, pb = DT.untied_pairs(r)
        n = len(t)
        q = DT.q_twovel(r)
        if n < NMIN or not np.isfinite(q):
            continue
        lg = EM.system_loglike_on_egrid_twovel(t - t.mean(), pa, pb, q, a.sigma, EGRID,
                                               M_per_e=a.draws, rng=rng)
    else:
        t = parse(r.mjd_per_visit)
        v1 = parse(r.v1_per_visit); v2 = parse(r.v2_per_visit)
        n = min(len(t), len(v1), len(v2))
        if n < NMIN:
            continue
        t, v1, v2 = t[:n], v1[:n], v2[:n]
        q = float(r.q_dyn) if np.isfinite(r.q_dyn) and 0.2 < r.q_dyn < 1.5 else float(r.q_spec)
        q = float(np.clip(q, 0.2, 1.0))
        gam = float(r.gamma) if np.isfinite(r.gamma) else float(np.mean(v1))
        v1_alt = gam + (gam - v2) * q          # tie inversion: primary if labels swapped
        lg = EM.system_loglike_on_egrid(t - t.mean(), v1, a.sigma, EGRID,
                                        M_per_e=a.draws, rng=rng, v_alt=v1_alt)
    if not np.isfinite(lg).any():
        continue
    rows.append(dict(sdss_id=int(sid), twin=bool(float(r.best_q) > 0.95),
                     P50=float(r.P50), K1=float(r.K1), n_ep=int(n),
                     loglike=";".join("%.5f" % x for x in (lg - np.nanmax(lg[np.isfinite(lg)]))),
                     method=a.method))
    if len(rows) % 100 == 0:
        print("done", len(rows), flush=True)
pd.DataFrame(rows, columns=["sdss_id", "twin", "P50", "K1", "n_ep", "loglike", "method"]).to_csv(a.out, index=False)
print("wrote %s (%d systems, method %s)" % (a.out, len(rows), a.method))
