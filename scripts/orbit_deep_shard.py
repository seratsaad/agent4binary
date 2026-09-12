#!/usr/bin/env python3
"""Deep orbit sampling: Keplerian posteriors for the 16-visit refits.

--method twovel (default): two-velocity sampler (orbit_sampler.sample_twovel_refined)
    on the UNTIED per-visit component velocities rv1_untied_per_visit /
    rv2_untied_per_visit. Each visit's pair is unordered; the label assignment is
    summed over per visit, and both velocities must be matched, so a label
    exchange cannot fold the curve. q = q_dyn clipped to [0.1, 1.5], or q_spec
    when q_dyn is missing or at a bound (deep_table.q_twovel). Visits with a
    missing or non-finite pair are dropped; at least 6 usable visits.
    See docs/ecc_twovelocity_method.md.

--method v1: the previous primary-velocity-only sampler (sample_v1_refined).
    The per-visit v2 is the exact momentum tie of v1 and carries no independent
    information; component swaps are handled by the tie inversion
    v1_alt = gamma - (v1 - gamma)/q with the SAME gamma and q the stage-2 fit used.
    Kept so the earlier result stays reproducible. It can fold the curve about
    gamma at q near 1 (the twin K1 is then too small).

Output columns are the same for both methods, plus `method`. Contact prior per
system from orbit_run_params where available.
"""
import os, sys, argparse
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from orbit_sampler import sample_v1_refined, sample_twovel_refined

NMIN = 6

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--n-shards", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--method", choices=["twovel", "v1"], default="twovel")
a = ap.parse_args()

import deep_table as DT
# Epochs come from each row's mjd_per_visit (same order as v1_per_visit and the
# untied lists); load_deep checks the column exists and that every row has one
# epoch per velocity. A4B_DEEP_TABLE overrides the default resources/census/stage2_deep.csv.
deep = DT.load_deep().set_index("sdss_id")
try:
    par = pd.read_csv(os.path.join(_ROOT, "resources/census/orbit_run_params.csv")).set_index("sdss_id")
except Exception:
    par = pd.DataFrame()


def run_v1(sid, r, mtot, rsum):
    v1 = np.array([float(x) for x in str(r.v1_per_visit).split(";") if x not in ("", "nan")])
    t = np.array([float(x) for x in str(r.mjd_per_visit).split(";") if x not in ("", "nan")])
    n = min(len(v1), len(t))
    if n < NMIN:
        return None, n
    v1, t = v1[:n], t[:n]
    q = float(r.q_dyn) if np.isfinite(r.q_dyn) and r.q_dyn > 0 else float(r.q_spec)
    q = float(np.clip(q, 0.15, 1.0))
    g = float(r.gamma)
    v1_alt = g - (v1 - g) / q
    return sample_v1_refined(t, v1, v1_alt, seed=sid % 2**31, mtot=mtot, rsum=rsum), n


def run_twovel(sid, r, mtot, rsum):
    t, pa, pb = DT.untied_pairs(r)
    n = len(t)
    q = DT.q_twovel(r)
    if n < NMIN or not np.isfinite(q):
        return None, n
    return sample_twovel_refined(t, pa, pb, q, seed=sid % 2**31, mtot=mtot, rsum=rsum), n


ids = sorted(deep.index)[a.shard::a.n_shards]
rows = []
for sid in ids:
    try:
        r = deep.loc[sid]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        mtot, rsum = 1.6, 2.0
        if sid in par.index:
            mtot, rsum = float(par.loc[sid, "mtot"]), float(par.loc[sid, "rsum"])
        s, n = (run_twovel if a.method == "twovel" else run_v1)(sid, r, mtot, rsum)
        if s is None:
            rows.append(dict(sdss_id=sid, status="too_few_epochs", n_ep=n, method=a.method)); continue
        if s["n_kept"] < 8:
            rows.append(dict(sdss_id=sid, status="unconstrained", n_ep=n,
                             n_kept=int(s["n_kept"]), method=a.method)); continue
        k = min(256, s["n_kept"])
        pick = np.random.default_rng(sid % 2**31).choice(s["n_kept"], k, replace=False)
        pc = lambda x: np.percentile(x, [16, 50, 84])
        P16, P50, P84 = pc(s["P"]); e16, e50, e84 = pc(s["e"])
        rows.append(dict(
            sdss_id=sid, status="ok", n_ep=n, n_kept=int(s["n_kept"]),
            P16=P16, P50=P50, P84=P84, e16=e16, e50=e50, e84=e84,
            K1=float(np.median(s["K1"])), v_sys=float(np.median(s["v_sys"])),
            s_jit=float(np.median(s["s_jit"])), chi2_min=s["chi2_min"],
            P_samp=";".join("%.4f" % x for x in s["P"][pick]),
            e_samp=";".join("%.4f" % x for x in s["e"][pick]), method=a.method))
    except Exception as exc:
        rows.append(dict(sdss_id=sid, status="error:%s" % type(exc).__name__, method=a.method))
cols = ["sdss_id", "status", "n_ep", "n_kept", "P16", "P50", "P84", "e16", "e50", "e84",
        "K1", "v_sys", "s_jit", "chi2_min", "P_samp", "e_samp", "method"]
pd.DataFrame(rows).reindex(columns=cols).to_csv(a.out, index=False)
print("shard %d: %d systems (method %s)" % (a.shard, len(rows), a.method))
