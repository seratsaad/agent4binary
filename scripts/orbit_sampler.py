#!/usr/bin/env python3
"""Joint two-component Keplerian rejection sampler for the SB2 per-visit velocities.

The Joker-style approach (Price-Whelan et al. 2017), adapted to SB2: both
components are fit at once,

    v1(t) = v_sys + K1 [cos(nu + omega) + e cos(omega)]
    v2(t) = v_sys - (K1/q) [cos(nu + omega) + e cos(omega)],

so each epoch contributes two data points tied by the mass ratio q. Nonlinear
parameters (P, e, omega, M0) are drawn from priors; the linear pair (K1, v_sys)
is solved per draw by weighted least squares; draws are kept with probability
prop. to the likelihood. Output: posterior samples per system.

Usage:
  python scripts/orbit_sampler.py --sdss-id 80836051            # one system, verbose
  python scripts/orbit_sampler.py --ids-file ids.txt --out o.csv  # batch summary
Env: AB_NSAMP (prior draws, default 200000), AB_RVERR (per-epoch km/s, default 1.5).
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NSAMP = int(os.environ.get("AB_NSAMP", 200000))
RVERR = float(os.environ.get("AB_RVERR", 1.5))      # km/s per epoch per component
PMIN, PMAX = 0.5, 5000.0                            # days, log-uniform prior


def kepler_E(M, e, iters=20):
    """Solve Kepler's equation E - e sinE = M (vectorized Newton)."""
    E = M + e * np.sin(M)
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def rv_shape(t, P, e, omega, M0):
    """cos(nu+omega) + e cos(omega) evaluated at times t (the K=1, v_sys=0 curve)."""
    M = 2.0 * np.pi * (t[None, :] / P[:, None]) + M0[:, None]
    E = kepler_E(np.mod(M, 2 * np.pi), e[:, None])
    nu = 2.0 * np.arctan2(np.sqrt(1 + e[:, None]) * np.sin(E / 2),
                          np.sqrt(1 - e[:, None]) * np.cos(E / 2))
    return np.cos(nu + omega[:, None]) + (e * np.cos(omega))[:, None]


def sample_system(t, v1, v2, q, nsamp=NSAMP, rverr=RVERR, seed=0):
    """Rejection-sample joint SB2 Keplerian. Returns dict of posterior samples."""
    rng = np.random.default_rng(seed)
    n = len(t)
    P = np.exp(rng.uniform(np.log(PMIN), np.log(PMAX), nsamp))
    e = rng.uniform(0.0, 0.95, nsamp)               # flat interim prior; population f(e) reweights it
    omega = rng.uniform(0, 2 * np.pi, nsamp)
    M0 = rng.uniform(0, 2 * np.pi, nsamp)

    S = rv_shape(t, P, e, omega, M0)                # (nsamp, n)
    w = 1.0 / rverr**2
    # Joint linear solve for (K1, v_sys):
    #   v1_i = v_sys + K1 S_i          v2_i = v_sys - (K1/q) S_i
    # Design per system: rows [S_i, 1] for v1 and [-S_i/q, 1] for v2.
    a11 = (S**2).sum(1) * (1 + 1.0 / q**2) * w
    a12 = S.sum(1) * (1 - 1.0 / q) * w
    a22 = 2 * n * w
    b1 = w * ((S * v1[None, :]).sum(1) - (S * v2[None, :]).sum(1) / q)
    b2 = w * (v1.sum() + v2.sum()) * np.ones(nsamp)
    det = a11 * a22 - a12**2
    det[det == 0] = 1e-30
    K1 = (b1 * a22 - b2 * a12) / det
    v_sys = (a11 * b2 - a12 * b1) / det
    # chi2 of the joint fit
    m1 = v_sys[:, None] + K1[:, None] * S
    m2 = v_sys[:, None] - (K1[:, None] / q) * S
    chi2 = w * (((v1[None, :] - m1) ** 2).sum(1) + ((v2[None, :] - m2) ** 2).sum(1))
    # reject unphysical (retrograde K) and rejection-sample on likelihood
    ok = K1 > 0
    chi2[~ok] = np.inf
    logL = -0.5 * chi2
    logL -= logL.max()
    keep = rng.random(nsamp) < np.exp(logL)
    return dict(P=P[keep], e=e[keep], K1=K1[keep], v_sys=v_sys[keep],
                omega=omega[keep], chi2=chi2[keep], n_kept=keep.sum(),
                chi2_min=chi2.min(), ndata=2 * n)


def load_system(sid):
    s2 = pd.read_csv(os.path.join(_ROOT, "resources/census/stage2_catalog_full.csv"))
    r = s2[s2.sdss_id == sid].iloc[0]
    v1 = np.array([float(x) for x in str(r.v1_per_visit).split(";") if x not in ("", "nan")])
    v2 = np.array([float(x) for x in str(r.v2_per_visit).split(";") if x not in ("", "nan")])
    z = np.load(os.path.join(_ROOT, f"data/dr19_raw_visits/{sid}.npz"), allow_pickle=True)
    t = np.asarray(z["mjd"], float)
    nn = min(len(v1), len(v2), len(t))
    q = float(r.q_dyn) if np.isfinite(r.q_dyn) and r.q_dyn > 0 else float(r.q_spec)
    return t[:nn], v1[:nn], v2[:nn], np.clip(q, 0.1, 1.0)


def summarize(sid, s):
    if s["n_kept"] < 10:
        return dict(sdss_id=sid, n_kept=int(s["n_kept"]), status="unconstrained")
    pctl = lambda x: np.percentile(x, [16, 50, 84])
    P16, P50, P84 = pctl(s["P"]); e16, e50, e84 = pctl(s["e"])
    return dict(sdss_id=sid, n_kept=int(s["n_kept"]), status="ok",
                P50=P50, P16=P16, P84=P84, e50=e50, e16=e16, e84=e84,
                K1=np.median(s["K1"]), v_sys=np.median(s["v_sys"]),
                chi2_min=s["chi2_min"], ndata=s["ndata"])


def sample_system_refined(t, v1, v2, q, nsamp=NSAMP, rverr=RVERR, seed=0, target=256, mtot=1.6, rsum=2.0):
    """Global rejection pass, then local re-draws around the surviving period
    modes until `target` posterior samples are kept. Acceptance is always
    anchored to the global best chi2, so the combined kept set stays a fair
    (locally exact, mode-weighted) posterior sample."""
    rng = np.random.default_rng(seed)
    n = len(t)

    def _batch(P, e, omega, M0):
        """Linear solve for (K1, v_sys) per draw, with per-epoch component-swap
        marginalization: at q near 1 the pipeline can exchange the two
        components' velocity labels between visits, so each epoch is evaluated
        under both assignments and the better one is kept (then the linear pair
        is re-solved once under the chosen assignments)."""
        S = rv_shape(t, P, e, omega, M0)
        w = 1.0 / rverr**2
        a11 = (S**2).sum(1) * (1 + 1.0 / q**2) * w
        a12 = S.sum(1) * (1 - 1.0 / q) * w
        a22 = 2 * n * w
        det = a11 * a22 - a12**2
        det[det == 0] = 1e-30

        def _solve(b1, b2):
            K1 = (b1 * a22 - b2 * a12) / det
            v_sys = (a11 * b2 - a12 * b1) / det
            return K1, v_sys

        b2 = w * (v1.sum() + v2.sum()) * np.ones(len(P))
        # pass 1: nominal assignment
        b1 = w * ((S * v1[None, :]).sum(1) - (S * v2[None, :]).sum(1) / q)
        K1, v_sys = _solve(b1, b2)
        m1 = v_sys[:, None] + K1[:, None] * S
        m2 = v_sys[:, None] - (K1[:, None] / q) * S
        c_nom = (v1[None, :] - m1) ** 2 + (v2[None, :] - m2) ** 2
        c_swp = (v2[None, :] - m1) ** 2 + (v1[None, :] - m2) ** 2
        swap = c_swp < c_nom                       # (ndraw, nepoch)
        # pass 2: re-solve with the chosen per-epoch assignment
        d1 = np.where(swap, v2[None, :], v1[None, :])
        d2 = np.where(swap, v1[None, :], v2[None, :])
        b1 = w * ((S * d1).sum(1) - (S * d2).sum(1) / q)
        K1, v_sys = _solve(b1, b2)
        m1 = v_sys[:, None] + K1[:, None] * S
        m2 = v_sys[:, None] - (K1[:, None] / q) * S
        chi2 = w * (((d1 - m1) ** 2).sum(1) + ((d2 - m2) ** 2).sum(1))
        chi2[K1 <= 0] = np.inf
        # periapsis-contact rejection: a(1-e) must exceed R1+R2. Kepler III with
        # M_tot in Msun, P in days -> a in Rsun (1 au = 215.032 Rsun).
        a_rsun = 215.032 * (mtot * (P / 365.25) ** 2) ** (1.0 / 3.0)
        chi2[a_rsun * (1.0 - e) < rsum] = np.inf
        return K1, v_sys, chi2

    # stage 1: global
    P = np.exp(rng.uniform(np.log(PMIN), np.log(PMAX), nsamp))
    e = rng.uniform(0.0, 0.95, nsamp)
    om = rng.uniform(0, 2 * np.pi, nsamp)
    M0 = rng.uniform(0, 2 * np.pi, nsamp)
    K1, v_sys, chi2 = _batch(P, e, om, M0)
    gbest = chi2.min()
    keep = rng.random(nsamp) < np.exp(-0.5 * (chi2 - gbest))
    seeds_P = P[chi2 < gbest + 25.0]           # survivors define the modes
    if len(seeds_P) == 0:
        seeds_P = P[np.argsort(chi2)[:8]]
    kept = {k: [v[keep]] for k, v in dict(P=P, e=e, K1=K1, v_sys=v_sys, omega=om, chi2=chi2).items()}
    total = keep.sum()

    # stage 2: local refinement around surviving modes
    for _ in range(12):
        if total >= target:
            break
        m = nsamp // 2
        base = rng.choice(seeds_P, m)
        Pl = base * np.exp(rng.normal(0, 0.01, m))
        el = rng.uniform(0.0, 0.95, m)
        oml = rng.uniform(0, 2 * np.pi, m)
        M0l = rng.uniform(0, 2 * np.pi, m)
        K1l, gl, c2l = _batch(Pl, el, oml, M0l)
        gbest = min(gbest, c2l.min())
        kl = rng.random(m) < np.exp(-0.5 * (c2l - gbest))
        for k, v in dict(P=Pl, e=el, K1=K1l, v_sys=gl, omega=oml, chi2=c2l).items():
            kept[k].append(v[kl])
        total += kl.sum()
    out = {k: np.concatenate(v) for k, v in kept.items()}
    out["n_kept"] = len(out["P"]); out["chi2_min"] = gbest; out["ndata"] = 2 * n
    return out


def sample_system_refined(t, v1, v2, q, nsamp=NSAMP, rverr=RVERR, seed=0, target=256, mtot=1.6, rsum=2.0):
    """Global rejection pass, then local re-draws around the surviving period
    modes until `target` posterior samples are kept. Acceptance is always
    anchored to the global best chi2, so the combined kept set stays a fair
    (locally exact, mode-weighted) posterior sample."""
    rng = np.random.default_rng(seed)
    n = len(t)

    def _batch(P, e, omega, M0):
        """Linear solve for (K1, v_sys) per draw, with per-epoch component-swap
        marginalization: at q near 1 the pipeline can exchange the two
        components' velocity labels between visits, so each epoch is evaluated
        under both assignments and the better one is kept (then the linear pair
        is re-solved once under the chosen assignments)."""
        S = rv_shape(t, P, e, omega, M0)
        w = 1.0 / rverr**2
        a11 = (S**2).sum(1) * (1 + 1.0 / q**2) * w
        a12 = S.sum(1) * (1 - 1.0 / q) * w
        a22 = 2 * n * w
        det = a11 * a22 - a12**2
        det[det == 0] = 1e-30

        def _solve(b1, b2):
            K1 = (b1 * a22 - b2 * a12) / det
            v_sys = (a11 * b2 - a12 * b1) / det
            return K1, v_sys

        b2 = w * (v1.sum() + v2.sum()) * np.ones(len(P))
        # pass 1: nominal assignment
        b1 = w * ((S * v1[None, :]).sum(1) - (S * v2[None, :]).sum(1) / q)
        K1, v_sys = _solve(b1, b2)
        m1 = v_sys[:, None] + K1[:, None] * S
        m2 = v_sys[:, None] - (K1[:, None] / q) * S
        c_nom = (v1[None, :] - m1) ** 2 + (v2[None, :] - m2) ** 2
        c_swp = (v2[None, :] - m1) ** 2 + (v1[None, :] - m2) ** 2
        swap = c_swp < c_nom                       # (ndraw, nepoch)
        # pass 2: re-solve with the chosen per-epoch assignment
        d1 = np.where(swap, v2[None, :], v1[None, :])
        d2 = np.where(swap, v1[None, :], v2[None, :])
        b1 = w * ((S * d1).sum(1) - (S * d2).sum(1) / q)
        K1, v_sys = _solve(b1, b2)
        m1 = v_sys[:, None] + K1[:, None] * S
        m2 = v_sys[:, None] - (K1[:, None] / q) * S
        chi2 = w * (((d1 - m1) ** 2).sum(1) + ((d2 - m2) ** 2).sum(1))
        chi2[K1 <= 0] = np.inf
        # periapsis-contact rejection: a(1-e) must exceed R1+R2. Kepler III with
        # M_tot in Msun, P in days -> a in Rsun (1 au = 215.032 Rsun).
        a_rsun = 215.032 * (mtot * (P / 365.25) ** 2) ** (1.0 / 3.0)
        chi2[a_rsun * (1.0 - e) < rsum] = np.inf
        return K1, v_sys, chi2

    # stage 1: global
    P = np.exp(rng.uniform(np.log(PMIN), np.log(PMAX), nsamp))
    e = rng.uniform(0.0, 0.95, nsamp)
    om = rng.uniform(0, 2 * np.pi, nsamp)
    M0 = rng.uniform(0, 2 * np.pi, nsamp)
    K1, v_sys, chi2 = _batch(P, e, om, M0)
    gbest = chi2.min()
    keep = rng.random(nsamp) < np.exp(-0.5 * (chi2 - gbest))
    seeds_P = P[chi2 < gbest + 25.0]           # survivors define the modes
    if len(seeds_P) == 0:
        seeds_P = P[np.argsort(chi2)[:8]]
    kept = {k: [v[keep]] for k, v in dict(P=P, e=e, K1=K1, v_sys=v_sys, omega=om, chi2=chi2).items()}
    total = keep.sum()

    # stage 2: local refinement around surviving modes
    for _ in range(12):
        if total >= target:
            break
        m = nsamp // 2
        base = rng.choice(seeds_P, m)
        Pl = base * np.exp(rng.normal(0, 0.01, m))
        el = rng.uniform(0.0, 0.95, m)
        oml = rng.uniform(0, 2 * np.pi, m)
        M0l = rng.uniform(0, 2 * np.pi, m)
        K1l, gl, c2l = _batch(Pl, el, oml, M0l)
        gbest = min(gbest, c2l.min())
        kl = rng.random(m) < np.exp(-0.5 * (c2l - gbest))
        for k, v in dict(P=Pl, e=el, K1=K1l, v_sys=gl, omega=oml, chi2=c2l).items():
            kept[k].append(v[kl])
        total += kl.sum()
    out = {k: np.concatenate(v) for k, v in kept.items()}
    out["n_kept"] = len(out["P"]); out["chi2_min"] = gbest; out["ndata"] = 2 * n
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdss-id", type=int)
    ap.add_argument("--ids-file")
    ap.add_argument("--out")
    a = ap.parse_args()
    ids = [a.sdss_id] if a.sdss_id else [int(x) for x in open(a.ids_file).read().split()]
    rows = []
    for sid in ids:
        try:
            t, v1, v2, q = load_system(sid)
            s = sample_system_refined(t, v1, v2, q, seed=sid % 2**31)
            r = summarize(sid, s)
        except Exception as exc:
            r = dict(sdss_id=sid, status="error:%s" % type(exc).__name__)
        rows.append(r)
        if a.sdss_id:
            print(r)
    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print("wrote", a.out, len(rows))





def sample_v1_refined(t, v1_obs, v1_alt, nsamp=NSAMP, rverr=RVERR, seed=0,
                      target=256, mtot=1.6, rsum=2.0):
    """Primary-velocity-only sampler. The stage-2 per-visit v2 is the exact
    momentum tie of v1 and carries no independent information, so the
    likelihood uses v1 alone. Component swaps are handled by giving each epoch
    two candidate primary velocities: the reported v1 and its tie inversion
    v1_alt (what the primary was if the labels were exchanged that epoch); the
    better of the two is kept per epoch per draw and the linear pair
    (K1, v_sys) is re-solved once under the chosen assignment."""
    rng = np.random.default_rng(seed)
    n = len(t)

    def _batch(P, e, omega, M0, s_jit):
        """s_jit: per-draw jitter (km/s), marginalized; the acceptance statistic
        is -2 ln L = chi2(s) + 2 n ln s, so noisy solutions pay the Occam cost."""
        S = rv_shape(t, P, e, omega, M0)
        # linear solve is jitter-independent (uniform weights cancel)
        a11 = (S**2).sum(1)
        a12 = S.sum(1)
        a22 = float(n)
        det = a11 * a22 - a12**2
        det[det == 0] = 1e-30

        def _solve(d):
            b1 = (S * d).sum(1)
            b2 = d.sum(1)
            K1 = (b1 * a22 - b2 * a12) / det
            v_sys = (a11 * b2 - a12 * b1) / det
            return K1, v_sys

        d0 = np.broadcast_to(v1_obs, (len(P), n))
        K1, v_sys = _solve(d0)
        m = v_sys[:, None] + K1[:, None] * S
        use_alt = (v1_alt[None, :] - m) ** 2 < (v1_obs[None, :] - m) ** 2
        d = np.where(use_alt, v1_alt[None, :], v1_obs[None, :])
        K1, v_sys = _solve(d)
        m = v_sys[:, None] + K1[:, None] * S
        rss = ((d - m) ** 2).sum(1)
        nll2 = rss / s_jit**2 + 2.0 * n * np.log(s_jit)   # -2 lnL up to const
        nll2[K1 <= 0] = np.inf
        a_rsun = 215.032 * (mtot * (P / 365.25) ** 2) ** (1.0 / 3.0)
        nll2[a_rsun * (1.0 - e) < rsum] = np.inf
        return K1, v_sys, nll2

    P = np.exp(rng.uniform(np.log(PMIN), np.log(PMAX), nsamp))
    e = rng.uniform(0.0, 0.95, nsamp)
    om = rng.uniform(0, 2 * np.pi, nsamp)
    M0 = rng.uniform(0, 2 * np.pi, nsamp)
    sj = np.exp(rng.uniform(np.log(0.5), np.log(10.0), nsamp))
    K1, v_sys, chi2 = _batch(P, e, om, M0, sj)
    gbest = chi2.min()
    keep = rng.random(nsamp) < np.exp(-0.5 * (chi2 - gbest))
    seeds_P = P[chi2 < gbest + 25.0]
    if len(seeds_P) == 0:
        seeds_P = P[np.argsort(chi2)[:8]]
    kept = {k: [v[keep]] for k, v in dict(P=P, e=e, K1=K1, v_sys=v_sys, omega=om, chi2=chi2, s_jit=sj).items()}
    total = keep.sum()
    for _ in range(12):
        if total >= target:
            break
        m_ = nsamp // 2
        base = rng.choice(seeds_P, m_)
        Pl = base * np.exp(rng.normal(0, 0.01, m_))
        el = rng.uniform(0.0, 0.95, m_)
        oml = rng.uniform(0, 2 * np.pi, m_)
        M0l = rng.uniform(0, 2 * np.pi, m_)
        sjl = np.exp(rng.uniform(np.log(0.5), np.log(10.0), m_))
        K1l, gl, c2l = _batch(Pl, el, oml, M0l, sjl)
        gbest = min(gbest, c2l.min())
        kl = rng.random(m_) < np.exp(-0.5 * (c2l - gbest))
        for k, v in dict(P=Pl, e=el, K1=K1l, v_sys=gl, omega=oml, chi2=c2l, s_jit=sjl).items():
            kept[k].append(v[kl])
        total += kl.sum()
    out = {k: np.concatenate(v) for k, v in kept.items()}
    out["n_kept"] = len(out["P"]); out["chi2_min"] = gbest; out["ndata"] = n
    return out


def sample_v1_t(t, v1_obs, v1_alt, nsamp=NSAMP, seed=0, target=256,
                mtot=1.6, rsum=2.0, nu=4.0):
    """Student-t variant of sample_v1_refined for outlier-contaminated epochs.

    Same structure (global pass, local refinement, tie-inversion swaps, jitter
    marginalization, contact prior), but the acceptance statistic is the
    Student-t log likelihood with nu degrees of freedom, and the linear pair
    (K1, v_sys) is re-solved once with the t-weights so a single outlier epoch
    does not drag the amplitude."""
    rng = np.random.default_rng(seed)
    n = len(t)

    def _batch(P, e, omega, M0, s_jit):
        S = rv_shape(t, P, e, omega, M0)
        a11 = (S**2).sum(1); a12 = S.sum(1); a22 = float(n)
        det = a11 * a22 - a12**2
        det[det == 0] = 1e-30

        def _solve(d, W=None):
            if W is None:
                b1 = (S * d).sum(1); b2 = d.sum(1)
                K1 = (b1 * a22 - b2 * a12) / det
                v_sys = (a11 * b2 - a12 * b1) / det
                return K1, v_sys
            A11 = (W * S**2).sum(1); A12 = (W * S).sum(1); A22 = W.sum(1)
            Dt = A11 * A22 - A12**2
            Dt[Dt == 0] = 1e-30
            b1 = (W * S * d).sum(1); b2 = (W * d).sum(1)
            K1 = (b1 * A22 - b2 * A12) / Dt
            v_sys = (A11 * b2 - A12 * b1) / Dt
            return K1, v_sys

        # pass 1: plain solve, pick swap assignment
        d0 = np.broadcast_to(v1_obs, (len(P), n))
        K1, v_sys = _solve(d0)
        m = v_sys[:, None] + K1[:, None] * S
        use_alt = (v1_alt[None, :] - m) ** 2 < (v1_obs[None, :] - m) ** 2
        d = np.where(use_alt, v1_alt[None, :], v1_obs[None, :])
        # pass 2: robust re-solve with t-weights
        K1, v_sys = _solve(d)
        m = v_sys[:, None] + K1[:, None] * S
        r2 = (d - m) ** 2 / s_jit[:, None] ** 2
        W = (nu + 1.0) / (nu + r2)
        K1, v_sys = _solve(d, W)
        m = v_sys[:, None] + K1[:, None] * S
        r2 = (d - m) ** 2 / s_jit[:, None] ** 2
        # -2 lnL for Student-t (up to const): (nu+1) sum ln(1+r2/nu) + 2 n ln s
        nll2 = (nu + 1.0) * np.log1p(r2 / nu).sum(1) + 2.0 * n * np.log(s_jit)
        nll2[K1 <= 0] = np.inf
        a_rsun = 215.032 * (mtot * (P / 365.25) ** 2) ** (1.0 / 3.0)
        nll2[a_rsun * (1.0 - e) < rsum] = np.inf
        return K1, v_sys, nll2

    P = np.exp(rng.uniform(np.log(PMIN), np.log(PMAX), nsamp))
    e = rng.uniform(0.0, 0.95, nsamp)
    om = rng.uniform(0, 2 * np.pi, nsamp)
    M0 = rng.uniform(0, 2 * np.pi, nsamp)
    sj = np.exp(rng.uniform(np.log(0.5), np.log(10.0), nsamp))
    K1, v_sys, chi2 = _batch(P, e, om, M0, sj)
    gbest = chi2.min()
    keep = rng.random(nsamp) < np.exp(-0.5 * (chi2 - gbest))
    seeds_P = P[chi2 < gbest + 25.0]
    if len(seeds_P) == 0:
        seeds_P = P[np.argsort(chi2)[:8]]
    kept = {k: [v[keep]] for k, v in dict(P=P, e=e, K1=K1, v_sys=v_sys,
                                          omega=om, M0=M0, chi2=chi2, s_jit=sj).items()}
    total = keep.sum()
    for _ in range(12):
        if total >= target:
            break
        m_ = nsamp // 2
        base = rng.choice(seeds_P, m_)
        Pl = base * np.exp(rng.normal(0, 0.01, m_))
        el = rng.uniform(0.0, 0.95, m_)
        oml = rng.uniform(0, 2 * np.pi, m_)
        M0l = rng.uniform(0, 2 * np.pi, m_)
        sjl = np.exp(rng.uniform(np.log(0.5), np.log(10.0), m_))
        K1l, gl, c2l = _batch(Pl, el, oml, M0l, sjl)
        gbest = min(gbest, c2l.min())
        kl = rng.random(m_) < np.exp(-0.5 * (c2l - gbest))
        for k, v in dict(P=Pl, e=el, K1=K1l, v_sys=gl, omega=oml, M0=M0l,
                         chi2=c2l, s_jit=sjl).items():
            kept[k].append(v[kl])
        total += kl.sum()
    out = {k: np.concatenate(v) for k, v in kept.items()}
    out["n_kept"] = len(out["P"]); out["chi2_min"] = gbest; out["ndata"] = n
    return out
