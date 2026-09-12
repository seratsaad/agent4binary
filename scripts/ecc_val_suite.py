#!/usr/bin/env python3
"""Validation suite for the marginalized eccentricity estimator: repeat the
injection at several true differences, several realizations each, on matched
cadences, and report the recovered difference. Establishes both the transfer
slope and the realization-to-realization scatter, so the real measurement can
be quoted with an honest error.

--method twovel (default): each mock is built like a real untied pair and fitted
    with the same two-velocity likelihood as the data
    (ecc_marginal.system_loglike_on_egrid_twovel): real cadence, P, e, omega, M0,
    K1, gamma drawn as before; q drawn from the real q values (deep_table.q_twovel)
    of the twins for the twin population and of the non-twins for the non-twin
    population; m1 = gamma + K1 S, m2 = gamma - (K1/q) S, Gaussian noise sigma on
    each; then the two velocities are exchanged at each visit with probability 0.5.
--method v1: the previous primary-velocity-only mocks and likelihood (kept for
    reproducibility).
Output format unchanged."""
import os, sys, argparse
import numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_ROOT, "src"))
import orbit_sampler as OS, ecc_marginal as EM

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n-shards", type=int, default=1)
ap.add_argument("--nsys", type=int, default=250)
ap.add_argument("--draws", type=int, default=60_000)   # per e-grid point
ap.add_argument("--sigma", type=float, default=1.5)
ap.add_argument("--method", choices=["twovel", "v1"], default="twovel")
ap.add_argument("--out", required=True)
a = ap.parse_args()

import deep_table as DT
# Real cadences: the fitted epochs (mjd_per_visit) of each deep-table system with
# 8-20 epochs. A4B_DEEP_TABLE overrides the default resources/census/stage2_deep.csv.
_deep = DT.load_deep()
cad = DT.cadences(_deep, nmin=8, nmax=20)
assert cad, "no deep-table system with 8-20 epochs"
EGRID = EM.default_egrid()

QPOOL = {}
if a.method == "twovel":
    # real mass ratios (the same choice the data use), split by the catalog twin flag
    _cat = pd.read_csv(os.path.join(_ROOT, "resources/census/dr19_sb2_catalog_open.csv"),
                       usecols=["sdss_id", "best_q"])
    _dq = _deep.merge(_cat, on="sdss_id", how="inner")
    _dq = _dq[_dq.best_q.notna()]
    _q = np.array([DT.q_twovel(r) for r in _dq.itertuples()])
    _tw = (_dq.best_q > 0.95).values
    QPOOL = {True: _q[_tw & np.isfinite(_q)], False: _q[~_tw & np.isfinite(_q)]}
    assert len(QPOOL[True]) and len(QPOOL[False]), "empty twin or non-twin q pool"

def draw_e(alpha, rng, n=1):
    u = rng.uniform(0, 1, n)
    return (u * EM.EMAX ** (1 + alpha)) ** (1.0 / (1 + alpha))

def population(alpha, nsys, seed, offset):
    """previous primary-velocity-only mocks (method v1)"""
    rng = np.random.default_rng(seed)
    lgs = []
    for k in range(nsys):
        t0 = cad[(k + offset) % len(cad)]        # matched cadences across populations
        P = np.exp(rng.uniform(np.log(6), np.log(400)))
        e = float(draw_e(alpha, rng)[0])
        om = rng.uniform(0, 2*np.pi); M0 = rng.uniform(0, 2*np.pi)
        K1 = rng.uniform(15, 60); vsys = rng.uniform(-30, 30)
        S = OS.rv_shape(t0, np.array([P]), np.array([e]), np.array([om]), np.array([M0]))[0]
        v = vsys + K1 * S + rng.normal(0, a.sigma, len(t0))
        lg = EM.system_loglike_on_egrid(t0 - t0.mean(), v, a.sigma, EGRID,
                                        M_per_e=a.draws, rng=rng)
        lgs.append(lg)
    return lgs

def population_twovel(alpha, nsys, seed, offset, twin):
    """two-velocity mocks: unordered per-visit pairs, same likelihood as the data"""
    rng = np.random.default_rng(seed)
    qpool = QPOOL[twin]
    lgs = []
    for k in range(nsys):
        t0 = cad[(k + offset) % len(cad)]        # matched cadences across populations
        P = np.exp(rng.uniform(np.log(6), np.log(400)))
        e = float(draw_e(alpha, rng)[0])
        om = rng.uniform(0, 2*np.pi); M0 = rng.uniform(0, 2*np.pi)
        K1 = rng.uniform(15, 60); vsys = rng.uniform(-30, 30)
        q = float(rng.choice(qpool))
        S = OS.rv_shape(t0, np.array([P]), np.array([e]), np.array([om]), np.array([M0]))[0]
        m1 = vsys + K1 * S + rng.normal(0, a.sigma, len(t0))
        m2 = vsys - (K1 / q) * S + rng.normal(0, a.sigma, len(t0))
        pa, pb = DT.exchange_pairs(m1, m2, rng)
        lg = EM.system_loglike_on_egrid_twovel(t0 - t0.mean(), pa, pb, q, a.sigma, EGRID,
                                               M_per_e=a.draws, rng=rng)
        lgs.append(lg)
    return lgs

def pop(alpha, nsys, seed, offset, twin):
    if a.method == "twovel":
        return population_twovel(alpha, nsys, seed, offset, twin)
    return population(alpha, nsys, seed, offset)

CONFIGS = []
for da, atw, ant in [(0.0, 0.0, 0.0), (0.3, 0.3, 0.0), (0.6, 0.6, 0.0), (1.0, 1.0, 0.0)]:
    for rep in range(4):
        CONFIGS.append((da, atw, ant, rep))
rows = []
for i, (da, atw, ant, rep) in enumerate(CONFIGS):
    if i % a.n_shards != a.shard:
        continue
    seed = 1000 * (rep + 1) + int(100 * da)
    at, lo_t, hi_t, _, _ = EM.fit_alpha_grid(pop(atw, a.nsys, seed, rep * 37, True), EGRID)
    an, lo_n, hi_n, _, _ = EM.fit_alpha_grid(pop(ant, a.nsys, seed + 7, rep * 37, False), EGRID)
    rows.append(dict(da_true=da, rep=rep, a_twin=at, a_nontwin=an, da_rec=at - an,
                     twin_lo=lo_t, twin_hi=hi_t, nt_lo=lo_n, nt_hi=hi_n))
    print("da_true=%.1f rep=%d -> da_rec=%+.3f" % (da, rep, at - an), flush=True)
pd.DataFrame(rows).to_csv(a.out, index=False)
print("wrote", a.out, "(method %s)" % a.method)
