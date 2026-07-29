#!/usr/bin/env python3
"""Population eccentricity index by proper hierarchical marginalization.

The published two-step chain fails a noiseless injection (a true difference of
1.0 in the index is recovered as +0.15, identical to a true difference of 0),
because rejection sampling discards period modes and keeps too few draws.

This module replaces it with the same architecture as the wide-binary gamma
analysis: one population parameter, per-system orbital elements marginalized
rather than point-estimated. For system j,

    p(D_j | alpha) = Int p(D_j | theta) pi(P) pi(omega) pi(phi) f(e | alpha) dtheta

is evaluated by importance sampling from the interim prior (flat in e), with
the linear pair (K1, v_sys) marginalized analytically. Every draw contributes
its weight, so no mode is eliminated and no system is discarded. The
population likelihood is the product over systems, giving

    ln L(alpha) = sum_j ln sum_m w_jm (1 + alpha) e_jm^alpha  + const.
"""
import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import orbit_sampler as OS

EMAX = 0.95


def draw_prior(M, rng, pmin=0.5, pmax=5000.0):
    P = np.exp(rng.uniform(np.log(pmin), np.log(pmax), M))
    e = rng.uniform(0.0, EMAX, M)
    om = rng.uniform(0, 2 * np.pi, M)
    M0 = rng.uniform(0, 2 * np.pi, M)
    return P, e, om, M0


def system_weights(t, v, sigma, M=200_000, seed=0, rng=None, chunk=50_000,
                   v_alt=None):
    """Return (e_m, log w_m) for one system: log marginal likelihood of each
    prior draw with the linear (K1, v_sys) pair integrated out analytically.

    If v_alt is given (the tie inversion of v, i.e. what the primary velocity
    would be had the component labels been exchanged at that epoch), the
    per-epoch label assignment z_i is a genuine binary latent and is summed
    over rather than optimized: each epoch contributes
    log[exp(-r_v^2/2) + exp(-r_alt^2/2)] - log 2, so both assignments enter
    the marginal likelihood with equal prior weight."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    n = len(t)
    e_all, lw_all = [], []
    done = 0
    while done < M:
        m = min(chunk, M - done); done += m
        P, e, om, M0 = draw_prior(m, rng)
        S = OS.rv_shape(t, P, e, om, M0)              # (m, n)
        w = 1.0 / sigma**2
        # design [S, 1]: analytic Gaussian marginal over (K1, v_sys)
        a11 = (S**2).sum(1) * w
        a12 = S.sum(1) * w
        a22 = float(n) * w
        det = a11 * a22 - a12**2
        det = np.where(np.abs(det) < 1e-30, 1e-30, det)
        b1 = (S * v[None, :]).sum(1) * w
        b2 = float(v.sum()) * w
        K1 = (b1 * a22 - b2 * a12) / det
        vs = (a11 * b2 - a12 * b1) / det
        mod = vs[:, None] + K1[:, None] * S
        if v_alt is None:
            chi2 = ((v[None, :] - mod) ** 2).sum(1) * w
        else:
            # marginalize the binary label assignment per epoch (logsumexp)
            r0 = -0.5 * ((v[None, :] - mod) ** 2) * w
            r1 = -0.5 * ((v_alt[None, :] - mod) ** 2) * w
            mx = np.maximum(r0, r1)
            lse = mx + np.log(np.exp(r0 - mx) + np.exp(r1 - mx)) - np.log(2.0)
            chi2 = -2.0 * lse.sum(1)
        # marginal over the linear pair adds -0.5 ln det (flat prior)
        lw = -0.5 * chi2 - 0.5 * np.log(np.abs(det))
        lw = np.where(K1 > 0, lw, -np.inf)            # K1 > 0 convention
        e_all.append(e); lw_all.append(lw)
    return np.concatenate(e_all), np.concatenate(lw_all)


def alpha_loglike(esets, lwsets, alphas):
    """Population log-likelihood over alpha from per-system weighted draws."""
    ll = np.zeros_like(alphas)
    for e, lw in zip(esets, lwsets):
        ok = np.isfinite(lw)
        if not ok.any():
            continue
        e_, lw_ = e[ok], lw[ok]
        lw_ = lw_ - lw_.max()                        # per-system stabilization
        w = np.exp(lw_)
        lo = np.log(np.clip(e_, 1e-6, None))
        for i, a in enumerate(alphas):
            # sum_m w_m (1+a) e_m^a, in log space
            terms = lw_ + a * lo
            mx = terms.max()
            ll[i] += mx + np.log(np.sum(np.exp(terms - mx))) + np.log1p(a)
    return ll


def fit_alpha(esets, lwsets, amin=-0.95, amax=3.0, n=200):
    A = np.linspace(amin, amax, n)
    ll = alpha_loglike(esets, lwsets, A)
    i = int(np.argmax(ll))
    ok = ll > ll.max() - 0.5
    return A[i], A[ok].min(), A[ok].max(), A, ll


# --------------------------------------------------------------------------- #
# Direct (grid-in-e) population inference.
#
# The weighted form above draws e from a flat interim prior and reweights by
# f(e | alpha). That is exact in the limit of infinite draws but degrades as
# alpha moves away from the proposal, which compresses the recovered index
# toward the interim value. The joint hierarchical model of the wide-binary
# analysis avoids this by sampling the population parameter together with the
# per-system elements, so no reweighting is needed.
#
# The same property is obtained here without a joint sampler: evaluate each
# system's marginal likelihood on a GRID in e,
#
#     L_j(e) = Int p(D_j | e, P, omega, phi) pi(P) pi(omega) pi(phi) dP domega dphi,
#
# which is independent of alpha, and then integrate the population prior
# against it by quadrature,
#
#     p(D_j | alpha) = Int L_j(e) f(e | alpha) de.
#
# Nothing is reweighted in e, so the estimator does not pull toward the
# interim prior and alpha is recovered directly.
# --------------------------------------------------------------------------- #

def system_loglike_on_egrid(t, v, sigma, e_grid, M_per_e=20_000, rng=None,
                            seed=0, v_alt=None, pmin=0.5, pmax=5000.0):
    """log L_j(e) on the supplied eccentricity grid."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    n = len(t)
    w = 1.0 / sigma**2
    out = np.empty(len(e_grid))
    for j, e0 in enumerate(e_grid):
        P = np.exp(rng.uniform(np.log(pmin), np.log(pmax), M_per_e))
        om = rng.uniform(0, 2 * np.pi, M_per_e)
        M0 = rng.uniform(0, 2 * np.pi, M_per_e)
        e = np.full(M_per_e, e0)
        S = OS.rv_shape(t, P, e, om, M0)
        a11 = (S**2).sum(1) * w
        a12 = S.sum(1) * w
        a22 = float(n) * w
        det = a11 * a22 - a12**2
        det = np.where(np.abs(det) < 1e-30, 1e-30, det)
        b1 = (S * v[None, :]).sum(1) * w
        b2 = float(v.sum()) * w
        K1 = (b1 * a22 - b2 * a12) / det
        vs = (a11 * b2 - a12 * b1) / det
        mod = vs[:, None] + K1[:, None] * S
        if v_alt is None:
            chi2 = ((v[None, :] - mod) ** 2).sum(1) * w
        else:
            r0 = -0.5 * ((v[None, :] - mod) ** 2) * w
            r1 = -0.5 * ((v_alt[None, :] - mod) ** 2) * w
            mx = np.maximum(r0, r1)
            chi2 = -2.0 * (mx + np.log(np.exp(r0 - mx) + np.exp(r1 - mx))
                           - np.log(2.0)).sum(1)
        lw = -0.5 * chi2 - 0.5 * np.log(np.abs(det))
        lw = np.where(K1 > 0, lw, -np.inf)
        ok = np.isfinite(lw)
        if not ok.any():
            out[j] = -np.inf
            continue
        m = lw[ok].max()
        out[j] = m + np.log(np.mean(np.exp(lw[ok] - m)))   # MC average over P, om, phi
    return out


def default_egrid(n_low=16, n_hi=20, e_break=0.12, e_min=1e-3):
    """Grid dense toward e = 0. For alpha < 0 the population density e^alpha
    concentrates near zero, so a linear grid starting well above zero misses
    the mass and biases the recovered index low."""
    lo = np.logspace(np.log10(e_min), np.log10(e_break), n_low, endpoint=False)
    hi = np.linspace(e_break, EMAX - 0.01, n_hi)
    return np.concatenate([lo, hi])


def alpha_loglike_grid(loglikes, e_grid, alphas):
    """Population log-likelihood from per-system log L_j(e) by quadrature."""
    de = np.gradient(e_grid)
    lo = np.log(np.clip(e_grid, 1e-6, None))
    ll = np.zeros_like(alphas)
    for lg in loglikes:
        ok = np.isfinite(lg)
        if not ok.any():
            continue
        for i, al in enumerate(alphas):
            terms = lg[ok] + al * lo[ok] + np.log1p(al) + np.log(de[ok])
            mx = terms.max()
            ll[i] += mx + np.log(np.sum(np.exp(terms - mx)))
    return ll


def fit_alpha_grid(loglikes, e_grid, amin=-0.95, amax=3.0, n=200):
    A = np.linspace(amin, amax, n)
    ll = alpha_loglike_grid(loglikes, e_grid, A)
    i = int(np.argmax(ll))
    ok = ll > ll.max() - 0.5
    return A[i], A[ok].min(), A[ok].max(), A, ll
