#!/usr/bin/env python3
# =========================================================================== #
# eval_sc_net.py
#
# Regression check for the SC single-star net (models/payne_dr19_sc.pt). Reports
# how well the net reproduces REAL spectra, the quantity that gates the whole
# single-vs-binary detector: a net that misfits singles by >> photon noise makes
# Delta-chi2 / f_imp meaningless (diagnosed 2026-06-23; the pre-retrain net sat at
# fitted reduced chi2 ~5-46 on controls vs binspec ~1-2).
#
# Two views, both on the SAME continuum_normalize + normalize_like_model path the
# detector uses:
#   (1) CONTROLS (data/dr19_raw_controls): FIT the 5-label single net to each true
#       single and report fitted reduced chi2 + residual-RMS / median-sigma (an
#       error-calibration-independent statement of net inadequacy) + v_macro rails.
#   (2) VAL SET (models/val_set_dr19_sc.npz, if it carries fluxes): the net's
#       reduced chi2 AT the catalog labels on truly held-out training-domain stars.
#
# CLI:  python src/eval_sc_net.py [--n-controls 30]
# =========================================================================== #

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _sc_obs(physics, flux_raw, ivar):
    # C-space (continuum_normalize) obs, matching the production detector after the
    # double-normalization fix: NO second normalize_like_model pass.
    sc_flux, sc_err = physics.continuum_normalize(flux_raw, ivar, return_error=True)
    sc_flux = np.where(np.isfinite(sc_flux), sc_flux, 1.0)
    sc_err_finite = np.where(np.isfinite(sc_err), sc_err, 1e6)
    obs_err = physics.mask_bad_pixels(sc_flux, sc_err_finite)
    obs_err = np.where(np.isfinite(obs_err), obs_err, 1e6)
    return sc_flux, obs_err


def eval_controls(physics, n):
    files = sorted(glob.glob(os.path.join(_ROOT, "data/dr19_raw_controls/*.npz")))[:n]
    redchi2, rms_over_sig, vmacro_rail = [], [], 0
    for f in files:
        d = np.load(f)
        obs, oe = _sc_obs(physics, np.asarray(d["flux_raw"], float),
                          np.asarray(d["ivar"], float))
        g = oe < 1e5
        ng = int(g.sum())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ps, ms, cs = physics.fit_single5_sc(obs, oe, 5500, 4.5, 0, 0, 5)
        resid = (obs - ms)[g]
        rms = float(np.sqrt(np.mean(resid ** 2)))
        medsig = float(np.median(oe[g]))
        redchi2.append(cs / max(ng, 1))
        rms_over_sig.append(rms / medsig if medsig > 0 else np.nan)
        if abs(ps[4] - physics._LMAXSC[4]) < 0.3:
            vmacro_rail += 1
    return (np.array(redchi2), np.array(rms_over_sig), vmacro_rail, len(files))


def eval_valset(physics):
    p = os.path.join(_ROOT, "models/val_set_dr19_sc.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    if "fluxes" not in d.files:
        return None
    labels = np.asarray(d["labels"], float)
    flux = np.asarray(d["fluxes"], float)
    err = np.asarray(d["errors"], float)
    rc = []
    for i in range(labels.shape[0]):
        t, g, h, m, v = labels[i]
        pred = physics.payne_predict5_sc(t, g, h, m, v)
        good = np.isfinite(flux[i]) & np.isfinite(err[i]) & (err[i] > 0) & (err[i] < 100)
        if not np.any(good):
            continue
        r = (flux[i][good] - pred[good]) / err[i][good]
        rc.append(float(np.sum(r * r) / good.sum()))
    return np.array(rc)


def _stats(a):
    a = a[np.isfinite(a)]
    return (float(np.median(a)), float(np.mean(a)),
            float(np.percentile(a, 90)), float(np.max(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-controls", type=int, default=30)
    args = ap.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import physics

    print("=== SC net regression check ===")
    rc, rs, rail, nf = eval_controls(physics, args.n_controls)
    med, mean, p90, mx = _stats(rc)
    print("CONTROLS (n=%d), FITTED single net:" % nf)
    print("  reduced chi2 : median=%.2f mean=%.2f p90=%.2f max=%.2f" % (med, mean, p90, mx))
    med, mean, p90, mx = _stats(rs)
    print("  residual-RMS / median-sigma : median=%.2f mean=%.2f p90=%.2f max=%.2f"
          % (med, mean, p90, mx))
    print("  v_macro railed to box edge : %d/%d" % (rail, nf))

    rcv = eval_valset(physics)
    if rcv is not None and len(rcv):
        med, mean, p90, mx = _stats(rcv)
        print("VAL SET (n=%d) at catalog labels, net reduced chi2:" % len(rcv))
        print("  median=%.2f mean=%.2f p90=%.2f max=%.2f" % (med, mean, p90, mx))
    else:
        print("VAL SET: no fluxes stored (old schema) — skipped.")


if __name__ == "__main__":
    main()
