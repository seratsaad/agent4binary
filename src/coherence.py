#!/usr/bin/env python3
"""
AB-2: velocity-coherence statistic for SB2 detection, robust to net misfit.

The production SC detector's false positives come from the binary model exploiting
the real-data net's ~1% broadband line misfit: a near-equal-mass binary mops up the
residual and a true single scores a spurious binary (METHODS.md.5, 2.8). Those
residuals are NOT velocity-coherent. A real SB2, by contrast, has a second component
whose absorption lines are ALL doubled by a single relative velocity -- a coherent,
velocity-localized signature.

This module measures that. After the single-star fit, the residual

    d = obs - m1                (continuum-normalized space)

is matched-filtered against the primary line template t = m1 - 1 over a grid of
velocity shifts. A genuine secondary at relative velocity dv contributes shifted
absorption that correlates strongly with t shifted by dv -> a sharp peak in the
matched-filter SNR S(v) at |v| ~ |dv|. Net misfit sits at v ~ 0 (the primary's own
line positions) and is incoherent elsewhere, so it produces no peak away from 0.

S(v) is normalized to unit variance for pure photon noise (matched-filter SNR), so
the statistic `coherence` is directly a detection significance in sigma.

The APOGEE 8575-pixel grid is log-linear, so a constant velocity is a constant pixel
shift: KMS_PER_PIX = c * ln(10) * 6e-6 ~ 4.14 km/s/pixel.
"""
import numpy as np

NPIX = 8575
C_KMS = 299792.458
KMS_PER_PIX = C_KMS * np.log(10.0) * 6e-6     # ~4.144 km/s per pixel (log-lambda grid)


def _shift_pix(arr, dpix):
    """Shift `arr` by dpix pixels (positive = toward higher index) via linear interp."""
    n = arr.size
    src = np.arange(n) - dpix
    return np.interp(src, np.arange(n), arr, left=arr[0], right=arr[-1])


def coherence_profile(obs, err, single_model, vmax=400.0, dv=2.0):
    """Return (vs, S) -- the matched-filter SNR of a velocity-shifted secondary in the
    single-fit residual, over v in [-vmax, vmax] km/s.

    obs, single_model : continuum-normalized flux (length 8575), ~1 in the continuum.
    err               : normalized 1-sigma error; non-finite / <=0 -> masked (zero weight).
    """
    obs = np.asarray(obs, float); single_model = np.asarray(single_model, float)
    err = np.asarray(err, float)
    w = np.where(np.isfinite(err) & (err > 0), 1.0 / err ** 2, 0.0)
    d = np.where(w > 0, obs - single_model, 0.0)
    t = np.where(w > 0, single_model - 1.0, 0.0)      # primary absorption pattern
    vs = np.arange(-vmax, vmax + dv, dv)
    S = np.zeros(vs.size)
    for i, v in enumerate(vs):
        tv = _shift_pix(t, v / KMS_PER_PIX)
        tv = np.where(w > 0, tv, 0.0)
        denom = np.sqrt(np.sum(w * tv * tv))
        if denom > 0:
            S[i] = np.sum(w * d * tv) / denom         # ~ N(0,1) for pure noise
    return vs, S


def coherence_score(obs, err, single_model, vmin=25.0, vmax=400.0, dv=2.0):
    """Velocity-coherence statistic for one spectrum.

    Returns a dict:
      coherence    : peak matched-filter SNR of a shifted secondary at |v| >= vmin
                     (in sigma; ~3-4 for a single/noise, large for a real SB2).
      v_peak       : the velocity (km/s) of that peak (an estimate of the secondary's
                     relative velocity).
      coh_over_base: peak / median(|S|) in the far region (a scale-free version).

    vmin excludes the near-zero region where the primary's own residual / net misfit
    lives, so the statistic only rewards a genuinely velocity-SHIFTED second component.
    """
    vs, S = coherence_profile(obs, err, single_model, vmax=vmax, dv=dv)
    far = np.abs(vs) >= vmin
    if not far.any():
        return {"coherence": 0.0, "v_peak": 0.0, "coh_over_base": 0.0}
    Sf = S[far]; vf = vs[far]
    k = int(np.argmax(Sf))
    peak = float(Sf[k])
    base = float(np.median(np.abs(Sf)))
    return {"coherence": peak, "v_peak": float(vf[k]),
            "coh_over_base": float(peak / base) if base > 0 else 0.0}
