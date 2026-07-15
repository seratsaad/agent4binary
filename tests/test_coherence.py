#!/usr/bin/env python3
"""Unit test for src/coherence.py -- the velocity-coherence statistic must score a
real (velocity-shifted) secondary HIGH and both a pure single and a broadband
net-misfit false positive LOW. Pure numpy; no net, no torch. Run:

    python tests/test_coherence.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import coherence as C

NPIX = C.NPIX
rng = np.random.default_rng(0)


def make_template():
    """A continuum-normalized single-star model ~1 with ~300 random absorption lines."""
    m = np.ones(NPIX)
    centers = rng.integers(50, NPIX - 50, size=300)
    depths = rng.uniform(0.05, 0.5, size=300)
    widths = rng.uniform(1.5, 3.5, size=300)
    x = np.arange(NPIX)
    for c, d, wdt in zip(centers, depths, widths):
        lo, hi = max(0, c - 15), min(NPIX, c + 15)
        m[lo:hi] -= d * np.exp(-0.5 * ((x[lo:hi] - c) / wdt) ** 2)
    return np.clip(m, 0.2, 1.0)


def shift_kms(arr, v):
    return C._shift_pix(arr, v / C.KMS_PER_PIX)


def main():
    fails = 0
    def check(cond, msg, val=None):
        nonlocal fails
        tag = "PASS" if cond else "FAIL"
        print("%s: %s%s" % (tag, msg, "" if val is None else "  (%.2f)" % val))
        fails += 0 if cond else 1

    m1 = make_template()
    snr = 200.0
    err = np.full(NPIX, 1.0 / snr)
    # mask a few chip-gap-like regions (zero weight)
    err[3400:3460] = np.inf
    err[6200:6260] = np.inf

    def noise():
        e = np.where(np.isfinite(err), err, 0.0)
        return rng.normal(0, 1, NPIX) * e

    # (1) pure single: obs = m1 + noise
    obs_single = m1 + noise()
    cs = C.coherence_score(obs_single, err, m1)["coherence"]

    # (2) real SB2: add a shifted, scaled secondary absorption at dv = +70 km/s
    sec_lines = (m1 - 1.0) * 0.4                     # secondary ~40% line strength
    obs_sb2 = m1 + shift_kms(sec_lines, 70.0) + noise()
    r = C.coherence_score(obs_sb2, err, m1)
    csb2 = r["coherence"]

    # (3) net-misfit FP: broadband ~1% smooth distortion of the lines (no shifted 2nd comp)
    bump = 0.01 * np.sin(np.linspace(0, 40 * np.pi, NPIX))      # broadband wiggle
    obs_fp = m1 + bump * (1.0 - m1) * 5 + noise()              # distort line depths, v~0
    cfp = C.coherence_score(obs_fp, err, m1)["coherence"]

    print("scores: single=%.2f  SB2=%.2f  FP=%.2f  (SB2 v_peak=%.0f km/s)"
          % (cs, csb2, cfp, r["v_peak"]))
    check(csb2 > 8.0, "real SB2 scores high (>8 sigma)", csb2)
    check(abs(r["v_peak"] - 70.0) <= 6.0, "SB2 v_peak recovers injected 70 km/s", r["v_peak"])
    check(cs < 5.0, "pure single scores low (<5)", cs)
    check(cfp < 5.0, "broadband net-misfit FP scores low (<5)", cfp)
    check(csb2 > 3 * max(cs, cfp), "SB2 well separated from single & FP", csb2)

    print("\n%d FAIL" % fails if fails else "\nALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
