#!/usr/bin/env python3
"""Compare the legacy visit frame (-bc) with input_frame="rest" (+v_rad) on a few stars.

The referee's star 115423945 (one visit, Astra v_rad -45.75 km/s), the SB2
114862938, and control stars with three or more visits and a bc spread above
15 km/s. On rest-frame input the legacy path should return velocities that follow
-bc, and the rest path velocities that follow Astra's v_rad.
"""
import glob, os, sys, time
import numpy as np
from astropy.io import fits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import physics as P  # noqa: E402

SCRATCH = sys.argv[1] if len(sys.argv) > 1 else "."


def from_fits(path):
    t = fits.open(path)[3].data
    return {k: np.array(t[c], float) for k, c in
            (("flux_raw", "flux"), ("ivar", "ivar"), ("v_rad", "v_rad"),
             ("v_rel", "v_rel"), ("bc", "bc"), ("snr", "snr"), ("mjd", "mjd"))}


stars = [("referee", "115423945", from_fits(os.path.join(SCRATCH, "mwmVisit-0.6.0-115423945.fits"))),
         ("sb2", "114862938", dict(np.load(os.path.join(ROOT, "data/dr19_raw_visits/114862938.npz"))))]
cands = [f for f in sorted(glob.glob(os.path.join(ROOT, "data/dr19_visits_ctl/*.npz")))
         if len(np.atleast_1d(np.load(f)["bc"])) >= 3 and np.ptp(np.load(f)["bc"]) > 15]
for f in np.random.default_rng(7).choice(cands, 6, replace=False):
    stars.append(("control", os.path.basename(f)[:-4], dict(np.load(f))))

fmt = lambda a: "[" + " ".join("%+6.1f" % x for x in np.atleast_1d(a)) + "]"
for label, sid, vis in stars:
    print("\n%s %s   Astra v_rad %s   -bc %s" % (label, sid, fmt(vis["v_rad"]), fmt(-np.asarray(vis["bc"]))))
    for frame in ("observed", "rest"):
        t0 = time.time()
        try:
            tups = [(np.asarray(vis["flux_raw"], float)[i], np.asarray(vis["ivar"], float)[i],
                     float(np.atleast_1d(vis["v_rad"])[i]), float(np.atleast_1d(vis["bc"])[i]))
                    for i in range(np.atleast_2d(vis["flux_raw"]).shape[0])]
            r = P.dr19_visit_single_vs_binary(tups, input_frame=frame)
        except Exception as e:
            print("   %-8s ERROR %s" % (frame, e)); continue
        print("   %-8s binary=%-5s dchi2=%9.1f gamma=%+7.2f  v1 %s  v2 %s  (%.0fs)" % (
            frame, r.get("prefers_binary"), r.get("delta_chi2", np.nan), r.get("gamma", np.nan),
            fmt(r.get("v1_per_visit", [])), fmt(r.get("v2_per_visit", [])), time.time() - t0), flush=True)
