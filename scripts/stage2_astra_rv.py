#!/usr/bin/env python3
"""Per-star summary of the Astra per-visit velocities for the stage-2 stars.

DR19 mwmVisit files carry each visit's heliocentric velocity (v_rad) from the
pipeline's own Doppler fit, which is correct for single-lined stars. This lists,
for every stage-2 star, the number of visits, the largest velocity change, the
scatter and the time span, so single-lined velocity variables can be selected
without refitting. Writes resources/census/stage2_astra_rv.csv.

The *_ok columns repeat the count and the largest change over the visits whose
pipeline velocity is at most VMAX_OK = 400 km/s in absolute value. Larger values
are failures of the pipeline's velocity fit (they reach +-1000 km/s, beyond any
Galactic dwarf) and would otherwise make a single star look velocity-variable.
"""
import csv, os
import numpy as np
VMAX_OK = 400.0
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ids = [x.strip() for x in open(f"{R}/resources/census/stage2_candidates_all.txt").read().split()]
with open(f"{R}/resources/census/stage2_astra_rv.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["sdss_id", "n_visits_rv", "dv_rad_max", "v_rad_std", "v_rad_median", "mjd_span",
                "n_visits_rv_ok", "dv_rad_max_ok"])
    for s in ids:
        p = f"{R}/data/dr19_raw_visits/{s}.npz"
        if not os.path.exists(p):
            w.writerow([s, 0, "", "", "", "", 0, ""]); continue
        z = np.load(p)
        v = np.atleast_1d(z["v_rad"]).astype(float); m = np.atleast_1d(z["mjd"]).astype(float)
        ok = np.isfinite(v)
        v, m = v[ok], m[ok]
        n = len(v)
        vk = v[np.abs(v) <= VMAX_OK]
        w.writerow([s, n, "%.3f" % np.ptp(v) if n >= 2 else "", "%.3f" % np.std(v) if n >= 2 else "",
                    "%.3f" % np.median(v) if n else "", "%.2f" % np.ptp(m) if n >= 2 else "",
                    len(vk), "%.3f" % np.ptp(vk) if len(vk) >= 2 else ""])
print("done", len(ids))
