#!/usr/bin/env python3
"""Pipeline per-visit velocities for the random single-star sample.

Applies the SB1 selection of the multi-epoch supplement (DR19 pipeline v_rad
changing by more than 10 km/s over three or more visits, as in stage2_astra_rv.py
and stage2_aggregate_fixed.py) to the 3,000 random census dwarfs of
stage2_singles_ids.txt, which lie outside the catalog, the sigma_v list and the
benchmark. The fraction selected is the false-positive rate of the SB1 flag.

Reads each mwmVisit file from the SAS with the same parser as the visit fit
(the rule of download_dr19_visits.read_visit_spectra, every visit with nonzero
ivar) and keeps only the velocities. Writes
resources/census/stage2_singles_astra_rv.csv.
"""
import csv, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from io import BytesIO
import urllib.request, urllib.error
from astropy.io import fits

BASE = "https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/spectra/visit"


def url_for(s):
    s = int(s)
    return "%s/%02d/%02d/mwmVisit-0.6.0-%d.fits" % (BASE, (s // 100) % 100, s % 100, s)


def read_rv(content):
    """Same visit rule as download_dr19_visits.read_visit_spectra: APOGEE/APO and
    APOGEE/LCO rows with a full-length flux and any positive ivar."""
    v, m = [], []
    with fits.open(BytesIO(content)) as h:
        for idx in (3, 4):
            if idx >= len(h) or h[idx].data is None or h[idx].header.get("NAXIS2", 0) <= 0:
                continue
            for row in h[idx].data:
                iv = np.asarray(row["ivar"], float)
                if np.asarray(row["flux"]).size != 8575 or iv.size != 8575 or not np.any(iv > 0):
                    continue
                v.append(float(row["v_rad"])); m.append(float(row["mjd"]))
    return np.array(v), np.array(m)

ids = [x.strip() for x in open(f"{R}/resources/census/stage2_singles_ids.txt").read().split() if x.strip()]
OUT = f"{R}/resources/census/stage2_singles_astra_rv.csv"


def one(s):
    for attempt in range(4):
        try:
            try:
                content = urllib.request.urlopen(url_for(s), timeout=60).read()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return [s, 0, "", "", "", "no_visit_file"]
                raise
            if len(content) < 1000:
                time.sleep(1 + attempt); continue
            v, m = read_rv(content)
            if v.size == 0:
                return [s, 0, "", "", "", "no_usable_visit"]
            ok = np.isfinite(v)
            v, m = v[ok], m[ok]
            n = len(v)
            return [s, n, "%.3f" % np.ptp(v) if n >= 2 else "", "%.3f" % np.std(v) if n >= 2 else "",
                    "%.2f" % np.ptp(m) if n >= 2 else "", ""]
        except Exception:
            time.sleep(1 + attempt)
    return [s, 0, "", "", "", "fetch_failed"]


with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(one, ids))
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["sdss_id", "n_visits_rv", "dv_rad_max", "v_rad_std", "mjd_span", "error"])
    w.writerows(rows)

ok = [r for r in rows if not r[5]]
n3 = [r for r in ok if r[1] >= 3]
sel = [r for r in n3 if float(r[2]) > 10.0]
p = len(sel) / max(len(n3), 1)
print("fetched %d/%d | >=3 pipeline visits %d | dv_rad_max > 10 km/s %d = %.2f%% +- %.2f"
      % (len(ok), len(ids), len(n3), len(sel), 100 * p, 100 * np.sqrt(p * (1 - p) / max(len(n3), 1))))
