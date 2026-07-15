#!/usr/bin/env python3
"""Build a DR19 dwarf census sample manifest for the binspec production census.

Pages the_payne_apogee_star (the DR19 ASPCAP/Payne catalog) for dwarfs --
logg 4-5, Teff 4000-7000, snr>60, flag_bad=0 -- via the SkyServer JSON API
(GET; same robust path as dr13_parity_rerun). UNLIKE the training-set builder it
does NOT exclude El-Badry binaries: a census must include them. Writes
resources/dr19_census_manifest.csv (sdss_id + 5 labels + snr).

Usage: python scripts/build_census_sample.py --n 20000
"""
import argparse
import os
import time

import pandas as pd
import requests

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(_ROOT, "resources", "dr19_census_manifest.csv")
_S = requests.Session(); _S.headers["User-Agent"] = "agent4binary-census/0.1"


def sql(q, tries=4):
    url = "https://skyserver.sdss.org/dr19/SkyServerWS/SearchTools/SqlSearch"
    for k in range(tries):
        try:
            r = _S.get(url, params={"cmd": q, "format": "json"}, timeout=120)
            if r.status_code != 200:
                time.sleep(2 * (k + 1)); continue
            rows = []
            for b in r.json():
                if b.get("TableName") != "SqlQuery":
                    rows.extend(b.get("Rows", []))
            return rows
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    out_rows = {}
    cursor = 0
    fcols = ["teff", "logg", "fe_h", "mg_h", "v_macro", "snr",
             "g_mag", "bp_mag", "rp_mag", "plx", "e_plx"]
    while len(out_rows) < a.n:
        q = ("SELECT TOP 1000 sdss_id, teff, logg, fe_h, mg_h, v_macro, snr, "
             "g_mag, bp_mag, rp_mag, plx, e_plx, gaia_dr3_source_id "
             "FROM the_payne_apogee_star "
             "WHERE logg BETWEEN 4.0 AND 5.0 AND teff BETWEEN 4000 AND 7000 "
             "AND snr > 60 AND flag_bad = 0 AND sdss_id > %d ORDER BY sdss_id" % cursor)
        rows = sql(q)
        if not rows:
            break
        adv = 0
        for r in rows:
            sid = int(r["sdss_id"]); cursor = max(cursor, sid); adv += 1
            try:
                rec = {"sdss_id": sid}
                for c in fcols:
                    rec[c] = float(r[c]) if r.get(c) not in (None, "") else float("nan")
                rec["gaia_dr3_source_id"] = r.get("gaia_dr3_source_id", "")
            except Exception:
                continue
            out_rows[sid] = rec
        print("  census candidates=%d (cursor=%d)" % (len(out_rows), cursor), flush=True)
        if adv == 0:
            break
    df = pd.DataFrame(list(out_rows.values())[:a.n])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print("wrote %s (%d dwarfs)" % (os.path.relpath(a.out, _ROOT), len(df)), flush=True)


if __name__ == "__main__":
    main()
