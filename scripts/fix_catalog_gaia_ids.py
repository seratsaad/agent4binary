#!/usr/bin/env python3
"""Replace the float-rounded Gaia DR3 ids in the released catalog.

The gaia_dr3_source_id column of dr19_sb2_catalog_open.csv passed through float64,
which rounds ids above 2**53, so about half point at no source. The census manifest
keeps them as exact strings. Each id is checked against Gaia DR3 (VizieR I/355/gaiadr3): it must exist and lie
within 2 arcsec of the catalog position, which allows for proper motion between
the APOGEE epoch and 2016.0. Writes the corrected
catalog next to the original and a stage-2 table of verified ids and positions.
"""
import csv, io, math, os, sys, time, urllib.parse, urllib.request
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# VizieR's copy of Gaia DR3 (I/355/gaiadr3): positions at epoch 2016.0, ids exact
TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
man = {}
for r in csv.DictReader(open(f"{R}/resources/dr19_census272k_manifest.csv")):
    v = (r.get("gaia_dr3_source_id") or "").strip()
    if v and v not in ("0", "-1") and "e" not in v.lower():
        man[r["sdss_id"].split(".")[0]] = v.split(".")[0]
cat = list(csv.DictReader(open(f"{R}/resources/census/dr19_sb2_catalog_open.csv")))
fields = list(cat[0].keys())
s2 = [x.strip() for x in open(f"{R}/resources/census/stage2_candidates_all.txt").read().split()]
want = sorted({man[s] for s in [r["sdss_id"].split(".")[0] for r in cat] + s2 if s in man})
pos = {}
STEP = 800
for i in range(0, len(want), STEP):
    q = 'SELECT "Source", "RA_ICRS", "DE_ICRS" FROM "I/355/gaiadr3" WHERE "Source" IN (%s)' % ",".join(want[i:i + STEP])
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": q}).encode()
    for attempt in range(5):
        try:
            txt = urllib.request.urlopen(urllib.request.Request(TAP, data=data), timeout=300).read().decode()
            break
        except Exception as e:  # the archive returns transient HTTP 500s
            sys.stderr.write("  retry %d after %s\n" % (attempt + 1, e))
            time.sleep(20 * (attempt + 1))
    else:
        raise SystemExit("Gaia archive failed 5 times at batch %d" % i)
    for r in csv.DictReader(io.StringIO(txt)):
        pos[r["Source"]] = (float(r["RA_ICRS"]), float(r["DE_ICRS"]))
    sys.stderr.write("  gaia %d/%d\n" % (min(i + STEP, len(want)), len(want)))
sep = lambda a, b: math.degrees(math.hypot(math.radians(a[0] - b[0]) * math.cos(math.radians(a[1])), math.radians(a[1] - b[1]))) * 3600
n_ok = n_changed = n_far = n_missing = 0
for r in cat:
    s = r["sdss_id"].split(".")[0]; old = (r.get("gaia_dr3_source_id") or "").split(".")[0]
    new = man.get(s, "")
    ok = new in pos and r.get("ra") and r.get("dec") and sep(pos[new], (float(r["ra"]), float(r["dec"]))) < 2.0
    if new and new in pos and not ok and r.get("ra"):
        n_far += 1
    if ok:
        n_ok += 1; n_changed += (new != old); r["gaia_dr3_source_id"] = new
    else:
        n_missing += 1; r["gaia_dr3_source_id"] = ""
out = f"{R}/resources/census/dr19_sb2_catalog_open.gaiafix.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(cat)
print("catalog %d | verified %d (changed from released %d) | no verified id %d (of which positional mismatch %d) -> %s" % (len(cat), n_ok, n_changed, n_missing, n_far, out))
with open(f"{R}/resources/census/stage2_gaia_coords.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["sdss_id", "gaia_dr3_source_id", "ra", "dec"])
    k = 0
    for s in s2:
        g = man.get(s, ""); p = pos.get(g)
        k += bool(p)
        w.writerow([s, g if p else "", "%.7f" % p[0] if p else "", "%.7f" % p[1] if p else ""])
print("stage-2 stars %d | verified Gaia id + position %d" % (len(s2), k))
