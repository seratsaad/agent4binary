#!/usr/bin/env python3
"""Build the second-revision release of the DR19 SB2 catalog.

Starts from dr19_sb2_catalog_open.gaiafix.csv (the catalog with Gaia DR3 ids and
positions verified by fix_catalog_gaia_ids.py) and adds our own fitted broadening
from the seeded refit (vmacro_refit_*.csv). The previous v_macro column, which was
the DR19 pipeline's value, is kept under the name v_macro_pipeline. Detection
columns (verdict, delta_chi2, f_imp, best_q, best_rv1, best_rv2, ...) are the
catalog values and are not changed. The previous file is kept as
dr19_sb2_catalog_open_round1.csv.
"""
import csv, glob, os, shutil
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = lambda p: os.path.join(R, "resources", "census", p)

vm = {}
for f in sorted(glob.glob(C("vmacro_refit_*.csv"))):
    for r in csv.DictReader(open(f)):
        if not r.get("error"):
            vm[r["sdss_id"].split(".")[0]] = r
rows = list(csv.DictReader(open(C("dr19_sb2_catalog_open.gaiafix.csv"))))
s2 = {r["sdss_id"].split(".")[0]: r for r in csv.DictReader(open(C("stage2_catalog_full.csv")))}
out, nv, nedge = [], 0, 0
for r in rows:
    s = r["sdss_id"].split(".")[0]
    v = vm.get(s, {})
    r["v_macro_pipeline"] = r.pop("v_macro", "")
    for k in ("vmacro_single", "vmacro_1", "vmacro_2"):
        r[k] = ("%.3f" % float(v[k])) if v.get(k) not in (None, "") else ""
    edge = r["vmacro_single"] != "" and float(r["vmacro_single"]) >= 30.0
    r["vmacro_at_edge"] = int(edge) if r["vmacro_single"] != "" else ""
    nv += r["vmacro_single"] != ""; nedge += edge
    # coadd_only: the coadd shows a split the visits do not. Two or more fitted
    # visits, coadd separation above 15 km/s, steady pipeline velocity (std below
    # 1 km/s), and a visit fit that prefers one star. A pair that far apart would
    # show two sets of lines in every visit; the visit fit prefers two stars for
    # 92-95% of the published SB2 at every coadd separation. Replaces the second-
    # revision coadd_symmetric flag, which also marked real twins.
    v = s2.get(s, {})
    try:
        nvis = int(float(v.get("n_visits") or 0))
        sep = abs(float(r["best_rv1"]) - float(r["best_rv2"]))
        std = float(v.get("v_rad_std_astra") or "nan")
        r["coadd_only"] = int(nvis >= 2 and sep > 15.0 and std < 1.0 and v.get("prefers_binary") == "False")
    except ValueError:
        r["coadd_only"] = ""
    out.append(r)
if not os.path.exists(C("dr19_sb2_catalog_open_round1.csv")):
    shutil.copy(C("dr19_sb2_catalog_open.csv"), C("dr19_sb2_catalog_open_round1.csv"))
cols = [c for c in out[0] if c not in ("vmacro_single", "vmacro_1", "vmacro_2", "vmacro_at_edge", "v_macro_pipeline")]
cols += ["vmacro_single", "vmacro_1", "vmacro_2", "vmacro_at_edge", "v_macro_pipeline"]
with open(C("dr19_sb2_catalog_open.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(out)
ng = sum(r["gaia_dr3_source_id"] != "" for r in out)
print("rows %d | verified Gaia id %d | our v_macro %d | at the 30 km/s edge %d (%.1f%%)"
      % (len(out), ng, nv, nedge, 100.0 * nedge / max(nv, 1)))
