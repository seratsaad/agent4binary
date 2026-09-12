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
    # A real SB2 sits off-centre in the pipeline frame, which follows the brighter
    # star; a single star fitted as two components splits about zero.
    try:
        r["coadd_symmetric"] = int(abs(float(r["best_rv1"]) + float(r["best_rv2"])) <= 2.0)
    except ValueError:
        r["coadd_symmetric"] = ""
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
