#!/usr/bin/env python3
"""Aggregate the corrected stage-2 run into the released multi-epoch table.

Inputs
  resources/census/stage2_fixed_shards/shard_*.csv   corrected per-visit fits
  resources/census/stage2_gaia_coords.csv            verified Gaia DR3 id + position
  resources/census/stage2_astra_rv.csv               Astra per-visit velocity summary
  resources/census/dr19_sb2_catalog_open.csv         released combined-spectrum catalog

Outputs
  resources/census/stage2_catalog_full.csv   (previous version kept as *_legacy.csv)
  resources/census/stage2_mjds.csv           epochs in the order of v1_per_visit

Definitions follow the paper. Confirmation: a catalog SB2 with two or more epochs
whose joint visit fit prefers two components. Orbit-ready: a catalog SB2 confirmed
by the visit fit, with three or more epochs whose primary velocity spans more than
20 km/s. SB1: a star outside
the catalog whose Astra per-visit velocities change by more than 10 km/s over three
or more visits. The legacy SB1 selection used the fitted primary velocity, which is
constant for a single-star fit and so cannot identify a single-lined variable.
"""
import csv, glob, os, shutil, sys
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = lambda p: os.path.join(R, "resources", "census", p)
F = lambda x: float(x) if x not in (None, "", "nan") else np.nan
series = lambda s: np.array([float(x) for x in str(s).split(";") if x not in ("", "nan")])

SHARDS = os.environ.get("A4B_S2_SHARDS", "stage2_fixed_shards")
rows = {}
for f in sorted(glob.glob(C(SHARDS + "/shard_*.csv"))):
    for r in csv.DictReader(open(f)):
        rows[r["sdss_id"].split(".")[0]] = r
if not rows:
    sys.exit("no corrected shards found")
cat = {r["sdss_id"].split(".")[0]: r for r in csv.DictReader(open(C("dr19_sb2_catalog_open.csv")))}
gaia = {r["sdss_id"]: r for r in csv.DictReader(open(C("stage2_gaia_coords.csv")))}
arv = {r["sdss_id"]: r for r in csv.DictReader(open(C("stage2_astra_rv.csv")))}

ids_all = [x.strip() for x in open(C("stage2_candidates_all.txt")).read().split()]
for s in ids_all:
    if s not in rows:
        rows[s] = {"sdss_id": s, "error": "not refit in the second revision"}
out = []
for s, r in rows.items():
    nv = int(F(r.get("n_visits"))) if np.isfinite(F(r.get("n_visits"))) else 0
    v1 = series(r.get("v1_per_visit", ""))
    span = float(np.ptp(v1)) if v1.size >= 2 else 0.0
    binary = r.get("prefers_binary") == "True"
    in_cat = s in cat
    a = arv.get(s, {})
    n_rv = int(F(a.get("n_visits_rv")) or 0) if a.get("n_visits_rv") not in ("", None) else 0
    dvr = F(a.get("dv_rad_max"))
    g = gaia.get(s, {})
    rec = dict(r)
    rec.update(refit=r.get("error") != "not refit in the second revision",
               gaia_dr3_source_id=g.get("gaia_dr3_source_id", ""), ra=g.get("ra", ""), dec=g.get("dec", ""),
               in_sb2_catalog=in_cat, n_visits_rv=n_rv,
               dv_rad_max_astra="" if not np.isfinite(dvr) else "%.3f" % dvr,
               v_rad_median_astra=a.get("v_rad_median", ""), v_rad_std_astra=a.get("v_rad_std", ""),
               orbit_ready=bool(in_cat and binary and nv >= 3 and span > 20.0),
               sb1=bool((not in_cat) and n_rv >= 3 and np.isfinite(dvr) and dvr > 10.0))
    # q_dyn means something only for a system the visit fit confirms. A fit that
    # falls back to the single-star solution sets q to one, which in the first
    # revision showed up as a large spurious group at q_dyn = 1.
    if not binary:
        rec["q_dyn"] = ""
        rec["q_dyn_at_bound"] = ""
    out.append(rec)

# ---------------------------------------------------------------- statistics
def pct(x):
    return 100.0 * np.mean(x) if len(x) else float("nan")
multi = [o for o in out if o["in_sb2_catalog"] and o["refit"] and not o.get("error")
         and np.isfinite(F(o.get("n_visits"))) and int(F(o["n_visits"])) >= 2]
conf = np.array([o["prefers_binary"] == "True" for o in multi])
nvs = np.array([int(F(o["n_visits"])) for o in multi])
print("catalog SB2 with >= 2 epochs fitted: %d | confirmed by the visit fit: %.1f%%" % (len(multi), pct(conf)))
for lo, hi, lab in ((2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 8, "6-8")):
    m = (nvs >= lo) & (nvs <= hi)
    print("   %-4s epochs: n=%5d  confirmed %.1f%%" % (lab, m.sum(), pct(conf[m])))
spans = np.array([np.ptp(series(o["v1_per_visit"])) for o in multi if series(o["v1_per_visit"]).size >= 2])
print("median primary-velocity change across visits: %.1f km/s" % np.median(spans))
print("catalog SB2 with >= 3 epochs (Figure 14 sample): %d" % sum(int(F(o["n_visits"])) >= 3 for o in multi))
print("orbit-ready (>= 3 epochs, primary span > 20 km/s): %d | of these confirmed: %d"
      % (sum(o["orbit_ready"] for o in out), sum(o["orbit_ready"] and o["prefers_binary"] == "True" for o in out)))
print("SB1 from Astra velocities (outside catalog, >= 3 visits, dv_rad > 10 km/s): %d" % sum(o["sb1"] for o in out))

# point 3: near-equal twins that the combined spectrum flags but the visits do not
tw = [o for o in multi if F(cat[o["sdss_id"].split(".")[0]]["best_q"]) >= 0.95]
tw_un = [o for o in tw if o["prefers_binary"] != "True"]
asym = lambda o: abs(F(cat[o["sdss_id"]]["best_rv1"]) + F(cat[o["sdss_id"]]["best_rv2"]))
split = lambda o: abs(F(cat[o["sdss_id"]]["best_rv1"]) - F(cat[o["sdss_id"]]["best_rv2"]))
print("catalog twins (q >= 0.95) with >= 2 epochs: %d | not confirmed by visits: %d (%.1f%%)"
      % (len(tw), len(tw_un), 100.0 * len(tw_un) / max(len(tw), 1)))
a_un = np.array([asym(o) for o in tw_un]); a_cf = np.array([asym(o) for o in tw if o["prefers_binary"] == "True"])
print("   |rv1+rv2| > 2 km/s: unconfirmed twins %.1f%% | confirmed twins %.1f%%" % (pct(a_un > 2), pct(a_cf > 2)))
sig = [o for o in tw_un if asym(o) > 2 and np.isfinite(F(o["dv_rad_max_astra"])) and F(o["dv_rad_max_astra"]) > 0.5 * split(o)]
print("   unconfirmed twins whose Astra velocities jump by more than half the coadd split: %d" % len(sig))
for s in ("61585132", "76096409"):
    o = next((x for x in out if x["sdss_id"].split(".")[0] == s), None)
    if o:
        print("   referee example %s: visit fit binary=%s, n_visits=%s, Astra dv_rad=%s, coadd rv1 %+.1f rv2 %+.1f"
              % (s, o["prefers_binary"], o["n_visits"], o["dv_rad_max_astra"], F(cat[s]["best_rv1"]), F(cat[s]["best_rv2"])))

# point 4b: dynamical against spectroscopic mass ratio where q_dyn is constrained
qd = [(F(o["q_dyn"]), F(cat[o["sdss_id"]]["best_q"])) for o in multi
      if o["prefers_binary"] == "True" and int(F(o["n_visits"])) >= 3 and o.get("q_dyn_at_bound") != "True" and np.isfinite(F(o["q_dyn"]))]
if qd:
    q = np.array(qd)
    print("q_dyn vs q_spec (confirmed, >= 3 epochs, off the bounds): n=%d  Spearman %.3f  median |dq| %.3f"
          % (len(q), np.corrcoef(np.argsort(np.argsort(q[:, 0])), np.argsort(np.argsort(q[:, 1])))[0, 1], np.median(np.abs(q[:, 0] - q[:, 1]))))

def wilson_q(o):
    """Mass ratio from the untied per-visit velocities: -1/slope of v2 against v1.
    Each visit's pair is oriented against the fitted primary velocity, since the
    per-visit fits can exchange the two components."""
    a, b, v1 = series(o.get("rv1_untied_per_visit", "")), series(o.get("rv2_untied_per_visit", "")), series(o["v1_per_visit"])
    if not (a.size == b.size == v1.size >= 3):
        return np.nan
    swap = np.abs(a - v1) > np.abs(b - v1)
    x, y = np.where(swap, b, a), np.where(swap, a, b)
    if np.ptp(x) < 10.0:
        return np.nan
    slope = np.polyfit(x, y, 1)[0]
    return -1.0 / slope if slope < 0 else np.nan
wq = [(wilson_q(o), F(o["q_dyn"]), F(cat[o["sdss_id"]]["best_q"])) for o in multi
      if o["prefers_binary"] == "True" and int(F(o["n_visits"])) >= 3]
wq = np.array([w for w in wq if np.isfinite(w[0]) and 0.05 < w[0] < 3])
if len(wq):
    rk = lambda v: np.argsort(np.argsort(v))
    for j, lab in ((1, "q_dyn (tied joint fit)"), (2, "q_spec (combined spectrum)")):
        m = np.isfinite(wq[:, j])
        print("independent Wilson q vs %-26s n=%d  Spearman %.3f  median |dq| %.3f"
              % (lab, m.sum(), np.corrcoef(rk(wq[m, 0]), rk(wq[m, j]))[0, 1], np.median(np.abs(wq[m, 0] - wq[m, j]))))
for o in out:
    o["q_wilson"] = ("%.3f" % wilson_q(o)) if (o.get("prefers_binary") == "True" and o["in_sb2_catalog"]
                                              and np.isfinite(wilson_q(o))) else ""

# ---------------------------------------------------------------- write
legacy = C("stage2_catalog_full_legacy.csv")
if not os.path.exists(legacy) and os.path.exists(C("stage2_catalog_full.csv")):
    shutil.copy(C("stage2_catalog_full.csv"), legacy)
cols = ["sdss_id", "gaia_dr3_source_id", "ra", "dec", "in_sb2_catalog", "n_visits", "delta_chi2", "f_imp",
        "prefers_binary", "q_spec", "q_dyn", "q_dyn_at_bound", "v_single", "gamma", "v1_range",
        "v1_per_visit", "v2_per_visit", "mjd_per_visit", "visit_index_per_visit",
        "rv1_untied_per_visit", "rv2_untied_per_visit", "q_wilson", "refit", "n_visits_rv",
        "v_rad_median_astra", "v_rad_std_astra", "dv_rad_max_astra", "sb1", "orbit_ready", "error"]
with open(C("stage2_catalog_full.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(out)
with open(C("stage2_mjds.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["sdss_id", "mjd_per_visit"])
    for o in out:
        w.writerow([o["sdss_id"], o.get("mjd_per_visit", "")])
print("wrote stage2_catalog_full.csv (%d rows) and stage2_mjds.csv; legacy copy kept" % len(out))
