#!/usr/bin/env python3
"""False-positive rate of the joint multi-epoch fit on a single-star sample.

  python scripts/stage2_fpr.py controls   benchmark controls (stage2_controls_shards)
  python scripts/stage2_fpr.py singles    random unflagged dwarfs outside the training
                                          set of the fit's network (stage2_singles_shards)

Reports the rate for stars with two or more fitted visits, then for the subsets that
are the better single-star samples: pipeline velocities stable to 1 km/s (a control
whose own DR19 velocities move is a velocity variable, not a single star) and
astrometrically quiet (RUWE < 1.1, no Gaia non-single-star solution, where known).
For the controls it also splits by whether the star was in the training set of
the visit fit's network or rejected from it. Writes resources/census/stage2_<sample>.csv.
"""
import csv, glob, os, sys
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = lambda p: os.path.join(R, "resources", "census", p)
F = lambda x: float(x) if x not in (None, "", "nan") else np.nan
ser = lambda s: np.array([float(x) for x in str(s).split(";") if x not in ("", "nan")])
sample = sys.argv[1] if len(sys.argv) > 1 else "controls"
vdir = {"controls": os.path.join(R, "data", "dr19_visits_ctl"), "singles": os.path.join(R, "data", "dr19_raw_visits")}[sample]

rows = []
for f in sorted(glob.glob(C("stage2_%s_shards/shard_*.csv" % sample))):
    for r in csv.DictReader(open(f)):
        r["sdss_id"] = r["sdss_id"].split(".")[0]; rows.append(r)
split = {r["sdss_id"]: r["split"] for r in csv.DictReader(open(C("stage2_controls_split.csv")))} if sample == "controls" else {}
rej = {r[list(r)[0]].split(".")[0] for r in csv.DictReader(open(os.path.join(R, "resources", "payne_dr19_sc_rejected.csv")))}
gaia = {}
for p, kr, kn in ((C("benchmark_with_gaia.csv"), "ruwe", "gaia_nss"), (os.path.join(R, "resources", "gaia_dr19_dwarfs.parquet"), "ruwe", "non_single_star")):
    if p.endswith(".csv") and os.path.exists(p):
        for r in csv.DictReader(open(p)):
            gaia.setdefault(r["sdss_id"].split(".")[0], (F(r.get(kr)), F(r.get(kn))))
    elif os.path.exists(p):
        import pandas as pd
        for s, ru, ns in pd.read_parquet(p)[["sdss_id", kr, kn]].itertuples(index=False):
            gaia.setdefault(str(s).split(".")[0], (float(ru) if ru == ru else np.nan, float(ns) if ns == ns else np.nan))


_vtab = {}
if os.path.exists(C("stage2_%s_vstd.csv" % sample)):   # scatter computed where the visit files live
    _vtab = {r["sdss_id"]: F(r["v_rad_std_pipeline"]) for r in csv.DictReader(open(C("stage2_%s_vstd.csv" % sample)))}


def vstd(s):
    if s in _vtab:
        return _vtab[s]
    p = os.path.join(vdir, "%s.npz" % s)
    if not os.path.exists(p):
        return np.nan
    v = np.atleast_1d(np.load(p, allow_pickle=True)["v_rad"]).astype(float)
    return float(np.std(v)) if v.size >= 2 else np.nan


ok = [r for r in rows if not r.get("error") and np.isfinite(F(r.get("n_visits"))) and int(F(r["n_visits"])) >= 2]
pb = np.array([r["prefers_binary"] == "True" for r in ok]); nv = np.array([int(F(r["n_visits"])) for r in ok])
sid = [r["sdss_id"] for r in ok]; vs = np.array([vstd(s) for s in sid])
ru = np.array([gaia.get(s, (np.nan, np.nan))[0] for s in sid]); ns = np.array([gaia.get(s, (np.nan, np.nan))[1] for s in sid])
isrej = np.array([s in rej for s in sid])
sp = np.array([np.ptp(ser(r["v1_per_visit"])) if r["prefers_binary"] == "True" else 0.0 for r in ok])
stable = vs < 1.0; quiet = (ru < 1.1) & (ns == 0)
P = lambda m: "%.1f%% (n=%d)" % (100 * pb[m].mean(), m.sum()) if m.sum() else "n/a"
print("%s: rows %d | fitted with >= 2 visits %d | pipeline v_rad std known %d | Gaia known %d" % (sample, len(rows), len(ok), np.isfinite(vs).sum(), np.isfinite(ru).sum()))
print("false-positive rate, all fitted:", P(np.ones(len(ok), bool)))
for lo, hi, lab in ((2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 8, "6-8")):
    m = (nv >= lo) & (nv <= hi); print("   %-4s epochs: %s" % (lab, P(m)))
print("pipeline v_rad std < 1 km/s:", P(stable), "| >= 1 km/s:", P(np.isfinite(vs) & ~stable))
print("astrometrically quiet:", P(quiet), "| not quiet:", P(np.isfinite(ru) & ~quiet))
print("stable AND quiet:", P(stable & quiet))
if sample == "controls":
    print("in SC training set:", P(~isrej), "| rejected from it:", P(isrej), "| stable, in training:", P(stable & ~isrej))
    for h in ("heldout", "training"):
        m = np.array([split.get(s) == h for s in sid]); print("   %s half: %s | stable %s" % (h, P(m), P(m & stable)))
fp3 = pb & (nv >= 3)
if fp3.any():
    print("false positives with >= 3 epochs: %d | primary change < 1 km/s %.1f%% | > 10 km/s %.1f%% | median %.2f km/s"
          % (fp3.sum(), 100 * np.mean(sp[fp3] < 1), 100 * np.mean(sp[fp3] > 10), np.median(sp[fp3])))
    m3 = nv >= 3; print("rate with >= 3 epochs AND a motion cut of 10 km/s: %.1f%% (n=%d)" % (100 * np.mean((pb & (sp > 10))[m3]), m3.sum()))
cols = ["sdss_id", "split", "n_visits", "delta_chi2", "f_imp", "prefers_binary", "q_spec", "q_dyn", "v_single", "gamma", "v1_range",
        "v1_per_visit", "v2_per_visit", "mjd_per_visit", "rv1_untied_per_visit", "rv2_untied_per_visit", "v_rad_std_pipeline", "ruwe", "gaia_nss", "in_sc_training", "error"]
vsd = dict(zip(sid, vs))
for r in rows:
    s = r["sdss_id"]; r["split"] = split.get(s, "")
    r["v_rad_std_pipeline"] = "%.3f" % vsd[s] if s in vsd and np.isfinite(vsd[s]) else ""
    g = gaia.get(s, (np.nan, np.nan)); r["ruwe"] = "%.3f" % g[0] if np.isfinite(g[0]) else ""; r["gaia_nss"] = "%d" % g[1] if np.isfinite(g[1]) else ""
    r["in_sc_training"] = int(s not in rej) if sample == "controls" else 0
with open(C("stage2_%s.csv" % sample), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print("wrote stage2_%s.csv (%d rows)" % (sample, len(rows)))
