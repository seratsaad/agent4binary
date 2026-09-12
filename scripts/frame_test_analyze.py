#!/usr/bin/env python3
"""Classify each DR19 mwmVisit spectrum by velocity frame.

Reads resources/frame_test/visit_vs_coadd.csv, written by
scripts/frame_test_visit_vs_coadd.py, and compares each measured visit-vs-coadd
shift with the three predictions 0 (rest frame), v_rad (barycentric) and v_rel
(observed). The coadd itself was checked to sit in the rest frame. A visit is
classified only when the three predictions are at least SEP km/s apart, i.e.
|v_rad|, |v_rel| and |bc| all exceed SEP, and it is assigned to a frame only when
its residual to that prediction is below TOL km/s.
"""
import collections, csv, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "resources", "frame_test", "visit_vs_coadd.csv")
OUT = os.path.join(ROOT, "resources", "frame_test", "frame_summary.txt")
SEP, TOL, PEAK = 6.0, 3.0, 0.5
ERA_SPLIT = 59160.0   # SDSS-IV APOGEE ends before this MJD, SDSS-V starts after
CLASSES = ["rest", "bary", "topo", "none", "ambiguous", "bad"]


def f(r, k):
    try:
        return float(r[k])
    except (ValueError, KeyError):
        return np.nan


def classify(r):
    s, vr, vl, bc, pk = (f(r, k) for k in ("shift", "v_rad", "v_rel", "bc", "peak"))
    if not np.all(np.isfinite([s, vr, vl, bc, pk])) or pk < PEAK:
        return "bad"
    if min(abs(vr), abs(vl), abs(bc)) < SEP:
        return "ambiguous"
    res = {"rest": abs(s), "bary": abs(s - vr), "topo": abs(s - vl)}
    k = min(res, key=res.get)
    return k if res[k] < TOL else "none"


def fit_alpha_beta(rows):
    """Least squares for shift = alpha * v_rad - beta * bc.

    rest frame: alpha = beta = 0; barycentric: alpha = 1, beta = 0;
    observed frame: alpha = beta = 1 (since v_rel = v_rad - bc)."""
    X, y = [], []
    for r in rows:
        s, vr, bc, pk = f(r, "shift"), f(r, "v_rad"), f(r, "bc"), f(r, "peak")
        if np.all(np.isfinite([s, vr, bc, pk])) and pk >= PEAK and abs(s) < 170:
            X.append([vr, -bc]); y.append(s)
    if len(y) < 10:
        return np.nan, np.nan, len(y)
    sol, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    return float(sol[0]), float(sol[1]), len(y)


def main():
    rows = list(csv.DictReader(open(IN)))
    for r in rows:
        r["cls"] = classify(r)
        m = f(r, "mjd")
        r["era"] = "SDSS-IV" if m < ERA_SPLIT else "SDSS-V"
    lines = []
    P = lines.append
    P("DR19 mwmVisit frame test: each visit against its own rest-frame coadd")
    P("classified only when |v_rad|, |v_rel|, |bc| > %.0f km/s; frame assigned when "
      "residual < %.0f km/s; CCF peak >= %.1f" % (SEP, TOL, PEAK))
    P("")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["sample"], r["era"], r["telescope"])].append(r)
        groups[(r["sample"], "all", "all")].append(r)
    P("%-8s %-8s %-7s %6s | %6s %6s %6s %6s | %5s %5s" %
      ("sample", "era", "tel", "n", "rest", "bary", "topo", "none", "alpha", "beta"))
    for key in sorted(groups):
        g = groups[key]
        c = collections.Counter(r["cls"] for r in g)
        n_cl = sum(c[k] for k in ("rest", "bary", "topo", "none"))
        a, b, _ = fit_alpha_beta(g)
        pct = lambda k: "%5.1f%%" % (100.0 * c[k] / n_cl) if n_cl else "   - "
        P("%-8s %-8s %-7s %6d | %6s %6s %6s %6s | %+5.2f %+5.2f   (classified %d)" %
          (key + (len(g),) + (pct("rest"), pct("bary"), pct("topo"), pct("none"), a, b, n_cl)))
    P("")
    by_star = collections.defaultdict(list)
    for r in rows:
        if r["cls"] in ("rest", "bary", "topo"):
            by_star[(r["sample"], r["sdss_id"])].append(r["cls"])
    multi = {k: v for k, v in by_star.items() if len(v) >= 2}
    mixed = {k: v for k, v in multi.items() if len(set(v)) > 1}
    P("stars with >= 2 classified visits: %d ; of these with visits in DIFFERENT frames: %d"
      % (len(multi), len(mixed)))
    for k, v in list(mixed.items())[:12]:
        P("   mixed: %s %s %s" % (k[0], k[1], collections.Counter(v)))
    P("")
    for cls in ("rest", "bary", "topo"):
        ex = [r for r in rows if r["cls"] == cls][:6]
        P("examples %-4s: %s" % (cls, ", ".join(
            "%s(mjd %.0f, shift %+.1f, v_rad %+.1f, v_rel %+.1f)" %
            (r["sdss_id"], f(r, "mjd"), f(r, "shift"), f(r, "v_rad"), f(r, "v_rel")) for r in ex)))
    txt = "\n".join(lines)
    open(OUT, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
