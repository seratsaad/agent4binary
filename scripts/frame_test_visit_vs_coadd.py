#!/usr/bin/env python3
"""Which velocity frame is DR19 mwmVisit flux in?  Visit-vs-own-coadd CCF.

Every visit of a star is cross-correlated against that star's own mwmStar coadd.
The template is the star itself, so template mismatch, the problem that made the
August model-based test unreliable, cancels. For visit i the measured shift of
the visit relative to the coadd is predicted to be

    0          if the visit flux is in the source rest frame
    v_rad_i    if it is in the barycentric frame
    v_rel_i    if it is in the observed (topocentric) frame, v_rel = v_rad - bc

assuming the coadd itself sits in the rest frame (checked separately).

Modes
    --validate   inject known Doppler shifts into real visits and recover them;
                 the measurement is only trusted if this passes.
    --measure    measure every visit of the control and SB2 samples and write
                 resources/frame_test/visit_vs_coadd.csv
"""
import argparse, csv, glob, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import physics as P  # noqa: E402

SAMPLES = {
    "control": ("data/dr19_visits_ctl", "data/dr19_raw_controls_bench"),
    "sb2": ("data/dr19_raw_visits", "data/dr19_raw_sb2"),
}
COARSE = np.arange(-180.0, 180.01, 2.0)
FINE_HALF, FINE_STEP = 4.0, 0.1
OUT = os.path.join(ROOT, "resources", "frame_test", "visit_vs_coadd.csv")


def depth(flux, ivar):
    """Continuum-normalized line depth (1 - f) and a validity mask."""
    obs, err = P._prep_visit_sc(flux, ivar)
    ok = np.isfinite(obs) & (err < 1e5) & (obs > 0.2) & (obs < 1.3)
    return np.where(ok, 1.0 - obs, 0.0), ok.astype(float)


def score(vis_d, vis_m, tpl_d, tpl_m, v):
    """Normalized CCF of the visit against the template shifted by +v km/s."""
    t = P._doppler_shift(tpl_d, v)
    tm = P._doppler_shift(tpl_m, v) > 0.99
    w = (vis_m > 0) & tm
    if w.sum() < 500:
        return np.nan
    a, b = vis_d[w], t[w]
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def measure_shift(vis_d, vis_m, tpl_d, tpl_m):
    """Velocity of the visit relative to the template (positive = redshift)."""
    s = np.array([score(vis_d, vis_m, tpl_d, tpl_m, v) for v in COARSE])
    if not np.isfinite(s).any():
        return np.nan, np.nan
    v0 = COARSE[int(np.nanargmax(s))]
    fine = np.arange(v0 - FINE_HALF, v0 + FINE_HALF + 1e-9, FINE_STEP)
    f = np.array([score(vis_d, vis_m, tpl_d, tpl_m, v) for v in fine])
    k = int(np.nanargmax(f))
    if 0 < k < len(fine) - 1:
        y0, y1, y2 = f[k - 1], f[k], f[k + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        return float(fine[k] + off * FINE_STEP), float(y1)
    return float(fine[k]), float(f[k])


def load(sample, sid):
    vdir, cdir = SAMPLES[sample]
    v = np.load(os.path.join(ROOT, vdir, "%s.npz" % sid), allow_pickle=True)
    c = np.load(os.path.join(ROOT, cdir, "%s.npz" % sid), allow_pickle=True)
    return v, c


def star_ids(sample):
    vdir, cdir = SAMPLES[sample]
    a = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(ROOT, vdir, "*.npz"))}
    b = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(ROOT, cdir, "*.npz"))}
    return sorted(a & b)


def validate(n_stars=40, seed=1):
    """Inject known shifts into real visits; compare m(shifted) - m(original)."""
    rng = np.random.default_rng(seed)
    ids = star_ids("control")
    rng.shuffle(ids)
    injections = [-60.0, -23.0, -6.5, 2.7, 15.0, 44.0]
    errs = []
    used = 0
    for sid in ids:
        if used >= n_stars:
            break
        v, c = load("control", sid)
        tpl_d, tpl_m = depth(c["flux_raw"], c["ivar"])
        fr = np.atleast_2d(v["flux_raw"])[0]
        iv = np.atleast_2d(v["ivar"])[0]
        d0, m0 = depth(fr, iv)
        base, pk = measure_shift(d0, m0, tpl_d, tpl_m)
        if not np.isfinite(base) or pk < 0.5:
            continue
        used += 1
        for inj in injections:
            ds, ms = depth(P._doppler_shift(fr, inj), P._doppler_shift(iv, inj))
            got, _ = measure_shift(ds, ms, tpl_d, tpl_m)
            errs.append((inj, got - base - inj))
    e = np.array(errs)
    print("VALIDATION on %d real control visits, %d injections" % (used, len(e)))
    for inj in injections:
        r = e[e[:, 0] == inj, 1]
        print("  inject %+6.1f km/s : bias %+6.3f  scatter %5.3f  worst %5.2f"
              % (inj, np.nanmedian(r), np.nanstd(r), np.nanmax(np.abs(r))))
    allr = e[:, 1]
    # The hypotheses differ by |v_rad|, |v_rel| and |bc|, and visits are only
    # classified where all three exceed 6 km/s, so what matters is bias and rms,
    # not the worst single visit at modest S/N.
    ok = abs(np.nanmedian(allr)) < 0.1 and np.sqrt(np.nanmean(allr ** 2)) < 1.0
    print("  overall: median %+.3f, rms %.3f, worst %.2f km/s  -> %s"
          % (np.nanmedian(allr), np.sqrt(np.nanmean(allr ** 2)),
             np.nanmax(np.abs(allr)), "PASS" if ok else "FAIL"))
    return ok


def measure():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = ["sample", "sdss_id", "visit", "mjd", "telescope", "snr", "v_rad",
            "v_rel", "bc", "shift", "peak"]
    n = 0
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for sample in ("control", "sb2"):
            ids = star_ids(sample)
            for j, sid in enumerate(ids):
                v, c = load(sample, sid)
                try:
                    tpl_d, tpl_m = depth(c["flux_raw"], c["ivar"])
                except (np.linalg.LinAlgError, ValueError):
                    continue
                F = np.atleast_2d(v["flux_raw"])
                I = np.atleast_2d(v["ivar"])
                for i in range(F.shape[0]):
                    if not np.any(I[i] > 0):
                        continue
                    try:
                        d, m = depth(F[i], I[i])
                        sh, pk = measure_shift(d, m, tpl_d, tpl_m)
                    except (np.linalg.LinAlgError, ValueError):
                        sh, pk = np.nan, np.nan
                    g = lambda k: float(np.atleast_1d(v[k])[i]) if k in v.files else np.nan
                    tel = str(np.atleast_1d(v["telescope"])[i]) if "telescope" in v.files else ""
                    w.writerow([sample, sid, i, g("mjd"), tel, g("snr"), g("v_rad"),
                                g("v_rel"), g("bc"), "%.3f" % sh, "%.4f" % pk])
                    n += 1
                if (j + 1) % 250 == 0:
                    print("  %s %d/%d stars, %d visits" % (sample, j + 1, len(ids), n), flush=True)
    print("wrote", OUT, "visits:", n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--measure", action="store_true")
    a = ap.parse_args()
    if a.validate:
        validate()
    if a.measure:
        measure()
