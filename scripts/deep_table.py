#!/usr/bin/env python3
"""Shared access to the 16-visit stage-2 deep table (resources/census/stage2_deep.csv).

Every eccentricity script takes a system's epochs from the mjd_per_visit column
of its own deep-table row, which the fitter writes in the same order as
v1_per_visit / v2_per_visit. Do not read epochs from stage2_mjds.csv: that file
comes from the main run, which caps at 8 visits and covers other stars.

The path can be overridden with the environment variable A4B_DEEP_TABLE
(absolute, or relative to the repository root).
"""
import os
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = "resources/census/stage2_deep.csv"


def deep_path():
    p = os.environ.get("A4B_DEEP_TABLE", DEFAULT)
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


def parse(s):
    return np.array([float(x) for x in str(s).split(";") if x not in ("", "nan")])


def load_deep(path=None):
    """Read the deep table and check that every fitted row carries one epoch per
    velocity. Rows without velocities (failed fits) are passed through untouched."""
    p = path or deep_path()
    if not os.path.exists(p):
        raise SystemExit("deep table not found: %s (build it with scripts/build_stage2_deep.py "
                         "or set A4B_DEEP_TABLE)" % p)
    d = pd.read_csv(p)
    if "mjd_per_visit" not in d.columns:
        raise SystemExit("%s has no mjd_per_visit column; it predates the epoch fix" % p)
    bad = []
    for r in d.itertuples():
        n1 = len(parse(r.v1_per_visit)) if isinstance(r.v1_per_visit, str) else 0
        if n1 == 0:
            continue
        nt = len(parse(r.mjd_per_visit))
        n2 = len(parse(r.v2_per_visit))
        if nt != n1 or n2 != n1:
            bad.append((r.sdss_id, n1, n2, nt))
    assert not bad, ("%d rows of %s have velocity and epoch lists of different length, "
                     "e.g. (sdss_id, n_v1, n_v2, n_mjd) = %s" % (len(bad), p, bad[:3]))
    print("deep table: %s (%d rows)" % (p, len(d)), flush=True)
    return d


def epochs_by_id(d):
    """{sdss_id: MJD array in the order of v1_per_visit}."""
    return {int(sid): parse(m) for sid, m in zip(d.sdss_id, d.mjd_per_visit)
            if isinstance(m, str) and m}


def cadences(d, nmin=8, nmax=20):
    """Real cadences for mock populations: one MJD list per deep-table system,
    in table order (build_stage2_deep.py sorts by sdss_id)."""
    out = []
    for m in d.mjd_per_visit:
        if not isinstance(m, str):
            continue
        t = parse(m)
        if nmin <= len(t) <= nmax:
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
# Two-velocity method: untied per-visit pairs and the mass ratio used with them.
# --------------------------------------------------------------------------- #
Q_LO, Q_HI = 0.1, 1.5          # the stage-2 fitter's bounds on q_dyn


def parse_keep(s):
    """Like parse, but keeps every position ('nan' or '' -> nan), so the list stays
    aligned with mjd_per_visit."""
    if not isinstance(s, str) or not s:
        return np.array([])
    out = []
    for x in s.split(";"):
        try:
            out.append(float(x))
        except ValueError:
            out.append(np.nan)
    return np.array(out)


def _at_bound(r):
    v = getattr(r, "q_dyn_at_bound", False) if not isinstance(r, dict) else r.get("q_dyn_at_bound", False)
    return str(v).strip().lower() in ("true", "1", "1.0")


def q_twovel(r):
    """Mass ratio for the two-velocity model: the row's q_dyn, clipped to
    [0.1, 1.5]; q_spec (same clip) if q_dyn is missing, non-positive, flagged
    q_dyn_at_bound, or within 1e-3 of a bound. nan if neither is usable."""
    get = (lambda k: r.get(k, np.nan)) if isinstance(r, dict) else (lambda k: getattr(r, k, np.nan))
    try:
        qd = float(get("q_dyn"))
    except (TypeError, ValueError):
        qd = np.nan
    if (np.isfinite(qd) and qd > 0 and not _at_bound(r)
            and Q_LO + 1e-3 < qd < Q_HI - 1e-3):
        return float(np.clip(qd, Q_LO, Q_HI))
    try:
        qs = float(get("q_spec"))
    except (TypeError, ValueError):
        qs = np.nan
    return float(np.clip(qs, Q_LO, Q_HI)) if np.isfinite(qs) and qs > 0 else np.nan


def untied_pairs(r):
    """(t, a, b) for one deep-table row: epochs and the untied per-visit component
    velocities (rv1_untied_per_visit, rv2_untied_per_visit), in the order of
    mjd_per_visit. The pair at each visit is unordered. Visits whose epoch or
    either velocity is missing or non-finite are dropped. Lists of a different
    length from mjd_per_visit cannot be aligned, so the row then yields nothing."""
    get = (lambda k: r.get(k)) if isinstance(r, dict) else (lambda k: getattr(r, k, None))
    t = parse_keep(get("mjd_per_visit"))
    a = parse_keep(get("rv1_untied_per_visit"))
    b = parse_keep(get("rv2_untied_per_visit"))
    if not (len(t) == len(a) == len(b)) or len(t) == 0:
        z = np.array([])
        return z, z, z
    ok = np.isfinite(t) & np.isfinite(a) & np.isfinite(b)
    return t[ok], a[ok], b[ok]


def exchange_pairs(v1, v2, rng, p=0.5):
    """Mock helper: exchange the two velocities at each visit with probability p,
    so the mock pair carries no label information, like a real untied pair."""
    sw = rng.random(len(v1)) < p
    return np.where(sw, v2, v1), np.where(sw, v1, v2)


def observed_dv(r, method):
    """Observed velocity-spread scale for the 'observed only' matching variants
    (ecc_final_table.py, ecc_robust.py). method 'v1': range of v1_per_visit (the
    earlier choice; v1 comes from the momentum-tied joint fit). method 'twovel':
    max_i |a_i - b_i| of the untied pairs, which a label exchange does not change
    and which uses no fitted quantity."""
    if method == "twovel":
        _, a, b = untied_pairs(r)
        return float(np.max(np.abs(a - b))) if len(a) else np.nan
    get = (lambda k: r.get(k)) if isinstance(r, dict) else (lambda k: r[k] if hasattr(r, "__getitem__") and not hasattr(r, "_fields") else getattr(r, k))
    v = parse(get("v1_per_visit"))
    return float(v.max() - v.min()) if len(v) else np.nan


def table_method(d):
    """Method recorded in an ecc_marginal_real.csv-like table ('v1' if no column)."""
    if "method" not in d.columns:
        return "v1"
    m = sorted(set(d["method"].dropna().astype(str)))
    assert len(m) == 1, "mixed methods in one table: %s" % m
    return m[0]
