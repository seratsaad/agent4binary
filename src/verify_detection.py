#!/usr/bin/env python3
"""
Verification harness for the self-consistent (SC) SB2 detector.

The pipeline under test is the production path RAW -> continuum_normalize ->
detect, with NO survey continuum and NO binspec net anywhere:
  - observed spectra load as RAW flux + ivar (data/dr19_raw_sb2/,
    data/dr19_raw_controls/ for the retained single controls),
  - physics.dr19_sc_single_vs_binary runs continuum_normalize (per-chip sigma-
    clipped Chebyshev from raw flux), fits OUR SC-trained 5-label net, composes the
    binary with OUR OWN Teff-keyed un-normalization continuum, and applies the
    EXACT El-Badry et al. (2018b) Eq. B1 f_imp + Table B1 thresholds.

We score N_PER_CLASS DR19 SB2 (true positives) + N_PER_CLASS held-out single
controls (true negatives) and report:
  - completeness  = fraction of SB2 flagged prefers_binary,
  - contamination = fraction of controls flagged prefers_binary,
  - per-class median Delta-chi2 and median f_imp,
  - per-star wall-clock.

This script is a small smoke harness in the minimal checkout. Full completeness
and purity claims must come from the binspec fair test and the post-fix
held-out-control rerun, not from this 25+25 sample.

Run:  python src/verify_detection.py          (defaults to 25 per class)
      N_PER_CLASS=3 python src/verify_detection.py   (quick smoke test)

SB2 positives  : data/dr19_raw_sb2/{sdss_id}.npz {wl, flux_raw, ivar}.
Single controls: retained local control spectra in data/dr19_raw_controls/.
                 When validation-set IDs overlap this directory, they are sampled
                 across the SNR distribution; otherwise the retained controls are
                 sampled deterministically.
Seeds          : the catalog five labels from the SB2 / training manifests.
"""

import os
import csv
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# This script lives in src/, so the project root is one level up. We add src/ to
# sys.path and import physics the same way the MCP servers do.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import physics  # noqa: E402

SB2_DIR = os.path.join(ROOT, "data", "dr19_raw_sb2")
CTRL_DIR = os.path.join(ROOT, "data", "dr19_raw_controls")
SB2_MANIFEST = os.path.join(ROOT, "resources", "dr19_sb2_manifest.csv")
TRAIN_MANIFEST = os.path.join(ROOT, "resources", "dr19_raw_train_manifest.csv")
VAL_SET = os.path.join(ROOT, "models", "val_set_dr19_sc.npz")  # held-out singles

N_PER_CLASS = int(os.environ.get("N_PER_CLASS", "25"))


def load_seed_map(manifest):
    """sdss_id -> (teff, logg, fe_h, mg_h, v_macro) for the single-fit seed."""
    m = {}
    for r in csv.DictReader(open(manifest)):
        try:
            m[int(r["sdss_id"])] = (float(r["teff"]), float(r["logg"]),
                                    float(r["fe_h"]), float(r["mg_h"]),
                                    float(r["v_macro"]))
        except Exception:
            continue
    return m


def pick_sb2(n):
    """Up to n SB2 sdss_ids with a raw npz present, highest snr first."""
    rows = list(csv.DictReader(open(SB2_MANIFEST)))
    try:
        rows.sort(key=lambda r: -float(r.get("snr", "0") or 0))
    except Exception:
        pass
    out = []
    for r in rows:
        sid = int(r["sdss_id"])
        if os.path.exists(os.path.join(SB2_DIR, "%d.npz" % sid)):
            out.append(sid)
        if len(out) >= n:
            break
    return out


def pick_controls(n):
    """n retained single controls, sampled deterministically.

    Prefer validation-set IDs when they are present locally; otherwise fall back
    to all .npz files in data/dr19_raw_controls.
    """
    try:
        v = np.load(VAL_SET)
        val_ids = [int(x) for x in v["ids"]]
    except Exception:
        val_ids = []
    snr_map = {int(r["sdss_id"]): float(r.get("snr", "0") or 0)
               for r in csv.DictReader(open(TRAIN_MANIFEST))}
    present = [s for s in val_ids
               if os.path.exists(os.path.join(CTRL_DIR, "%d.npz" % s))]
    if not present:
        present = [int(os.path.splitext(fn)[0]) for fn in os.listdir(CTRL_DIR)
                   if fn.endswith(".npz") and os.path.splitext(fn)[0].isdigit()]
    present.sort(key=lambda s: snr_map.get(s, 0.0))
    if len(present) > n:
        idx = np.linspace(0, len(present) - 1, n).round().astype(int)
        return [present[i] for i in idx]
    return present


def run_one(sid, npz_dir, seed_map):
    """Run the SC detector on one star from its RAW npz; (result, wall_seconds)."""
    d = np.load(os.path.join(npz_dir, "%d.npz" % sid))
    flux_raw = np.asarray(d["flux_raw"], float)
    ivar = np.asarray(d["ivar"], float)
    t0 = time.time()
    r = physics.dr19_sc_single_vs_binary(flux_raw, ivar, seed=seed_map.get(sid))
    return r, time.time() - t0


def summarize(sb2, ctl, wall):
    """Print completeness / contamination / per-class medians / timing."""
    n_sb2, n_ctl = len(sb2), len(ctl)
    comp = np.mean([r["prefers_binary"] for r in sb2]) if n_sb2 else float("nan")
    cont = np.mean([r["prefers_binary"] for r in ctl]) if n_ctl else float("nan")
    print("\n=== SELF-CONSISTENT detector (RAW -> continuum_normalize -> SC net) ===")
    print("  N SB2=%d  N control=%d" % (n_sb2, n_ctl))
    print("  COMPLETENESS  (SB2 flagged binary)     = %.1f%%  (%d/%d)"
          % (100 * comp, int(round(comp * n_sb2)), n_sb2))
    print("  CONTAMINATION (controls flagged binary) = %.1f%%  (%d/%d)"
          % (100 * cont, int(round(cont * n_ctl)), n_ctl))
    for cls, rs in (("SB2", sb2), ("control", ctl)):
        if not rs:
            continue
        dch = np.median([r["delta_chi2"] for r in rs])
        fim = np.median([r["f_imp"] for r in rs])
        print("  %-8s median Delta-chi2=%9.1f  median f_imp=%6.3f"
              % (cls, dch, fim))
    print("  per-star wall-clock: median=%.2fs  mean=%.2fs  (N=%d)"
          % (np.median(wall), np.mean(wall), len(wall)))
    print("\n  NOTE: this is a minimal-checkout smoke harness. Use the 250-star "
          "binspec fair test for the decisive completeness number.")


def main():
    sb2_ids = pick_sb2(N_PER_CLASS)
    ctl_ids = pick_controls(N_PER_CLASS)
    print("SB2 test stars: %d   single controls: %d" % (len(sb2_ids), len(ctl_ids)),
          flush=True)

    sb2_seed = load_seed_map(SB2_MANIFEST)
    ctl_seed = load_seed_map(TRAIN_MANIFEST)

    sb2, ctl, wall = [], [], []
    print("\n--- scoring SB2 positives ---", flush=True)
    for k, sid in enumerate(sb2_ids):
        r, wc = run_one(sid, SB2_DIR, sb2_seed)
        sb2.append(r)
        wall.append(wc)
        print("  SB2 %2d/%d sid=%d  dchi2=%8.1f f_imp=%5.3f -> %s (%.1fs)"
              % (k + 1, len(sb2_ids), sid, r["delta_chi2"], r["f_imp"],
                 "BINARY" if r["prefers_binary"] else "single", wc), flush=True)
    print("\n--- scoring single controls ---", flush=True)
    for k, sid in enumerate(ctl_ids):
        r, wc = run_one(sid, CTRL_DIR, ctl_seed)
        ctl.append(r)
        wall.append(wc)
        print("  CTL %2d/%d sid=%d  dchi2=%8.1f f_imp=%5.3f -> %s (%.1fs)"
              % (k + 1, len(ctl_ids), sid, r["delta_chi2"], r["f_imp"],
                 "BINARY" if r["prefers_binary"] else "single", wc), flush=True)
    summarize(sb2, ctl, wall)


if __name__ == "__main__":
    main()
