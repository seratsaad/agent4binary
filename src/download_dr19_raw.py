#!/usr/bin/env python3
"""
Stage 1: fetch RAW mwmStar APOGEE flux + ivar (NOT flux/continuum).

The data layer for the self-consistent continuum pivot. We save the RAW `flux` and
`ivar` columns straight from the mwmStar APOGEE HDU, with NO survey continuum.
Normalization is done LATER and IDENTICALLY on training and observed spectra by
physics.continuum_normalize, so the survey continuum never enters the pipeline.

THREE sets are produced (each npz holds {wl, flux_raw, ivar} on physics.WAVELENGTH):

  1. data/dr19_raw/        ~8000 DR19 dwarfs for TRAINING the 5-label Payne.
       SQL: the_payne_apogee_star, logg 4-5, teff 4000-7000, snr>60, flag_bad=0,
       paged on an sdss_id cursor (GET-URL-length-safe paging). El-Badry SB2
       apogee_ids are EXCLUDED so no known binary leaks into the single-star
       training set.

  2. data/dr19_raw_sb2/    the El-Badry (2018b) SB2 DWARF test set (type==SB2,
       logg>4) cross-matched to DR19. Read from resources/dr19_sb2_manifest.csv
       (already cross-matched + labelled), re-fetched as RAW flux/ivar.

  3. data/dr19_raw_controls/  ~60 held-out single-star CONTROLS, drawn from the
       training-pool dwarfs but written to a SEPARATE directory and EXCLUDED from
       the training manifest, so the verify step (Stage 5) scores controls the
       trainer never saw.

LABELS. The five labels (teff, logg, fe_h, mg_h, v_macro) are read from the
existing manifests (resources/dr19_dwarfs_train_manifest.csv and
resources/dr19_sb2_manifest.csv), which already carry them per sdss_id. When we
page NEW dwarfs to reach ~8000, we pull the five labels in the SQL itself and
append them to a RAW manifest. The raw manifest written here,
resources/dr19_raw_train_manifest.csv, is the one Stage 3 trains from.

Idempotent: existing non-trivial npz are skipped, so re-runs resume.

mwmStar RAW columns and HDU selection: APOGEE HDU 3=APO / 4=LCO, columns
`flux`, `ivar`.
"""
import csv
import os
import sys
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physics  # for NPIX (8575) and the shared WAVELENGTH grid

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

RAW_TRAIN_DIR = os.path.join(_ROOT, "data", "dr19_raw")
RAW_SB2_DIR = os.path.join(_ROOT, "data", "dr19_raw_sb2")
RAW_CTRL_DIR = os.path.join(_ROOT, "data", "dr19_raw_controls")
for d in (RAW_TRAIN_DIR, RAW_SB2_DIR, RAW_CTRL_DIR):
    os.makedirs(d, exist_ok=True)

DWARF_MANIFEST = os.path.join(_ROOT, "resources", "dr19_raw_train_manifest.csv")
SB2_MANIFEST = os.path.join(_ROOT, "resources", "dr19_sb2_manifest.csv")
ELBADRY = os.path.join(_ROOT, "resources", "elbadry2018b_binaries.csv")
# The RAW training manifest Stage 3 reads (sdss_id + the five labels + snr).
RAW_TRAIN_MANIFEST = os.path.join(_ROOT, "resources", "dr19_raw_train_manifest.csv")
RAW_CTRL_MANIFEST = os.path.join(_ROOT, "resources", "dr19_raw_controls_manifest.csv")

MWM_BASE = "https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/spectra/star"
WORKERS = 48
SQL_PAGE = 5000          # rows per catalog page
CHUNK = 1000             # progress log cadence
TARGET_DWARFS = 8000     # ~8000 training dwarfs
N_CONTROLS = 60          # held-out single controls
LABEL_COLS = ["teff", "logg", "fe_h", "mg_h", "v_macro"]

_S = requests.Session()
_S.headers["User-Agent"] = "agent4binary-raw/0.1"


def mwm_url(sdss_id):
    """mwmStar URL. Sharding XX=(id//100)%100, YY=id%100 (confirmed)."""
    xx = (sdss_id // 100) % 100
    yy = sdss_id % 100
    return "%s/%02d/%02d/mwmStar-0.6.0-%d.fits" % (MWM_BASE, xx, yy, sdss_id)


def read_raw_spectrum(fits_bytes):
    """mwmStar bytes -> (flux_raw, ivar) on the 8575 grid, or None.

    Opens the populated APOGEE HDU (3=APO, 4=LCO) and returns the RAW `flux` and
    `ivar` columns UNCHANGED -- no survey continuum, no normalization. ivar==0
    pixels (chip gaps, masked) are left as-is so continuum_normalize can drop them.
    """
    with fits.open(BytesIO(fits_bytes)) as hdul:
        row = None
        for idx in (3, 4):                       # APOGEE/APO, APOGEE/LCO
            if idx < len(hdul) and hdul[idx].header.get("NAXIS2", 0) > 0:
                row = hdul[idx].data[0]
                break
        if row is None:
            return None
        flux = np.asarray(row["flux"], dtype=np.float64)
        ivar = np.asarray(row["ivar"], dtype=np.float64)
    if flux.size != physics.NPIX or ivar.size != physics.NPIX:
        return None
    return flux, ivar


def fetch_one(sdss_id, out_dir):
    """Download + save one RAW spectrum. Returns 'ok' | 'skip' | 'fail'."""
    dest = os.path.join(out_dir, "%d.npz" % sdss_id)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return "skip"
    try:
        r = _S.get(mwm_url(sdss_id), timeout=60)
        if r.status_code != 200 or len(r.content) < 1000:
            return "fail"
        parsed = read_raw_spectrum(r.content)
        if parsed is None:
            return "fail"
        flux_raw, ivar = parsed
        np.savez_compressed(dest, wl=physics.WAVELENGTH,
                            flux_raw=flux_raw, ivar=ivar)
        return "ok"
    except Exception:
        return "fail"


def fetch_many(id_label_rows, out_dir, label=""):
    """Download a list of (sdss_id, label_dict) in parallel into out_dir.

    Returns the set of sdss_ids that are present on disk afterwards (ok or skip),
    so a caller can write a manifest of exactly the spectra that exist.
    """
    t1 = time.time()
    ok = skip = fail = 0
    done = 0
    present = set()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_one, sid, out_dir): sid
                for sid, _ in id_label_rows}
        for fut in as_completed(futs):
            sid = futs[fut]
            res = fut.result()
            if res in ("ok", "skip"):
                present.add(sid)
            ok += res == "ok"
            skip += res == "skip"
            fail += res == "fail"
            done += 1
            if done % CHUNK == 0:
                el = time.time() - t1
                rate = done / el if el else 0
                eta = (len(futs) - done) / rate / 60.0 if rate else 0
                print("  [%s] %d/%d  ok=%d skip=%d fail=%d  %.1f f/s  ETA=%.0f min"
                      % (label, done, len(futs), ok, skip, fail, rate, eta),
                      flush=True)
    el = time.time() - t1
    print("  [%s] DONE ok=%d skip=%d fail=%d of %d in %.0fs"
          % (label, ok, skip, fail, len(id_label_rows), el), flush=True)
    return present


# --------------------------------------------------------------------------- #
# Catalog paging on an sdss_id cursor (pulls the five labels alongside the ids).
# --------------------------------------------------------------------------- #
def page_dwarf_rows(target, exclude_apogee_ids):
    """Page the_payne_apogee_star for dwarf rows (sdss_id + five labels).

    Cursor on sdss_id avoids the GET-URL-length 404 that an IN-list hits. Pulls
    the five labels in the SQL so a NEW dwarf carries its labels without a second
    query. Rows whose apogee_id is a known El-Badry binary are skipped (no binary
    in the single-star training set). Stops once `target` clean rows are gathered.

    Returns a list of (sdss_id, {teff, logg, fe_h, mg_h, v_macro, snr}) dicts.
    """
    from astroquery.sdss import SDSS
    rows = []
    cursor = -1
    while len(rows) < target:
        sql = (
            "SELECT TOP %d sdss_id, sdss4_apogee_id, teff, logg, fe_h, mg_h, "
            "v_macro, snr FROM the_payne_apogee_star "
            "WHERE logg BETWEEN 4.0 AND 5.0 AND teff BETWEEN 4000 AND 7000 "
            "AND snr > 60 AND flag_bad = 0 AND sdss_id > %d "
            "ORDER BY sdss_id" % (SQL_PAGE, cursor))
        tab = SDSS.query_sql(sql, data_release=19)
        if tab is None or len(tab) == 0:
            break
        for r in tab:
            apid = str(r["sdss4_apogee_id"]).strip()
            if apid in exclude_apogee_ids:
                continue
            try:
                lab = {c: float(r[c]) for c in LABEL_COLS}
                lab["snr"] = float(r["snr"])
            except Exception:
                continue
            if not all(np.isfinite(list(lab.values()))):
                continue
            rows.append((int(r["sdss_id"]), lab))
        cursor = int(tab["sdss_id"][-1])
        print("  paged %d clean dwarf rows (cursor=%d)" % (len(rows), cursor),
              flush=True)
        if len(tab) < SQL_PAGE:
            break
    return rows[:target]


def load_manifest_rows(path):
    """Read an existing manifest -> [(sdss_id, label_dict), ...] with finite labels."""
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                lab = {c: float(r[c]) for c in LABEL_COLS}
                lab["snr"] = float(r.get("snr", "0") or 0)
            except Exception:
                continue
            if not all(np.isfinite([lab[c] for c in LABEL_COLS])):
                continue
            out.append((int(r["sdss_id"]), lab))
    return out


def write_manifest(path, id_label_rows, present):
    """Write a RAW manifest of the spectra that exist on disk (present set)."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sdss_id"] + LABEL_COLS + ["snr"])
        for sid, lab in id_label_rows:
            if sid in present:
                w.writerow([sid] + [lab[c] for c in LABEL_COLS] + [lab["snr"]])


def build_training_set():
    """Fetch ~8000 training dwarfs (RAW) and write the raw training manifest."""
    # Exclude every El-Badry apogee_id (any type) so no known binary is trained on.
    exclude = set()
    if os.path.exists(ELBADRY):
        for r in csv.DictReader(open(ELBADRY)):
            exclude.add(r["apogee_id"].strip())

    # Seed from the existing dwarf manifest (already has labels + ids), then top up
    # by paging the catalog for new dwarfs until we reach ~8000 distinct ids.
    seed = load_manifest_rows(DWARF_MANIFEST)
    seed_ids = {sid for sid, _ in seed}
    print("seed dwarfs from existing manifest: %d" % len(seed), flush=True)

    rows = list(seed)
    if len(rows) < TARGET_DWARFS:
        need = TARGET_DWARFS - len(rows)
        print("paging catalog for ~%d more dwarfs ..." % need, flush=True)
        extra = page_dwarf_rows(TARGET_DWARFS + 2000, exclude)
        for sid, lab in extra:
            if sid not in seed_ids:
                rows.append((sid, lab))
                seed_ids.add(sid)
                if len(rows) >= TARGET_DWARFS:
                    break
    rows = rows[:TARGET_DWARFS]
    print("training dwarf rows to fetch: %d" % len(rows), flush=True)

    present = fetch_many(rows, RAW_TRAIN_DIR, label="train")
    write_manifest(RAW_TRAIN_MANIFEST, rows, present)
    print("raw training manifest: %s  (%d spectra present)"
          % (RAW_TRAIN_MANIFEST, len(present)), flush=True)
    return rows, present


def build_sb2_set():
    """Fetch the El-Badry SB2 dwarf test set (RAW) from the existing manifest."""
    rows = load_manifest_rows(SB2_MANIFEST)
    print("SB2 dwarf rows (from manifest): %d" % len(rows), flush=True)
    present = fetch_many(rows, RAW_SB2_DIR, label="sb2")
    print("SB2 raw spectra present: %d" % len(present), flush=True)
    return rows, present


def build_controls(train_rows, train_present):
    """Pick ~60 single-star controls, fetch RAW, EXCLUDE them from training.

    Controls are dwarfs that ARE in the training pool but get their own directory
    and manifest and are NOT listed in the raw training manifest, so the Stage 3
    trainer never sees them and the Stage 5 verify scores genuinely held-out
    singles.

    We pick controls that SPAN the SNR distribution of the training pool (evenly
    spaced quantiles in SNR), NOT the highest-SNR stars. Two reasons: a control set
    must be representative of normal targets, and at very high SNR (>1000) the
    absolute chi^2 of even a tiny single-star model misfit balloons past the Table
    B1 Delta-chi2 threshold, which would over-state contamination on a non-
    representative sample. The pre-pivot baseline contamination (7.5%) was measured
    on moderate-SNR (median ~120) held-out singles, so we match that regime here.
    """
    pool = [(sid, lab) for sid, lab in train_rows if sid in train_present]
    # Sort ascending in SNR and sample evenly across the distribution so the
    # controls span faint -> bright, centred on the typical target SNR.
    pool.sort(key=lambda kv: kv[1]["snr"])
    if len(pool) > N_CONTROLS:
        idx = np.linspace(0, len(pool) - 1, N_CONTROLS).round().astype(int)
        ctrl = [pool[i] for i in idx]
    else:
        ctrl = pool
    ctrl_ids = {sid for sid, _ in ctrl}
    present = fetch_many(ctrl, RAW_CTRL_DIR, label="ctrl")
    write_manifest(RAW_CTRL_MANIFEST, ctrl, present)
    print("controls present: %d  manifest: %s" % (len(present), RAW_CTRL_MANIFEST),
          flush=True)

    # Rewrite the training manifest WITHOUT the controls so they are held out.
    kept = [(sid, lab) for sid, lab in train_rows
            if sid in train_present and sid not in ctrl_ids]
    write_manifest(RAW_TRAIN_MANIFEST, kept,
                   {sid for sid, _ in kept})
    print("training manifest rewritten without controls: %d train rows"
          % len(kept), flush=True)
    return present


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("train", "all"):
        print("\n=== RAW TRAINING SET (~8000 dwarfs) ===", flush=True)
        train_rows, train_present = build_training_set()
    if which in ("sb2", "all"):
        print("\n=== RAW SB2 TEST SET ===", flush=True)
        build_sb2_set()
    if which in ("controls", "all"):
        print("\n=== RAW SINGLE CONTROLS (~60, held out) ===", flush=True)
        if which == "controls":
            train_rows = load_manifest_rows(RAW_TRAIN_MANIFEST)
            train_present = {sid for sid, _ in train_rows}
        build_controls(train_rows, train_present)
    print("\nStage 1 total runtime: %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
