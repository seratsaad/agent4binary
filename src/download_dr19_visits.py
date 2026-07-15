#!/usr/bin/env python3
"""
Stage 2 data layer: fetch RAW per-VISIT APOGEE flux + ivar from the DR19 Astra
``mwmVisit`` product (the per-visit analog of ``mwmStar``).

WHY A SEPARATE PRODUCT. Our Stage-1 detector
(physics.dr19_sc_single_vs_binary) fits the COMBINED mwmStar coadd. The close
binaries it misses have a small velocity separation IN THE COADD: their two
components are nearly aligned in velocity at the epoch the coadd represents, so
the coadded spectrum looks single. The ORBITAL velocity change between visits is
the signal the coadd lacks. El-Badry's Stage 2 re-fits the individual visit
spectra jointly to recover exactly these systems. To do that we need the visit
spectra, which live in ``mwmVisit``, not ``mwmStar``.

SAS PATH (verified live, DR19, Astra v0.6.0). Same sharding as mwmStar
(src/download_dr19_raw.mwm_url): XX = (sdss_id // 100) % 100, YY = sdss_id % 100.

    https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/spectra/visit/
        XX/YY/mwmVisit-0.6.0-<sdss_id>.fits

FILE STRUCTURE (verified on real files). Five HDUs:
    HDU 0  PrimaryHDU (empty)
    HDU 1  BOSS/APO    (empty for APOGEE dwarfs)
    HDU 2  BOSS/LCO    (empty)
    HDU 3  APOGEE/APO  -> one ROW PER VISIT taken from APO
    HDU 4  APOGEE/LCO  -> one ROW PER VISIT taken from LCO
Each APOGEE visit row carries, among many columns:
    flux, ivar   : length-8575 arrays on the SAME log-lambda grid as mwmStar
                   (physics.WAVELENGTH). These are the RAW per-visit flux/ivar,
                   in the OBSERVED (topocentric) frame -- the per-visit Doppler
                   shift is PRESERVED (verified by cross-correlating visits: the
                   relative shift between two visits tracks their v_rel
                   difference, NOT zero). This is the orbital signal Stage 2
                   needs; we do NOT shift the flux to a common rest frame.
    snr          : Astra median S/N of the visit (El-Badry skips visits < 30).
    v_rad        : doppler heliocentric radial velocity of the visit (km/s).
    v_rel        : relative (topocentric) velocity (km/s); v_rad = v_rel + bc.
    bc           : barycentric correction applied at this visit (km/s).
    mjd, fiber   : visit identifiers (kept for provenance / dedup).
    valid        : Astra visit-validity flag.

WHAT WE SAVE. One npz per sdss_id under data/dr19_raw_visits/<sdss_id>.npz with
stacked per-visit arrays so the joint fit reads it with a single np.load:

    wl        : (8575,)            physics.WAVELENGTH (stored once)
    flux_raw  : (n_visit, 8575)    RAW per-visit flux (observed frame)
    ivar      : (n_visit, 8575)    RAW per-visit ivar
    v_rad     : (n_visit,)         heliocentric RV per visit (km/s)
    v_rel     : (n_visit,)         topocentric/relative velocity per visit (km/s)
    bc        : (n_visit,)         barycentric correction per visit (km/s)
    snr       : (n_visit,)         Astra visit S/N
    mjd       : (n_visit,)         visit MJD
    telescope : (n_visit,)         'apo25m' / 'lco25m' (for provenance)

We keep BOTH telescopes' visits (APO HDU 3 and LCO HDU 4) stacked, since a
system can have visits from each. Visits with all-zero ivar (empty rows) are
dropped. NO survey continuum, NO normalization here -- identical to Stage 1, so
physics.continuum_normalize is the only normalizer and it runs per visit later.

Auth / path logic mirrors src/download_dr19_raw.py (public SAS, no auth token;
a plain requests session with a User-Agent). Idempotent: a non-trivial existing
npz is skipped, so re-runs resume.

CLI:
    python src/download_dr19_visits.py <sdss_id> [<sdss_id> ...]
    python src/download_dr19_visits.py --from-csv path.csv [--col sdss_id]
"""
import argparse
import csv
import os
import sys
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physics  # NPIX (8575) + the shared WAVELENGTH grid

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

VISIT_DIR = os.path.join(_ROOT, "data", "dr19_raw_visits")
os.makedirs(VISIT_DIR, exist_ok=True)

# Same Astra 0.6.0 tree as mwmStar; the product folder is 'visit' not 'star'.
MWM_VISIT_BASE = "https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/spectra/visit"
# Concurrency for fetch_many's thread pool. data.sdss.org intermittently REJECTS
# requests under burst load: at 24 simultaneous connections a whole batch can come
# back connection-reset (RemoteDisconnected / ConnectionResetError), which the
# urllib3 Retry adapter does NOT cover (it only retries HTTP 5xx, not transport-
# level resets), so every one surfaced as a swallowed exception -> "fail" (verified
# 2026-06: 24 threads -> 61/61 fail; the SAME ids succeed serially). On the cloud
# each VM runs this from MANY worker processes at once, so we keep the per-process
# pool modest AND add an explicit per-id retry below. 8 is reliable in testing.
WORKERS = 8

_S = requests.Session()
_S.headers["User-Agent"] = "agent4binary-visits/0.1"
# data.sdss.org occasionally stalls the TLS handshake on the larger visit files;
# a retry adapter + a (connect, read) timeout tuple recovers cleanly (verified:
# without it ~1 in 3 GETs hangs to the read timeout, with it every GET succeeds).
# connect=read=status retries cover BOTH the TLS stall AND a transport-level reset
# on a fresh connection (a reset mid-stream still surfaces to fetch_one's loop).
_S.mount("https://", HTTPAdapter(pool_maxsize=WORKERS, max_retries=Retry(
    total=5, connect=5, read=3, backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"])))
_TIMEOUT = (15, 180)            # (connect, read) seconds
_FETCH_ATTEMPTS = 4             # explicit per-id retries on top of the adapter


def mwm_visit_url(sdss_id):
    """mwmVisit URL. Sharding XX=(id//100)%100, YY=id%100 (same as mwmStar)."""
    sdss_id = int(sdss_id)
    xx = (sdss_id // 100) % 100
    yy = sdss_id % 100
    return "%s/%02d/%02d/mwmVisit-0.6.0-%d.fits" % (MWM_VISIT_BASE, xx, yy, sdss_id)


def read_visit_spectra(fits_bytes):
    """mwmVisit bytes -> dict of stacked per-visit arrays, or None.

    Reads the APOGEE/APO (HDU 3) and APOGEE/LCO (HDU 4) tables, one row per
    visit, and stacks the RAW flux/ivar (no normalization) plus the per-visit
    velocities and metadata. Drops any visit whose ivar is entirely zero (an
    empty / unmeasured row). Returns None if no usable visit is found.
    """
    fluxes, ivars = [], []
    v_rad, v_rel, bc, snr, mjd, tele = [], [], [], [], [], []
    with fits.open(BytesIO(fits_bytes)) as hdul:
        for idx, scope in ((3, "apo25m"), (4, "lco25m")):  # APOGEE/APO, APOGEE/LCO
            if idx >= len(hdul):
                continue
            data = hdul[idx].data
            if data is None or hdul[idx].header.get("NAXIS2", 0) <= 0:
                continue
            for row in data:
                flux = np.asarray(row["flux"], dtype=np.float64)
                ivar = np.asarray(row["ivar"], dtype=np.float64)
                if flux.size != physics.NPIX or ivar.size != physics.NPIX:
                    continue
                if not np.any(ivar > 0):           # empty / unmeasured visit row
                    continue
                fluxes.append(flux)
                ivars.append(ivar)
                # Velocity columns. Missing/non-finite -> NaN (handled downstream).
                v_rad.append(_col(row, "v_rad"))
                v_rel.append(_col(row, "v_rel"))
                bc.append(_col(row, "bc"))
                snr.append(_col(row, "snr"))
                mjd.append(_col(row, "mjd"))
                tele.append(scope)
    if not fluxes:
        return None
    return {
        "wl": physics.WAVELENGTH,
        "flux_raw": np.asarray(fluxes, dtype=np.float64),
        "ivar": np.asarray(ivars, dtype=np.float64),
        "v_rad": np.asarray(v_rad, dtype=np.float64),
        "v_rel": np.asarray(v_rel, dtype=np.float64),
        "bc": np.asarray(bc, dtype=np.float64),
        "snr": np.asarray(snr, dtype=np.float64),
        "mjd": np.asarray(mjd, dtype=np.float64),
        "telescope": np.asarray(tele),
    }


def _col(row, name):
    """Read a scalar FITS column tolerantly -> float, NaN if absent / non-numeric."""
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def fetch_one(sdss_id, out_dir=VISIT_DIR):
    """Download + save one sdss_id's visit spectra. Returns 'ok' | 'skip' | 'fail'.

    Retries transient transport failures (connection reset / read timeout / a 404
    that is really the SAS load-shedding under burst) up to _FETCH_ATTEMPTS times
    with a short backoff. A genuine 404 (no mwmVisit product for the star) is NOT
    retried -- it is a stable "fail" the caller records as no_visit_file. Only the
    transport exceptions and 5xx/429 (which the adapter already retried) and a
    too-short body are retried, since those are the load-induced failures.
    """
    sdss_id = int(sdss_id)
    dest = os.path.join(out_dir, "%d.npz" % sdss_id)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return "skip"
    url = mwm_visit_url(sdss_id)
    for attempt in range(_FETCH_ATTEMPTS):
        try:
            r = _S.get(url, timeout=_TIMEOUT)
            if r.status_code == 404:
                return "fail"               # stable: no visit product for this id
            if r.status_code != 200 or len(r.content) < 1000:
                time.sleep(0.5 * (attempt + 1))   # transient: back off and retry
                continue
            parsed = read_visit_spectra(r.content)
            if parsed is None:
                return "fail"               # 200 but no usable visit row
            np.savez_compressed(dest, **parsed)
            return "ok"
        except Exception:                   # connection reset / read timeout / TLS
            time.sleep(0.5 * (attempt + 1))
    return "fail"


def fetch_many(sdss_ids, out_dir=VISIT_DIR, label="visits"):
    """Download a list of sdss_ids in parallel. Returns the set present on disk."""
    sdss_ids = [int(s) for s in sdss_ids]
    t1 = time.time()
    ok = skip = fail = 0
    present = set()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_one, sid, out_dir): sid for sid in sdss_ids}
        for fut in as_completed(futs):
            sid = futs[fut]
            res = fut.result()
            if res in ("ok", "skip"):
                present.add(sid)
            ok += res == "ok"
            skip += res == "skip"
            fail += res == "fail"
    print("  [%s] ok=%d skip=%d fail=%d of %d in %.0fs"
          % (label, ok, skip, fail, len(sdss_ids), time.time() - t1), flush=True)
    return present


def load_visits(sdss_id, in_dir=VISIT_DIR):
    """Load a saved visit npz -> dict of arrays (see read_visit_spectra). Or None."""
    path = os.path.join(in_dir, "%d.npz" % int(sdss_id))
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def n_visits(sdss_id, in_dir=VISIT_DIR):
    """Number of stored visits for an sdss_id (0 if not on disk)."""
    d = load_visits(sdss_id, in_dir)
    return 0 if d is None else int(d["flux_raw"].shape[0])


def main():
    ap = argparse.ArgumentParser(description="Fetch DR19 mwmVisit per-visit spectra.")
    ap.add_argument("sdss_ids", nargs="*", help="one or more sdss_id")
    ap.add_argument("--from-csv", help="read sdss_ids from a CSV column")
    ap.add_argument("--col", default="sdss_id", help="CSV column name (default sdss_id)")
    args = ap.parse_args()

    ids = list(args.sdss_ids)
    if args.from_csv:
        with open(args.from_csv) as fh:
            for r in csv.DictReader(fh):
                v = str(r.get(args.col, "")).strip()
                if v:
                    ids.append(v)
    ids = [int(s) for s in ids if str(s).strip()]
    if not ids:
        ap.error("no sdss_ids given (positional or --from-csv)")

    print("fetching mwmVisit for %d sdss_ids into %s" % (len(ids), VISIT_DIR),
          flush=True)
    present = fetch_many(ids, label="cli")
    print("present: %d / %d" % (len(present), len(set(ids))), flush=True)


if __name__ == "__main__":
    main()
