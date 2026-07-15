#!/usr/bin/env python3
"""
Data I/O and format policy for agent4binary.

Format policy (modern, sliceable, cloud/Vertex-friendly):
  - Spectra (8575-float flux/ivar per star)  -> Parquet (one table; sdss_id +
    list<float32> flux/ivar). Columnar, compressed, sliceable by sdss_id, and
    readable by BigQuery / Vertex / DuckDB. JSON is avoided for numeric arrays.
  - Catalogs / manifests                      -> Parquet (+ JSONL for small ones).
  - Detection results                         -> Parquet (+ a JSON run-metadata
    sidecar).
  - Small configs / metadata                  -> JSON.
  - Model weights (.pt, .npz)                 -> KEPT as-is. .npz is a good format
    for opaque numeric weight blobs; there is no slicing/cloud benefit to
    converting them, so we do not.
  - FITS                                       -> upstream SDSS mwmStar only (we do
    not control it); all DERIVED products use the formats above.

The APOGEE wavelength grid is identical for every spectrum, so it is stored ONCE
as a sidecar (data/apogee_grid.json), not per row.

CLI: `python src/data_io.py convert` writes Parquet next to the existing files
(it does not delete the .npz/.csv; the loader switch is a separate step).
"""
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# APOGEE 8575-pixel log-linear grid, rebuilt analytically (no physics import).
NPIX = 8575
WAVELENGTH = 10.0 ** (4.179 + 6e-6 * np.arange(NPIX))


# --------------------------------------------------------------------------- #
# Spectra <-> Parquet
# --------------------------------------------------------------------------- #
def spectra_dir_to_parquet(npz_dir, out_parquet, flux_key="flux_raw"):
    """Pack a directory of <sdss_id>.npz {wl, flux/ivar} into one Parquet table.

    Columns: sdss_id (int64), flux (list<float32>), ivar (list<float32>). The
    wavelength grid is global (WAVELENGTH); it is not stored per row. Returns the
    number of spectra written.
    """
    ids, flux, ivar = [], [], []
    for fn in sorted(os.listdir(npz_dir)):
        if not (fn.endswith(".npz") and fn[:-4].isdigit()):
            continue
        d = np.load(os.path.join(npz_dir, fn))
        fk = flux_key if flux_key in d.files else ("flux" if "flux" in d.files else None)
        if fk is None:
            continue
        ids.append(int(fn[:-4]))
        flux.append(np.asarray(d[fk], dtype=np.float32))
        ivar.append(np.asarray(d["ivar"], dtype=np.float32) if "ivar" in d.files
                    else np.asarray(d.get("error", np.ones(NPIX)), dtype=np.float32))
    table = pa.table({
        "sdss_id": pa.array(ids, pa.int64()),
        "flux": pa.array(flux, pa.list_(pa.float32())),
        "ivar": pa.array(ivar, pa.list_(pa.float32())),
    })
    pq.write_table(table, out_parquet, compression="snappy")
    return len(ids)


def read_spectrum_parquet(parquet_path, sdss_id):
    """Read one spectrum (wl, flux, ivar) from a Parquet spectra table by id."""
    tbl = pq.read_table(parquet_path, filters=[("sdss_id", "=", int(sdss_id))])
    if tbl.num_rows == 0:
        raise KeyError(f"sdss_id {sdss_id} not in {parquet_path}")
    row = tbl.to_pylist()[0]
    return WAVELENGTH, np.asarray(row["flux"], float), np.asarray(row["ivar"], float)


# --------------------------------------------------------------------------- #
# DR13 apStar adapter (M1 DR13 parity rerun).
#
# DR13 predates Astra / mwmStar: spectra come from the old APOGEE-2 pipeline as
# `apStar` (combined) FITS rather than DR19's `mwmStar`. The APOGEE 8575-pixel
# log-lambda grid (CRVAL1=4.179, CDELT1=6e-6) is IDENTICAL between DR13 and DR19,
# so no regridding is needed -- this adapter only handles the different FITS
# layout so the detector can read either release. The return contract matches
# download_dr19_raw.read_raw_spectrum: (flux, ivar) on the 8575 grid, or None.
# --------------------------------------------------------------------------- #

# APOGEE_PIXMASK bad bits (apStar HDU3). BADPIX, CRPIX, SATPIX, UNFIXABLE,
# BADDARK, BADFLAT, BADERR, NOSKY = bits 0-7. Used only when apply_mask=True;
# the DR19 reader does not apply mask bits (it relies on ivar==0 / huge error),
# so apply_mask defaults False to keep the two release paths self-consistent.
APOGEE_BADPIX = (1 << 8) - 1  # 0xFF = bits 0..7

# apStar masks chip gaps / unusable pixels with a huge error sentinel (~1e10 in
# flux units) rather than zero ivar. Snap those to ivar==0 so the adapter's output
# uses the SAME "gap == ivar 0" convention as DR19 mwmStar, which continuum_normalize
# gates on (ivar <= 0 -> error inf). Real apStar errors never approach this.
APOGEE_ERR_BADVAL = 1e9


def load_dr13_apstar(fits_bytes, apply_mask=False):
    """DR13-era apStar bytes -> (flux_raw, ivar) on the 8575 grid, or None.

    apStar is multi-extension FITS: HDU1 = flux, HDU2 = error (sigma, NOT ivar),
    HDU3 = bitmask. For a star with several visits the flux/error arrays are 2D
    (n_combine + n_visit, 8575); row 0 is the pixel-weighted COMBINED spectrum --
    the coadd the Stage-1 detector wants -- so we take row 0. A single-visit
    apStar may be 1D (8575,), handled too.

    ivar is derived as 1/sigma**2; apStar encodes chip gaps and unusable pixels
    with a huge error rather than zero ivar, so sigma<=0 / non-finite -> ivar=0
    leaves exactly the pixels continuum_normalize already drops. With
    apply_mask=True the APOGEE_PIXMASK bad bits in HDU3 additionally zero ivar.
    """
    from io import BytesIO
    from astropy.io import fits

    with fits.open(BytesIO(fits_bytes)) as hdul:
        if len(hdul) < 3 or hdul[1].data is None or hdul[2].data is None:
            return None
        flux = np.asarray(hdul[1].data, dtype=np.float64)
        err = np.asarray(hdul[2].data, dtype=np.float64)
        mask = None
        if apply_mask and len(hdul) > 3 and hdul[3].data is not None:
            mask = np.asarray(hdul[3].data)
        if flux.ndim == 2:                       # multi-visit: row 0 = combined coadd
            flux, err = flux[0], err[0]
            if mask is not None and mask.ndim == 2:
                mask = mask[0]
    if flux.size != NPIX or err.size != NPIX:
        return None
    ivar = np.zeros(NPIX, dtype=np.float64)
    good = np.isfinite(err) & (err > 0) & (err < APOGEE_ERR_BADVAL) & np.isfinite(flux)
    ivar[good] = 1.0 / (err[good] ** 2)
    if mask is not None and mask.size == NPIX:
        ivar[(mask.astype(np.int64) & APOGEE_BADPIX) != 0] = 0.0
    return flux, ivar


# --------------------------------------------------------------------------- #
# Tables / JSON
# --------------------------------------------------------------------------- #
def csv_to_parquet(csv_path, out_parquet):
    """Convert a CSV catalog/manifest to Parquet. Returns row count."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df.to_parquet(out_parquet, index=False)
    return len(df)


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# CLI: convert the existing derived data in place (Parquet alongside the files).
# --------------------------------------------------------------------------- #
def _convert_all():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Store the shared grid once.
    write_json({"npix": NPIX, "wl_min": float(WAVELENGTH[0]),
                "wl_max": float(WAVELENGTH[-1]), "grid": "log-linear",
                "crval1": 4.179, "cdelt1": 6e-6},
               os.path.join(root, "data", "apogee_grid.json"))
    # Spectra dirs -> Parquet.
    for d, out in [("dr19_raw", "dr19_raw.parquet"),
                   ("dr19_raw_sb2", "dr19_raw_sb2.parquet"),
                   ("dr19_raw_controls", "dr19_raw_controls.parquet")]:
        src = os.path.join(root, "data", d)
        if os.path.isdir(src):
            n = spectra_dir_to_parquet(src, os.path.join(root, "data", out))
            print(f"  {d}: {n} spectra -> data/{out}", flush=True)
    # Manifests -> Parquet.
    res = os.path.join(root, "resources")
    for fn in os.listdir(res):
        if fn.endswith(".csv"):
            n = csv_to_parquet(os.path.join(res, fn),
                               os.path.join(res, fn[:-4] + ".parquet"))
            print(f"  {fn}: {n} rows -> resources/{fn[:-4]}.parquet", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        _convert_all()
    else:
        print("usage: python src/data_io.py convert")
