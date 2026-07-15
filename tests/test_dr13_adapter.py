#!/usr/bin/env python3
"""Unit test for the DR13 apStar adapter (src/data_io.load_dr13_apstar).

Self-contained: builds a synthetic apStar-format FITS in memory (no network, no
torch, no real spectra) and checks the adapter's format handling -- combined-row
selection, sigma->ivar conversion, bad-pixel handling, optional bitmask. Run:

    python tests/test_dr13_adapter.py
"""
import os
import sys
from io import BytesIO

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import data_io

NPIX = data_io.NPIX


def _make_apstar_bytes(n_rows=3, with_mask=True):
    """Build a minimal multi-row apStar FITS: HDU0 primary, HDU1 flux, HDU2 err,
    HDU3 mask. Row 0 of flux/err is the 'combined' coadd the adapter must pick."""
    rng = np.random.default_rng(0)
    flux = rng.normal(1.0, 0.02, size=(n_rows, NPIX)).astype(np.float32)
    err = np.full((n_rows, NPIX), 0.02, dtype=np.float32)
    # Encode a chip gap in the combined row the apStar way: huge error.
    err[0, 100:110] = 1e10
    # And a couple of genuinely bad (<=0) error pixels.
    err[0, 200] = 0.0
    err[0, 201] = -1.0
    mask = np.zeros((n_rows, NPIX), dtype=np.int32)
    mask[0, 300:305] = 1 << 0   # BADPIX bit on the combined row
    mask[0, 305] = 1 << 9       # PERSIST_HIGH (NOT in APOGEE_BADPIX) -> must survive

    hdr = fits.Header()
    hdr["CRVAL1"] = 4.179
    hdr["CDELT1"] = 6e-6
    hdr["NWAVE"] = NPIX
    hdus = [fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(flux, name="FLUX"),
            fits.ImageHDU(err, name="ERROR")]
    if with_mask:
        hdus.append(fits.ImageHDU(mask, name="MASK"))
    buf = BytesIO()
    fits.HDUList(hdus).writeto(buf)
    return buf.getvalue(), flux, err


def main():
    fails = 0

    def check(cond, msg):
        nonlocal fails
        print(("PASS" if cond else "FAIL") + ": " + msg)
        fails += 0 if cond else 1

    # --- multi-row: combined row 0 selected, sigma->ivar, bad pixels -> ivar 0 ---
    raw, flux, err = _make_apstar_bytes(n_rows=3, with_mask=True)
    out = data_io.load_dr13_apstar(raw)
    check(out is not None, "adapter returns a result on valid apStar bytes")
    f, ivar = out
    check(f.shape == (NPIX,) and ivar.shape == (NPIX,), "flux/ivar length == 8575")
    check(np.allclose(f, flux[0]), "combined row 0 is selected (not a visit row)")
    check(np.all(ivar[100:110] == 0.0), "huge-error chip-gap pixels -> ivar 0")
    check(ivar[200] == 0.0 and ivar[201] == 0.0, "err<=0 pixels -> ivar 0")
    g = 5000  # a clean pixel
    check(abs(ivar[g] - 1.0 / err[0, g] ** 2) < 1e-3, "ivar == 1/sigma**2 on good pixels")

    # --- mask handling is opt-in and bit-selective ---
    check(np.all(ivar[300:305] > 0), "without apply_mask, BADPIX pixels keep ivar")
    f2, ivar2 = data_io.load_dr13_apstar(raw, apply_mask=True)
    check(np.all(ivar2[300:305] == 0.0), "apply_mask zeros BADPIX pixels")
    check(ivar2[305] > 0, "apply_mask leaves non-bad bits (PERSIST_HIGH) untouched")

    # --- single-row (1D) apStar handled ---
    rng = np.random.default_rng(1)
    flux1 = rng.normal(1.0, 0.02, size=NPIX).astype(np.float32)
    err1 = np.full(NPIX, 0.05, dtype=np.float32)
    buf = BytesIO()
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(flux1, name="FLUX"),
                  fits.ImageHDU(err1, name="ERROR")]).writeto(buf)
    out1 = data_io.load_dr13_apstar(buf.getvalue())
    check(out1 is not None and out1[0].shape == (NPIX,), "single-visit 1D apStar handled")

    # --- wrong-length / malformed -> None, not a crash ---
    buf = BytesIO()
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.ones(10, dtype=np.float32)),
                  fits.ImageHDU(np.ones(10, dtype=np.float32))]).writeto(buf)
    check(data_io.load_dr13_apstar(buf.getvalue()) is None, "wrong-length apStar -> None")

    print("\n%d FAIL" % fails if fails else "\nALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
