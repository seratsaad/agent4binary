"""Build an H-band continuum-pixel mask for the APOGEE 8575-pixel grid.

This script constructs a continuum-pixel mask from first principles by combining
two independent signals.

(1) A line-list signal from the Kurucz gfallvac line list. For each pixel we sum
the oscillator strengths (10**log_gf) of nearby lines. Pixels with little nearby
line opacity are line-free continuum candidates.

(2) An empirical signal from a sample of DR19 dwarf spectra. We normalize each
spectrum with a per-chip running-percentile continuum, then compute the
cross-star median and a robust scatter at each pixel. Pixels whose median sits at
1 and whose cross-star scatter is small are flat across the dwarf sample, which is
the empirical signature of a continuum pixel; pixels with high scatter track
stellar labels and carry line information.

A pixel is labelled continuum when it is in the low tail of both signals.

The script reads inputs and writes three outputs. It does not import or edit any
existing project module. The mask is not wired into continuum normalization here;
that is a separate later step.

Outputs:
    models/continuum_pixels.npz       the mask and its diagnostic arrays
    notebooks/continuum_pixels_diag.png   a diagnostic figure

Inputs:
    ~/pykurucz/lines/gfallvac.latest          Kurucz gfall line list (nm, log gf)
    data/dr19_raw/*.npz                        DR19 raw dwarf spectra
    data/binspec_cannon_cont_pixels_apogee.npz Ness/Cannon mask for cross-check
    data/binspec_apogee_wavelength.npz         binspec 7214-pixel wavelength grid
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Paths. The project root is the parent of this src/ directory. The line list
# lives under the user home in ~/pykurucz.
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
LINE_LIST = Path.home() / "pykurucz" / "lines" / "gfallvac.latest"
DR19_GLOB = str(ROOT / "data" / "dr19_raw" / "*.npz")
CANNON_NPZ = ROOT / "data" / "binspec_cannon_cont_pixels_apogee.npz"
BINSPEC_WL_NPZ = ROOT / "data" / "binspec_apogee_wavelength.npz"
OUT_MASK = ROOT / "models" / "continuum_pixels.npz"
OUT_PLOT = ROOT / "notebooks" / "continuum_pixels_diag.png"

# ----------------------------------------------------------------------------
# Grid definition. The APOGEE combined spectrum is sampled on a log-linear grid
# of 8575 pixels. Wavelengths are in Angstrom.
# ----------------------------------------------------------------------------
NPIX = 8575
PIX = np.arange(NPIX)
WL_A = 10.0 ** (4.179 + 6e-6 * PIX)  # 15100.8 .. 16999.8 Angstrom
DLOG = 6e-6  # log10 step per pixel; constant across the grid

# Three detector chips. The gaps near pixels 3430 and 6212, plus the trimmed
# blue and red edges, are not part of any chip. The boundaries below come from
# the empirical ivar<=0 regions measured across the DR19 sample (see the chip-gap
# scan in the project notes): edge [0,159], gap1 [3318,3503], gap2 [6123,6264],
# edge [8376,8574]. The interior detector ranges are therefore:
CHIPS = {
    "blue": (160, 3317),
    "green": (3504, 6122),
    "red": (6265, 8375),
}
# A pixel is "on a chip" when it falls inside one of the three ranges. Pixels in
# the gaps and edges are never continuum candidates.
ON_CHIP = np.zeros(NPIX, dtype=bool)
for _a, _b in CHIPS.values():
    ON_CHIP[_a : _b + 1] = True

# ----------------------------------------------------------------------------
# Tunable parameters.
# ----------------------------------------------------------------------------
N_DR19 = 400  # number of DR19 dwarf spectra to draw
CONT_WIN = 151  # window (pixels) for the inline running-percentile continuum
CONT_PCTL = 90.0  # percentile used as the per-window continuum estimate
# APOGEE resolution R ~ 22500. One resolution element spans dln(wl) ~ 1/R, which
# at DLOG = 6e-6 per pixel is (1/R)/ln(10)/DLOG ~ 1.5 pixels. We spread each
# line's oscillator strength over a Gaussian of this width so a line contaminates
# the pixel it lands on plus its immediate neighbours.
RES_SIGMA_PIX = (1.0 / 22500.0) / np.log(10.0) / DLOG  # ~1.5 pixels
# Percentile thresholds for the two signals. A pixel is a continuum candidate
# when its line strength is below LS_PCTL and its cross-star scatter is below
# SCAT_PCTL, with both percentiles taken over on-chip pixels only.
LS_PCTL = 25.0
SCAT_PCTL = 25.0
# A continuum pixel must also sit at the upper flux envelope, since a flat but
# low pixel is a saturated line core, not continuum. The simple running-
# percentile normalization places the true continuum slightly below 1 (the high
# percentile rides just above the envelope), so we do not test against 1
# directly. Instead we require the cross-star median to be at or above a high
# percentile of the on-chip median-flux distribution, which is the empirical
# continuum level for this sample.
MEDIAN_LEVEL_PCTL = 60.0

RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# (1) Line-list signal.
# ----------------------------------------------------------------------------
def build_line_strength() -> np.ndarray:
    """Return per-pixel integrated line strength on the 8575-pixel grid.

    The gfallvac file is fixed-width and sorted by wavelength in nanometres. We
    stream it, keep only lines in the H band (1510-1700 nm), and stop once we
    pass the upper edge. Each line's gf = 10**log_gf is deposited onto the grid
    with a Gaussian of width RES_SIGMA_PIX centred on the line's pixel, so the
    result is the local sum of gf within roughly one resolution element of each
    pixel.
    """
    wl_min_nm = 1510.0
    wl_max_nm = 1700.0

    line_pix = []  # fractional pixel coordinate of each line
    line_gf = []  # oscillator strength 10**log_gf of each line

    # Convert a wavelength in nm to a fractional pixel index on the log grid.
    # pixel = (log10(wl_A) - 4.179) / DLOG, with wl_A = 10 * wl_nm.
    def nm_to_pix(wl_nm: float) -> float:
        return (np.log10(wl_nm * 10.0) - 4.179) / DLOG

    with open(LINE_LIST, "r", encoding="ascii", errors="ignore") as fh:
        for line in fh:
            if len(line) < 18:
                continue
            # Column 1 (chars 0..10) is wavelength in nm; column 2 (11..17) is
            # log gf. These are the only fields the line-strength proxy needs.
            try:
                wl_nm = float(line[0:11])
            except ValueError:
                continue
            if wl_nm < wl_min_nm:
                continue  # not in band yet; keep streaming
            if wl_nm > wl_max_nm:
                break  # sorted file: past the band, nothing more to read
            try:
                log_gf = float(line[11:18])
            except ValueError:
                continue
            line_pix.append(nm_to_pix(wl_nm))
            line_gf.append(10.0 ** log_gf)

    line_pix = np.asarray(line_pix)
    line_gf = np.asarray(line_gf)

    # Deposit each line onto the grid with a Gaussian kernel. We only touch
    # pixels within +/- 4 sigma of the line centre, which keeps the loop cheap.
    strength = np.zeros(NPIX)
    half = int(np.ceil(4.0 * RES_SIGMA_PIX))
    inv2s2 = 1.0 / (2.0 * RES_SIGMA_PIX * RES_SIGMA_PIX)
    for p0, gf in zip(line_pix, line_gf):
        c = int(round(p0))
        lo = max(0, c - half)
        hi = min(NPIX - 1, c + half)
        if hi < lo:
            continue
        idx = np.arange(lo, hi + 1)
        w = np.exp(-((idx - p0) ** 2) * inv2s2)
        strength[idx] += gf * w

    return strength


# ----------------------------------------------------------------------------
# (2) Empirical low-variance signal.
# ----------------------------------------------------------------------------
def running_percentile_continuum(flux: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Return a smooth per-chip continuum estimate for one spectrum.

    For each chip we slide a window of CONT_WIN pixels and take the CONT_PCTL
    percentile of the good (ivar>0) flux inside it as the local continuum. The
    high percentile rides the upper envelope of the flux, which is the continuum
    for an absorption spectrum. The estimate is only used to remove the smooth
    pseudo-continuum shape so that cross-star scatter is comparable pixel to
    pixel; it does not need to be a careful normalization.
    """
    cont = np.ones(NPIX)
    half = CONT_WIN // 2
    for a, b in CHIPS.values():
        for i in range(a, b + 1):
            lo = max(a, i - half)
            hi = min(b, i + half)
            seg = flux[lo : hi + 1]
            seg_good = good[lo : hi + 1]
            vals = seg[seg_good]
            if vals.size >= 5:
                cont[i] = np.percentile(vals, CONT_PCTL)
            elif vals.size > 0:
                cont[i] = np.median(vals)
            # else leave at 1.0
    # Guard against zeros or negatives in the divisor.
    cont[cont <= 0] = 1.0
    return cont


def build_empirical_signals() -> tuple[np.ndarray, np.ndarray, int]:
    """Return (median_flux, cross_star_scatter, n_used) over the DR19 sample.

    Each spectrum is normalized by its running-percentile continuum, masking
    ivar<=0 pixels. We then take the cross-star median and a robust scatter (the
    median absolute deviation scaled to a Gaussian sigma) at each pixel, ignoring
    masked entries.
    """
    files = sorted(glob.glob(DR19_GLOB))
    if len(files) == 0:
        raise FileNotFoundError(f"no DR19 spectra found under {DR19_GLOB}")
    n_pick = min(N_DR19, len(files))
    pick = RNG.choice(len(files), size=n_pick, replace=False)

    # Accumulate normalized flux into a (n_used, NPIX) array with NaN for masked
    # pixels, so cross-star statistics can ignore them per pixel.
    norm = np.full((n_pick, NPIX), np.nan)
    n_used = 0
    for k, j in enumerate(pick):
        d = np.load(files[j])
        flux = np.asarray(d["flux_raw"], dtype=float)
        ivar = np.asarray(d["ivar"], dtype=float)
        good = ivar > 0
        if good.sum() < NPIX // 2:
            continue  # too much masked; skip this star
        cont = running_percentile_continuum(flux, good)
        nf = flux / cont
        nf[~good] = np.nan
        # Reject absurd values from the division.
        nf[(nf < 0.0) | (nf > 2.0)] = np.nan
        norm[n_used] = nf
        n_used += 1
    norm = norm[:n_used]

    # Per-pixel cross-star statistics, ignoring NaN. Gap and edge pixels are NaN
    # in every star and yield an all-NaN slice; that is expected, so silence the
    # warning and handle those pixels below.
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_flux = np.nanmedian(norm, axis=0)
            mad = np.nanmedian(np.abs(norm - median_flux[None, :]), axis=0)
    scatter = 1.4826 * mad  # MAD scaled to a Gaussian-equivalent sigma

    # Pixels that were masked in essentially every star have NaN statistics.
    # Mark their scatter as large so they can never be continuum candidates.
    bad = ~np.isfinite(median_flux) | ~np.isfinite(scatter)
    median_flux[bad] = 0.0
    scatter[bad] = np.inf
    return median_flux, scatter, n_used


# ----------------------------------------------------------------------------
# Cross-check against the Ness/Cannon mask.
# ----------------------------------------------------------------------------
def cannon_mask_on_grid() -> np.ndarray:
    """Lift the binspec 7214-pixel Cannon mask onto the 8575-pixel grid.

    The binspec wavelength grid is the subset of our grid with the chip gaps and
    edges trimmed; each binspec pixel maps to a unique nearest pixel of our grid
    (residuals ~1e-8 Angstrom). We place each Cannon continuum pixel at its
    nearest grid pixel and return a bool array of length 8575.
    """
    cannon = np.load(CANNON_NPZ)["pixels_cannon"].astype(bool)
    bwl = np.load(BINSPEC_WL_NPZ)["wavelength"]
    # Nearest grid pixel for each binspec wavelength.
    idx = np.searchsorted(WL_A, bwl)
    idx = np.clip(idx, 1, NPIX - 1)
    left = WL_A[idx - 1]
    right = WL_A[idx]
    nearest = np.where(np.abs(bwl - left) < np.abs(bwl - right), idx - 1, idx)
    out = np.zeros(NPIX, dtype=bool)
    out[nearest] = cannon
    # Also return which grid pixels are covered by the binspec grid at all, so
    # the overlap is computed only where the Cannon mask is defined.
    covered = np.zeros(NPIX, dtype=bool)
    covered[nearest] = True
    return out, covered


# ----------------------------------------------------------------------------
# Diagnostic figure.
# ----------------------------------------------------------------------------
def make_plot(line_strength, scatter, median_flux, mask):
    """Render line strength, cross-star scatter, and the mask over a window."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Window 15700-15850 Angstrom (a busy region inside the blue chip).
    sel = (WL_A >= 15700.0) & (WL_A <= 15850.0)
    w = WL_A[sel]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    ax.semilogy(w, line_strength[sel] + 1e-30, color="C0", lw=0.8)
    ax.set_ylabel("line strength\n(sum gf)")
    ax.set_title("Continuum-pixel diagnostics, 15700-15850 Angstrom")

    ax = axes[1]
    ax.plot(w, scatter[sel], color="C1", lw=0.8, label="cross-star scatter")
    ax.plot(w, median_flux[sel], color="C2", lw=0.8, alpha=0.7, label="median flux")
    ax.set_ylabel("scatter / median")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    # Show the mask as vertical bands and the median flux for context.
    ax.plot(w, median_flux[sel], color="0.6", lw=0.8)
    msel = mask[sel]
    ax.fill_between(w, 0, 1, where=msel, transform=ax.get_xaxis_transform(),
                    color="C3", alpha=0.3, step="mid", label="continuum pixel")
    ax.set_ylabel("median flux")
    ax.set_xlabel("wavelength (Angstrom)")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------
def main():
    print("(1) building line-strength signal from gfallvac ...")
    line_strength = build_line_strength()

    print("(2) building empirical low-variance signal from DR19 dwarfs ...")
    median_flux, scatter, n_used = build_empirical_signals()
    print(f"    used {n_used} DR19 spectra")

    # Thresholds are percentiles taken over on-chip pixels only, so the chip gaps
    # and edges do not skew them.
    ls_on = line_strength[ON_CHIP]
    sc_on = scatter[ON_CHIP & np.isfinite(scatter)]
    ls_thresh = np.percentile(ls_on, LS_PCTL)
    sc_thresh = np.percentile(sc_on, SCAT_PCTL)
    # Continuum level: a high percentile of the on-chip median-flux distribution.
    # Pixels at or above it sit on the empirical upper flux envelope.
    med_on = median_flux[ON_CHIP & np.isfinite(median_flux)]
    med_thresh = np.percentile(med_on, MEDIAN_LEVEL_PCTL)

    # Final mask: on a chip, low line strength, low scatter, at the upper flux
    # envelope.
    low_line = line_strength <= ls_thresh
    low_scatter = scatter <= sc_thresh
    flat_median = median_flux >= med_thresh
    mask = ON_CHIP & low_line & low_scatter & flat_median

    # Per-chip counts and coverage.
    n_cont = int(mask.sum())
    per_chip = {name: int(mask[a : b + 1].sum()) for name, (a, b) in CHIPS.items()}
    coverage = n_cont / NPIX

    # Cross-check against the Cannon mask, restricted to where it is defined.
    cannon_grid, cannon_covered = cannon_mask_on_grid()
    # Overlap: of the Cannon continuum pixels (within the covered grid), what
    # fraction are also flagged continuum here.
    cannon_cont = cannon_grid & cannon_covered
    inter = mask & cannon_cont
    overlap_of_cannon = inter.sum() / max(1, cannon_cont.sum())
    # And of our continuum pixels that lie within the covered grid, what fraction
    # the Cannon mask also flags.
    ours_in_cov = mask & cannon_covered
    overlap_of_ours = inter.sum() / max(1, ours_in_cov.sum())

    # Save the mask and all diagnostic arrays.
    OUT_MASK.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_MASK,
        pixels=mask,
        line_strength=line_strength,
        cross_star_scatter=scatter,
        median_flux=median_flux,
        wl=WL_A,
        ls_threshold=np.float64(ls_thresh),
        scatter_threshold=np.float64(sc_thresh),
        median_threshold=np.float64(med_thresh),
        ls_percentile=np.float64(LS_PCTL),
        scatter_percentile=np.float64(SCAT_PCTL),
        median_level_percentile=np.float64(MEDIAN_LEVEL_PCTL),
        n_dr19_used=np.int64(n_used),
    )

    print("(3) rendering diagnostic figure ...")
    make_plot(line_strength, scatter, median_flux, mask)

    # Report.
    print("")
    print("=== continuum-pixel mask report ===")
    print(f"thresholds: line_strength <= {ls_thresh:.4g} (P{LS_PCTL:.0f} on-chip), "
          f"scatter <= {sc_thresh:.4g} (P{SCAT_PCTL:.0f} on-chip), "
          f"median >= {med_thresh:.4f} (P{MEDIAN_LEVEL_PCTL:.0f} on-chip)")
    print(f"continuum pixels: {n_cont} of {NPIX} "
          f"(coverage {coverage:.3%} of full grid)")
    on_chip_total = int(ON_CHIP.sum())
    print(f"  on-chip coverage: {n_cont / on_chip_total:.3%} "
          f"of {on_chip_total} on-chip pixels")
    for name, (a, b) in CHIPS.items():
        nchip = b - a + 1
        print(f"  {name:5s}: {per_chip[name]:4d} continuum pixels "
              f"of {nchip} ({per_chip[name] / nchip:.2%})")
    print(f"overlap with Cannon mask:")
    print(f"  {overlap_of_cannon:.2%} of Cannon continuum pixels are continuum here")
    print(f"  {overlap_of_ours:.2%} of our (covered) continuum pixels are Cannon too")
    print(f"saved mask: {OUT_MASK}")
    print(f"saved plot: {OUT_PLOT}")


if __name__ == "__main__":
    main()
