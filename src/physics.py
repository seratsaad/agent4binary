#!/usr/bin/env python3
"""
Shared science module for binary-star spectral disentangling.

Centralizes the El-Badry et al. (2018b) forward model and chi-squared logic so the
MCP servers stay thin wrappers. IMPORT-ONLY: it defines functions and loads the
trained nets plus the isochrone grid at import, with no CLI / MCP surface.

Method ported from El-Badry et al. (2018a/b); reference code
github.com/kareemelbadry/binspec.

DETECTORS (three entry points, same compact result dict, drop-in comparable):
  - dr19_sc_single_vs_binary (section 7c): the PRODUCTION path. RAW flux+ivar in,
    normalized by continuum_normalize (per-chip sigma-clipped Chebyshev from the
    raw flux itself), single-star model = OUR SC-trained 5-label Payne, binary
    composite uses OUR OWN Teff-keyed un-normalization continuum. No survey
    continuum and no binspec net anywhere.
  - dr19_single_vs_binary (section 7b): same OUR 5-label net, but trained on the
    survey-normalized flux; kept as a comparison.
  - binspec_single_vs_binary (section 8): the ORACLE. El-Badry's VENDORED trained
    nets (vendor.binspec) on binspec's own 7214-pixel grid; head-to-head reference.

The 3-label Payne (payne_predict) and its compose_binary / fit_single / fit_binary
path are KEPT for the teaching / notebook section, not for detection.

Pieces reproduced here:
  1. payne_predict / payne_predict5 / payne_predict5_sc : single-star NORMALIZED
     rest-frame spectrum from labels. Forward pass is PURE NUMPY (weights exported
     from the checkpoint once at import); torch reads the checkpoint only, so the
     fit loop is torch-free.
  2. continuum_normalize / pseudo_continuum_sc : OUR self-consistent normalization
     and un-normalization (section 3b / Stage 4); no Cannon mask, no survey column.
  3. pseudo_continuum : the binspec-PORTED un-normalization continuum (trained flux
     net + Cannon-pixel Chebyshev), used by the survey-normalized and 3-label paths.
  4. secondary_from_q : the isochrone q -> (Teff2, logg2, R2, R1) map; OUR OWN MIST
     v1.2 interpolation (isochrone_mist), not a binspec net.
  5. compose_binary* : El-Badry Eq. 2, R1^2 f1 + R2^2 f2 summed in flux then
     re-normalized.
  6. chi2 : inverse-variance chi^2 over good pixels.
  7. f_imp : the improvement-fraction statistic, El-Badry et al. 2018b Eq. B1 (the
     binspec repo does not ship this).

Binspec-PORTED continuum (pseudo_continuum, section 3): binspec un-normalizes the
normalized spectrum by a synthetic continuum before summing components in flux
(Eq. 2). It runs a trained per-pixel flux net (NN_unnormalized_spectra) on
[Teff, logg, feh, alpha], then fits a 4th-order per-chip Chebyshev through Melissa
Ness' Cannon continuum pixels (utils.get_apogee_continuum). We load the flux-net
weights (models/binspec_NN_unnormalized_spectra.npz) and the Cannon mask
(data/binspec_cannon_cont_pixels_apogee.npz), reproduce both on binspec's 7214-pixel
grid, then interpolate the smooth continuum onto our 8575-pixel grid. The PRODUCTION
SC path does not use this; it uses pseudo_continuum_sc instead.

Simplifications in the binspec-ported continuum (it feeds the non-production paths):
  - the flux net has alpha (=[Mg/Fe]) as a 4th label; we run it at alpha=0, which
    shifts only the line-blanketing the Chebyshev continuum averages out,
  - the flux net runs on the 7214-pixel sub-grid; the smooth Chebyshev continuum is
    interpolated to 8575 pixels.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from scipy.ndimage import percentile_filter, uniform_filter1d

# --------------------------------------------------------------------------- #
# Paths. This file lives in src/, so the project root is one level up.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
MODEL_PATH = os.path.join(_ROOT, "models", "payne.pt")

# --------------------------------------------------------------------------- #
# OUR OWN MIST isochrone (the PRODUCTION DR19 detection path uses this, NOT any
# binspec neural net). isochrone_mist replaces the binspec NN_Teff2_logg2 +
# NN_radius (q -> secondary labels and stellar radii) with a direct interpolation
# of genuine MIST v1.2 isochrones we downloaded. The module lives in src/ next to
# this file, so we ensure src/ is importable before importing it (the MCP servers
# also add src/ to sys.path). The SC un-normalization continuum is data-derived
# (pseudo_continuum_sc, from models/dr19_sc_continuum.npz), not a separate module.
# --------------------------------------------------------------------------- #
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import isochrone_mist as _iso_ours        # noqa: E402  OUR MIST q->secondary + radii

# --------------------------------------------------------------------------- #
# Ported binspec isochrone neural networks. Vendored (copied from binspec) so
# import needs no clone; the production q -> secondary map uses OUR MIST
# interpolation instead, but these stay for the isochrone MCP server and for
# agreement checks. The numpy forward passes below reproduce
# spectral_model.get_Teff2_logg2_NN / get_radius_NN exactly.
#
#   binspec_NN_Teff2_logg2.npz : El-Badry et al. (2018b) net mapping
#     (Teff1, logg1, [Fe/H], q) -> (Teff2, logg2). Trained on MIST main-sequence
#     pairs that share age and composition, so it encodes the equal-age,
#     equal-[Fe/H] binary constraint. One hidden layer of 20 sigmoid units;
#     predicts Teff2/1000.
#   binspec_NN_radius.npz : (Teff, logg, [Fe/H]) -> R (Rsun) for a single MS star,
#     MIST-trained, two hidden layers of 100 sigmoid units. Used for R1 and R2.
# --------------------------------------------------------------------------- #
ISO_TEFF2_NN_PATH = os.path.join(_ROOT, "models", "binspec_NN_Teff2_logg2.npz")
ISO_RADIUS_NN_PATH = os.path.join(_ROOT, "models", "binspec_NN_radius.npz")

# binspec ships no mass network, so primary_mass recovers m1 from (Teff1, logg1,
# [Fe/H]) against the raw MIST single-star table binspec distributes (genuine MIST
# v1.2 output). Used for reporting and the q floor, not the spectral forward model.
MIST_TABLE_PATH = os.path.join(_ROOT, "data", "binspec_MIST_single_stars.npz")

# Ported binspec flux net + grid + Cannon mask for pseudo_continuum (section 3).
FLUX_NN_PATH = os.path.join(_ROOT, "models", "binspec_NN_unnormalized_spectra.npz")
BINSPEC_WL_PATH = os.path.join(_ROOT, "data", "binspec_apogee_wavelength.npz")
CONT_PIX_PATH = os.path.join(_ROOT, "data", "binspec_cannon_cont_pixels_apogee.npz")

# Representative main-sequence age (Gyr) for the isochrone q -> secondary map.
# OUR MIST relation needs one age for the APOGEE dwarf sample; the forward model
# passes it through the whole chain so the age is set in one place. The 4.0 Gyr
# choice is justified on secondary_from_q.
_MS_REPR_AGE_GYR = 4.0

# Number of pixels on the APOGEE combined (aspcapStar) grid.
NPIX = 8575

# Speed of light, km/s. Used for Doppler shifts on the log-lambda grid.
C_KMS = 299792.458

# The APOGEE log-linear wavelength grid (Angstrom). The aspcapStar combined
# spectrum spans 15100.8 - 16999.8 A on a log10-linear grid of 8575 pixels.
# We rebuild it analytically so physics.py does not depend on a cached FITS
# header; the endpoints below match the cached wl arrays to ~1e-3 A.
_WL_MIN = 15100.801541641493
_WL_MAX = 16999.807358923248
# log10-linear: equal spacing in log10(lambda). linspace in log space, then 10**.
WAVELENGTH = 10.0 ** np.linspace(np.log10(_WL_MIN), np.log10(_WL_MAX), NPIX)


# --------------------------------------------------------------------------- #
# q-bias fix: per-component H-band surface-brightness level vs Teff.
#
# pseudo_continuum_sc (and the dr19_sc_continuum table behind it) returns a
# MEAN-1 SED SHAPE at EVERY Teff: each real continuum was divided by its own
# mean when the table was built, so the absolute surface-brightness LEVEL vs
# Teff was divided out. The binary composite R1^2 f1 cont1 + R2^2 f2 cont2 then
# gives the cool secondary the SAME continuum mean (1.0) as the hot primary, so
# the secondary is over-bright at low q and the fit recovers q biased LOW. The
# closed q-calibration scratch cache was purged; the MIST-H fix remains here.
#
# binspec keeps this level: its flux net predicts un-normalized 1e6*f_lambda, so
# surfflux2/surfflux1 carries the temperature dilution. We restore it by
# weighting each component's mean-1 continuum by its H-band Planck surface
# brightness B(Teff), normalized to the PRIMARY -- B(Teff2)/B(Teff1). The ratio
# is 1 at q=1 (Teff2==Teff1 -> B2/B1=1), so the q=1 single-star identity stays
# EXACT and the detection logic is untouched. The Planck H-band ratio matches
# the binspec flux-net surfflux ratio to ~6-15% over 3400-5500 K (verified in
# the diagnosis): B2/B1 ~ 0.30/0.38/0.55/0.84 vs binspec 0.32/0.41/0.64/0.89 at
# q = 0.3/0.5/0.7/0.9.
_PLANCK_H = 6.62607015e-34     # J s
_PLANCK_C = 2.99792458e8       # m / s
_PLANCK_KB = 1.380649e-23      # J / K
# WAVELENGTH (Angstrom) -> metres, evaluated once. The H-band Planck mean is a
# smooth function of Teff; the per-pixel structure cancels in the ratio, but we
# average over the full grid so the band weighting matches the composite's band.
_WAVELENGTH_M = WAVELENGTH * 1e-10


def _band_surface_brightness(teff):
    """Mean Planck H-band surface brightness B_lambda(Teff) (arb. units).

    The absolute scale is irrelevant -- only the RATIO B(Teff2)/B(Teff1) enters
    the binary composite, and the constant prefactor 2 h c^2 cancels there, so we
    drop it. teff is floored at 2000 K (below the isochrone MS range) to keep the
    exponent finite. Returns the band-mean of 1 / (lambda^5 (exp(x) - 1)) with
    x = h c / (lambda kB Teff).
    """
    lam = _WAVELENGTH_M
    x = _PLANCK_H * _PLANCK_C / (lam * _PLANCK_KB * max(float(teff), 2000.0))
    return float(np.mean(1.0 / (lam ** 5 * (np.exp(x) - 1.0))))


# --------------------------------------------------------------------------- #
# 1. Payne single-star emulator.
#    Architecture and label normalization are reused exactly from payne.pt so
#    predictions are identical to payne_server.predict_spectrum.
# --------------------------------------------------------------------------- #
class _Payne(nn.Module):
    """The trained single-star emulator: 3 labels -> 8575 normalized pixels.

    Architecture must match the one saved in payne.pt:
    Linear(n_label, n_hidden) -> LeakyReLU -> Linear(n_hidden, n_hidden)
    -> LeakyReLU -> Linear(n_hidden, n_pix).
    """

    def __init__(self, n_label=3, n_hidden=300, n_pix=NPIX):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_label, n_hidden), nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(),
            nn.Linear(n_hidden, n_pix),
        )

    def forward(self, x):
        return self.net(x)


# Load the checkpoint once at import. CPU is sufficient. REQUIREMENT 1: torch is
# used here ONLY to read the checkpoint and (if training) to instantiate the net.
# We immediately EXPORT every weight/bias to a numpy array so the per-spectrum fit
# never touches torch. The torch module/object is kept available for a training
# path but is not called in payne_predict.
_CKPT = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
_NET = _Payne(_CKPT["n_label"], _CKPT["n_hidden"], _CKPT["n_pix"])
_NET.load_state_dict(_CKPT["state_dict"])
_NET.eval()  # the torch path, kept only for training / reference.

# Label normalization bounds saved alongside the weights. The Payne was trained
# on labels scaled as (label - min)/(max - min) - 0.5, the same convention as
# binspec (get_spectrum_from_neural_net). We must apply the identical scaling.
_LMIN = np.asarray(_CKPT["label_min"], dtype=np.float32)
_LMAX = np.asarray(_CKPT["label_max"], dtype=np.float32)

# --- Export the Payne weights to numpy ONCE (REQUIREMENT 1) ---------------- #
# The architecture is Linear -> LeakyReLU -> Linear -> LeakyReLU -> Linear. We
# pull each Linear layer's weight/bias from the loaded state_dict and store them
# as float64 numpy arrays. Torch stores Linear weight as (out, in) and computes
# x @ W.T + b; we transpose to (in, out) here so the numpy forward is plain
# x @ W + b. This is done at import, not per call, so the fit loop is torch-free.
_state = _NET.state_dict()
# nn.Sequential names the Linear layers net.0, net.2, net.4 (the LeakyReLU at
# net.1 / net.3 has no parameters).
_W0 = _state["net.0.weight"].numpy().astype(np.float64).T   # (n_label, n_hidden)
_B0 = _state["net.0.bias"].numpy().astype(np.float64)       # (n_hidden,)
_W1 = _state["net.2.weight"].numpy().astype(np.float64).T   # (n_hidden, n_hidden)
_B1 = _state["net.2.bias"].numpy().astype(np.float64)       # (n_hidden,)
_W2 = _state["net.4.weight"].numpy().astype(np.float64).T   # (n_hidden, n_pix)
_B2 = _state["net.4.bias"].numpy().astype(np.float64)       # (n_pix,)
# LeakyReLU default negative slope in torch is 0.01; _Payne uses the default, so
# the numpy activation must use the SAME slope to match the torch output exactly.
_LRELU_SLOPE = 0.01


def _scale_labels(labels):
    """Scale raw labels into the network's input range, exactly as in training."""
    return (labels - _LMIN) / (_LMAX - _LMIN) - 0.5


def _leaky_relu(z):
    """LeakyReLU(z) = z for z>=0, slope*z for z<0; matches torch.nn.LeakyReLU."""
    return np.where(z >= 0.0, z, _LRELU_SLOPE * z)


def _payne_forward_numpy(scaled):
    """Pure-numpy Payne forward pass on a SCALED 1-D label vector (length 3).

    Reproduces _Payne.forward exactly: two LeakyReLU hidden layers and a linear
    output, using the weights exported from payne.pt at import. Returns the raw
    8575-pixel normalized spectrum. This is the function the scipy fit calls; it
    contains no torch, so it runs at numpy/BLAS speed inside least_squares.
    """
    h0 = _leaky_relu(scaled @ _W0 + _B0)   # first hidden layer
    h1 = _leaky_relu(h0 @ _W1 + _B1)       # second hidden layer
    return h1 @ _W2 + _B2                  # linear output, length n_pix


def payne_predict(teff, logg, feh):
    """Predict the rest-frame NORMALIZED APOGEE spectrum of a single star.

    Runs the trained Payne MLP (in PURE NUMPY) on (Teff, logg, [Fe/H]). The output
    is the continuum-normalized flux (~1 in the continuum, dips in lines), in the
    rest frame, on the 8575-pixel APOGEE grid. Broadening and RV are applied
    separately (RV via Doppler shift inside compose_binary).

    Parameters are scalars; returns a 1-D float array of length 8575. The result
    is bit-for-bit equivalent (to float tolerance) to the old torch path.
    """
    lab = np.array([teff, logg, feh], dtype=np.float64)
    scaled = _scale_labels(lab)
    return _payne_forward_numpy(scaled).astype(np.float64)


def label_in_range(teff, logg, feh):
    """True if (Teff, logg, [Fe/H]) lies inside the trained label box."""
    lab = np.array([teff, logg, feh], dtype=np.float32)
    return bool(np.all(lab >= _LMIN) and np.all(lab <= _LMAX))


# --------------------------------------------------------------------------- #
# 1b. OUR DR19 5-label Payne, SURVEY-normalized (the comparison single-star model;
#     the PRODUCTION model is the SC net in 1c).
#
#     A 5-label net WE trained on ~4000 SURVEY-normalized DR19 dwarf single-star
#     spectra. It maps the five El-Badry et al. (2018b) labels
#         (Teff, logg, [Fe/H], [Mg/H], v_macro)  ->  flux[8575]
#     directly on OUR 8575-pixel grid (= physics.WAVELENGTH), so the detector needs
#     no regridding. The model='dr19' detector (7b) uses this; the production
#     model='dr19_sc' detector (7c) uses the SC net (1c) instead.
#
#     The checkpoint schema is identical to payne.pt (Linear layers net.0/net.2/
#     net.4, label_min, label_max, n_label=5, n_hidden=300, n_pix=8575, same
#     (x-min)/(max-min)-0.5 scaling), so we load + export it to numpy exactly as
#     the 3-label net above and the fit stays torch-free. If models/payne_dr19.pt
#     is absent we set _HAVE_DR19=False and the model='dr19' path raises a clear
#     error; the other paths still import.
# --------------------------------------------------------------------------- #
DR19_MODEL_PATH = os.path.join(_ROOT, "models", "payne_dr19.pt")

try:
    _CKPT5 = torch.load(DR19_MODEL_PATH, map_location="cpu", weights_only=False)
    _HAVE_DR19 = True
except FileNotFoundError:
    _CKPT5 = None
    _HAVE_DR19 = False

if _HAVE_DR19:
    # Build the torch module ONCE (to read the weights), then export every layer
    # to numpy so the fit loop never touches torch -- the same trick as the
    # 3-label net. _Payne already takes n_label, so it builds the 5-input net.
    _NET5 = _Payne(_CKPT5["n_label"], _CKPT5["n_hidden"], _CKPT5["n_pix"])
    _NET5.load_state_dict(_CKPT5["state_dict"])
    _NET5.eval()
    _LMIN5 = np.asarray(_CKPT5["label_min"], dtype=np.float64)   # 5-vector
    _LMAX5 = np.asarray(_CKPT5["label_max"], dtype=np.float64)
    _s5 = _NET5.state_dict()
    # Same (out,in)->(in,out) transpose so the numpy forward is x @ W + b.
    _W0_5 = _s5["net.0.weight"].numpy().astype(np.float64).T   # (5, n_hidden)
    _B0_5 = _s5["net.0.bias"].numpy().astype(np.float64)
    _W1_5 = _s5["net.2.weight"].numpy().astype(np.float64).T   # (n_hidden, n_hidden)
    _B1_5 = _s5["net.2.bias"].numpy().astype(np.float64)
    _W2_5 = _s5["net.4.weight"].numpy().astype(np.float64).T   # (n_hidden, n_pix)
    _B2_5 = _s5["net.4.bias"].numpy().astype(np.float64)
else:
    _NET5 = None
    _LMIN5 = _LMAX5 = None
    _W0_5 = _B0_5 = _W1_5 = _B1_5 = _W2_5 = _B2_5 = None


def _payne5_forward_numpy(scaled):
    """Pure-numpy forward pass for OUR 5-label net on a SCALED label vector.

    Reproduces _Payne.forward exactly (two LeakyReLU hidden layers, linear
    output) using the DR19 weights exported above. Returns the raw 8575-pixel
    normalized spectrum on physics.WAVELENGTH. No torch, so it runs at numpy/BLAS
    speed inside least_squares.
    """
    h0 = _leaky_relu(scaled @ _W0_5 + _B0_5)
    h1 = _leaky_relu(h0 @ _W1_5 + _B1_5)
    return h1 @ _W2_5 + _B2_5


def payne_predict5(teff, logg, feh, mgh, vmacro):
    """Predict the rest-frame NORMALIZED APOGEE spectrum with OUR DR19 5-label net.

    Runs the network WE trained (PURE NUMPY) on the five labels
    (Teff, logg, [Fe/H], [Mg/H], v_macro). The output is the continuum-normalized
    flux (~1 in the continuum) in the rest frame on the 8575-pixel grid
    (= physics.WAVELENGTH); RV is applied separately (Doppler shift). This is the
    single-star spectral model the DETECTION path uses. Labels are CLIPPED into
    the trained box first so an out-of-box primary/secondary stays where the net
    was trained. Returns a 1-D float array of length 8575.
    """
    if not _HAVE_DR19:
        raise RuntimeError(
            "models/payne_dr19.pt not found (the committed survey-normalized "
            "5-label net for the model='dr19' comparison path).")
    lab = np.array([teff, logg, feh, mgh, vmacro], dtype=np.float64)
    lab = np.clip(lab, _LMIN5, _LMAX5)
    scaled = (lab - _LMIN5) / (_LMAX5 - _LMIN5) - 0.5
    return _payne5_forward_numpy(scaled).astype(np.float64)


def label5_in_range(teff, logg, feh, mgh, vmacro):
    """True if the five labels lie inside OUR DR19 net's trained label box."""
    if not _HAVE_DR19:
        return False
    lab = np.array([teff, logg, feh, mgh, vmacro], dtype=np.float64)
    return bool(np.all(lab >= _LMIN5) and np.all(lab <= _LMAX5))


# --------------------------------------------------------------------------- #
# 1c. OUR DR19 5-label Payne trained on SELF-CONSISTENT-normalized flux (THE
#     PRODUCTION DETECTION SINGLE-STAR MODEL after the continuum pivot).
#
#     payne_dr19_sc.pt is the SAME 5->300->300->8575 schema as payne_dr19.pt, but
#     it was trained (src/train_payne_dr19_sc.py) on flux normalized by
#     continuum_normalize -- the per-chip sigma-clipped Chebyshev continuum derived
#     from RAW flux -- NOT the survey continuum. The production detector
#     (dr19_single_vs_binary) uses THIS net so that the normalization on the
#     training side and the observed side is the identical self-consistent
#     operator. We load + export it to numpy exactly as the survey-trained net
#     above, so the per-spectrum fit stays torch-free. If the file is absent we set
#     _HAVE_SC=False and the SC path raises a clear error.
# --------------------------------------------------------------------------- #
DR19_SC_MODEL_PATH = os.environ.get(
    "AGENT4BINARY_SC_MODEL",
    os.path.join(_ROOT, "models", "payne_dr19_sc.pt"))

try:
    _CKPTSC = torch.load(DR19_SC_MODEL_PATH, map_location="cpu",
                         weights_only=False)
    _HAVE_SC = True
except FileNotFoundError:
    _CKPTSC = None
    _HAVE_SC = False

if _HAVE_SC:
    _NETSC = _Payne(_CKPTSC["n_label"], _CKPTSC["n_hidden"], _CKPTSC["n_pix"])
    _NETSC.load_state_dict(_CKPTSC["state_dict"])
    _NETSC.eval()
    _LMINSC = np.asarray(_CKPTSC["label_min"], dtype=np.float64)   # 5-vector
    _LMAXSC = np.asarray(_CKPTSC["label_max"], dtype=np.float64)
    _ssc = _NETSC.state_dict()
    _W0_SC = _ssc["net.0.weight"].numpy().astype(np.float64).T
    _B0_SC = _ssc["net.0.bias"].numpy().astype(np.float64)
    _W1_SC = _ssc["net.2.weight"].numpy().astype(np.float64).T
    _B1_SC = _ssc["net.2.bias"].numpy().astype(np.float64)
    _W2_SC = _ssc["net.4.weight"].numpy().astype(np.float64).T
    _B2_SC = _ssc["net.4.bias"].numpy().astype(np.float64)
else:
    _NETSC = None
    _LMINSC = _LMAXSC = None
    _W0_SC = _B0_SC = _W1_SC = _B1_SC = _W2_SC = _B2_SC = None


def _payne_sc_forward_numpy(scaled):
    """Pure-numpy forward pass for OUR SC-trained 5-label net (torch-free)."""
    h0 = _leaky_relu(scaled @ _W0_SC + _B0_SC)
    h1 = _leaky_relu(h0 @ _W1_SC + _B1_SC)
    return h1 @ _W2_SC + _B2_SC


def payne_predict5_sc(teff, logg, feh, mgh, vmacro):
    """Predict the rest-frame SC-normalized spectrum with OUR SC-trained net.

    Identical interface to payne_predict5, but the net was trained on flux that
    continuum_normalize produced (per-chip sigma-clipped Chebyshev from raw flux),
    so its predictions live in the SAME normalized space the detector puts the
    observed spectrum in. Labels are clipped into the trained box. Length 8575.
    """
    if not _HAVE_SC:
        raise RuntimeError(
            "models/payne_dr19_sc.pt not found; train the SC 5-label net first "
            "(src/train_payne_dr19_sc.py).")
    lab = np.array([teff, logg, feh, mgh, vmacro], dtype=np.float64)
    lab = np.clip(lab, _LMINSC, _LMAXSC)
    scaled = (lab - _LMINSC) / (_LMAXSC - _LMINSC) - 0.5
    return _payne_sc_forward_numpy(scaled).astype(np.float64)


def label5sc_in_range(teff, logg, feh, mgh, vmacro):
    """True if the five labels lie inside OUR SC net's trained label box."""
    if not _HAVE_SC:
        return False
    lab = np.array([teff, logg, feh, mgh, vmacro], dtype=np.float64)
    return bool(np.all(lab >= _LMINSC) and np.all(lab <= _LMAXSC))


# --------------------------------------------------------------------------- #
# 2. Isochrone relation: q -> secondary labels / radii.
#
#    The PRODUCTION q -> secondary map (secondary_from_q, below) is now OUR OWN
#    interpolation of genuine MIST v1.2 isochrones (isochrone_mist, imported as
#    _iso_ours). The production detector imports NO binspec isochrone net.
#
#    The binspec neural-net forward passes (get_Teff2_logg2_NN, get_radius_NN,
#    primary_mass) are KEPT in this section but are NO LONGER called by
#    secondary_from_q. They remain for two non-production uses only:
#      - the isochrone MCP server (src/mcp_servers/isochrone_server.py), a
#        reference / inspection tool, and
#      - head-to-head agreement checks against OUR MIST relation.
#    The binspec single-vs-binary ORACLE (binspec_single_vs_binary, section 8)
#    runs its own vendored binspec nets (vendor.binspec) and does not touch these
#    helpers either. Two networks are reproduced here, both loaded once at import:
#      - NN_Teff2_logg2 : (Teff1, logg1, [Fe/H], q) -> (Teff2, logg2)
#      - NN_radius      : (Teff, logg, [Fe/H])      -> R (Rsun)
#    A primary mass (for the MCP server and the q floor) is read from the raw MIST
#    single-star table. See the path block above for provenance.
# --------------------------------------------------------------------------- #

# --- Load the ported isochrone networks at import. ------------------------- #
# NN_Teff2_logg2 is a single-hidden-layer network. binspec stores it with the
# keys w_array_0 (n_hidden, n_label), w_array_1 (n_out, n_hidden), b_array_0
# (n_hidden,), b_array_1 (n_out,), and the label scaling x_min / x_max over the
# four labels [Teff1, logg1, [Fe/H], q].
_ISO_T = np.load(ISO_TEFF2_NN_PATH)
_ITW0 = _ISO_T["w_array_0"].astype(np.float64)   # (n_hidden, 4)
_ITW1 = _ISO_T["w_array_1"].astype(np.float64)   # (2, n_hidden)
_ITB0 = _ISO_T["b_array_0"].astype(np.float64)   # (n_hidden,)
_ITB1 = _ISO_T["b_array_1"].astype(np.float64)   # (2,)
_ITXMIN = _ISO_T["x_min"].astype(np.float64)     # [Teff1, logg1, feh, q]
_ITXMAX = _ISO_T["x_max"].astype(np.float64)
_ISO_T.close()

# NN_radius is a two-hidden-layer network with keys w_array_0 (n_h, 3),
# w_array_1 (n_h, n_h), w_array_2 (1, n_h) and matching biases, over the three
# labels [Teff, logg, [Fe/H]].
_ISO_R = np.load(ISO_RADIUS_NN_PATH)
_IRW0 = _ISO_R["w_array_0"].astype(np.float64)   # (n_h, 3)
_IRW1 = _ISO_R["w_array_1"].astype(np.float64)   # (n_h, n_h)
_IRW2 = _ISO_R["w_array_2"].astype(np.float64)   # (1, n_h)
_IRB0 = _ISO_R["b_array_0"].astype(np.float64)   # (n_h,)
_IRB1 = _ISO_R["b_array_1"].astype(np.float64)   # (n_h,)
_IRB2 = _ISO_R["b_array_2"].astype(np.float64)   # (1,)
_IRXMIN = _ISO_R["x_min"].astype(np.float64)     # [Teff, logg, feh]
_IRXMAX = _ISO_R["x_max"].astype(np.float64)
_ISO_R.close()

# Label ranges the Teff2/logg2 network was trained over. We clip primary inputs
# into this box before the forward pass so an out-of-range primary (e.g. a hot
# turnoff star or a metal-poor outlier) does not push the network into a region
# it never saw. The box is, per the weight file: Teff1 in ~[3032, 7499] K, logg1
# in ~[3.80, 5.25], [Fe/H] in ~[-0.99, 0.47], q in ~[0.074, 1.0].
_ISO_LABEL_LO = _ITXMIN.copy()
_ISO_LABEL_HI = _ITXMAX.copy()


def _sigmoid(z):
    """Logistic sigmoid, the activation binspec uses in every isochrone net."""
    return 1.0 / (1.0 + np.exp(-z))


# --- Raw MIST single-star table (for primary mass recovery only). ---------- #
# Columns are 1-D arrays over a feh x age x mass grid of MS single stars. We snap
# (feh, age) to the nearest node that actually exists, then match (Teff, logg).
_MIST = np.load(MIST_TABLE_PATH)
_MIST_TEFF = _MIST["Teff"].astype(np.float64)
_MIST_LOGG = _MIST["logg"].astype(np.float64)
_MIST_FEH = _MIST["feh"].astype(np.float64)
_MIST_MASS = _MIST["mass"].astype(np.float64)
_MIST_AGE = _MIST["age"].astype(np.float64)   # already in Gyr
_MIST.close()
_MIST_FEH_NODES = np.unique(_MIST_FEH)


def get_radius_NN(teff, logg, feh):
    """Port of binspec spectral_model.get_radius_NN: (Teff, logg, [Fe/H]) -> R.

    Reproduces the two-hidden-layer sigmoid forward pass exactly. Returns the
    radius in Rsun for a single main-sequence star. Inputs are clipped into the
    radius network's own training box so a cool / faint secondary stays inside
    the region the network was trained on.
    """
    labels = np.array([teff, logg, feh], dtype=np.float64)
    labels = np.clip(labels, _IRXMIN, _IRXMAX)
    # Same (label - x_min)/(x_max - x_min) - 0.5 scaling used in training.
    scaled = (labels - _IRXMIN) / (_IRXMAX - _IRXMIN) - 0.5
    h0 = _sigmoid(_IRW0 @ scaled + _IRB0)        # first hidden layer
    h1 = _sigmoid(_IRW1 @ h0 + _IRB1)            # second hidden layer
    out = _IRW2 @ h1 + _IRB2                     # linear output, length 1
    return float(out[0])


def get_Teff2_logg2_NN(teff1, logg1, feh, q, force_lower_teff=True):
    """Port of binspec spectral_model.get_Teff2_logg2_NN.

    Maps (Teff1, logg1, [Fe/H], q) -> (Teff2, logg2) with the MIST-trained
    single-hidden-layer network. Reproduces binspec's behaviour:
      - q ~ 1 returns the primary labels unchanged, so an equal-mass binary is
        identical to a single star (the q=1 identity the detector relies on),
      - the network predicts Teff2/1000, so its first output is scaled by 1000,
      - with force_lower_teff (binspec default) the secondary is forced to be no
        hotter and no less compact than the primary, the equal-age MS ordering.
    Returns (Teff2, logg2).
    """
    # q=1 identity, copied from binspec: guarantee the equal-mass limit is exact.
    if np.isclose(q, 1.0):
        return float(teff1), float(logg1)

    # Clip the primary labels and q into the network's training box.
    labels = np.array([teff1, logg1, feh, q], dtype=np.float64)
    labels = np.clip(labels, _ISO_LABEL_LO, _ISO_LABEL_HI)
    # Same scaling as training, then a single sigmoid hidden layer.
    scaled = (labels - _ITXMIN) / (_ITXMAX - _ITXMIN) - 0.5
    inside = _ITW0 @ scaled + _ITB0
    outside = _ITW1 @ _sigmoid(inside) + _ITB1
    teff2 = float(outside[0] * 1000.0)   # network predicts Teff2/1000
    logg2 = float(outside[1])

    # Equal-age, equal-composition MS pair: the secondary is the lower-mass star,
    # so it is cooler (lower Teff) and more compact (higher logg). Force this for
    # stability exactly as binspec does.
    if force_lower_teff:
        if teff2 > teff1:
            teff2 = float(teff1)
        if logg2 < logg1:
            logg2 = float(logg1)
    return teff2, logg2


def primary_mass(teff1, logg1, feh, age_gyr=_MS_REPR_AGE_GYR):
    """Recover the primary's MIST mass (Msun) from (Teff1, logg1, [Fe/H]).

    binspec's networks never expose a mass, so we read it off the raw MIST
    single-star table. We snap [Fe/H] to the nearest table node, then snap the
    age to the nearest age that EXISTS at that node (the table is sparse: some
    ages appear only at some metallicities). On that single isochrone track,
    which is monotonic in mass on the MS, we densely interpolate (Teff, logg) in
    mass and return the mass whose (Teff, logg) is closest to the primary in a
    scaled metric (Teff scaled by 500 K, logg by 0.3 dex). This is used for
    reporting and for the q floor, not inside the spectral forward model.
    """
    # Snap [Fe/H] to the nearest available node.
    feh_n = _MIST_FEH_NODES[np.argmin(np.abs(_MIST_FEH_NODES - feh))]
    feh_sel = np.abs(_MIST_FEH - feh_n) < 1e-9
    # Among ages that exist at this feh, snap to the nearest one.
    ages_here = np.unique(_MIST_AGE[feh_sel])
    age_n = ages_here[np.argmin(np.abs(ages_here - age_gyr))]
    sel = feh_sel & (np.abs(_MIST_AGE - age_n) < 1e-9)
    m = _MIST_MASS[sel]
    t = _MIST_TEFF[sel]
    g = _MIST_LOGG[sel]
    order = np.argsort(m)
    m, t, g = m[order], t[order], g[order]
    # Dense mass sampling of the (monotonic-in-mass) MS track, scaled match.
    mm = np.linspace(m.min(), m.max(), 4000)
    tt = np.interp(mm, m, t)
    gg = np.interp(mm, m, g)
    dt = (tt - teff1) / 500.0
    dg = (gg - logg1) / 0.3
    d2 = dt ** 2 + dg ** 2
    return float(mm[int(np.argmin(d2))])


def secondary_from_q(teff1, logg1, feh, q, age_gyr=_MS_REPR_AGE_GYR):
    """Map the mass ratio q -> secondary labels and both radii (PRODUCTION path).

    This is the PRODUCTION isochrone relation, called by compose_binary5 and
    compose_binary5_sc. It is now OUR OWN MIST v1.2 interpolation
    (isochrone_mist.secondary_from_q_ours), NOT the binspec neural net. The binspec
    isochrone-net forward passes (the Teff2/logg2 and radius ports defined above)
    are kept for the MCP server and for agreement checks, but are no longer called
    here, so the production detector depends on no binspec isochrone net.

    OUR MIST relation uses a coeval, equal-[Fe/H] main-sequence pair (El-Badry et
    al. 2018b): it finds the primary mass m1 on the (age, [Fe/H]) isochrone from
    (Teff1, logg1), sets m2 = q * m1, reads the secondary (Teff2, logg2, L2) at m2
    on the SAME isochrone, and takes both radii from the MIST L-Teff relation.
    age_gyr selects the isochrone (snapped to the nearest MIST log-age node);
    unlike the binspec net (which folded age into training and ignored it), OUR
    relation takes age as a real input.

    The default age is 4.0 Gyr, a representative main-sequence age for the APOGEE
    dwarf sample. It was set by the detector re-verify, not by hand: the flux ratio
    that drives the binary signal is R2^2 / R1^2, and at 4 Gyr OUR (R2 / R1)^2 over
    q tracks the binspec NN it replaces and the SC detector reproduces the binspec-
    NN completeness (76% on the 25 SB2 / 25 control set). At 4 Gyr a solar primary
    also recovers m1 ~ 1.0 Msun. Older isochrones swell R1 and suppress the flux
    ratio (lower completeness); younger ones raise contamination.

    The q = 1 / equal-mass limit is EXACT: secondary_from_q_ours returns the
    REQUESTED primary labels and R2 == R1, so the binary at q = 1 reduces to the
    single star at the same labels (the identity the detector's nesting relies on).

    Returns (Teff2, logg2, R2_rsun, R1_rsun) -- the same tuple order as before, so
    every caller (compose_binary5, compose_binary5_sc, compose_binary) is unchanged.
    """
    return _iso_ours.secondary_from_q_ours(teff1, logg1, feh, q, age_gyr=age_gyr)


# --------------------------------------------------------------------------- #
# 3. Pseudo-continuum: physical un-normalization via the PORTED binspec flux net.
#    binspec multiplies the normalized spectrum by a synthetic continuum before
#    summing components in flux (Eq. 2): a TRAINED per-pixel flux net
#    (NN_unnormalized_spectra) followed by a Cannon-pixel Chebyshev fit. We port
#    both: (a) run the flux net (numpy forward), (b) fit binspec's 4th-order
#    per-chip Chebyshev on Cannon pixels, (c) interpolate onto our 8575-pixel grid.
#    The production SC path uses pseudo_continuum_sc instead (Stage 4).
# --------------------------------------------------------------------------- #

# --- Load the ported binspec flux net + grid + cont-pixel mask at import. --- #
_FLUX = np.load(FLUX_NN_PATH)
# binspec flux net: single hidden layer, per-pixel. Shapes:
#   w_array_0 (n_pix, n_hidden, n_label), w_array_1 (n_pix, n_hidden),
#   b_array_0 (n_pix, n_hidden), b_array_1 (n_pix,), x_min/x_max (n_label,).
_FW0 = _FLUX["w_array_0"].astype(np.float64)
_FW1 = _FLUX["w_array_1"].astype(np.float64)
_FB0 = _FLUX["b_array_0"].astype(np.float64)
_FB1 = _FLUX["b_array_1"].astype(np.float64)
_FXMIN = _FLUX["x_min"].astype(np.float64)   # [Teff, logg, feh, alpha]
_FXMAX = _FLUX["x_max"].astype(np.float64)
_FLUX.close()

# The binspec wavelength grid the flux net predicts on (7214 pixels, distinct
# from our 8575-pixel aspcapStar grid). We need it to fit the Chebyshev continuum
# and to interpolate the result onto WAVELENGTH.
_BWL = np.load(BINSPEC_WL_PATH)["wavelength"].astype(np.float64)

# Melissa Ness' Cannon continuum-pixel boolean mask (length 7214), split by chip
# exactly as binspec's get_apogee_continuum does (blue 0:2920, green 2920:5320,
# red 5320:). True = a continuum pixel used in the Chebyshev fit.
_CONT_PIX = np.load(CONT_PIX_PATH)["pixels_cannon"].astype(bool)


def _flux_net_predict(teff, logg, feh, alpha=0.0):
    """Run the ported binspec flux net -> un-normalized surface flux (7214 pix).

    Reproduces spectral_model.get_spectrum_from_neural_net (normalized=False):
      scaled = (labels - x_min)/(x_max - x_min) - 0.5
      inside  = einsum('ijk,k->ij', w0, scaled) + b0   # (n_pix, n_hidden)
      outside = einsum('ij,ij->i', w1, sigmoid(inside)) + b1   # (n_pix,)
      spectrum = outside / 1e6   # the net predicts 1e6 * f_lambda
    The net's 4 labels are [Teff, logg, feh, alpha]; our Payne has no alpha, so we
    pass alpha=0 (solar). Sigmoid activation, matching binspec exactly.

    The flux net was trained over Teff ~ [3529, 6983] K, logg ~ [0.34, 5.22],
    [Fe/H] ~ [-0.99, 0.52], alpha ~ [-0.56, 0.91]. A cool, low-q secondary can
    fall below the Teff floor (e.g. Teff2 ~ 3400 K at q = 0.3 for a solar
    primary). We clip the labels into the training box so the surface-flux
    continuum stays physical (a colder star keeps a redder, weaker H-band
    continuum); this clipping affects only the smooth continuum used for the
    flux ratio, not the line spectrum.
    """
    labels = np.array([teff, logg, feh, alpha], dtype=np.float64)
    labels = np.clip(labels, _FXMIN, _FXMAX)
    scaled = (labels - _FXMIN) / (_FXMAX - _FXMIN) - 0.5
    inside = np.einsum("ijk,k->ij", _FW0, scaled) + _FB0
    sig = 1.0 / (1.0 + np.exp(-inside))           # logistic sigmoid
    outside = np.einsum("ij,ij->i", _FW1, sig) + _FB1
    return outside / 1e6                           # undo the 1e6 training scale


def _chebyshev_continuum_binspec(spec):
    """Port of utils.get_apogee_continuum: 4th-order per-chip Chebyshev on Cannon
    continuum pixels, returned on the binspec 7214-pixel grid.

    For each of the three APOGEE chips, fit a degree-4 Chebyshev polynomial to the
    flux at the Cannon continuum pixels (equal weights, since the flux net output
    is noiseless) over a rescaled [-1, 1] wavelength axis, then evaluate it over
    every pixel of that chip. This is the smooth continuum binspec divides by.
    """
    cont = np.empty_like(spec)
    deg = 4
    # The per-chip rescaled wavelength axes binspec uses (chip lengths 2920/2400/
    # 1894 sum to 7214). Each chip maps its pixel index to [-1, 1].
    bluewav = 2 * np.arange(2920) / 2919 - 1
    greenwav = 2 * np.arange(2400) / 2399 - 1
    redwav = 2 * np.arange(1894) / 1893 - 1

    def _fit(wav, s, pix):
        # Chebyshev.fit through the continuum pixels only, then evaluate over all.
        cheb = np.polynomial.Chebyshev.fit(wav[pix], s[pix], deg)
        return cheb(wav)

    cont[:2920] = _fit(bluewav, spec[:2920], _CONT_PIX[:2920])
    cont[2920:5320] = _fit(greenwav, spec[2920:5320], _CONT_PIX[2920:5320])
    cont[5320:] = _fit(redwav, spec[5320:], _CONT_PIX[5320:])
    return cont


from functools import lru_cache as _lru_cache  # noqa: E402


@_lru_cache(maxsize=4096)
def _pseudo_continuum_cached(teff_r, logg_r, feh_r):
    """Cached core of pseudo_continuum, keyed on ROUNDED labels.

    The continuum is a degree-4 per-chip Chebyshev, so it varies smoothly and
    slowly with the labels; rounding to 1 K / 0.01 dex changes it imperceptibly.
    The binary grid scan calls pseudo_continuum with the SAME primary labels 150+
    times, so caching collapses the flux-net + Chebyshev cost (the detector's
    dominant per-star cost) to one evaluation per distinct (rounded) label triple.
    Returns a length-8575 array, mean 1.
    """
    flux = _flux_net_predict(teff_r, logg_r, feh_r)
    cont_bwl = _chebyshev_continuum_binspec(flux)
    cont = np.interp(WAVELENGTH, _BWL, cont_bwl)
    return cont / cont.mean()


def pseudo_continuum(teff, logg=4.5, feh=0.0):
    """Trained-flux-net continuum on the 8575-pixel grid (PORTED from binspec).

    This is the El-Badry un-normalization continuum, NOT a blackbody. Steps:
      1. run the ported binspec flux net at (Teff, logg, feh, alpha=0) to get an
         un-normalized surface flux on binspec's 7214-pixel grid,
      2. fit binspec's 4th-order per-chip Chebyshev continuum through the Cannon
         continuum pixels of that flux (the get_apogee_continuum path),
      3. interpolate the smooth Chebyshev continuum onto our 8575-pixel grid.
    The continuum is divided by its own mean so only its SHAPE and the relative
    level between two temperatures matter (the absolute scale cancels in
    compose_binary's final re-normalization).

    logg/feh default to a dwarf, solar value so legacy single-argument calls keep
    working; compose_binary passes the component's true logg/feh.

    The heavy work is memoized on rounded labels (_pseudo_continuum_cached); the
    result returned is a COPY so callers that mutate it in place cannot corrupt
    the cache. Returns a length-8575 array, mean 1.
    """
    cont = _pseudo_continuum_cached(round(float(teff), 0),
                                    round(float(logg), 2),
                                    round(float(feh), 2))
    return cont.copy()


# =========================================================================== #
# 3b. SELF-CONSISTENT continuum normalization (THE production normalization).
#
#     Replaces the survey (mwmStar / ASPCAP) pre-normalization. It builds a
#     continuum from the RAW flux itself (per-chip, inverse-variance-weighted,
#     iteratively sigma-clipped Chebyshev) and divides the raw flux by it. NO
#     external continuum-pixel mask, NO survey continuum column.
#
#     "Self-consistent" means the SAME operator runs on the training spectra
#     (Stage 3) and on every observed spectrum at detection (Stage 5). Both sides
#     of the chi^2 pass through the identical estimator, so any continuum error is
#     common-mode and cancels in chi^2 / Delta-chi2 / f_imp; this is what lets the
#     q=1 limit of the binary model coincide with the single-star model.
#
#     The per-chip fit and the asymmetric sigma-clip are documented on
#     _fit_chebyshev_chip_sigmaclip below.
# =========================================================================== #

# APOGEE detector chip boundaries on the 8575-pixel grid. The aspcapStar combined
# spectrum is three physical detectors (blue/green/red) with two gaps between
# them; the gaps carry no signal (ivar == 0). We split at the MIDPOINTS of those
# two gaps so each chip's continuum is fit on its own detector, never across a
# gap. The split is by PIXEL INDEX (the grid is fixed), confirmed against a raw
# mwmStar ivar profile: blue ends ~px 3315, green starts ~px 3545; green ends
# ~px 6122, red starts ~px 6302. Midpoints 3430 and 6212.
CHIP_EDGES = (3430, 6212)
# Convenience: the three (start, stop) pixel slices, stop exclusive.
CHIP_SLICES = ((0, CHIP_EDGES[0]),
               (CHIP_EDGES[0], CHIP_EDGES[1]),
               (CHIP_EDGES[1], NPIX))

# --------------------------------------------------------------------------- #
# Chip-boundary guard band + systematic error floor (Refinement 2 -- TRIED AND
# REVERTED; constants kept only to record what was tested, NOT wired in).
#
# Motivation: the documented DR19 false positives pile Delta-chi2 onto a few
# pixels at the detector / chip-gap boundaries, where the single model predicts
# flux ~ 0 against observed ~ 1 (the detector-edge drop gemini-3.5-flash flagged):
# blue start px ~161-172, blue/green edge px ~3303-3524, green/red edge px
# ~6273-6282. The measured on-chip interiors (DR19 ivar profile, see
# build_continuum_pixels.py) are blue [160,3317], green [3504,6122], red
# [6265,8375]; the gaps (3318-3503, 6123-6264) and outer edges already carry
# ivar == 0. Two guards were tried in mask_bad_pixels: (1) mask EDGE_GUARD pixels
# just inside each chip boundary, (2) a systematic error floor (SYS_FLOOR of the
# flux, in quadrature; Li et al. 2025 App A.1).
#
# RESULT on the 25+25 verify set: both regressed contamination. Sys floor +
# edge guard -> 64% completeness / 12% contamination; edge guard alone -> 76% /
# 12%. Removing the edge pixels (or reweighting via the sys floor) DISTURBS the
# f_imp balance the existing positive f_imp floor already relies on: the
# previously-rejected controls (55142338, 54635100, 54873275) had their f_imp jump
# from ~0.05 / -0.13 to ~0.25 once the edge pixels were dropped, crossing the
# floor. The documented chip-edge mode is already neutralized at 0% contamination
# by table_b1_min_fimp's FIMP_FLOOR, so this refinement is NOT applied.
CHIP_INTERIORS = ((160, 3317), (3504, 6122), (6265, 8375))
EDGE_GUARD = 25              # pixels that WOULD be masked inside each chip boundary
SYS_FLOOR = 0.005           # systematic per-pixel floor that WAS tried (0.5% flux)

# --------------------------------------------------------------------------- #
# Chip-edge guard mask (Refinement 2, REDONE CORRECTLY -- the chip-edge
# false-positive fix, WIRED IN via mask_bad_pixels).
#
# ROOT CAUSE (historical chip-edge audit; summarized in METHODS.md).
# The SC Payne net (and the un-normalization continuum it shares) predicts
# normalized flux ~ 0 in the first / last pixels of each chip: the SC training
# spectra were continuum_normalize'd with ZERO (gap / shoulder) flux in those
# chip-edge pixels, so the net LEARNED ~ 0 there. The observed continuum_normalize
# flux at the SAME pixels is real (~ 1, nonzero ivar), so the single model is ~ 0
# against obs ~ 1 -> a huge per-pixel chi2. The velocity-shifted binary SECONDARY
# template partially fills that step (binary ~ 0.6 vs single ~ 0), so the binary
# model buys a large, SPURIOUS Delta-chi2 (and a depressed/inflated f_imp) at a
# handful of chip-edge pixels -- not a real companion. This was 188/212 of the
# vision-cull rejects.
#
# WHY THE OLD EDGE_GUARD FAILED. Refinement 2 masked a FIXED, SYMMETRIC EDGE_GUARD
# (= 25 px) just inside each CHIP_EDGES boundary. But the artifact is ASYMMETRIC
# and lives at the DATA-region edges (CHIP_INTERIORS), not the gap midpoints: the
# net's depressed band runs ~ 30-38 px into each chip LEADING edge but only ~ 6-10
# px at each TRAILING edge. The
# fixed symmetric guard therefore masked good interior pixels on the wrong side
# while missing the deep leading band, which is what disturbed the f_imp balance
# and regressed contamination.
#
# THE FIX. Build the guard DIRECTLY from where the model itself is depressed: take
# the median SC single model over a grid of dwarf labels and, at each of the six
# data-region edges (CHIP_INTERIORS start/end), mask the contiguous run where the
# model falls below CHIP_GUARD_FRAC of the chip-interior level, plus a small margin.
# The mask is a PROPERTY OF THE MODEL, so it is the SAME pixels for every spectrum.
# It is applied by SETTING THE OBSERVED ERROR TO inf on those pixels in
# mask_bad_pixels, so the SAME pixels drop out of BOTH the single and the binary
# chi2 AND the f_imp sums (they cancel) AND out of both least_squares fits -- one
# shared point, no call-signature change. Because identical pixels leave both
# chi2 sums, the binary-nests-single identity (Delta-chi2 >= 0) is preserved
# exactly. If the SC net is absent we fall back to a fixed conservative band.
CHIP_GUARD_FRAC = 0.6       # mask where the model continuum < this * chip-interior level
CHIP_GUARD_MARGIN = 3       # extra guard pixels beyond the detected depressed run
_CHIP_EDGE_GUARD = None     # lazily built boolean mask (True = guard / exclude)


def _build_chip_edge_guard():
    """Build the fixed chip-edge guard mask from the SC model's own depressed band.

    Returns a length-NPIX boolean array, True on the pixels to EXCLUDE. The band
    is found by sampling the SC single model (payne5_single_model_sc) over a grid
    of dwarf labels, taking the per-pixel median model continuum, and, at each
    CHIP_INTERIORS data-region edge, marking the contiguous run where the model is
    below CHIP_GUARD_FRAC of the chip-interior reference level (plus a margin).
    This is the model-intrinsic chip-edge artifact (see the block comment above).

    Built lazily on first use so it can reference payne5_single_model_sc (defined
    later in the module) and so importing physics never forces a net evaluation.
    If the SC net is unavailable, falls back to a fixed conservative guard (the
    leading ~38 px and trailing ~10 px of each chip data-region).
    """
    guard = np.zeros(NPIX, dtype=bool)
    if not _HAVE_SC:
        # Fallback: conservative fixed bands matching the measured net behaviour.
        for a, b in CHIP_INTERIORS:
            guard[a:min(b + 1, a + 40)] = True       # leading ~40 px
            guard[max(a, b - 10):b + 1] = True       # trailing ~10 px
        # Mask the two grid ends as well (never carry real data).
        guard[:CHIP_INTERIORS[0][0]] = True
        guard[CHIP_INTERIORS[-1][1] + 1:] = True
        return guard

    # Median SC single model over a dwarf-label grid (the model continuum). Teff
    # spans the dwarf range inside the SC net box; logg/feh a small spread. The
    # depressed chip-edge band is the same across all of these.
    teffs = np.linspace(max(4500.0, _LMINSC[0]), min(6500.0, _LMAXSC[0]), 5)
    models = []
    for t in teffs:
        for g in (4.0, 4.5):
            for h in (-0.3, 0.0, 0.3):
                models.append(payne5_single_model_sc(float(t), float(g),
                                                     float(h), 0.0, 5.0))
    med = np.median(np.asarray(models), axis=0)
    # Chip-interior reference level (robust mid-chip slab mean per chip).
    ref = float(np.median([med[a + 200:b - 200].mean()
                           for a, b in CHIP_INTERIORS]))
    if not np.isfinite(ref) or ref <= 0:
        ref = 1.0
    thr = CHIP_GUARD_FRAC * ref
    m = CHIP_GUARD_MARGIN
    for a, b in CHIP_INTERIORS:
        # LEADING edge: walk inward from a until the model recovers above thr.
        px = a
        while px < b and med[px] < thr:
            px += 1
        if px - 1 >= a:
            guard[a:min(b + 1, px + m)] = True
        # TRAILING edge: walk inward from b until the model recovers above thr.
        px = b
        while px > a and med[px] < thr:
            px -= 1
        if px + 1 <= b:
            guard[max(a, px + 1 - m):b + 1] = True
    # Also guard the two grid ends outside the data regions (never real data).
    guard[:CHIP_INTERIORS[0][0]] = True
    guard[CHIP_INTERIORS[-1][1] + 1:] = True
    return guard


def chip_edge_guard_mask():
    """Return the fixed chip-edge guard mask (True = exclude), building it once.

    The mask is cached at module level after the first build. See
    _build_chip_edge_guard and the block comment above for the artifact it removes.
    """
    global _CHIP_EDGE_GUARD
    if _CHIP_EDGE_GUARD is None:
        _CHIP_EDGE_GUARD = _build_chip_edge_guard()
    return _CHIP_EDGE_GUARD

# --------------------------------------------------------------------------- #
# Line-list + cross-star-variance continuum-pixel mask (Refinement 3 evaluation).
#
# models/continuum_pixels.npz (build_continuum_pixels.py) is a principled
# continuum-pixel mask on the 8575-pixel grid: 620 pixels that are both line-free
# (low summed Kurucz gf within a resolution element) and flat across a DR19 dwarf
# sample (low cross-star scatter at the upper flux envelope). It is evaluated here
# as an ALTERNATIVE continuum-pixel set for the Chebyshev fit, versus the default
# sigma-clip. It is OFF by default (continuum_normalize uses the sigma-clip): the
# production SC net was trained on the sigma-clip normalization, so switching the
# normalization at detection would break self-consistency.
CONT_PIX_MASK_PATH = os.path.join(_ROOT, "models", "continuum_pixels.npz")
try:
    _CPM = np.load(CONT_PIX_MASK_PATH)
    _CONT_MASK = _CPM["pixels"].astype(bool)        # (8575,) True = continuum pixel
    _CPM.close()
    _HAVE_CONT_MASK = bool(_CONT_MASK.size == NPIX)
except (FileNotFoundError, OSError):
    _CONT_MASK = None
    _HAVE_CONT_MASK = False


def _fit_chebyshev_chip_sigmaclip(wav_chip, flux_chip, ivar_chip,
                                  deg=4, n_iter=5,
                                  low_sigma=1.5, high_sigma=4.0):
    """Fit ONE chip's continuum: ivar-weighted, asymmetrically sigma-clipped Chebyshev.

    Returns the continuum evaluated over EVERY pixel of the chip (same length as
    wav_chip). Pure numpy / numpy.polynomial; no survey continuum, no line mask.

    Parameters
    ----------
    wav_chip  : (n,) wavelength of this chip (Angstrom).
    flux_chip : (n,) RAW flux of this chip.
    ivar_chip : (n,) inverse variance of this chip (0 where unmeasured).
    deg       : Chebyshev degree (4 = broad blaze/SED, no line following).
    n_iter    : maximum sigma-clip refit passes.
    low_sigma : reject pixels BELOW the fit by more than this many sigma
                (tight: absorption lines pull flux down).
    high_sigma: reject pixels ABOVE the fit by more than this many sigma
                (loose: keep continuum that noises high).

    The wavelength axis is rescaled to [-1, 1] over the chip so the Chebyshev
    basis is well conditioned. The fit is weighted by sqrt(ivar) (numpy's
    Chebyshev.fit takes per-point weights w with cost sum (w*(y-f))^2, so w =
    sqrt(ivar) realizes inverse-variance weighting). The clip set starts as all
    finite, ivar>0 pixels and only ever shrinks.
    """
    n = wav_chip.size
    # Map this chip's wavelengths to [-1, 1] for a conditioned Chebyshev basis.
    w0, w1 = wav_chip[0], wav_chip[-1]
    xx = 2.0 * (wav_chip - w0) / (w1 - w0) - 1.0

    # sqrt(ivar) weights realize inverse-variance least squares in Chebyshev.fit.
    sqrt_iv = np.sqrt(np.where(ivar_chip > 0, ivar_chip, 0.0))

    # Initial good set: finite flux, positive ivar. Lines are still IN here; the
    # sigma-clip below removes them. If too few pixels survive (a near-empty chip),
    # fall back to a flat continuum at the median so the division is defined.
    good = np.isfinite(flux_chip) & (ivar_chip > 0)
    if np.count_nonzero(good) < deg + 2:
        med = np.median(flux_chip[np.isfinite(flux_chip)]) if np.any(
            np.isfinite(flux_chip)) else 1.0
        return np.full(n, med if med != 0 else 1.0)

    cheb = None
    for _ in range(n_iter):
        # Inverse-variance-weighted Chebyshev fit on the CURRENT survivor set.
        cheb = np.polynomial.Chebyshev.fit(
            xx[good], flux_chip[good], deg, w=sqrt_iv[good])
        fit_all = cheb(xx)                          # continuum at every pixel
        resid = flux_chip - fit_all                 # raw - continuum
        # Robust sigma from the survivors only (median absolute deviation ->
        # Gaussian sigma). MAD is insensitive to the line cores we are removing.
        r_good = resid[good]
        med_r = np.median(r_good)
        mad = np.median(np.abs(r_good - med_r))
        sigma = 1.4826 * mad
        if sigma <= 0:
            break                                   # already a perfect fit
        # ASYMMETRIC clip: below the fit is clipped tight (lines), above loose.
        # Centre on med_r so a small overall offset does not bias the bounds.
        lo = med_r - low_sigma * sigma
        hi = med_r + high_sigma * sigma
        new_good = (np.isfinite(flux_chip) & (ivar_chip > 0)
                    & (resid >= lo) & (resid <= hi))
        if np.array_equal(new_good, good):
            break                                   # converged: set unchanged
        if np.count_nonzero(new_good) < deg + 2:
            break                                   # do not clip away the chip
        good = new_good

    cont = cheb(xx)
    # Floor the continuum away from zero so the division cannot blow up on a chip
    # whose blaze dips low; uses the chip's own median scale.
    scale = np.median(np.abs(cont[np.isfinite(cont)])) if np.any(
        np.isfinite(cont)) else 1.0
    floor = 0.05 * (scale if scale > 0 else 1.0)
    cont = np.where(np.isfinite(cont) & (cont > floor), cont, floor)
    return cont


def _fit_chebyshev_chip_maskpix(wav_chip, flux_chip, ivar_chip, mask_chip,
                                deg=4):
    """Fit ONE chip's continuum on the line-list+variance MASK pixels (Refinement 3).

    Instead of finding continuum pixels by sigma-clipping the flux, this uses a
    FIXED continuum-pixel mask (models/continuum_pixels.npz) and fits the same
    degree-deg, ivar-weighted Chebyshev through the mask pixels of this chip, then
    evaluates it over every pixel. This is the Cannon-type "fit only on
    label-insensitive continuum pixels" recipe (El-Badry et al. 2018b Sec 2.3).
    Falls back to the sigma-clip fit when too few mask pixels survive on the chip
    (so the division stays defined). Same return contract as
    _fit_chebyshev_chip_sigmaclip.
    """
    n = wav_chip.size
    w0, w1 = wav_chip[0], wav_chip[-1]
    xx = 2.0 * (wav_chip - w0) / (w1 - w0) - 1.0
    sqrt_iv = np.sqrt(np.where(ivar_chip > 0, ivar_chip, 0.0))
    good = (np.isfinite(flux_chip) & (ivar_chip > 0) & mask_chip)
    if np.count_nonzero(good) < deg + 2:
        # Not enough mask pixels on this chip: fall back to the sigma-clip fit.
        return _fit_chebyshev_chip_sigmaclip(wav_chip, flux_chip, ivar_chip,
                                             deg=deg)
    cheb = np.polynomial.Chebyshev.fit(xx[good], flux_chip[good], deg,
                                       w=sqrt_iv[good])
    cont = cheb(xx)
    scale = np.median(np.abs(cont[np.isfinite(cont)])) if np.any(
        np.isfinite(cont)) else 1.0
    floor = 0.05 * (scale if scale > 0 else 1.0)
    cont = np.where(np.isfinite(cont) & (cont > floor), cont, floor)
    return cont


def continuum_normalize(flux_raw, ivar, deg=4, n_iter=5,
                        low_sigma=1.5, high_sigma=4.0, return_error=False,
                        use_mask=False):
    """Self-consistent continuum normalization of a RAW APOGEE spectrum.

    Divides the raw flux by a per-chip, inverse-variance-weighted, iteratively
    (asymmetrically) sigma-clipped Chebyshev continuum derived from the RAW flux
    ITSELF. NO survey continuum, NO external continuum-pixel mask. See the section
    header above for the full method and rationale.

    Parameters
    ----------
    flux_raw : (8575,) raw flux (mwmStar APOGEE `flux` column; median ~1e3-1e4).
    ivar     : (8575,) inverse variance (mwmStar `ivar`; 0 in gaps / bad pixels).
    deg, n_iter, low_sigma, high_sigma : per-chip fit controls (see
        _fit_chebyshev_chip_sigmaclip).
    return_error : if True, also return the normalized 1/sqrt(ivar) error so the
        detector can build inverse-variance weights on the SAME continuum.
    use_mask : Refinement 3 evaluation toggle. When True (and the mask file is
        present), the per-chip Chebyshev is fit on the FIXED line-list+variance
        continuum-pixel mask (_CONT_MASK) instead of the sigma-clipped continuum.
        OFF by default so the production path (and the SC net trained on it) keeps
        the sigma-clip normalization.

    Returns
    -------
    norm_flux                         if return_error is False
    (norm_flux, norm_err)             if return_error is True

    norm_flux is length 8575, ~1 in the continuum. norm_err is the raw photon
    error divided by the SAME continuum (S/N preserved); pixels with ivar<=0 or
    non-finite flux get norm_err = inf so the detector ignores them (the continuum
    is still defined there, but the data are not trusted).
    """
    flux_raw = np.asarray(flux_raw, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    if flux_raw.size != NPIX or ivar.size != NPIX:
        raise ValueError("continuum_normalize expects length-%d arrays" % NPIX)

    use_mask = bool(use_mask and _HAVE_CONT_MASK)

    # Build the continuum chip by chip; each chip is fit on its own detector so a
    # gap never enters a fit and the three blazes are independent.
    cont = np.empty(NPIX, dtype=float)
    for a, b in CHIP_SLICES:
        if use_mask:
            cont[a:b] = _fit_chebyshev_chip_maskpix(
                WAVELENGTH[a:b], flux_raw[a:b], ivar[a:b],
                _CONT_MASK[a:b], deg=deg)
        else:
            cont[a:b] = _fit_chebyshev_chip_sigmaclip(
                WAVELENGTH[a:b], flux_raw[a:b], ivar[a:b],
                deg=deg, n_iter=n_iter, low_sigma=low_sigma, high_sigma=high_sigma)

    # Normalize. The continuum is floored away from zero per chip, so this is safe.
    norm_flux = flux_raw / cont

    if not return_error:
        return norm_flux

    # Propagate the photon error through the SAME continuum so S/N is preserved.
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = 1.0 / np.sqrt(ivar)                 # ivar<=0 -> inf
    norm_err = sigma / cont
    # Untrusted pixels (no ivar, non-finite flux/continuum) -> err = inf so the
    # inverse-variance chi^2 / f_imp / fits skip them. The flux value is left as
    # computed (the continuum is defined everywhere); only the error gates it.
    bad = (ivar <= 0) | ~np.isfinite(flux_raw) | ~np.isfinite(norm_flux) \
        | ~np.isfinite(norm_err)
    norm_err = np.where(bad, np.inf, norm_err)
    return norm_flux, norm_err


# --------------------------------------------------------------------------- #
# Doppler shift on the APOGEE grid.
# --------------------------------------------------------------------------- #
def _doppler_shift(flux, dv_kms):
    """Shift a spectrum by radial velocity dv (km/s) and resample to the grid.

    Uses the relativistic Doppler convention from binspec
    (sqrt((1 - dv/c)/(1 + dv/c))), positive dv = moving away (redshift). Pixels
    that fall outside the original grid are filled by np.interp edge values.
    """
    factor = np.sqrt((1.0 - dv_kms / C_KMS) / (1.0 + dv_kms / C_KMS))
    new_wl = WAVELENGTH * factor
    return np.interp(new_wl, WAVELENGTH, flux)


def _running_continuum(flux, win=151, pct=95.0):
    """Smooth running upper-envelope continuum used to re-normalize the flux sum.

    A coarse stand-in for binspec's 4th-order per-chip Chebyshev continuum on
    Cannon continuum pixels. We slide a window and take a HIGH PERCENTILE (the
    upper envelope) rather than the median: absorption lines pull a median DOWN
    into the line forest, which would distort the re-normalized spectrum (and
    break the q=1 identity), whereas the upper envelope tracks the true
    continuum level. A wide window keeps the envelope smooth.

    Returns a continuum array of the same shape; floored away from zero so the
    final division cannot blow up.
    """
    flux = np.asarray(flux, dtype=float)
    # Upper-envelope percentile. This was a python strided np.percentile over a
    # (8575, win) view (~11 ms/call), which dominated compose_binary and made the
    # binary fit ~50 s/star. scipy.ndimage.percentile_filter is the SAME running
    # high-percentile in C (~1 ms), so the detector drops ~10x with no change of
    # intent. 'nearest' replicates the edge padding the old code used.
    cont = percentile_filter(flux, percentile=pct, size=win, mode="nearest")
    # Smooth the envelope so it is continuous, not stepwise (boxcar, in C).
    cont = uniform_filter1d(cont, size=win, mode="nearest")
    # Floor away from zero (deep blends / noisy edges) to avoid division spikes.
    scale = np.median(np.abs(cont[cont > 0])) if np.any(cont > 0) else 1.0
    floor = 0.05 * scale
    cont = np.where(cont < floor, floor, cont)
    return cont


# --------------------------------------------------------------------------- #
# 4. Binary composite: El-Badry Eq. 2.
# --------------------------------------------------------------------------- #
def compose_binary_flux(teff1, logg1, feh, q, rv1_kms, rv2_kms,
                        age_gyr=_MS_REPR_AGE_GYR):
    """Return the UN-normalized binary flux sum R1^2 f1 + R2^2 f2 (pre-continuum).

    This is the physical flux sum before the single re-normalization step. It is
    useful for injecting a realistic OBSERVED-like binary spectrum into a test
    (an observed spectrum is un-normalized flux; the detector normalizes it). For
    the model used in fitting, call compose_binary, which normalizes this sum.
    """
    f1 = payne_predict(teff1, logg1, feh)
    teff2, logg2, R2, R1 = secondary_from_q(teff1, logg1, feh, q, age_gyr)
    f2 = payne_predict(teff2, logg2, feh)
    # Un-normalize each component by the PORTED binspec flux-net continuum,
    # evaluated at that component's own (Teff, logg, feh). A cooler secondary gets
    # a redder, weaker H-band continuum, exactly the dilution Eq. 2 needs.
    # q-bias fix (parity with compose_binary5_sc): pseudo_continuum is mean-1 at
    # every Teff, so the cross-Teff surface-brightness LEVEL is divided out. Weight
    # each component by its MIST 2MASS H LUMINOSITY 10^(-0.4 M_H) (folds in BOTH
    # R^2 and the true surface brightness / bolometric correction), REPLACING the
    # old R^2 x continuum-level weighting. Normalized to the primary, the secondary
    # weight is 10^(-0.4 (M_H2 - M_H1)) (=1 at q=1).
    MH1, MH2 = _iso_ours.mh_from_q_ours(teff1, logg1, feh, q, age_gyr)
    w_sec = 10.0 ** (-0.4 * (MH2 - MH1))
    f1_phys = f1 * pseudo_continuum(teff1, logg1, feh)
    f2_phys = f2 * pseudo_continuum(teff2, logg2, feh)
    f1_shift = _doppler_shift(f1_phys, rv1_kms)
    f2_shift = _doppler_shift(f2_phys, rv2_kms)
    return f1_shift + w_sec * f2_shift


def compose_binary(teff1, logg1, feh, q, rv1_kms, rv2_kms,
                   age_gyr=_MS_REPR_AGE_GYR):
    """Compose a normalized binary spectrum via El-Badry et al. (2018b) Eq. 2.

    Path (binspec get_normalized_spectrum_binary):
      1. f1 = payne_predict(Teff1, logg1, feh)  -> primary NORMALIZED spectrum.
      2. (Teff2, logg2, R2, R1) = secondary_from_q(...); the q -> secondary map.
      3. f2 = payne_predict(Teff2, logg2, feh)  -> secondary NORMALIZED spectrum.
      4. Un-normalize each by its trained-flux-net pseudo_continuum (ported from
         binspec) so the two components are in physical flux, NOT normalized space.
      5. Doppler-shift each un-normalized component by its own RV.
      6. Sum in flux weighted by emitting area:
             F = R1^2 * f1_unnorm(shift1) + R2^2 * f2_unnorm(shift2).
         (Eq. 2 is (1/D^2)[R1^2 f1 + R2^2 f2]; the distance D cancels in the
         final re-normalization, so we omit it.)
      7. Re-normalize once by a smooth running-median continuum of the sum.

    This replaces the previous heuristic that blended two ALREADY-normalized
    spectra with a free flux_ratio. The flux ratio is now fixed physically by
    the isochrone (R^2 and the trained-flux-net surface brightness), tying q to the
    line-depth dilution exactly as the paper does.

    Returns a length-8575 normalized composite (~1 in the continuum).
    """
    # Physical flux sum R1^2 f1 + R2^2 f2 (each component un-normalized by its
    # trained-flux-net pseudo-continuum and Doppler-shifted by its RV).
    # Important: do NOT collapse q~1 to a single shifted component unless the two
    # RVs are equal. An equal-mass binary with rv1 != rv2 is still a line-doubled
    # spectrum. The q=1 nesting is preserved by secondary_from_q / mh_from_q
    # returning identical component labels and weights; when rv1 == rv2, the flux
    # sum is just a scaled single-star spectrum and the final normalization cancels
    # the scale exactly.
    flux_sum = compose_binary_flux(
        teff1, logg1, feh, q, rv1_kms, rv2_kms, age_gyr)

    # Re-normalize once by the running upper-envelope continuum of the SUM. This
    # is the single re-normalization step (not per-component), matching binspec.
    cont = _running_continuum(flux_sum)
    return flux_sum / cont


def normalize_like_model(flux, err=None):
    """Re-normalize a spectrum (and its errors) with the SAME continuum routine.

    binspec normalizes the model and the observed spectrum with the IDENTICAL
    continuum routine so any normalization distortion cancels in chi^2 (it is
    not a perfect continuum, but it is the SAME imperfect continuum on both
    sides). The detector must pass the observed spectrum (and a single-star Payne
    model) through this before comparing them to a compose_binary output. This is
    what makes the q=1 / single-star limit self-consistent.

    The continuum is computed from the FLUX; the errors are divided by the SAME
    continuum so the signal-to-noise ratio is preserved after normalization
    (binspec normalizes spec and spec_err together for this reason).

    Returns the normalized flux if err is None, else (norm_flux, norm_err).
    """
    flux = np.asarray(flux, dtype=float)
    cont = _running_continuum(flux)
    if err is None:
        return flux / cont
    return flux / cont, np.asarray(err, dtype=float) / cont


_MODEL_IVAR = np.ones(NPIX, dtype=float)   # uniform weights for model normalization


def _cont_norm_model(flux):
    """Normalize a MODEL flux with the SAME per-chip sigma-clipped Chebyshev that
    continuum_normalize applies to OBSERVED flux, so model and data share one
    continuum operator -- the operator the SC net was TRAINED in.

    WHY (the double-normalization fix, 2026-06-23). The SC net is trained to predict
    continuum_normalize(flux) (per-chip Chebyshev, CN-space). The SC detector used to
    re-normalize both the model and the observed spectrum a SECOND time with
    normalize_like_model (a 151-px running upper-envelope continuum), a DIFFERENT
    operator. On real spectra those two continua do not cancel: the single net fit a
    true single at reduced chi2 ~22 through the running-continuum path but ~5-11 when
    compared in CN-space (same-star A/B). That mismatch -- not the net -- dominated
    chi2_single and corrupted Delta-chi2 / f_imp. Putting obs, the single model, and
    the binary composite ALL through continuum_normalize removes the second,
    inconsistent normalization. Uniform ivar (a model is noiseless) and n_iter=3
    (fewer sigma-clip passes than the observed path needs) keep it fast inside the
    fit loop. The overall scale cancels in the normalization, so the q=1 / equal-RV
    identity (payne5_single_model_sc == compose_binary5_sc(q=1)) is preserved exactly.
    """
    return continuum_normalize(np.asarray(flux, dtype=float), _MODEL_IVAR, n_iter=3,
                               use_mask=SC_USE_MASK)


def mask_bad_pixels(norm_flux, norm_err, snr_cap=200.0,
                    flux_lo=0.1, flux_hi=1.2, chip_edge_guard=True):
    """Return a masked error array for an OBSERVED normalized spectrum.

    APOGEE aspcapStar spectra carry sky-line residuals, chip gaps, and persistence
    artifacts that show up as normalized-flux SPIKES (up to ~1.4) or drops to ~0.
    El-Badry et al. 2018b mask these with the APOGEE bitmask and S/N cuts before
    fitting; we do not cache the bitmask, so we reproduce the effect from the flux
    and error themselves. Three operations:

      1. S/N>200 cap: floor the error at norm_flux/200 so a few very-high-S/N
         pixels cannot dominate chi^2 (apogee_binaries.tex line 144).
      2. Bad-pixel reject: set the error to +inf where the spectrum is non-finite,
         already has err<=0, or the normalized flux falls outside [flux_lo, flux_hi]
         (a real absorption line stays well inside this; a value of 0 or 1.4 is an
         artifact). chi2 / f_imp / the least_squares fits all skip err=inf pixels,
         so this removes the artifacts WITHOUT changing any call signature.
      3. Chip-edge guard (chip_edge_guard=True, default): set the error to +inf on
         the fixed chip-edge guard mask (chip_edge_guard()), the pixels at each
         chip data-region edge where the SC model itself predicts flux ~ 0 (the
         net learned the zero-padded training shoulders) while the observed flux is
         real ~ 1. That model/data step is the DOMINANT chip-edge false positive:
         the velocity-shifted binary secondary fills it and buys spurious
         Delta-chi2 / f_imp. Masking it via the OBSERVED error drops the SAME
         pixels from BOTH the single and the binary chi2 AND the f_imp sums (they
         cancel) AND from both least_squares fits, so it removes only the artifact
         and preserves the binary-nests-single identity (Delta-chi2 >= 0). See the
         chip-edge guard block comment above CHIP_GUARD_FRAC.

    (The OLD Refinement 2 -- a FIXED symmetric guard at the CHIP_EDGES gap midpoints
    plus a quadrature error floor -- regressed contamination because it masked the
    wrong, symmetric pixels; the guard above is model-derived and asymmetric.)

    Returns the modified error array (same shape). Models are never masked; only
    the observed error is, because the mask encodes which DATA pixels to trust.
    """
    f = np.asarray(norm_flux, dtype=float)
    e = np.asarray(norm_err, dtype=float).copy()
    # 1. S/N>200 cap: err must be at least flux/200 (0.5% of the flux level).
    floor = np.abs(f) / snr_cap
    e = np.maximum(e, floor)
    # 2. reject artifacts and invalid pixels -> err = inf (ignored downstream).
    bad = ~np.isfinite(f) | ~np.isfinite(e) | (e <= 0) | (f < flux_lo) | (f > flux_hi)
    e[bad] = np.inf
    # 3. chip-edge guard: exclude the model-derived chip-edge band (see docstring).
    if chip_edge_guard:
        e[chip_edge_guard_mask()] = np.inf
    return e


# --------------------------------------------------------------------------- #
# 5. Inverse-variance chi-squared.
# --------------------------------------------------------------------------- #
def chi2(obs_flux, obs_err, model_flux, mask=None):
    """Inverse-variance chi^2 = sum(((obs - model)/err)^2) over good pixels.

    Good pixels are those with finite obs, model, and err, and err > 0. If a mask
    is supplied (True = use the pixel), it is combined with the good-pixel test.
    This is the binspec convention (utils.get_chi2_difference); it fixes our old
    UNWEIGHTED sum, which let high-noise / sky pixels dominate.
    """
    obs = np.asarray(obs_flux, dtype=float)
    err = np.asarray(obs_err, dtype=float)
    mod = np.asarray(model_flux, dtype=float)
    good = (np.isfinite(obs) & np.isfinite(err) & np.isfinite(mod) & (err > 0))
    if mask is not None:
        good &= np.asarray(mask, dtype=bool)
    if not np.any(good):
        return float("inf")
    r = (obs[good] - mod[good]) / err[good]
    return float(np.sum(r * r))


# --------------------------------------------------------------------------- #
# 6. Improvement fraction f_imp (El-Badry et al. 2018b, Eq. B1).
#    The binspec repo provides only the Delta-chi2 building block; this statistic
#    is defined in the paper, not the code.
# --------------------------------------------------------------------------- #
def f_imp(obs, single_model, binary_model, err, mask=None):
    """Improvement fraction f_imp, El-Badry et al. 2018b Eq. B1, VERBATIM.

    Quoted from the paper source (apogee_binaries.tex, lines 974-977):

        f_imp = sum_lambda [ (|f_single - f| - |f_binary - f|) / sigma ]
                ---------------------------------------------------------
                       sum_lambda [ |f_single - f_binary| / sigma ]

    where f is the observed normalized flux, f_single and f_binary the best-fit
    single-star and binary models, sigma the per-pixel flux uncertainty, and the
    sums run over all (good) wavelength pixels. NOTE the weighting is 1/sigma, NOT
    1/sigma^2, and the statistic is built from ABSOLUTE residuals, not a dot
    product. An earlier version here used a 1/sigma^2 projection sum(d*s)/sum(d^2);
    that was NOT Eq. B1 and is replaced.

    Reading of the statistic: the numerator is the total (inverse-sigma weighted)
    REDUCTION in absolute residual that the binary model buys over the single
    model; the denominator is the total absolute model difference between them.
    f_imp -> 1 means essentially every bit by which the binary model departs from
    the single model lands on top of the data (a real improvement spread across
    many pixels); f_imp near 0 (or negative) means the binary model differs from
    the single model in directions that do not track the data, so its Delta-chi2
    is not trustworthy. This is exactly why Table B1 pairs a Delta-chi2 cut with a
    minimum f_imp: it rejects large-Delta-chi2-but-spurious binary solutions.

    Returns a float; 0.0 when the two models are identical over good pixels.
    """
    obs = np.asarray(obs, dtype=float)
    single = np.asarray(single_model, dtype=float)
    binary = np.asarray(binary_model, dtype=float)
    err = np.asarray(err, dtype=float)
    good = (np.isfinite(obs) & np.isfinite(single) & np.isfinite(binary)
            & np.isfinite(err) & (err > 0))
    if mask is not None:
        good &= np.asarray(mask, dtype=bool)
    if not np.any(good):
        return 0.0
    inv_sigma = 1.0 / err[good]                       # 1/sigma weighting (Eq. B1)
    resid_single = np.abs(single[good] - obs[good])   # |f_single - f|
    resid_binary = np.abs(binary[good] - obs[good])   # |f_binary - f|
    model_diff = np.abs(single[good] - binary[good])  # |f_single - f_binary|
    numer = np.sum((resid_single - resid_binary) * inv_sigma)
    denom = np.sum(model_diff * inv_sigma)
    if denom <= 0:
        return 0.0
    return float(numer / denom)


# --------------------------------------------------------------------------- #
# 6b. Table B1 acceptance (El-Badry et al. 2018b, apogee_binaries.tex 992-1000).
#     A binary is accepted only if Delta-chi2 clears 300 AND f_imp clears the
#     minimum for its Delta-chi2 bin. This SLIDING scale replaces a single fixed
#     (Delta-chi2, f_imp) pair: a larger Delta-chi2 tolerates a smaller f_imp,
#     because a big, well-spread improvement is itself evidence. Below 300, or
#     below the bin's minimum f_imp, the system is inconclusive (treated as a
#     non-detection here). For multi-epoch fits the paper scales the Delta-chi2
#     axis as 300 * N_epochs per step in model complexity; single-epoch here.
# --------------------------------------------------------------------------- #
# (lower Delta-chi2 bound inclusive, minimum f_imp) sorted high -> low.
TABLE_B1 = [
    (3000.0, 0.0), (2500.0, 0.05), (2000.0, 0.075), (1500.0, 0.10),
    (1000.0, 0.125), (750.0, 0.15), (600.0, 0.175), (450.0, 0.20), (300.0, 0.225),
]

# DR19 positive-f_imp floor (validated 2026-06 on the verify set).
#
# Table B1's top bins (Delta-chi2 >= 3000 / 2500 / 2000) allow f_imp >= 0 / 0.05 /
# 0.075. On the DR19 verify set the contaminating controls land in exactly these
# bins: a high-SNR or chip-edge mismatch drives a large Delta-chi2 while f_imp
# stays barely above 0 (false positives at f_imp = 0.054 / 0.013). Their positive
# Delta-chi2 piles onto a handful of chip-boundary pixels where the single model
# predicts flux ~ 0 against observed ~ 1, so the improvement is spurious, not a
# spread-out second spectrum; real SB2 detections here carry f_imp 0.09 - 0.45.
# We therefore require a small POSITIVE f_imp wherever the table would accept
# f_imp < this floor. The exception is an EXTREME Delta-chi2: above
# FIMP_FLOOR_WAIVE_DCHI2 even a tiny f_imp reflects a pervasive real mismatch (the
# one true SB2 with f_imp = 0.008 sits at Delta-chi2 = 7.1e5), so the floor is
# waived there, as Table B1's own logic treats a huge spread improvement as
# evidence.
#
# CHIP-EDGE FIX RE-CALIBRATION. The 0.06 floor
# was calibrated against the artifact-PRESENT f_imp distribution. The chip-edge
# guard (mask_bad_pixels) now removes the chip-edge pixels from BOTH chi2 and
# f_imp, so the Delta-chi2 and f_imp are HONEST: artifact-dominated controls that
# used to carry Delta-chi2 ~ 1e5 with strongly NEGATIVE f_imp (the edge dragged
# f_imp down, which is what kept them below the 0.06 floor) collapse to Delta-chi2
# ~ 1e3-1e4 once the edge is masked, and their f_imp recomputes over the clean
# pixels. The floor must therefore be re-derived on the cleaned distribution. The
# guard-ON operating curve on the 25 SB2 / 150 val-set controls is:
#   floor 0.06 -> 80% / 16.7% ; 0.10 -> 80% / 12.0% ; 0.12 -> 76% / 10.7% ;
#   floor 0.14 -> 76% / 8.7%  ; 0.15 -> 72% / 6.7% .
# 0.14 holds completeness at the documented 76% baseline while dropping the FPR
# from the pre-fix 10.7% to 8.7% -- strictly better on both axes than the pre-fix
# (guard-off) operating point. It is the chosen floor.
FIMP_FLOOR = 0.14            # minimum trustworthy f_imp on the chip-edge-guarded f_imp
FIMP_FLOOR_WAIVE_DCHI2 = 1.0e5  # above this Delta-chi2 the floor is waived

# S/N cap for the SC detector's error floor (mask_bad_pixels). El-Badry/binspec use
# 200. We tested matching the floor to OUR ~1% net accuracy (cap 100/70): cap=100
# barely moved the control FPR (19.2->17.5%) because lowering the cap rescales ALL
# Delta-chi2 down (false-positive AND real-binary), so against the fixed Table B1
# threshold the population SEPARATION does not improve; cap=70 flips the false
# positives but also pushes real detections below Delta-chi2=300. A global error
# floor cannot fix a population OVERLAP -- only net accuracy can. Kept at the
# binspec-standard 200; the parameter stays configurable for diagnostics.
SC_SNR_CAP = 200.0

# Continuum operator for the SC path. False = per-chip sigma-clipped Chebyshev
# (continuum_normalize default). True = fit the Chebyshev on the FIXED continuum-
# pixel mask (models/continuum_pixels.npz, ~ binspec get_apogee_continuum).
#
# TESTED AND REJECTED (2026-06-23): on OUR 8575-pixel grid the fixed mask is too sparse
# per chip, so the per-chip deg-4 Chebyshev on the mask pixels is ill-conditioned and
# blows up -- the mask-normalized flux differs from the sigma-clip flux by RMS 2.7-5
# (on a ~1-level spectrum!) for a large fraction of stars, and a net trained on it fit
# ~14x worse (monitor 21 vs 1.5). The continuum-pixel-mask idea needs a denser/rebuilt
# mask (build_continuum_pixels.py) or a sigma-clip fallback where the mask is sparse
# before it is usable. Left FALSE (sigma-clip) -- the working operator. MUST match how
# the SC net was trained (train_payne_dr19_sc uses physics.SC_USE_MASK).
SC_USE_MASK = False


def table_b1_min_fimp(delta_chi2):
    """Minimum f_imp required at this Delta-chi2, or None if Delta-chi2 < 300.

    This is the El-Badry et al. 2018b Table B1 sliding scale, with a DR19 positive
    f_imp floor (FIMP_FLOOR) applied so a large Delta-chi2 alone can never accept a
    near-zero-f_imp binary (the verified DR19 contamination mode); the floor is
    waived only at an extreme Delta-chi2 (FIMP_FLOOR_WAIVE_DCHI2), where a tiny
    f_imp still reflects a pervasive real mismatch. See the block comment above.
    """
    for lo, min_fimp in TABLE_B1:
        if delta_chi2 >= lo:
            if delta_chi2 >= FIMP_FLOOR_WAIVE_DCHI2:
                return min_fimp                  # extreme Delta-chi2: table verbatim
            return max(min_fimp, FIMP_FLOOR)     # else enforce the positive floor
    return None


def passes_table_b1(delta_chi2, fimp):
    """True iff (delta_chi2, fimp) clears the El-Badry 2018b Table B1 scale."""
    min_fimp = table_b1_min_fimp(delta_chi2)
    return (min_fimp is not None) and (fimp >= min_fimp)


# Shared binary-search starts. The coadd high-q systems are often near-equal-mass
# pairs with small velocity separations (a few to ~20 km/s), so a grid containing
# only 0/45/90 km/s misses the narrow RV basin. Keep both the coarse large-split
# starts and fine near-zero starts, then force the best high-q seeds into the
# least_squares polish stage.
BINARY_Q_SCAN = np.array([0.30, 0.45, 0.60, 0.75, 0.87, 0.94, 0.99])
BINARY_RV_SCAN = np.array([-90.0, -45.0, -15.0, 0.0, 15.0, 45.0, 90.0])
_EXTRA_HIGH_Q_SEEDS = (
    (0.99, 0.0, 0.0),
    (0.99, -15.0, 15.0),
    (0.99, 15.0, -15.0),
)


def _select_binary_polish_seeds(scanned, top_n=6, high_q_min=0.85,
                                high_q_per_q=2):
    """Choose deterministic q/RV starts to polish after the coarse grid scan.

    `scanned` is a chi2-sorted list of (chi2, q, rv1, rv2). We keep the best
    overall starts, but also force the best starts at each high-q grid value into
    the polish list. Without that per-q retention, the q~0.9-1.0 basin can be
    absent from least_squares even when it exists in the grid, because lower-q
    coarse starts often have slightly better chi2 before the primary labels are
    refined.
    """
    selected = []
    seen = set()

    def add(q, rv1, rv2):
        key = (round(float(q), 6), round(float(rv1), 3), round(float(rv2), 3))
        if key in seen:
            return
        seen.add(key)
        selected.append((float(q), float(rv1), float(rv2)))

    for _, q0, rv1_0, rv2_0 in scanned[:top_n]:
        add(q0, rv1_0, rv2_0)

    q_values = sorted({round(float(row[1]), 6) for row in scanned})
    for qv in q_values:
        if qv < high_q_min:
            continue
        rows = [row for row in scanned if abs(float(row[1]) - qv) < 5e-7]
        for _, q0, rv1_0, rv2_0 in rows[:high_q_per_q]:
            add(q0, rv1_0, rv2_0)

    for q0, rv1_0, rv2_0 in _EXTRA_HIGH_Q_SEEDS:
        add(q0, rv1_0, rv2_0)

    return selected


# --------------------------------------------------------------------------- #
# 7. scipy least-squares fits (REQUIREMENT 1).
#    binspec fits with scipy.optimize.curve_fit on the per-pixel residual vector
#    (obs - model)/err, method='trf' (fitting.fit_all_p0s). We do the SAME with
#    scipy.optimize.least_squares (the engine curve_fit wraps) on the residual
#    vector r_i = (obs_i - model_i)/err_i. least_squares uses the Levenberg-
#    Marquardt / trust-region (trf) Jacobian, which converges far faster and more
#    robustly than the Nelder-Mead simplex on chi^2 the old server used, and the
#    numpy Payne forward (above) keeps each evaluation torch-free.
# --------------------------------------------------------------------------- #
def _single_model(teff, logg, feh):
    """Single-star NORMALIZED model on the same flux->normalize path as the binary.

    Predict the normalized Payne flux, un-normalize by the trained-flux-net
    continuum, then re-normalize with the shared continuum routine. This places
    the single model and the binary model in the same normalized-flux space (the
    single star is the q->1 / primary-only limit of the binary).
    """
    flux = payne_predict(teff, logg, feh) * pseudo_continuum(teff, logg, feh)
    return normalize_like_model(flux)


def _good_mask(obs, err, *models):
    """Boolean mask of pixels finite in obs, err (>0), and every model passed."""
    good = np.isfinite(obs) & np.isfinite(err) & (err > 0)
    for m in models:
        good &= np.isfinite(m)
    return good


def fit_single(obs, err, teff0, logg0, feh0):
    """Fit a single Payne model with scipy least_squares (trf) on (obs-model)/err.

    Optimizes (Teff, logg, [Fe/H]) inside the trained label box. obs is expected
    to already be normalized through normalize_like_model; the model uses the same
    routine so the two are self-consistent (binspec convention). Replaces the old
    Nelder-Mead-on-chi^2 fit. Returns (labels, normalized_model, chi2).
    """
    lmin = np.asarray(_LMIN, dtype=float)
    lmax = np.asarray(_LMAX, dtype=float)
    x0 = np.clip(np.array([teff0, logg0, feh0], float), lmin, lmax)
    obs = np.asarray(obs, float)
    err = np.asarray(err, float)

    # Residual vector r = (obs - model)/err over good pixels (zeros elsewhere so
    # the vector length stays fixed, which least_squares requires). This is the
    # exact quantity curve_fit minimizes the sum-of-squares of.
    base_good = np.isfinite(obs) & np.isfinite(err) & (err > 0)

    def resid(p):
        model = _single_model(p[0], p[1], p[2])
        good = base_good & np.isfinite(model)
        r = np.zeros_like(obs)
        r[good] = (obs[good] - model[good]) / err[good]
        return r

    # trf honors the box bounds directly; ftol/xtol match binspec's 5e-4 tol.
    res = least_squares(resid, x0, method="trf", bounds=(lmin, lmax),
                        ftol=5e-4, xtol=5e-4, max_nfev=200)
    p = np.clip(res.x, lmin, lmax)
    model = _single_model(p[0], p[1], p[2])
    return p, model, chi2(obs, err, model)


def fit_binary(obs, err, teff1, logg1, feh, age_gyr=_MS_REPR_AGE_GYR,
               fit_primary=False):
    """Fit the binary model with scipy least_squares, grid-seeded over q + RV.

    Fits the mass ratio q and the two RVs on the per-pixel residual
    (obs - model)/err with trf, exactly as binspec fits its binary labels with
    curve_fit. When fit_primary is True it ALSO fits the primary (Teff, logg, feh)
    jointly, which matters for two reasons: a diluted SB2 biases a single-star fit,
    and the binary must be able to reach the single-star solution at q->1 (else a
    true single would spuriously prefer the binary, and a true binary that needs
    slightly different primary labels than the single fit could not improve on it).

    Because the q chi^2 surface is BIMODAL and the RV basin is narrow, we pre-scan
    a coarse (q, rv1, rv2) grid to locate the basin, then polish the best seeds.

    Parameters: teff1/logg1/feh seed the primary; feh and Teff are clamped into the
    isochrone MS range. Returns (best_params_dict, normalized_model, chi2), where
    best_params_dict has teff1, logg1, feh, q, rv1, rv2.
    """
    obs = np.asarray(obs, float)
    err = np.asarray(err, float)
    base_good = np.isfinite(obs) & np.isfinite(err) & (err > 0)

    # Keep the primary Teff/feh inside the isochrone-network MS training box.
    # The [Fe/H] limits come from the ported Teff2/logg2 network's own label box
    # (_ISO_LABEL_LO/_ISO_LABEL_HI index 2), so the secondary map is never
    # extrapolated in metallicity.
    feh_lo, feh_hi = float(_ISO_LABEL_LO[2]), float(_ISO_LABEL_HI[2])
    teff1 = float(np.clip(teff1, 4500.0, 6800.0))
    logg1 = float(np.clip(logg1, 3.5, 5.0))
    feh = float(np.clip(feh, feh_lo, feh_hi))

    def _model(params):
        # params layouts:
        #   fit_primary : [teff1, logg1, feh, q, rv1, rv2]
        #   else        : [q, rv1, rv2]
        if fit_primary:
            tp, gp, hp, q, rv1, rv2 = params
        else:
            tp, gp, hp = teff1, logg1, feh
            q, rv1, rv2 = params
        return compose_binary(float(tp), float(gp), float(hp),
                              float(q), float(rv1), float(rv2), age_gyr)

    def resid(params):
        try:
            model = _model(params)
        except ValueError:
            # Out-of-grid secondary: return a large residual so trf backs off.
            return np.full(obs.shape, 1e3)
        good = base_good & np.isfinite(model)
        r = np.zeros_like(obs)
        r[good] = (obs[good] - model[good]) / err[good]
        return r

    # Bounds. q in (0, 1]; RVs within +/-150 km/s; primary labels inside the MS
    # isochrone box when we fit them.
    if fit_primary:
        lo = [4500.0, 3.5, feh_lo, 0.1, -150.0, -150.0]
        hi = [6800.0, 5.0, feh_hi, 1.0, 150.0, 150.0]
    else:
        lo = [0.1, -150.0, -150.0]
        hi = [1.0, 150.0, 150.0]

    # The RV chi^2 basin is NARROW (~tens of km/s wide) and the q surface is
    # bimodal, so a finite-difference trf from a far seed cannot cross the RV
    # barrier. We therefore PRE-SCAN a coarse (q, rv1, rv2) grid (the binspec
    # grid-initialization idea) at the seed primary labels to find the basin, keep
    # the best few seeds, then polish each with least_squares (trf). This combines
    # a robust global search with the fast, accurate local trf refinement
    # REQUIREMENT 1 asks for. The primary labels enter trf as free parameters.
    q_scan = BINARY_Q_SCAN
    rv_scan = BINARY_RV_SCAN

    def _scan_model(q0, rv1_0, rv2_0):
        return compose_binary(teff1, logg1, feh, float(q0),
                              float(rv1_0), float(rv2_0), age_gyr)

    seeds = []
    for q0 in q_scan:
        for rv1_0 in rv_scan:
            for rv2_0 in rv_scan:
                try:
                    m = _scan_model(q0, rv1_0, rv2_0)
                except ValueError:
                    continue
                seeds.append((chi2(obs, err, m), q0, rv1_0, rv2_0))
    seeds.sort(key=lambda t: t[0])

    # Polish a compact but q-aware seed list with trf. Keeping the best overall
    # starts alone loses high-q, small-RV basins; _select_binary_polish_seeds
    # explicitly retains them while still seeding the q->1 nesting solution.
    polish_seeds = _select_binary_polish_seeds(seeds)

    best_chi2 = np.inf
    best_params = None
    best_model = None
    for q0, rv1_0, rv2_0 in polish_seeds:
        x0 = ([teff1, logg1, feh, q0, rv1_0, rv2_0] if fit_primary
              else [q0, rv1_0, rv2_0])
        try:
            res = least_squares(resid, x0, method="trf",
                                bounds=(lo, hi), ftol=5e-4, xtol=5e-4,
                                max_nfev=150)
        except Exception:
            continue
        model = _model(res.x)
        c = chi2(obs, err, model)
        if c < best_chi2:
            best_chi2 = c
            best_params = res.x
            best_model = model

    # SINGLE-STAR FLOOR. The q=1, equal-RV binary is the single star, so the
    # binary chi^2 can never physically be worse than the best single fit at the
    # same labels. trf can nonetheless descend from the q->1 seed into a shallow
    # worse q/RV minimum and report it as "best". We guard against that: evaluate
    # the exact q=1, rv1=rv2=0 model at the seed primary labels (= the single fit)
    # and keep it if it beats every polished candidate. This makes Delta-chi2 >= 0
    # for a true single (delta ~ 0) and lets a real binary win only when it
    # genuinely improves the fit.
    floor_model = compose_binary(teff1, logg1, feh, 1.0, 0.0, 0.0, age_gyr)
    floor_chi2 = chi2(obs, err, floor_model)
    if floor_chi2 < best_chi2 or best_params is None:
        out = {"teff1": teff1, "logg1": logg1, "feh": feh,
               "q": 1.0, "rv1": 0.0, "rv2": 0.0}
        return out, floor_model, floor_chi2

    if fit_primary:
        tp, gp, hp, q, rv1, rv2 = best_params
    else:
        tp, gp, hp = teff1, logg1, feh
        q, rv1, rv2 = best_params
    out = {"teff1": float(tp), "logg1": float(gp), "feh": float(hp),
           "q": float(q), "rv1": float(rv1), "rv2": float(rv2)}
    return out, best_model, best_chi2


# =========================================================================== #
# 7b. OUR DR19 5-label single-vs-binary detector, SURVEY-normalized (the
#     model='dr19' comparison path; the PRODUCTION path is the SC detector in 7c).
#
#     The single-star SPECTRAL model is OUR net (payne_predict5), trained on DR19
#     dwarfs, NOT binspec's vendored net. The isochrone q -> (Teff2, logg2, R1, R2)
#     map (secondary_from_q) is OUR OWN MIST v1.2 interpolation (isochrone_mist).
#     The un-normalization continuum here is the binspec-ported flux net
#     (pseudo_continuum); the production SC path (7c) uses OUR pseudo_continuum_sc
#     instead. Everything else mirrors the binspec oracle so the two are comparable:
#       - the binary composite is El-Badry Eq. 2 (R1^2 f1 + R2^2 f2, summed in
#         flux, re-normalized) with OUR net giving f1 and f2,
#       - the single fit optimizes the FIVE labels via scipy least_squares on
#         (obs - model)/err,
#       - the binary fit grid-seeds q + the two RVs, polishes with least_squares,
#         and refines the primary (its five labels) jointly,
#       - the same bad-pixel masking (mask_bad_pixels),
#       - the same binary-nests-single floor (Delta-chi2 >= 0),
#       - the EXACT El-Badry Eq. B1 f_imp and the Table B1 sliding thresholds.
#     The fit runs ON OUR 8575-pixel grid (no interpolation to binspec's 7214),
#     because OUR net predicts there directly.
# =========================================================================== #

def payne5_single_model(teff, logg, feh, mgh, vmacro):
    """Single-star NORMALIZED model from OUR 5-label net, on the binary's path.

    Predict the normalized flux with payne_predict5, un-normalize by the ported
    binspec trained-flux-net continuum (pseudo_continuum, evaluated at this star's
    Teff/logg/feh), then re-normalize with the shared running continuum. This
    places OUR single model in the SAME normalized-flux space as the binary
    composite (the single star is the q->1 / primary-only limit of the binary), so
    Delta-chi2 is self-consistent.
    """
    flux = payne_predict5(teff, logg, feh, mgh, vmacro) \
        * pseudo_continuum(teff, logg, feh)
    return normalize_like_model(flux)


def compose_binary5(teff1, logg1, feh, mgh, vmacro, q, rv1_kms, rv2_kms,
                    age_gyr=_MS_REPR_AGE_GYR):
    """El-Badry Eq. 2 binary composite using OUR DR19 5-label net.

    Path (mirrors compose_binary, but the single-star spectra come from OUR net):
      1. f1 = payne_predict5(Teff1, logg1, [Fe/H], [Mg/H], v_macro) -> primary
         NORMALIZED spectrum.
      2. (Teff2, logg2, R2, R1) = secondary_from_q(...): OUR MIST v1.2 isochrone
         map (equal-age, equal-composition MS pair), no binspec net.
      3. f2 = payne_predict5(Teff2, logg2, [Fe/H], [Mg/H], v_macro): the secondary
         shares the system's [Fe/H] and [Mg/H] (same chemistry) and we reuse the
         primary's v_macro as the secondary broadening (a faint-secondary
         nuisance; the flux ratio, not v_macro2, drives the dilution).
      4. un-normalize each component by its ported pseudo_continuum (physical flux),
      5. Doppler-shift each by its RV,
      6. sum F = R1^2 f1(shift1) + R2^2 f2(shift2) (Eq. 2; distance cancels),
      7. re-normalize once by the running continuum of the sum.

    The q=1, equal-RV limit nests the exact single model at the requested labels.
    A q=1 binary with rv1 != rv2 is still a valid equal-mass, line-doubled
    spectrum, so this path must not collapse near-equal-mass systems to one
    shifted component. Returns a length-8575 normalized composite.
    """
    f1 = payne_predict5(teff1, logg1, feh, mgh, vmacro)
    teff2, logg2, R2, R1 = secondary_from_q(teff1, logg1, feh, q, age_gyr)
    # Secondary shares [Fe/H], [Mg/H]; reuse the primary's v_macro broadening.
    f2 = payne_predict5(teff2, logg2, feh, mgh, vmacro)
    # q-bias fix (parity with compose_binary5_sc): pseudo_continuum is also mean-1
    # at every Teff (see _pseudo_continuum_cached: cont/cont.mean()), so it drops
    # the cross-Teff surface-brightness level. Weight each component by its MIST
    # 2MASS H LUMINOSITY 10^(-0.4 M_H) -- which folds in BOTH the emitting area
    # (R^2) and the real surface brightness / bolometric correction -- so this
    # REPLACES the old R^2 x continuum-level weighting. Normalized to the primary,
    # the secondary weight is 10^(-0.4 (M_H2 - M_H1)) (=1 at q=1, identity preserved
    # by secondary_from_q / mh_from_q returning identical component labels and
    # weights at q~1. If rv1 == rv2, the final normalization cancels the factor of
    # two and the model nests the single star exactly; if rv1 != rv2, it remains a
    # valid equal-mass line-doubled binary.
    MH1, MH2 = _iso_ours.mh_from_q_ours(teff1, logg1, feh, q, age_gyr)
    w_sec = 10.0 ** (-0.4 * (MH2 - MH1))
    f1_phys = f1 * pseudo_continuum(teff1, logg1, feh)
    f2_phys = f2 * pseudo_continuum(teff2, logg2, feh)
    f1_shift = _doppler_shift(f1_phys, rv1_kms)
    f2_shift = _doppler_shift(f2_phys, rv2_kms)
    flux_sum = f1_shift + w_sec * f2_shift
    return flux_sum / _running_continuum(flux_sum)


def fit_single5(obs, err, teff0, logg0, feh0, mgh0, vmacro0):
    """Fit OUR 5-label single-star model with scipy least_squares (trf).

    Optimizes (Teff, logg, [Fe/H], [Mg/H], v_macro) inside OUR net's trained label
    box on the per-pixel residual (obs - model)/err. obs is expected to already be
    normalized through normalize_like_model; the model uses the same routine, so
    the two are self-consistent. Returns (labels5, normalized_model, chi2).
    """
    lmin = _LMIN5.astype(float)
    lmax = _LMAX5.astype(float)
    x0 = np.clip(np.array([teff0, logg0, feh0, mgh0, vmacro0], float),
                 lmin, lmax)
    obs = np.asarray(obs, float)
    err = np.asarray(err, float)
    base_good = np.isfinite(obs) & np.isfinite(err) & (err > 0)

    def resid(p):
        model = payne5_single_model(p[0], p[1], p[2], p[3], p[4])
        good = base_good & np.isfinite(model)
        r = np.zeros_like(obs)
        r[good] = (obs[good] - model[good]) / err[good]
        return r

    res = least_squares(resid, x0, method="trf", bounds=(lmin, lmax),
                        ftol=5e-4, xtol=5e-4, max_nfev=200)
    p = np.clip(res.x, lmin, lmax)
    model = payne5_single_model(p[0], p[1], p[2], p[3], p[4])
    return p, model, chi2(obs, err, model)


def fit_binary5(obs, err, teff1, logg1, feh, mgh, vmacro,
                age_gyr=_MS_REPR_AGE_GYR):
    """Fit OUR 5-label binary model with scipy least_squares, grid-seeded over q+RV.

    Mirrors fit_binary, but the single-star components come from OUR net and the
    primary carries all FIVE labels. Pre-scans a coarse (q, rv1, rv2) grid to find
    the narrow RV basin / bimodal q surface, polishes the best seeds with trf, and
    refines the primary's five labels jointly (a diluted SB2 biases a single-star
    fit, and the binary must be able to reach the single-star solution at q->1).

    Returns (best_params_dict, normalized_model, chi2) with keys teff1, logg1,
    feh, mgh, vmacro, q, rv1, rv2.
    """
    obs = np.asarray(obs, float)
    err = np.asarray(err, float)
    base_good = np.isfinite(obs) & np.isfinite(err) & (err > 0)

    # Keep the primary inside BOTH OUR net's box and the isochrone MS box. The
    # isochrone Teff2/logg2 network is only valid for MS primaries, so its [Fe/H]
    # limits gate the secondary map; OUR net's box gates the spectral prediction.
    feh_lo = max(float(_ISO_LABEL_LO[2]), float(_LMIN5[2]))
    feh_hi = min(float(_ISO_LABEL_HI[2]), float(_LMAX5[2]))
    t_lo = max(4500.0, float(_LMIN5[0]))
    t_hi = min(6800.0, float(_LMAX5[0]))
    g_lo = max(3.5, float(_LMIN5[1]))
    g_hi = min(5.0, float(_LMAX5[1]))
    mg_lo, mg_hi = float(_LMIN5[3]), float(_LMAX5[3])
    vm_lo, vm_hi = float(_LMIN5[4]), float(_LMAX5[4])

    teff1 = float(np.clip(teff1, t_lo, t_hi))
    logg1 = float(np.clip(logg1, g_lo, g_hi))
    feh = float(np.clip(feh, feh_lo, feh_hi))
    mgh = float(np.clip(mgh, mg_lo, mg_hi))
    vmacro = float(np.clip(vmacro, vm_lo, vm_hi))

    def _model(params):
        tp, gp, hp, mp, vp, q, rv1, rv2 = params
        return compose_binary5(float(tp), float(gp), float(hp), float(mp),
                               float(vp), float(q), float(rv1), float(rv2),
                               age_gyr)

    def resid(params):
        try:
            model = _model(params)
        except ValueError:
            return np.full(obs.shape, 1e3)
        good = base_good & np.isfinite(model)
        r = np.zeros_like(obs)
        r[good] = (obs[good] - model[good]) / err[good]
        return r

    # Bounds: five primary labels + q in (0,1] + two RVs.
    lo = [t_lo, g_lo, feh_lo, mg_lo, vm_lo, 0.1, -150.0, -150.0]
    hi = [t_hi, g_hi, feh_hi, mg_hi, vm_hi, 1.0, 150.0, 150.0]

    # Coarse (q, rv1, rv2) grid at the SEED labels to locate the basin.
    q_scan = BINARY_Q_SCAN
    rv_scan = BINARY_RV_SCAN
    seeds = []
    for q0 in q_scan:
        for rv1_0 in rv_scan:
            for rv2_0 in rv_scan:
                try:
                    m = compose_binary5(teff1, logg1, feh, mgh, vmacro,
                                        float(q0), float(rv1_0), float(rv2_0),
                                        age_gyr)
                except ValueError:
                    continue
                seeds.append((chi2(obs, err, m), q0, rv1_0, rv2_0))
    seeds.sort(key=lambda t: t[0])
    polish_seeds = _select_binary_polish_seeds(seeds)

    best_chi2 = np.inf
    best_params = None
    best_model = None
    for q0, rv1_0, rv2_0 in polish_seeds:
        x0 = [teff1, logg1, feh, mgh, vmacro, q0, rv1_0, rv2_0]
        try:
            res = least_squares(resid, x0, method="trf", bounds=(lo, hi),
                                ftol=5e-4, xtol=5e-4, max_nfev=150)
        except Exception:
            continue
        model = _model(res.x)
        c = chi2(obs, err, model)
        if c < best_chi2:
            best_chi2 = c
            best_params = res.x
            best_model = model

    # Binary-nests-single floor: evaluate the exact q=1 model at the seed labels
    # (= the single fit) and keep it if it beats every polished candidate, so a
    # true single sits at Delta-chi2 ~ 0 and never spuriously prefers the binary.
    floor_model = compose_binary5(teff1, logg1, feh, mgh, vmacro, 1.0, 0.0, 0.0,
                                  age_gyr)
    floor_chi2 = chi2(obs, err, floor_model)
    if floor_chi2 < best_chi2 or best_params is None:
        out = {"teff1": teff1, "logg1": logg1, "feh": feh, "mgh": mgh,
               "vmacro": vmacro, "q": 1.0, "rv1": 0.0, "rv2": 0.0}
        return out, floor_model, floor_chi2

    tp, gp, hp, mp, vp, q, rv1, rv2 = best_params
    out = {"teff1": float(tp), "logg1": float(gp), "feh": float(hp),
           "mgh": float(mp), "vmacro": float(vp), "q": float(q),
           "rv1": float(rv1), "rv2": float(rv2)}
    return out, best_model, best_chi2


def dr19_single_vs_binary(wl8575, flux8575, err8575, seed=None):
    """OUR survey-normalized 5-label single-vs-binary detector (model='dr19').

    Given a cached DR19 spectrum on physics.WAVELENGTH (the grid OUR net predicts
    on), decide single vs binary using OUR survey-normalized DR19 5-label net as
    the single-star forward model. The production path is dr19_sc_single_vs_binary
    (7c). Steps:

      1. Normalize the observed flux + error with the SHARED continuum routine
         (normalize_like_model) and mask bad pixels (mask_bad_pixels: S/N>200 cap
         + reject artifacts / out-of-range flux -> err=inf), so model and data
         share one normalization and the same pixels are trusted on both sides.
      2. Single fit: fit_single5 (five labels) seeded from `seed` (the catalog
         labels) or a dwarf default.
      3. Binary fit: fit_binary5 seeded by the single fit, grid q + RV scan,
         primary five-label refine, with the binary-nests-single floor.
      4. Statistics: chi2_single, chi2_binary; Delta-chi2 = single - binary; the
         EXACT El-Badry Eq. B1 f_imp (f_imp) and the Table B1 sliding thresholds
         (passes_table_b1). Clamp Delta-chi2 >= 0.

    Returns the SAME compact dict shape as binspec_single_vs_binary (scalars only)
    so the two are drop-in comparable:
      {chi2_single, chi2_binary, delta_chi2, f_imp, min_fimp_required,
       prefers_binary, best_q, best_rv1, best_rv2, best_teff1, teff_single,
       logg_single, feh_single}
    """
    if not _HAVE_DR19:
        raise RuntimeError(
            "models/payne_dr19.pt not found (the committed survey-normalized "
            "5-label net for the model='dr19' comparison path).")

    flux = np.asarray(flux8575, float)
    err = np.asarray(err8575, float)

    # 1. shared-continuum normalization + bad-pixel masking (model/data consistent).
    obs, obs_err = normalize_like_model(flux, err)
    obs_err = mask_bad_pixels(obs, obs_err)
    # least_squares cannot take err=inf in the residual; replace inf with a huge
    # finite value (the pixel is then negligibly weighted, same effect).
    obs_err = np.where(np.isfinite(obs_err), obs_err, 1e6)

    # Seed labels. Use the catalog seed if given; else a solar dwarf. Clip into
    # OUR net's box so the seed is always a valid start.
    if seed is None:
        seed = (5500.0, 4.5, 0.0, 0.0, 5.0)
    t0, g0, h0, m0, v0 = seed
    t0 = float(np.clip(t0, _LMIN5[0], _LMAX5[0]))
    g0 = float(np.clip(g0, _LMIN5[1], _LMAX5[1]))
    h0 = float(np.clip(h0, _LMIN5[2], _LMAX5[2]))
    m0 = float(np.clip(m0, _LMIN5[3], _LMAX5[3]))
    v0 = float(np.clip(v0, _LMIN5[4], _LMAX5[4]))

    # 2. single fit (five labels).
    p_s, model_s, chi2_s = fit_single5(obs, obs_err, t0, g0, h0, m0, v0)

    # 3. binary fit, seeded by the single fit.
    p_b, model_b, chi2_b = fit_binary5(
        obs, obs_err, p_s[0], p_s[1], p_s[2], p_s[3], p_s[4])

    # Binary-nests-single floor: the binary contains the single as its q->1 limit,
    # so its chi^2 cannot physically be worse; clamp so Delta-chi2 >= 0.
    if chi2_b > chi2_s:
        chi2_b = chi2_s
        model_b = model_s

    delta = chi2_s - chi2_b
    fimp = f_imp(obs, model_s, model_b, obs_err)
    min_fimp_required = table_b1_min_fimp(delta)
    prefers_binary = bool(passes_table_b1(delta, fimp))

    return {
        "chi2_single": float(chi2_s),
        "chi2_binary": float(chi2_b),
        "delta_chi2": float(delta),
        "f_imp": float(fimp),
        "min_fimp_required": min_fimp_required,
        "prefers_binary": prefers_binary,
        "best_q": float(p_b["q"]),
        "best_rv1": float(p_b["rv1"]),
        "best_rv2": float(p_b["rv2"]),
        "best_teff1": float(p_b["teff1"]),
        "teff_single": float(p_s[0]),
        "logg_single": float(p_s[1]),
        "feh_single": float(p_s[2]),
    }


# =========================================================================== #
# 7c. SELF-CONSISTENT detector: SC-normalized flux + SC Payne + OUR OWN
#     un-normalization continuum (THE production path after the continuum pivot).
#
#     This block is the end-to-end self-consistent pipeline:
#       - the single-star spectral model is payne_predict5_sc (trained on
#         continuum_normalize'd flux),
#       - the observed spectrum enters as RAW flux + ivar and is normalized by the
#         SAME continuum_normalize operator (Stage 5 wiring),
#       - the binary composite (compose_binary5_sc) sums two components in flux
#         (Eq. 2, R1^2 f1 + R2^2 f2) using OUR OWN un-normalization continuum
#         (Stage 4: pseudo_continuum_sc), NOT the survey continuum and NOT the
#         binspec flux net,
#       - the q=1 / equal-RV limit is EXACTLY the single model on the same flux,
#       - masking, the binary-nests-single floor, the EXACT Eq. B1 f_imp, and the
#         Table B1 thresholds are unchanged.
#     The isochrone q -> (Teff2, logg2, R1, R2) map (secondary_from_q) is OUR OWN
#     MIST v1.2 interpolation (isochrone_mist), not the binspec net. With the
#     SPECTRAL model, the NORMALIZATION, the un-normalization CONTINUUM, and now
#     the isochrone all de-coupled, this production path ports nothing from binspec.
# =========================================================================== #

# --- Stage 4: OUR OWN un-normalization continuum (no survey, no binspec net). - #
# compose_binary5_sc sums two stars in FLUX, so it needs the relative SED shape
# between a hot primary and a cooler secondary (a cooler star has a redder, weaker
# H-band continuum; that relative shape sets the line-depth dilution in Eq. 2).
# We derive that shape SELF-CONSISTENTLY from OUR training data: Stage 3
# (train_payne_dr19_sc.py) averages the per-chip sigma-clipped Chebyshev CONTINUUM
# that continuum_normalize fits to each raw training spectrum, binned by Teff, and
# saves a small (Teff-node -> mean-1 continuum shape) table to
# models/dr19_sc_continuum.npz. pseudo_continuum_sc(Teff) interpolates that table
# in Teff. This is OUR continuum (the inverse of the same Chebyshev estimator,
# averaged over real stars), not a survey column and not a trained flux net.
#
# If the table is absent (before Stage 3), we fall back to a smooth blackbody-like
# Teff scaling so import never fails; the table is the production path.
DR19_SC_CONT_PATH = os.environ.get(
    "AGENT4BINARY_SC_CONT",
    os.path.join(_ROOT, "models", "dr19_sc_continuum.npz"))

try:
    _SCC = np.load(DR19_SC_CONT_PATH)
    _SCC_TEFF = _SCC["teff_nodes"].astype(np.float64)        # (K,) Teff nodes
    _SCC_SHAPE = _SCC["cont_shapes"].astype(np.float64)      # (K, 8575) mean-1
    _SCC.close()
    _HAVE_SCC = True
except (FileNotFoundError, OSError):
    _SCC_TEFF = _SCC_SHAPE = None
    _HAVE_SCC = False


def _blackbody_shape(teff):
    """Fallback SED shape vs Teff (mean-1) when the SC continuum table is absent.

    A Planck B_lambda over the APOGEE band, normalized to mean 1. Only used before
    Stage 3 writes models/dr19_sc_continuum.npz; the data-derived table replaces
    it. Kept so physics.py imports and compose_binary5_sc is defined either way.
    """
    h = 6.62607015e-34
    c = 2.99792458e8
    kB = 1.380649e-23
    lam = WAVELENGTH * 1e-10                       # Angstrom -> metre
    x = h * c / (lam * kB * max(teff, 2000.0))
    b = 1.0 / (lam ** 5 * (np.exp(x) - 1.0))
    return b / b.mean()


def pseudo_continuum_sc(teff, logg=4.5, feh=0.0):
    """OUR OWN un-normalization continuum (mean-1) for the SC pipeline.

    Returns the relative SED shape at temperature Teff on the 8575-pixel grid,
    interpolated from the data-derived (Teff-node -> mean-1 continuum) table that
    Stage 3 builds from the SAME sigma-clipped Chebyshev continua continuum_normalize
    fits to real raw spectra. logg / feh are accepted for interface parity with
    pseudo_continuum but the table is keyed on Teff (the dominant SED driver in the
    H band). No survey continuum, no binspec flux net.

    The result is divided by its own mean, so only the SHAPE and the relative level
    between two temperatures matter (the absolute scale cancels in the final
    re-normalization, exactly as in compose_binary).
    """
    if not _HAVE_SCC:
        return _blackbody_shape(teff)
    # Interpolate each pixel across the Teff nodes (the table is sorted in Teff).
    t = float(np.clip(teff, _SCC_TEFF[0], _SCC_TEFF[-1]))
    # Find the bracketing nodes and linearly blend their shapes.
    j = int(np.searchsorted(_SCC_TEFF, t))
    if j <= 0:
        cont = _SCC_SHAPE[0].copy()
    elif j >= _SCC_TEFF.size:
        cont = _SCC_SHAPE[-1].copy()
    else:
        t0, t1 = _SCC_TEFF[j - 1], _SCC_TEFF[j]
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        cont = (1.0 - w) * _SCC_SHAPE[j - 1] + w * _SCC_SHAPE[j]
    return cont / cont.mean()


# --- Stage 4/5: SC single-star model + binary composite. ------------------- #
def payne5_single_model_sc(teff, logg, feh, mgh, vmacro, dv=0.0):
    """Single-star model for the SC pipeline, on the binary's flux->normalize path.

    Predict the SC-normalized flux with payne_predict5_sc, un-normalize by OUR OWN
    Teff-keyed continuum (pseudo_continuum_sc), optionally Doppler-shift by the
    radial velocity dv (km/s), then re-normalize with the SAME continuum_normalize
    operator the net was trained in (_cont_norm_model). This places the single model
    in the SAME continuum_normalize space as compose_binary5_sc: it is EXACTLY the
    q->1 / rv1==rv2==dv limit of the binary composite, so Delta-chi2 is self-
    consistent and the binary nests the single. dv gives the single star the SAME
    velocity freedom the binary's rv1/rv2 already have (El-Badry fits a velocity in
    the single model too); without it a residual systemic RV in the coadd misaligns
    every line of the rest-frame single, inflating chi2_single and letting a
    near-equal-mass binary win purely by shifting -> a false positive. No survey
    continuum anywhere.
    """
    flux = payne_predict5_sc(teff, logg, feh, mgh, vmacro) \
        * pseudo_continuum_sc(teff, logg, feh)
    if dv != 0.0:
        flux = _doppler_shift(flux, dv)
    return _cont_norm_model(flux)


def compose_binary5_sc(teff1, logg1, feh, mgh, vmacro, q, rv1_kms, rv2_kms,
                       age_gyr=_MS_REPR_AGE_GYR):
    """El-Badry Eq. 2 binary composite for the SC pipeline (OUR net + OUR continuum).

    Mirrors compose_binary5, but every survey / binspec-net dependence is removed:
      - f1, f2 come from OUR SC-trained net (payne_predict5_sc),
      - each component is un-normalized by OUR OWN Teff-keyed continuum
        (pseudo_continuum_sc), NOT the survey continuum and NOT the binspec flux net,
      - the sum F = R1^2 f1(shift1) + R2^2 f2(shift2) is re-normalized once by the
        shared running continuum.
    The isochrone q -> (Teff2, logg2, R1, R2) map (secondary_from_q) is OUR OWN
    MIST v1.2 interpolation (isochrone_mist), not the binspec net.

    The q=1 / equal-RV identity is EXACT: secondary_from_q / mh_from_q return
    identical component labels and weights, so if rv1 == rv2 the final
    normalization cancels the factor of two and compose_binary5_sc(q=1) equals
    payne5_single_model_sc. If rv1 != rv2, q=1 remains a valid equal-mass,
    line-doubled binary. Returns a length-8575 normalized composite.
    """
    f1 = payne_predict5_sc(teff1, logg1, feh, mgh, vmacro)
    teff2, logg2, R2, R1 = secondary_from_q(teff1, logg1, feh, q, age_gyr)
    f2 = payne_predict5_sc(teff2, logg2, feh, mgh, vmacro)
    # q-bias fix (self-contained MIST H-band): pseudo_continuum_sc is the mean-1
    # SED SHAPE per Teff (the absolute level vs Teff was divided out of the table),
    # so we must restore the per-component LEVEL. The correct H-band weight of each
    # component is its 2MASS H LUMINOSITY, 10^(-0.4 M_H), read off OUR MIST
    # isochrone -- M_H carries BOTH the emitting area (R^2) AND the true surface
    # brightness / bolometric correction (so a cool secondary's molecular H-band
    # absorption is included; a blackbody/Planck weight over-estimates it). This
    # H-band luminosity weight therefore REPLACES the old R^2 x continuum-level
    # weighting (R^2 is already inside the luminosity). Normalized to the primary
    # (w1 := 1), the secondary weight is 10^(-0.4 (M_H2 - M_H1)); the composite is
    # re-normalized once below, so the overall scale is irrelevant. At q = 1,
    # M_H2 == M_H1 -> ratio 1. With equal RVs the factor of two cancels in the
    # final normalization; with different RVs the equal-mass binary remains
    # line-doubled instead of being collapsed to a single component.
    MH1, MH2 = _iso_ours.mh_from_q_ours(teff1, logg1, feh, q, age_gyr)
    w_sec = 10.0 ** (-0.4 * (MH2 - MH1))
    f1_phys = f1 * pseudo_continuum_sc(teff1, logg1, feh)
    f2_phys = f2 * pseudo_continuum_sc(teff2, logg2, feh)
    f1_shift = _doppler_shift(f1_phys, rv1_kms)
    f2_shift = _doppler_shift(f2_phys, rv2_kms)
    flux_sum = f1_shift + w_sec * f2_shift
    # Normalize the composite with the SAME continuum_normalize operator the SC net
    # was trained in (see _cont_norm_model), NOT the running-continuum: model and
    # data then share one continuum so the normalization cancels in chi2. Scale-
    # invariant, so q=1/equal-RV still nests payne5_single_model_sc exactly.
    return _cont_norm_model(flux_sum)


SINGLE_DV_BOUND = 150.0    # km/s velocity bound for the single-star fit
_SINGLE_DV_SCAN = np.arange(-90.0, 90.1, 5.0)   # coarse RV scan grid


def fit_single5_sc(obs, err, teff0, logg0, feh0, mgh0, vmacro0, dv0=0.0):
    """Fit OUR SC 5-label single-star model + a radial velocity dv, DECOUPLED.

    Returns (params6 = [Teff, logg, feh, mgh, vmacro, dv], model, chi2). The dv gives
    the single model the SAME velocity freedom the binary's rv1/rv2 have, so a
    residual systemic RV in the coadd is absorbed by the single rather than mistaken
    for a near-equal-mass binary (the dominant control false positive).

    CRUCIAL: dv and the LABELS are fit DECOUPLED, not jointly. A joint 6-parameter
    fit lets the labels drift (raise Teff / v_macro) to partially absorb a real
    binary's line-DOUBLING, which both lowers chi2_single AND hands the binary fit a
    smeared primary seed -- on the canonical SB2 that wiped out a real detection
    (f_imp 0.317 -> -0.006). Our net is not accurate enough to forbid that the way
    binspec's synthetic net does. So:
      1. Fit the 5 labels at dv=0 (the stable label seed the binary inherits).
      2. Refine dv at THOSE fixed labels (coarse scan + fine scan); a model bulk-
         shift cannot absorb line-doubling, so this is safe for true binaries.
    The labels are NEVER re-fit at the new dv (we tried that conditional re-fit; it
    let the labels drift on velocity-offset SB2 and cost recovery, so it was removed).
    obs must be in continuum_normalize space.
    """
    lmin = _LMINSC.astype(float)
    lmax = _LMAXSC.astype(float)
    seed5 = np.clip(np.array([teff0, logg0, feh0, mgh0, vmacro0], float), lmin, lmax)
    obs = np.asarray(obs, float)
    err = np.asarray(err, float)
    base_good = np.isfinite(obs) & np.isfinite(err) & (err > 0)

    def fit_labels_at_dv(dv_fixed, seed):
        """5-label least_squares with dv held fixed; returns (labels5, model, chi2)."""
        def resid(p):
            m = payne5_single_model_sc(p[0], p[1], p[2], p[3], p[4], dv_fixed)
            good = base_good & np.isfinite(m)
            r = np.zeros_like(obs)
            r[good] = (obs[good] - m[good]) / err[good]
            return r
        res = least_squares(resid, np.clip(seed, lmin, lmax), method="trf",
                            bounds=(lmin, lmax), ftol=5e-4, xtol=5e-4, max_nfev=200)
        p5 = np.clip(res.x, lmin, lmax)
        m = payne5_single_model_sc(p5[0], p5[1], p5[2], p5[3], p5[4], dv_fixed)
        return p5, m, chi2(obs, err, m)

    def best_dv_at_labels(p5):
        """Coarse + fine RV scan at FIXED labels p5 (shift the physical flux)."""
        base_phys = payne_predict5_sc(*p5) * pseudo_continuum_sc(p5[0], p5[1], p5[2])
        bd, bc = 0.0, np.inf
        for dv in _SINGLE_DV_SCAN:
            c = chi2(obs, err, _cont_norm_model(_doppler_shift(base_phys, float(dv))))
            if c < bc:
                bc, bd = c, float(dv)
        for dv in np.arange(bd - 4.0, bd + 4.01, 1.0):
            c = chi2(obs, err, _cont_norm_model(_doppler_shift(base_phys, float(dv))))
            if c < bc:
                bc, bd = c, float(dv)
        return float(np.clip(bd, -SINGLE_DV_BOUND, SINGLE_DV_BOUND))

    # 1. labels at dv=0 (the stable seed the binary inherits).
    p5_0, m0, c0 = fit_labels_at_dv(0.0, seed5)
    # 2. dv refinement ONLY, at those fixed labels. We deliberately do NOT re-fit
    #    the labels at the new dv: with our (real-data, imperfect) net, any label
    #    freedom lets the single drift to absorb a real binary's line-doubling,
    #    which costs recovery (64%->58% on the fair test) without improving purity
    #    (the false positives are net-misfit-driven, not velocity-driven). A pure
    #    velocity shift cannot absorb line-doubling, so this is recovery-safe and
    #    still gives the single the El-Badry velocity freedom for the offset tail.
    best_dv = best_dv_at_labels(p5_0)
    m = payne5_single_model_sc(p5_0[0], p5_0[1], p5_0[2], p5_0[3], p5_0[4], best_dv)
    c = chi2(obs, err, m)
    return np.concatenate([p5_0, [best_dv]]), m, c


def fit_binary5_sc(obs, err, teff1, logg1, feh, mgh, vmacro,
                   age_gyr=_MS_REPR_AGE_GYR, dv_center=0.0):
    """Fit OUR SC 5-label binary model with scipy least_squares, grid-seeded q+RV.

    Mirrors fit_binary5 (coarse q+RV scan, polish the best seeds, refine the
    primary's five labels, binary-nests-single floor) but uses the SC net + SC
    continuum composite (compose_binary5_sc). Returns (params_dict, model, chi2)
    with keys teff1, logg1, feh, mgh, vmacro, q, rv1, rv2.

    dv_center (km/s) is the systemic velocity the single-star fit found: the coarse
    RV scan and the q=1 binary-nests-single floor are CENTERED on it, so the binary
    is evaluated in the SAME velocity frame as the single. Then Delta-chi2 reflects
    line-DOUBLING, not a bulk velocity shift the single could not make (which is what
    produced near-equal-mass false positives on velocity-offset true singles).
    """
    obs = np.asarray(obs, float)
    err = np.asarray(err, float)
    base_good = np.isfinite(obs) & np.isfinite(err) & (err > 0)

    # Primary kept inside BOTH the SC net's box and the isochrone MS box.
    feh_lo = max(float(_ISO_LABEL_LO[2]), float(_LMINSC[2]))
    feh_hi = min(float(_ISO_LABEL_HI[2]), float(_LMAXSC[2]))
    t_lo = max(4500.0, float(_LMINSC[0]))
    t_hi = min(6800.0, float(_LMAXSC[0]))
    g_lo = max(3.5, float(_LMINSC[1]))
    g_hi = min(5.0, float(_LMAXSC[1]))
    mg_lo, mg_hi = float(_LMINSC[3]), float(_LMAXSC[3])
    vm_lo, vm_hi = float(_LMINSC[4]), float(_LMAXSC[4])

    teff1 = float(np.clip(teff1, t_lo, t_hi))
    logg1 = float(np.clip(logg1, g_lo, g_hi))
    feh = float(np.clip(feh, feh_lo, feh_hi))
    mgh = float(np.clip(mgh, mg_lo, mg_hi))
    vmacro = float(np.clip(vmacro, vm_lo, vm_hi))

    def _model(params):
        tp, gp, hp, mp, vp, q, rv1, rv2 = params
        return compose_binary5_sc(float(tp), float(gp), float(hp), float(mp),
                                  float(vp), float(q), float(rv1), float(rv2),
                                  age_gyr)

    def resid(params):
        try:
            model = _model(params)
        except ValueError:
            return np.full(obs.shape, 1e3)
        good = base_good & np.isfinite(model)
        r = np.zeros_like(obs)
        r[good] = (obs[good] - model[good]) / err[good]
        return r

    lo = [t_lo, g_lo, feh_lo, mg_lo, vm_lo, 0.1, -150.0, -150.0]
    hi = [t_hi, g_hi, feh_hi, mg_hi, vm_hi, 1.0, 150.0, 150.0]

    q_scan = BINARY_Q_SCAN
    # Center the RV scan on the single's systemic velocity (absolute velocities),
    # clipped into the fit bounds, so a high-RV system is bracketed by the grid.
    rv_scan = np.clip(BINARY_RV_SCAN + float(dv_center), -150.0, 150.0)
    seeds = []
    for q0 in q_scan:
        for rv1_0 in rv_scan:
            for rv2_0 in rv_scan:
                try:
                    m = compose_binary5_sc(teff1, logg1, feh, mgh, vmacro,
                                           float(q0), float(rv1_0), float(rv2_0),
                                           age_gyr)
                except ValueError:
                    continue
                seeds.append((chi2(obs, err, m), q0, rv1_0, rv2_0))
    seeds.sort(key=lambda t: t[0])
    polish_seeds = _select_binary_polish_seeds(seeds)

    best_chi2 = np.inf
    best_params = None
    best_model = None
    for q0, rv1_0, rv2_0 in polish_seeds:
        x0 = [teff1, logg1, feh, mgh, vmacro, q0, rv1_0, rv2_0]
        try:
            res = least_squares(resid, x0, method="trf", bounds=(lo, hi),
                                ftol=5e-4, xtol=5e-4, max_nfev=150)
        except Exception:
            continue
        model = _model(res.x)
        c = chi2(obs, err, model)
        if c < best_chi2:
            best_chi2 = c
            best_params = res.x
            best_model = model

    # Binary-nests-single floor: the exact q=1 model at the seed labels, at the
    # single's systemic velocity (dv_center) so it equals the single model and the
    # floor is a true lower bound (binary can never be worse than the single).
    floor_model = compose_binary5_sc(teff1, logg1, feh, mgh, vmacro, 1.0,
                                     float(dv_center), float(dv_center), age_gyr)
    floor_chi2 = chi2(obs, err, floor_model)
    if floor_chi2 < best_chi2 or best_params is None:
        out = {"teff1": teff1, "logg1": logg1, "feh": feh, "mgh": mgh,
               "vmacro": vmacro, "q": 1.0, "rv1": float(dv_center),
               "rv2": float(dv_center)}
        return out, floor_model, floor_chi2

    tp, gp, hp, mp, vp, q, rv1, rv2 = best_params
    out = {"teff1": float(tp), "logg1": float(gp), "feh": float(hp),
           "mgh": float(mp), "vmacro": float(vp), "q": float(q),
           "rv1": float(rv1), "rv2": float(rv2)}
    return out, best_model, best_chi2


def dr19_sc_single_vs_binary(flux_raw, ivar, seed=None, snr_cap=SC_SNR_CAP):
    """SELF-CONSISTENT single-vs-binary detector (THE production path, Stage 5).

    Takes a RAW APOGEE spectrum (flux_raw + ivar straight off mwmStar, NO survey
    continuum) and decides single vs binary using OUR SC-trained net. Steps:

      1. continuum_normalize(flux_raw, ivar): per-chip sigma-clipped Chebyshev
         continuum from the RAW flux ITSELF -> normalized flux + normalized error.
         This is the IDENTICAL operator the training spectra were normalized with,
         so the normalization is common-mode and cancels in chi^2.
      2. normalize_like_model + mask_bad_pixels: put the observed flux on the same
         shared running-continuum the model uses and mask artifacts (S/N>200 cap +
         reject out-of-range / non-finite pixels -> err=inf).
      3. Single fit: fit_single5_sc (five labels), seeded by `seed` (catalog
         labels) or a dwarf default.
      4. Binary fit: fit_binary5_sc, grid q+RV, primary refine, nests-single floor.
      5. Statistics: Delta-chi2 (clamped >= 0), the EXACT El-Badry Eq. B1 f_imp,
         and the Table B1 sliding thresholds.

    Returns the SAME compact dict shape as dr19_single_vs_binary, so the server is
    a drop-in swap. NO mwmStar continuum, NO binspec net anywhere in this path.
    """
    if not _HAVE_SC:
        raise RuntimeError(
            "models/payne_dr19_sc.pt not found; train the SC net first "
            "(src/train_payne_dr19_sc.py).")

    flux_raw = np.asarray(flux_raw, float)
    ivar = np.asarray(ivar, float)

    # 1. SELF-CONSISTENT normalization from raw flux (the pivot). The normalized
    #    error carries inf in gaps / untrusted pixels; we keep that for masking.
    sc_flux, sc_err = continuum_normalize(flux_raw, ivar, return_error=True,
                                          use_mask=SC_USE_MASK)
    # continuum_normalize already floored the continuum, but lines / gaps can still
    # leave a few non-finite normalized-flux pixels; replace them with 1.0 so the
    # downstream running continuum stays finite (their error is already inf).
    sc_flux = np.where(np.isfinite(sc_flux), sc_flux, 1.0)
    # least_squares cannot take err=inf; turn inf into a huge finite error so those
    # pixels are negligibly weighted (same effect as masking) but the vector stays
    # finite.
    sc_err_finite = np.where(np.isfinite(sc_err), sc_err, 1e6)

    # 2. Compare in continuum_normalize SPACE -- the space the SC net was trained in.
    #    The observed spectrum is ALREADY continuum_normalize'd (step 1); the single
    #    model (payne5_single_model_sc) and the binary composite (compose_binary5_sc)
    #    are now normalized with the SAME operator (_cont_norm_model), so model and
    #    data share one continuum and it cancels in chi2. We do NOT re-normalize obs
    #    a second time with the running continuum (normalize_like_model) -- that
    #    inconsistent second operator was the dominant chi2_single inflation.
    obs = sc_flux
    obs_err = mask_bad_pixels(obs, sc_err_finite, snr_cap=snr_cap)
    obs_err = np.where(np.isfinite(obs_err), obs_err, 1e6)

    # Seed labels (catalog seed or a solar dwarf), clipped into the SC net's box.
    if seed is None:
        seed = (5500.0, 4.5, 0.0, 0.0, 5.0)
    t0, g0, h0, m0, v0 = seed
    t0 = float(np.clip(t0, _LMINSC[0], _LMAXSC[0]))
    g0 = float(np.clip(g0, _LMINSC[1], _LMAXSC[1]))
    h0 = float(np.clip(h0, _LMINSC[2], _LMAXSC[2]))
    m0 = float(np.clip(m0, _LMINSC[3], _LMAXSC[3]))
    v0 = float(np.clip(v0, _LMINSC[4], _LMAXSC[4]))

    # 3. single fit (5 labels + a radial velocity dv = p_s[5]).
    p_s, model_s, chi2_s = fit_single5_sc(obs, obs_err, t0, g0, h0, m0, v0)
    dv_single = float(p_s[5])
    # 4. binary fit, seeded by the single fit and CENTERED on its systemic velocity
    #    so Delta-chi2 measures line-doubling, not a bulk shift the single can't make.
    p_b, model_b, chi2_b = fit_binary5_sc(
        obs, obs_err, p_s[0], p_s[1], p_s[2], p_s[3], p_s[4], dv_center=dv_single)

    # Binary-nests-single floor (Delta-chi2 >= 0).
    if chi2_b > chi2_s:
        chi2_b = chi2_s
        model_b = model_s

    delta = chi2_s - chi2_b
    fimp = f_imp(obs, model_s, model_b, obs_err)
    min_fimp_required = table_b1_min_fimp(delta)
    prefers_binary = bool(passes_table_b1(delta, fimp))

    return {
        "chi2_single": float(chi2_s),
        "chi2_binary": float(chi2_b),
        "delta_chi2": float(delta),
        "f_imp": float(fimp),
        "min_fimp_required": min_fimp_required,
        "prefers_binary": prefers_binary,
        "best_q": float(p_b["q"]),
        "best_rv1": float(p_b["rv1"]),
        "best_rv2": float(p_b["rv2"]),
        "best_teff1": float(p_b["teff1"]),
        "teff_single": float(p_s[0]),
        "logg_single": float(p_s[1]),
        "feh_single": float(p_s[2]),
        "dv_single": dv_single,
    }


# =========================================================================== #
# 7d. Stage 2: MULTI-EPOCH (individual-visit) joint single-vs-SB2 detector.
#
#     The Stage-1 detector above (dr19_sc_single_vs_binary) fits the COMBINED
#     mwmStar coadd; it recovers the wide-Delta-v binaries but MISSES close
#     binaries whose two components are nearly velocity-aligned IN THE COADD
#     (the coadd looks single). El-Badry's Stage 2 ("Fitting Multi-Epoch
#     Spectra", apogee_binaries.tex sec:visit) re-fits the INDIVIDUAL VISIT
#     spectra SIMULTANEOUSLY: ONE set of stellar parameters shared across all
#     visits, a per-visit primary velocity, and the secondary velocity tied to
#     the primary by momentum conservation. The orbital velocity change between
#     visits is the signal the coadd lacks.
#
#     This reuses the SAME SC building blocks as Stage 1 (continuum_normalize +
#     normalize_like_model + mask_bad_pixels per visit; payne5_*_sc nets;
#     compose_binary5_sc composite; f_imp / Table B1), so the two stages are on
#     the same self-consistent normalized-flux footing. It does NOT modify any
#     Stage-1 function.
# =========================================================================== #
def _prep_visit_sc(flux_raw, ivar):
    """RAW one-visit flux/ivar -> SC-normalized (obs, err) ready for the SC fits.

    Identical preprocessing to the Stage-1 head of dr19_sc_single_vs_binary, but
    applied PER VISIT: per-chip sigma-clipped Chebyshev continuum from the raw visit
    flux itself (continuum_normalize, the SAME continuum_normalize SPACE the net was
    trained in) + the S/N>200 cap + bad-pixel mask. Returns (obs, err) with err = 1e6
    on masked/untrusted pixels (finite so least_squares is happy).

    NOTE: like Stage 1, obs is NOT re-normalized a second time with normalize_like_model
    (the running continuum) -- that double-normalization put obs in a different
    continuum space than the models (_cont_norm_model / CN-space) and inflated chi2.
    """
    flux_raw = np.asarray(flux_raw, float)
    ivar = np.asarray(ivar, float)
    sc_flux, sc_err = continuum_normalize(flux_raw, ivar, return_error=True,
                                          use_mask=SC_USE_MASK)
    sc_flux = np.where(np.isfinite(sc_flux), sc_flux, 1.0)
    sc_err_finite = np.where(np.isfinite(sc_err), sc_err, 1e6)
    obs = sc_flux
    obs_err = mask_bad_pixels(obs, sc_err_finite, snr_cap=SC_SNR_CAP)
    obs_err = np.where(np.isfinite(obs_err), obs_err, 1e6)
    return obs, obs_err


def _v2_from_momentum(v1, gamma, q_dyn):
    """Secondary heliocentric velocity from El-Badry Eq. vr1_vr2 (momentum cons.).

        v_Helio,2 = gamma + (gamma - v_Helio,1) / q_dyn

    In the center-of-mass frame momentum conservation gives v2 = -v1/q_dyn; with
    a systemic velocity gamma this is the heliocentric form above. q_dyn is the
    dynamical mass ratio, floored away from 0 so the division is finite.
    """
    return gamma + (gamma - v1) / max(q_dyn, 1e-3)


def dr19_visit_single_vs_binary(visits, seed=None, snr_min=30.0, max_visits=20):
    """STAGE 2: joint multi-epoch single-vs-SB2 detector (El-Badry sec:visit).

    Fits the individual VISIT spectra of one system SIMULTANEOUSLY and decides
    single vs SB2 using OUR SC-trained net, exactly as Stage 1 does per spectrum
    but with the velocity structure across visits that the coadd throws away.

    INPUT. `visits` is either an sdss_id (int/str) -- in which case the visit
    npz written by src/download_dr19_visits.py is loaded -- or a list of
    per-visit tuples (flux_raw, ivar, vhelio) where flux_raw/ivar are length-8575
    RAW arrays and vhelio is the visit heliocentric velocity (used only to SEED
    the per-visit velocity; the fit re-optimizes it). `seed` is the catalog
    5-label seed (teff, logg, feh, mgh, vmacro); None -> the dwarf default.

    MODELS (both share ONE set of five stellar labels across all visits):

      SINGLE : the Stage-1 single model with ONE velocity for ALL visits
               (El-Badry's single-star model forces v_Helio equal at every
               epoch). Free params: teff, logg, feh, mgh, vmacro, v.
      SB2    : shared (teff, logg, feh, mgh, vmacro, q_spec); a free systemic
               gamma and dynamical mass ratio q_dyn; and a per-visit PRIMARY
               velocity v1_i. The secondary velocity at each visit is TIED by
               momentum conservation, v2_i = gamma + (gamma - v1_i)/q_dyn
               (Eq. vr1_vr2). q_spec sets the spectral dilution (compose_binary5_sc);
               q_dyn sets the velocity tie. Free params: 8 + N_visit.

    JOINT CHI2. The total chi2 is the inverse-variance chi2 SUMMED over all
    visits (each visit compared to the model at its own velocity). The least_-
    squares residual vector is the per-visit (obs-model)/err vectors concatenated.

    GLOBAL CONVERGENCE (El-Badry's strategy). We first fit each visit ONE AT A
    TIME with the Stage-1 single and binary fits to estimate the per-visit
    velocity of each component, then seed the joint optimizer from those
    estimates (all velocities within a few km/s of their per-visit best fits).
    This is what keeps the (8+N)-dimensional SB2 fit out of local minima.

    Visits with median S/N < `snr_min` are DROPPED before fitting (El-Badry skip
    < 30 / pixel: low-S/N visits fit poorly, mostly from bad continuum). At most
    `max_visits` highest-S/N visits are kept (the optimizer cost grows with N).

    STATISTICS. Delta-chi2 = chi2_single_joint - chi2_sb2_joint (floored at 0 by
    the nests-single guard). f_imp is the El-Badry Eq. B1 statistic on the
    CONCATENATED joint residuals (all visits' good pixels). The Table B1
    acceptance Delta-chi2 axis is SCALED by N_visit (El-Badry: the per-step
    Delta-chi2 threshold is 300 * N_epochs), so a multi-epoch detection must
    clear a proportionally larger total Delta-chi2; f_imp uses the same floor.

    Returns a dict: n_visits_used, delta_chi2, f_imp, min_fimp_required,
    prefers_binary, q (q_spec), q_dyn, gamma, v_single, v1_per_visit (list),
    v2_per_visit (list), and the labels of both fits. A system with only ONE
    usable visit yields no multi-epoch gain (single == SB2 collapse to the
    coadd-like fit); we still return a result with prefers_binary from the
    single-epoch SC fit and note n_visits_used == 1.
    """
    if not _HAVE_SC:
        raise RuntimeError(
            "models/payne_dr19_sc.pt not found; train the SC net first "
            "(src/train_payne_dr19_sc.py).")

    # ----- 1. Assemble per-visit (flux_raw, ivar, vhelio) tuples. ----------- #
    if isinstance(visits, (int, str, np.integer)):
        import download_dr19_visits as _dlv
        d = _dlv.load_visits(int(visits))
        if d is None:
            raise FileNotFoundError(
                "no visit npz for sdss_id %s (run src/download_dr19_visits.py)"
                % visits)
        flux_all = np.atleast_2d(np.asarray(d["flux_raw"], float))
        ivar_all = np.atleast_2d(np.asarray(d["ivar"], float))
        v_rad = np.asarray(d.get("v_rad", np.full(flux_all.shape[0], np.nan)), float)
        bc_all = np.asarray(d.get("bc", np.full(flux_all.shape[0], np.nan)), float)
        snr_all = np.asarray(d.get("snr", np.full(flux_all.shape[0], np.nan)), float)
        raw = [(flux_all[i], ivar_all[i],
                (float(v_rad[i]) if i < v_rad.size else float("nan")),
                (float(snr_all[i]) if i < snr_all.size else float("nan")),
                (float(bc_all[i]) if i < bc_all.size else float("nan")))
               for i in range(flux_all.shape[0])]
    else:
        raw = []
        for tup in visits:
            f, iv = np.asarray(tup[0], float), np.asarray(tup[1], float)
            vh = float(tup[2]) if len(tup) > 2 and tup[2] is not None else float("nan")
            # Optional 4th tuple element = barycentric correction (km/s). If a
            # caller passes already-heliocentric flux it should pass bc = 0.
            bc = float(tup[3]) if len(tup) > 3 and tup[3] is not None else float("nan")
            raw.append((f, iv, vh, float("nan"), bc))

    # ----- 2. Per-visit preprocessing + S/N cut. ---------------------------- #
    # FRAME. mwmVisit flux is in the OBSERVED (topocentric) frame: the inter-visit
    # velocity difference is dominated by the BARYCENTRIC correction (Earth's
    # motion, up to ~60 km/s between visits), which is NOT orbital and would make
    # the single model -- forced to ONE velocity -- misfit every visit. We remove
    # it by Doppler-shifting each visit by -bc into the HELIOCENTRIC frame, so the
    # only velocity difference left between visits is the star's orbital motion
    # (v_Helio). This is the frame El-Badry's v_Helio,i and Eq. vr1_vr2 live in.
    prepped = []          # list of (obs, err, vhelio_seed, snr)
    for f, iv, vh, snr_meta, bc in raw:
        good = np.isfinite(f) & np.isfinite(iv) & (iv > 0)
        if not np.any(good):
            continue
        # Median pixel S/N (use the metadata column if finite, else compute it).
        if np.isfinite(snr_meta) and snr_meta > 0:
            snr = snr_meta
        else:
            snr = float(np.median((f * np.sqrt(iv))[good]))
        if not np.isfinite(snr) or snr < snr_min:
            continue
        # Heliocentric correction (remove the barycentric velocity). On the raw
        # flux/ivar BEFORE normalization so the per-chip continuum is fit on the
        # shifted spectrum exactly as for a coadd. bc unknown -> no shift.
        if np.isfinite(bc):
            f = _doppler_shift(f, -bc)
            iv = _doppler_shift(iv, -bc)
        obs, err = _prep_visit_sc(f, iv)
        prepped.append((obs, err, vh, snr))

    if not prepped:
        # No visit cleared S/N >= snr_min. El-Badry falls back to the coadd here;
        # for this prototype we instead keep the SINGLE highest-S/N visit so the
        # call still returns a (single-epoch) verdict rather than erroring. With
        # one visit there is no multi-epoch gain (flagged by n_visits_used == 1).
        scored = []
        for f, iv, vh, snr_meta, bc in raw:
            good = np.isfinite(f) & np.isfinite(iv) & (iv > 0)
            if not np.any(good):
                continue
            snr = (snr_meta if (np.isfinite(snr_meta) and snr_meta > 0)
                   else float(np.median((f * np.sqrt(iv))[good])))
            scored.append((snr, f, iv, vh, bc))
        if not scored:
            raise ValueError("no usable visit (all-zero ivar)")
        scored.sort(key=lambda t: -t[0])
        snr, f, iv, vh, bc = scored[0]
        if np.isfinite(bc):
            f = _doppler_shift(f, -bc)
            iv = _doppler_shift(iv, -bc)
        obs, err = _prep_visit_sc(f, iv)
        prepped.append((obs, err, vh, snr))

    # Keep the highest-S/N visits if there are many (cost grows with N).
    prepped.sort(key=lambda t: -t[3])
    prepped = prepped[:max_visits]
    N = len(prepped)
    obs_list = [p[0] for p in prepped]
    err_list = [p[1] for p in prepped]
    vseed_list = [p[2] for p in prepped]

    # Seed labels (catalog seed or solar dwarf), clipped into the SC net box.
    if seed is None:
        seed = (5500.0, 4.5, 0.0, 0.0, 5.0)
    t0, g0, h0, m0, v0 = seed
    t0 = float(np.clip(t0, _LMINSC[0], _LMAXSC[0]))
    g0 = float(np.clip(g0, _LMINSC[1], _LMAXSC[1]))
    h0 = float(np.clip(h0, _LMINSC[2], _LMAXSC[2]))
    m0 = float(np.clip(m0, _LMINSC[3], _LMAXSC[3]))
    v0 = float(np.clip(v0, _LMINSC[4], _LMAXSC[4]))

    # ----- 3. Per-visit single-epoch fits to estimate per-visit velocities. - #
    #         (El-Badry: initialize the joint optimizer from one-at-a-time fits,
    #         "with all velocities within +/- ~20 km/s of their true values".)
    # We seed each visit's primary velocity from the Astra-measured per-visit
    # HELIOCENTRIC RV (v_rad, carried in vseed_list): after the -bc correction the
    # flux of visit i still carries its v_rad, so the model for visit i must be
    # shifted by ~v_rad_i. This is the orbital velocity the coadd averaged away,
    # and using it as the seed is exactly El-Badry's strategy. For each visit we
    # then refine the velocity with a quick scan + fit (labels fixed) so the seed
    # is good even when the catalog v_rad is missing.
    per_v1 = []           # per-visit refined primary velocity (the v1 seed)
    per_q = []            # per-visit best q_spec
    labels_seed = np.array([t0, g0, h0, m0, v0], float)
    label_acc = np.zeros(5)
    for obs, err, vh in zip(obs_list, err_list, vseed_list):
        p_s, model_s, _ = fit_single5_sc(obs, err, *labels_seed)
        label_acc += p_s[:5]   # p_s is [Teff,logg,feh,mgh,vmacro,dv]; accumulate labels
        # Velocity seed: the measured v_rad if finite, else 0; refine by a coarse
        # scan of the REST-FRAME single model around it (the single visit alone
        # constrains the bulk Doppler shift well). Use p_s[:5] so base_single is at
        # rest (dv=0); the scan below supplies the velocity, not the single fit's dv.
        v_anchor = vh if np.isfinite(vh) else 0.0
        base_single = payne5_single_model_sc(*p_s[:5])
        best = (np.inf, v_anchor)
        for dv in v_anchor + np.arange(-25.0, 25.5, 2.5):
            c = chi2(obs, err, _doppler_shift(base_single, float(dv)))
            if c < best[0]:
                best = (c, float(dv))
        per_v1.append(best[1])
        # A quick q estimate from the single-visit binary fit (spectral dilution).
        p_b, _, _ = fit_binary5_sc(obs, err, p_s[0], p_s[1], p_s[2], p_s[3], p_s[4])
        per_q.append(float(p_b["q"]))
    # Shared-label seed = mean of the per-visit single-fit labels.
    lab0 = label_acc / N
    lab0 = np.clip(lab0, _LMINSC, _LMAXSC)
    # Median q from the per-visit binary fits seeds q_spec / q_dyn.
    q_seed = float(np.clip(np.median(per_q), 0.1, 0.99))
    v1_seed = np.array(per_v1, float)
    gamma_seed = float(np.median(v1_seed))        # systemic ~ median per-visit RV

    base_good = [np.isfinite(o) & np.isfinite(e) & (e > 0)
                 for o, e in zip(obs_list, err_list)]

    # ----- 4a. JOINT SINGLE fit: shared 5 labels + ONE velocity. ------------ #
    #          The single model forces v equal at every epoch (El-Badry Eq.
    #          singe_visit_labels), so it cannot absorb an orbital RV change.
    def _single_models(labels, v):
        m = payne5_single_model_sc(*labels)        # rest-frame normalized single
        return _doppler_shift(m, v)                # one velocity for all visits

    def resid_single(p):
        labels = np.clip(p[:5], _LMINSC, _LMAXSC)
        v = p[5]
        model = _single_models(labels, v)
        out = []
        for obs, err, gd in zip(obs_list, err_list, base_good):
            g = gd & np.isfinite(model)
            r = np.zeros_like(obs)
            r[g] = (obs[g] - model[g]) / err[g]
            out.append(r)
        return np.concatenate(out)

    lo_s = list(_LMINSC) + [-150.0]
    hi_s = list(_LMAXSC) + [150.0]
    x0_s = list(lab0) + [gamma_seed]
    res_s = least_squares(resid_single, np.clip(x0_s, lo_s, hi_s), method="trf",
                          bounds=(lo_s, hi_s), ftol=5e-4, xtol=5e-4, max_nfev=300)
    lab_s = np.clip(res_s.x[:5], _LMINSC, _LMAXSC)
    v_s = float(res_s.x[5])
    single_models = [_single_models(lab_s, v_s)] * N
    chi2_single = sum(chi2(o, e, single_models[0]) for o, e in zip(obs_list, err_list))

    # ----- 4b. JOINT SB2 fit: shared labels + q_spec + gamma + q_dyn + per-visit v1. #
    #          Secondary velocity tied per visit by Eq. vr1_vr2.
    n_par = 8 + N         # teff,logg,feh,mgh,vmacro,q_spec,gamma,q_dyn + N x v1
    IDX_Q, IDX_GAMMA, IDX_QDYN = 5, 6, 7

    def _sb2_models(p):
        labels = np.clip(p[:5], _LMINSC, _LMAXSC)
        q_spec = float(np.clip(p[IDX_Q], 0.1, 0.99))
        gamma = float(p[IDX_GAMMA])
        q_dyn = float(np.clip(p[IDX_QDYN], 0.1, 1.5))
        models = []
        for i in range(N):
            v1 = float(p[8 + i])
            v2 = _v2_from_momentum(v1, gamma, q_dyn)
            try:
                models.append(compose_binary5_sc(
                    labels[0], labels[1], labels[2], labels[3], labels[4],
                    q_spec, v1, v2))
            except ValueError:
                models.append(None)
        return models

    def resid_sb2(p):
        models = _sb2_models(p)
        out = []
        for obs, err, gd, model in zip(obs_list, err_list, base_good, models):
            if model is None:
                out.append(np.full(obs.shape, 1e3))
                continue
            g = gd & np.isfinite(model)
            r = np.zeros_like(obs)
            r[g] = (obs[g] - model[g]) / err[g]
            out.append(r)
        return np.concatenate(out)

    lo_b = list(_LMINSC) + [0.1, -150.0, 0.1] + [-150.0] * N
    hi_b = list(_LMAXSC) + [0.99, 150.0, 1.5] + [150.0] * N

    # Seed set: the per-visit-fit velocities (the El-Badry init), plus a couple
    # of robust fallbacks so a bad per-visit RV cannot strand the joint fit.
    seeds = []
    seeds.append(list(lab0) + [q_seed, gamma_seed, q_seed] + list(v1_seed))
    seeds.append(list(lab0) + [0.6, gamma_seed, 0.6] + [gamma_seed] * N)
    seeds.append(list(lab_s) + [0.5, v_s, 0.5] + [v_s] * N)   # seed from single fit

    best_chi2_b = np.inf
    best_p_b = None
    for s in seeds:
        x0 = np.clip(np.array(s, float), lo_b, hi_b)
        try:
            res_b = least_squares(resid_sb2, x0, method="trf", bounds=(lo_b, hi_b),
                                  ftol=5e-4, xtol=5e-4, max_nfev=400)
        except Exception:
            continue
        models = _sb2_models(res_b.x)
        c = sum(chi2(o, e, m) for o, e, m in zip(obs_list, err_list, models)
                if m is not None)
        if c < best_chi2_b:
            best_chi2_b = c
            best_p_b = res_b.x

    # Nests-single floor: SB2 can never beat the joint single by construction
    # (q -> 1 + equal v reproduces the single model at the same labels). If the
    # optimizer's SB2 chi2 is worse, fall back to the single fit (Delta >= 0).
    if best_p_b is None or best_chi2_b > chi2_single:
        sb2_models = single_models
        chi2_sb2 = chi2_single
        q_spec = 1.0
        q_dyn = 1.0
        gamma = v_s
        v1_pv = [v_s] * N
        v2_pv = [v_s] * N
        lab_b = lab_s
    else:
        sb2_models = _sb2_models(best_p_b)
        chi2_sb2 = best_chi2_b
        lab_b = np.clip(best_p_b[:5], _LMINSC, _LMAXSC)
        q_spec = float(np.clip(best_p_b[IDX_Q], 0.1, 0.99))
        gamma = float(best_p_b[IDX_GAMMA])
        q_dyn = float(np.clip(best_p_b[IDX_QDYN], 0.1, 1.5))
        v1_pv = [float(best_p_b[8 + i]) for i in range(N)]
        v2_pv = [_v2_from_momentum(v1_pv[i], gamma, q_dyn) for i in range(N)]

    # ----- 5. Joint statistics on the CONCATENATED residuals. --------------- #
    delta = float(max(chi2_single - chi2_sb2, 0.0))
    obs_cat = np.concatenate(obs_list)
    err_cat = np.concatenate(err_list)
    single_cat = np.concatenate(single_models)
    sb2_cat = np.concatenate(sb2_models)
    fimp = f_imp(obs_cat, single_cat, sb2_cat, err_cat)

    # Table B1 with the Delta-chi2 axis scaled by N visits (El-Badry: the per-
    # step threshold is 300 * N_epochs). We test the per-visit-equivalent
    # Delta-chi2 = delta / N against the single-epoch Table B1, which is the same
    # as scaling the axis by N.
    delta_per_visit = delta / max(N, 1)
    min_fimp_required = table_b1_min_fimp(delta_per_visit)
    # f_imp floor, ALWAYS enforced (no extreme-Delta-chi2 waiver) on the multi-
    # epoch path. The waiver Stage 1 uses (FIMP_FLOOR_WAIVE_DCHI2) assumes a huge
    # Delta-chi2 reflects a pervasive real second spectrum; here the per-visit
    # continuum normalization is poorer (El-Badry: low-S/N visits fit badly), so a
    # large Delta-chi2 paired with a near-zero f_imp is the SINGLE-star
    # contamination mode (verified on controls: f_imp ~ 0.01 at q ~ 0.99 with
    # Delta-chi2/visit > 1e5). Requiring f_imp >= FIMP_FLOOR rejects exactly those.
    if min_fimp_required is not None:
        min_fimp_required = max(min_fimp_required, FIMP_FLOOR)
    prefers_binary = bool(
        (min_fimp_required is not None) and (fimp >= min_fimp_required)
        # A binary verdict needs >= 2 usable visits: with one visit there is no
        # multi-epoch information, so any "detection" is just a single-epoch fit
        # (which Stage 1 already evaluated on the higher-S/N coadd).
        and (N >= 2))

    return {
        "n_visits_used": int(N),
        "chi2_single": float(chi2_single),
        "chi2_binary": float(chi2_sb2),
        "delta_chi2": delta,
        "delta_chi2_per_visit": float(delta_per_visit),
        "f_imp": float(fimp),
        "min_fimp_required": min_fimp_required,
        "prefers_binary": prefers_binary,
        "q": float(q_spec),
        "q_dyn": float(q_dyn),
        "gamma": float(gamma),
        "v_single": float(v_s),
        "v1_per_visit": [float(x) for x in v1_pv],
        "v2_per_visit": [float(x) for x in v2_pv],
        "vhelio_seed_per_visit": [float(x) for x in vseed_list],
        "teff_binary": float(lab_b[0]),
        "logg_binary": float(lab_b[1]),
        "feh_binary": float(lab_b[2]),
        "teff_single": float(lab_s[0]),
        "logg_single": float(lab_s[1]),
        "feh_single": float(lab_s[2]),
    }


# =========================================================================== #
# 8. binspec-faithful single-vs-binary detector (THE ORACLE, for head-to-head
#    comparison; not the default detection path -- the SC path in 7c is).
#
#    A documented cross-check found the SEPARATION comes from the single-star
#    FORWARD MODEL: binspec's 5-label net (Teff, logg, [Fe/H], [Mg/Fe], v_macro)
#    fits real dwarfs tightly (median Delta-chi2 ~18 on singles), so the binary
#    model has little to improve and cannot manufacture a false positive (binspec
#    on this set: 85% completeness, 0% contamination). A 3-label Payne underfits,
#    leaving a structured residual the over-parameterized binary fit absorbs on
#    singles too, collapsing the separation. Our OWN 5-label SC net (7c) closes
#    that gap; this oracle stays for comparison.
#
#    All four nets and the binspec wavelength / continuum machinery load from the
#    VENDORED package src/vendor/binspec (copied verbatim from
#    github.com/kareemelbadry/binspec), so the detector runs without the clone.
# =========================================================================== #

# --- Import the vendored binspec package once, at module import. ----------- #
# src/ is on sys.path (the MCP servers add it; physics.py lives in src/). The
# vendored package resolves its own weight / grid files relative to itself.
import sys as _sys  # local alias; physics.py otherwise has no sys dependency
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
from vendor.binspec import utils as _bs_utils          # noqa: E402
from vendor.binspec import spectral_model as _bs_sm    # noqa: E402
from vendor.binspec import fitting as _bs_fitting       # noqa: E402

# Load the four trained El-Badry nets once, from inside the vendored package.
#   NN_norm : 5-label normalized single-star net (Teff, logg, [Fe/H], [Mg/Fe], v_macro)
#   NN_flux : synthetic flux / continuum net (the un-normalization path)
#   NN_R    : (Teff, logg, [Fe/H]) -> R (Rsun)
#   NN_T2   : (Teff1, logg1, [Fe/H], q) -> (Teff2, logg2)
_BS_NN_NORM = _bs_utils.read_in_neural_network("normalized_spectra")
_BS_NN_FLUX = _bs_utils.read_in_neural_network("unnormalized_spectra")
_BS_NN_R = _bs_utils.read_in_neural_network("radius")
_BS_NN_T2 = _bs_utils.read_in_neural_network("Teff2_logg2")

# binspec's own 7214-pixel DR12/13 grid and Cannon continuum-pixel mask. The
# fits and the continuum renormalization all happen ON THIS GRID, exactly as in
# the cross-check (binspec's net and continuum machinery are defined here, not
# on our 8575-pixel aspcapStar grid).
_BS_WL = _bs_utils.load_wavelength_array()        # 7214 pixels
_BS_CONT_PIX = _bs_utils.load_cannon_contpixels()  # 7214-long boolean mask
_BS_NPIX = int(_BS_WL.size)


def ingest_to_binspec_grid(wl8575, flux8575, err8575):
    """DR17 aspcapStar arrays -> binspec-normalized (spec, spec_err) on the 7214 grid.

    The oracle's ingest. Our cached spectra are on the 8575-pixel ASPCAP grid (ext1
    flux already pseudo-continuum-normalized by ASPCAP, ext2 normalized error);
    binspec's net and continuum live on its 7214-pixel DR12/13 grid. Steps, all
    from El-Badry et al. 2018b / binspec process_spectra:

      1. Chip gaps / unmeasured pixels (flux==0, err<=0, or non-finite): set the
         error very large (100 x mean good flux) so binspec ignores them, and
         replace non-finite flux with 1.0 so the interpolation stays finite.
      2. S/N>200 cap: where flux/err>200, reset err to 0.005*|flux| (binspec
         process_spectra), so a few very-high-S/N pixels cannot dominate chi^2.
      3. Interpolate flux and error by wavelength onto binspec's 7214-pixel grid.
         The three DR17 chip ranges contain binspec's three chip ranges, so the
         interpolation is interior (no extrapolation).
      4. Renormalize through binspec's OWN get_apogee_continuum so the model and
         the data share one normalization (self-consistent, as binspec requires).
         Floor the error at 0.005 so capped / gap pixels stay sane.

    Returns (spec, spec_err), both length 7214.
    """
    wl8575 = np.asarray(wl8575, dtype=float)
    flux = np.asarray(flux8575, dtype=float)
    err = np.asarray(err8575, dtype=float)

    # 1. chip gaps / unmeasured pixels -> very large error; finite flux for interp.
    bad = (~np.isfinite(flux)) | (~np.isfinite(err)) | (flux == 0) | (err <= 0)
    flux = np.where(np.isfinite(flux), flux, 1.0)
    big = 100.0 * np.mean(flux[~bad]) if np.any(~bad) else 1e3
    err = np.where(bad, big, err)

    # 2. S/N>200 cap, exactly as binspec process_spectra (on the normalized flux).
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(err > 0, flux / err, 0.0)
    highsnr = snr > 200.0
    err = np.where(highsnr, 0.005 * np.abs(flux), err)

    # 3. interpolate flux and error onto binspec's 7214 grid by wavelength.
    fi = np.interp(_BS_WL, wl8575, flux)
    ei = np.interp(_BS_WL, wl8575, err)

    # 4. renormalize through binspec's own continuum routine (model/data self-consistent).
    cont = _bs_utils.get_apogee_continuum(
        wavelength=_BS_WL, spec=fi, spec_err=ei, cont_pixels=_BS_CONT_PIX)
    spec = fi / cont
    spec_err = ei / cont
    spec_err[~np.isfinite(spec_err)] = 1e3
    spec_err[spec_err < 0.005] = 0.005
    return spec, spec_err


def binspec_single_vs_binary(wl8575, flux8575, err8575, num_p0_binary=10):
    """The binspec-faithful single-vs-binary fit; the ORACLE (not the production path).

    Given DR17 aspcapStar arrays (8575-pixel grid), it ingests them to binspec's
    7214 grid (ingest_to_binspec_grid) with binspec's normalization + masking, then:

      1. Single-star fit: fitting.fit_normalized_spectrum_single_star_model with
         the 5-label net (Teff, logg, [Fe/H], [Mg/Fe], v_macro + dv). One start.
      2. Binary fit: fitting.fit_normalized_spectrum_binary_model seeded by the
         single-star fit, num_p0_binary q-starts (9 labels), with binspec's own
         single-star fallback guard (chi2_single < chi2_binary -> keep single).
      3. Statistics: chi2_single, chi2_binary on binspec's grid; Delta-chi2 =
         chi2_single - chi2_binary; the EXACT El-Badry Eq. B1 f_imp (physics.f_imp,
         on the binspec grid) and the Table B1 sliding thresholds
         (physics.passes_table_b1). Keep the binary-nests-single floor:
         Delta-chi2 >= 0 (a true single must not sit at a spurious negative).

    Returns a COMPACT dict (scalars only; no spectra):
      {chi2_single, chi2_binary, delta_chi2, f_imp, min_fimp_required,
       prefers_binary, best_q, best_rv1, best_rv2, best_teff1, teff_single,
       logg_single, feh_single}
    """
    # Ingest to binspec's grid with binspec's normalization + masking.
    spec, spec_err = ingest_to_binspec_grid(wl8575, flux8575, err8575)

    # 1. binspec single-star fit -> labels [Teff, logg, feh, alpha, vmacro, dv].
    popt_s, _, model_s = _bs_fitting.fit_normalized_spectrum_single_star_model(
        norm_spec=spec, spec_err=spec_err,
        NN_coeffs_norm=_BS_NN_NORM, NN_coeffs_flux=_BS_NN_FLUX, num_p0=1)

    # 2. binspec binary fit, seeded by the single-star fit -> 9 labels, q-starts.
    #    fit_normalized_spectrum_binary_model already applies binspec's own
    #    single-star fallback guard (keeps the single model if it has lower chi^2).
    popt_b, _, model_b = _bs_fitting.fit_normalized_spectrum_binary_model(
        norm_spec=spec, spec_err=spec_err,
        NN_coeffs_norm=_BS_NN_NORM, NN_coeffs_flux=_BS_NN_FLUX,
        NN_coeffs_Teff2_logg2=_BS_NN_T2, NN_coeffs_R=_BS_NN_R,
        p0_single=popt_s, num_p0=num_p0_binary)

    # 3. chi^2 on binspec's grid (inverse-variance, the binspec convention).
    chi2_s = float(np.sum((model_s - spec) ** 2 / spec_err ** 2))
    chi2_b = float(np.sum((model_b - spec) ** 2 / spec_err ** 2))

    # Binary-nests-single floor: the binary model contains the single star as its
    # q->1 / equal-RV limit, so the binary chi^2 can never physically be worse
    # than the single. binspec's own guard usually enforces this, but clamp here
    # too so Delta-chi2 is >= 0 (a true single must not sit at a spurious negative).
    if chi2_b > chi2_s:
        chi2_b = chi2_s
        model_b = model_s

    delta = chi2_s - chi2_b
    # EXACT El-Badry Eq. B1 f_imp (physics.f_imp), evaluated on the binspec grid.
    fimp = f_imp(spec, model_s, model_b, spec_err)
    # Table B1 sliding scale (physics.passes_table_b1 / table_b1_min_fimp).
    min_fimp_required = table_b1_min_fimp(delta)
    prefers_binary = bool(passes_table_b1(delta, fimp))

    # Best-fit binary labels (popt_b layout = [Teff1, logg1, feh, alpha, q,
    # vmacro1, vmacro2, dv1, dv2]). When the single-star fallback fired, popt_b is
    # the 6-label single vector ([..., 1, vmacro1, vmacro1, dv1, dv1]); guard the
    # index access so a fallback never raises.
    if len(popt_b) >= 9:
        best_teff1 = float(popt_b[0])
        best_q = float(popt_b[4])
        best_rv1 = float(popt_b[7])
        best_rv2 = float(popt_b[8])
    else:
        best_teff1 = float(popt_s[0])
        best_q = 1.0
        best_rv1 = float(popt_s[5])
        best_rv2 = float(popt_s[5])

    return {
        "chi2_single": chi2_s,
        "chi2_binary": chi2_b,
        "delta_chi2": delta,
        "f_imp": float(fimp),
        "min_fimp_required": min_fimp_required,
        "prefers_binary": prefers_binary,
        "best_q": best_q,
        "best_rv1": best_rv1,
        "best_rv2": best_rv2,
        "best_teff1": best_teff1,
        "teff_single": float(popt_s[0]),
        "logg_single": float(popt_s[1]),
        "feh_single": float(popt_s[2]),
    }
