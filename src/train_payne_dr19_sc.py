#!/usr/bin/env python3
"""
Stage 3: train OUR 5-label Payne on SELF-CONSISTENT-normalized flux.

The retrain that completes the continuum pivot. Trains the 5->300->300->8575 net
on flux normalized by physics.continuum_normalize (per-chip sigma-clipped Chebyshev
from the RAW flux itself). The detector normalizes OBSERVED spectra with the
IDENTICAL operator at fit time, so the normalization is common-mode and cancels in
chi^2; no survey continuum on either side. models/payne_dr19.pt (survey-trained) is
NOT touched.

INPUT (Stage 1 raw products)
----------------------------
  - resources/dr19_raw_train_manifest.csv : sdss_id + five labels + snr (controls
    already excluded by Stage 1).
  - data/dr19_raw/{sdss_id}.npz           : {wl[8575], flux_raw[8575], ivar[8575]}.

LABELS
------
  (Teff, logg, [Fe/H], [Mg/H], v_macro), the El-Badry et al. (2018b) label set. v_macro is
  the GENUINE ASTRA the_payne (Ting 2019) macroturbulence label (NOT vsini; the_payne has no
  vsini column), the same broadening construct El-Badry used (his v_macro also absorbs
  rotation). See docs/ab9_vmacro.md. Read from manifest columns teff/logg/fe_h/mg_h/
  v_macro. Each raw spectrum passes through continuum_normalize once at load; the
  normalized flux is the training target, the normalized error gates rejection.

SCHEMA (must match what physics.py loads)
-----------------------------------------
  Checkpoint keys: state_dict, label_min, label_max, label_names, n_label=5,
  n_hidden=300, n_pix=8575; label scaling (x-min)/(max-min)-0.5. This is what
  physics._payne_sc_forward_numpy reproduces bit-for-bit.

ITERATIVE BINARY / OUTLIER REJECTION (El-Badry et al. 2018b style)
------------------------------------------------------------------
  ~3 passes of train -> score by single-fit reduced chi^2 -> reject the worst
  (robust median + 5*MAD, capped at the worst ~10% per pass) -> retrain. An
  undetected SB2 / peculiar star a single-star model cannot reproduce scores high
  and is dropped. Cumulative rejects -> resources/payne_dr19_sc_rejected.csv.

STAGE 4 SIDE-PRODUCT: OUR OWN un-normalization continuum
--------------------------------------------------------
  compose_binary5_sc sums two stars in FLUX, so it needs the relative SED shape vs
  Teff. For each retained star we keep the per-chip Chebyshev CONTINUUM that
  continuum_normalize fit (the un-normalization curve), divide by its mean, and
  average these mean-1 shapes in Teff bins -> models/dr19_sc_continuum.npz, which
  physics.pseudo_continuum_sc interpolates. OUR continuum, not a survey column or
  a trained flux net.

OUTPUT
------
  - models/payne_dr19_sc.pt              : OUR SC 5-label net.
  - models/dr19_sc_continuum.npz         : Teff-binned mean-1 continuum shapes.
  - models/val_set_dr19_sc.npz           : held-out validation sdss_ids + labels.
  - resources/payne_dr19_sc_rejected.csv : cumulative rejected ids.

VALIDATION: ~150 dwarfs held out (fixed RNG); report median fractional and median
absolute residual.
"""

import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
import physics  # for continuum_normalize, NPIX, WAVELENGTH, CHIP_SLICES

# Inputs + output names are env-overridable so the SAME recipe retrains on a
# different release (e.g. M1 DR13 parity) without touching the DR19 artifacts.
TRAIN_DIR = os.environ.get("AGENT4BINARY_TRAIN_DIR",
                           os.path.join(_ROOT, "data", "dr19_raw"))
MANIFEST = os.environ.get("AGENT4BINARY_TRAIN_MANIFEST",
                          os.path.join(_ROOT, "resources", "dr19_raw_train_manifest.csv"))
MODELS_DIR = os.path.join(_ROOT, "models")
RES_DIR = os.path.join(_ROOT, "resources")
os.makedirs(MODELS_DIR, exist_ok=True)

# Output basenames (DR19 defaults; swap for DR13 via env).
SC_MODEL_OUT = os.environ.get("AGENT4BINARY_SC_MODEL_OUT", "payne_dr19_sc.pt")
SC_CONT_OUT = os.environ.get("AGENT4BINARY_SC_CONT_OUT", "dr19_sc_continuum.npz")
VAL_OUT = os.environ.get("AGENT4BINARY_VAL_OUT", "val_set_dr19_sc.npz")

NPIX = physics.NPIX
# Label set is env-configurable so the SAME recipe trains a richer-label net (the
# A3 gap experiment: 16 labels incl. abundances) without forking this file.
LABEL_COLS = os.environ.get(
    "AGENT4BINARY_LABEL_COLS", "teff,logg,fe_h,mg_h,v_macro").split(",")
N_LABEL = len(LABEL_COLS)
LABEL_NAMES = LABEL_COLS

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

N_VAL = 150            # held-out validation dwarfs
N_ITER = 3             # iterative-rejection passes


# --------------------------------------------------------------------------- #
# 1. Load + SELF-CONSISTENTLY normalize the raw training spectra.
# --------------------------------------------------------------------------- #
def load_dataset():
    """Read the raw manifest + npz, return labels + SC-normalized flux/err + the
    per-chip Chebyshev CONTINUUM (for the Stage 4 un-normalization table).

    For each star: read the five labels and the raw {flux_raw, ivar}, run
    physics.continuum_normalize to get (norm_flux, norm_err), and ALSO recompute
    the raw continuum itself (raw_flux / norm_flux) so we can build the Stage 4
    SED-shape table. A star is kept only if its npz exists, lengths are 8575, and
    the labels + normalized flux are finite.

    Returns (ids, labels, fluxes, errors, conts, teffs):
      ids     : (N,) int
      labels  : (N, 5) float32
      fluxes  : (N, 8575) float32 SC-normalized flux (training target)
      errors  : (N, 8575) float32 SC-normalized error (rejection diagnostic)
      conts   : (N, 8575) float32 mean-1 per-chip Chebyshev continuum (Stage 4)
      teffs   : (N,) float32 Teff (for Stage 4 binning)
    """
    ids, labels, fluxes, errors, conts, teffs = [], [], [], [], [], []
    n_missing = n_bad = 0
    with open(MANIFEST) as fh:
        rows = list(csv.DictReader(fh))
    for k, row in enumerate(rows):
        sid = int(row["sdss_id"])
        path = os.path.join(TRAIN_DIR, "%d.npz" % sid)
        if not os.path.exists(path):
            n_missing += 1
            continue
        try:
            lab = [float(row[c]) for c in LABEL_COLS]
        except Exception:
            n_bad += 1
            continue
        if not np.all(np.isfinite(lab)):
            n_bad += 1
            continue
        d = np.load(path)
        flux_raw = np.asarray(d["flux_raw"], dtype=np.float64)
        ivar = np.asarray(d["ivar"], dtype=np.float64)
        if flux_raw.size != NPIX or ivar.size != NPIX:
            n_bad += 1
            continue
        # THE PIVOT: self-consistent continuum normalization from raw flux, on the
        # SAME continuum operator the detector uses at inference (physics.SC_USE_MASK
        # -> fixed continuum-pixel mask, ~ binspec get_apogee_continuum, for a
        # star-independent continuum that cancels model-vs-data more cleanly).
        norm_flux, norm_err = physics.continuum_normalize(
            flux_raw, ivar, return_error=True, use_mask=physics.SC_USE_MASK)
        if not np.all(np.isfinite(norm_flux)):
            # A few non-finite pixels (deep cores / gaps) -> set flat with big err
            # so they self-down-weight; the net never sees a NaN target.
            bad = ~np.isfinite(norm_flux)
            norm_flux = np.where(bad, 1.0, norm_flux)
            norm_err = np.where(bad, np.inf, norm_err)
        # The raw continuum is raw_flux / norm_flux (the curve we divided by).
        with np.errstate(divide="ignore", invalid="ignore"):
            cont = np.where(norm_flux != 0, flux_raw / norm_flux, np.nan)
        cm = np.nanmean(cont) if np.any(np.isfinite(cont)) else 1.0
        cont_mean1 = (cont / cm) if cm != 0 else np.ones(NPIX)
        cont_mean1 = np.where(np.isfinite(cont_mean1), cont_mean1, 1.0)
        # Store. Replace inf error with a large finite value for storage/training.
        err_store = np.where(np.isfinite(norm_err), norm_err, 1e3)
        ids.append(sid)
        labels.append(lab)
        fluxes.append(norm_flux.astype(np.float32))
        errors.append(err_store.astype(np.float32))
        conts.append(cont_mean1.astype(np.float32))
        teffs.append(lab[0])
        if (k + 1) % 1000 == 0:
            print("  normalized %d/%d ..." % (k + 1, len(rows)), flush=True)
    print("Loaded %d dwarfs (missing npz=%d, bad=%d)"
          % (len(ids), n_missing, n_bad), flush=True)
    return (np.array(ids), np.array(labels, np.float32),
            np.array(fluxes, np.float32), np.array(errors, np.float32),
            np.array(conts, np.float32), np.array(teffs, np.float32))


# --------------------------------------------------------------------------- #
# 2. The Payne net (architecture matches physics._Payne, n_label=5).
# --------------------------------------------------------------------------- #
class Payne(nn.Module):
    def __init__(self, n_label=N_LABEL, n_hidden=300, n_pix=NPIX):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_label, n_hidden), nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(),
            nn.Linear(n_hidden, n_pix),
        )

    def forward(self, x):
        return self.net(x)


def norm_labels(labels, lmin, lmax):
    """Scale raw labels into ~[-0.5, 0.5], the SAME convention physics.py uses."""
    return (labels - lmin) / (lmax - lmin) - 0.5


# --------------------------------------------------------------------------- #
# 3. Training loop (per-pixel MLP, INVERSE-VARIANCE-weighted Huber, Adam, MPS).
#
#    WHY NOT PLAIN L1 (the previous recipe). The detector decides single-vs-binary
#    from a chi^2 = sum_pix ((obs-model)/sigma)^2 and the El-Badry Eq. B1 f_imp.
#    The training target is the NOISY observed normalized flux, whose per-pixel
#    noise sigma varies by an order of magnitude across the band. An unweighted L1
#    loss treats a 0.5%-noise line core and a 5%-noise telluric-residual pixel as
#    equally important, so the net spends capacity matching uninformative noisy
#    pixels and UNDER-FITS the high-S/N line cores that carry the binary signal.
#    The fitted single net then misses real spectra by ~5% RMS (>> photon noise),
#    chi2_single is dominated by net-misfit, and Delta-chi2 / f_imp stop isolating
#    the line-doubling. (Diagnosed 2026-06-23: SC net reduced chi2 17-46 on binary
#    primaries vs binspec ~1-2 on the SAME coadds.)
#
#    THE FIX. Minimize the SAME statistic the detector uses: a Huber loss on the
#    INVERSE-VARIANCE-normalized residual r = (pred - target)/sigma. This is exactly
#    a robustified per-pixel chi^2 -- the training objective now equals the
#    inference objective. The Huber transition (delta) keeps it chi^2-like in the
#    well-measured core (|r| <= delta) but L1-like in the tail (|r| > delta), so a
#    handful of label-noise / bad-pixel outliers cannot drag the fit (per-star
#    outliers are also removed by the iterative rejection below). Masked / gap
#    pixels (stored err >= ERR_MASK) get zero weight. A cosine-ish ReduceLROnPlateau
#    schedule + early stopping on a held-out MONITOR split replaces the fixed
#    250/400-epoch run so the net trains to convergence, not to an arbitrary stop.
# --------------------------------------------------------------------------- #
SIGMA_FLOOR = 0.005     # floor on normalized sigma (mirror the S/N>200 cap)
ERR_MASK = 100.0        # stored err >= this  -> masked pixel, zero training weight
HUBER_DELTA = 5.0       # |normalized residual| transition core(chi2)->tail(L1)


def _weighted_huber(pred, target, err):
    """Mean Huber loss on the inverse-variance-normalized residual (robust chi^2).

    r = (pred - target) / max(err, SIGMA_FLOOR); pixels with err >= ERR_MASK
    (gaps / masked) contribute zero. Returns a scalar torch loss.
    """
    sig = torch.clamp(err, min=SIGMA_FLOOR)
    good = (err < ERR_MASK).float()
    r = (pred - target) / sig
    absr = r.abs()
    quad = 0.5 * r * r
    lin = HUBER_DELTA * (absr - 0.5 * HUBER_DELTA)
    huber = torch.where(absr <= HUBER_DELTA, quad, lin)
    denom = good.sum().clamp(min=1.0)
    return (huber * good).sum() / denom


def train_net(X_norm, Y, E, X_mon, Y_mon, E_mon, max_epochs=1500, batch=256,
              lr=1e-3, patience=120, lr_patience=40, min_epochs=200):
    """Train a fresh Payne net minimizing the inverse-variance Huber (robust chi^2).

    X_norm/Y/E : scaled labels, SC-normalized flux target, SC-normalized error for
                 the TRAINING stars. X_mon/Y_mon/E_mon : the same for a held-out
                 MONITOR split used only for early stopping + LR scheduling (never
                 backpropagated). Returns the net at its BEST monitor loss.
    """
    net = Payne().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=lr_patience, min_lr=1e-5)
    Xt = torch.from_numpy(X_norm).to(DEVICE)
    Yt = torch.from_numpy(Y).to(DEVICE)
    Et = torch.from_numpy(E).to(DEVICE)
    Xm = torch.from_numpy(X_mon).to(DEVICE)
    Ym = torch.from_numpy(Y_mon).to(DEVICE)
    Em = torch.from_numpy(E_mon).to(DEVICE)
    n = Xt.shape[0]

    best_mon = float("inf")
    best_state = None
    best_ep = 0
    since_improve = 0
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(n, device=DEVICE)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = net(Xt[idx])
            loss = _weighted_huber(pred, Yt[idx], Et[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * idx.shape[0]
        # Monitor (held-out) loss drives the schedule + early stop.
        net.eval()
        with torch.no_grad():
            mon = float(_weighted_huber(net(Xm), Ym, Em).item())
        sched.step(mon)
        if mon < best_mon - 1e-4:   # ignore sub-1e-4 noise so early stop can fire
            best_mon = mon
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            best_ep = ep
            since_improve = 0
        else:
            since_improve += 1
        if (ep + 1) % 50 == 0 or ep == 0:
            lr_now = opt.param_groups[0]["lr"]
            print("    epoch %4d  train=%.5f  monitor=%.5f  best=%.5f@%d  lr=%.1e"
                  % (ep + 1, tot / n, mon, best_mon, best_ep + 1, lr_now),
                  flush=True)
        if ep + 1 >= min_epochs and since_improve >= patience:
            print("    early stop at epoch %d (no monitor gain for %d epochs)"
                  % (ep + 1, patience), flush=True)
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    print("    -> restored best monitor=%.5f from epoch %d"
          % (best_mon, best_ep + 1), flush=True)
    return net


# --------------------------------------------------------------------------- #
# 4. Single-fit residual reduced chi^2 (the rejection statistic).
# --------------------------------------------------------------------------- #
def net_reduced_chi2(net, labels, flux, err, lmin, lmax, batch=512):
    """Reduced chi^2 of net prediction vs SC-normalized flux, per star, good pix."""
    net.eval()
    lmin_t = torch.from_numpy(lmin).to(DEVICE)
    lmax_t = torch.from_numpy(lmax).to(DEVICE)
    preds = np.empty((labels.shape[0], NPIX), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, labels.shape[0], batch):
            Lt = torch.from_numpy(labels[i:i + batch]).to(DEVICE)
            Xn = (Lt - lmin_t) / (lmax_t - lmin_t) - 0.5
            preds[i:i + batch] = net(Xn).cpu().numpy()
    good = np.isfinite(flux) & np.isfinite(preds) & np.isfinite(err) & (err > 0)
    n = np.maximum(np.count_nonzero(good, axis=1), 1)
    r = np.where(good, (flux - preds) / err, 0.0)
    return np.sum(r * r, axis=1) / n


# --------------------------------------------------------------------------- #
# 5. Stage 4 continuum table: average the mean-1 per-chip Chebyshev continua in
#    Teff bins, on the RETAINED training stars only.
# --------------------------------------------------------------------------- #
def build_continuum_table(teffs, conts, n_nodes=12):
    """Average mean-1 continuum shapes in Teff bins -> (teff_nodes, cont_shapes).

    The shapes already each have mean 1 (the SED shape, scale removed). We bin in
    Teff, average within each bin, and re-normalize each averaged shape to mean 1.
    Empty bins are filled from the nearest non-empty bin so the node grid is dense.
    Returns (teff_nodes[K], cont_shapes[K, 8575]), sorted by Teff.
    """
    tmin, tmax = float(np.min(teffs)), float(np.max(teffs))
    edges = np.linspace(tmin, tmax, n_nodes + 1)
    nodes = 0.5 * (edges[:-1] + edges[1:])
    shapes = np.zeros((n_nodes, NPIX), dtype=np.float64)
    filled = np.zeros(n_nodes, dtype=bool)
    for i in range(n_nodes):
        sel = (teffs >= edges[i]) & (teffs <= edges[i + 1])
        if np.count_nonzero(sel) >= 5:
            m = np.mean(conts[sel].astype(np.float64), axis=0)
            shapes[i] = m / m.mean()
            filled[i] = True
    # Fill empty bins from the nearest filled bin (so interpolation is well posed).
    if not np.all(filled):
        fi = np.where(filled)[0]
        for i in range(n_nodes):
            if not filled[i]:
                j = fi[np.argmin(np.abs(fi - i))]
                shapes[i] = shapes[j]
    return nodes.astype(np.float64), shapes


def main():
    t0 = time.time()
    ids, labels, flux, err, conts, teffs = load_dataset()
    n = len(ids)

    # Hold out N_VAL dwarfs (never trained on). Fixed RNG for a stable split.
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    val_idx = perm[:N_VAL]
    train_pool = perm[N_VAL:]
    print("Validation held out: %d; training pool: %d"
          % (len(val_idx), len(train_pool)), flush=True)

    # Iterative binary / outlier rejection.
    active = list(train_pool)
    rejected = {}
    net = None
    final_lmin = final_lmax = None
    for it in range(N_ITER):
        idx = np.array(active)
        lmin = labels[idx].min(axis=0)
        lmax = labels[idx].max(axis=0)
        Xn = norm_labels(labels[idx], lmin, lmax)
        # Carve a MONITOR split (10%) from the active set for early stopping + the
        # LR schedule. It is held out of back-prop within this pass but is allowed
        # to re-enter training in the next rejection pass (it is NOT the final
        # N_VAL reporting set, which is held out of every pass). Fixed RNG/pass.
        mrng = np.random.default_rng(1000 + it)
        order = mrng.permutation(len(idx))
        n_mon = max(64, len(idx) // 10)
        mon_local = order[:n_mon]
        tr_local = order[n_mon:]
        # Budget: shorter for the rejection passes, full for the final net.
        max_ep = 600 if it < N_ITER - 1 else 1500
        print("[iter %d] N_train=%d (monitor=%d)  max_epochs=%d"
              % (it, len(tr_local), len(mon_local), max_ep), flush=True)
        net = train_net(
            Xn[tr_local], flux[idx][tr_local], err[idx][tr_local],
            Xn[mon_local], flux[idx][mon_local], err[idx][mon_local],
            max_epochs=max_ep)

        rchi2 = net_reduced_chi2(net, labels[idx], flux[idx], err[idx],
                                 lmin, lmax)
        med = float(np.median(rchi2))
        print("[iter %d] median single-fit reduced_chi2=%.3f" % (it, med),
              flush=True)
        final_lmin, final_lmax = lmin, lmax

        if it < N_ITER - 1:
            mad = float(np.median(np.abs(rchi2 - med)))
            sigma = 1.4826 * mad
            robust_thr = med + 5.0 * sigma
            pct90_thr = float(np.quantile(rchi2, 0.90))
            thr = max(robust_thr, pct90_thr)
            keep_mask = rchi2 <= thr
            rej = idx[~keep_mask]
            for j, c in zip(rej, rchi2[~keep_mask]):
                rejected[int(j)] = float(c)
            n_rej = int(np.count_nonzero(~keep_mask))
            active = list(idx[keep_mask])
            print("[iter %d] rejected=%d (thr=%.2f; robust=%.2f, p90=%.2f) "
                  "-> N_next=%d" % (it, n_rej, thr, robust_thr, pct90_thr,
                                    len(active)), flush=True)
        else:
            print("[iter %d] final iteration, no rejection" % it, flush=True)

    # Cumulative rejected (binary/peculiar) candidates.
    rej_csv = os.path.join(RES_DIR, "payne_dr19_sc_rejected.csv")
    order = sorted(rejected.items(), key=lambda kv: -kv[1])
    with open(rej_csv, "w") as fh:
        fh.write("sdss_id,reduced_chi2\n")
        for j, c in order:
            fh.write("%d,%.4f\n" % (ids[j], c))
    print("Wrote %d candidate binary/peculiar dwarfs to %s"
          % (len(order), rej_csv), flush=True)

    # Save the SC net in the schema physics.py loads.
    pt_path = os.path.join(MODELS_DIR, SC_MODEL_OUT)
    torch.save({
        "state_dict": net.state_dict(),
        "label_min": final_lmin.astype(np.float32),
        "label_max": final_lmax.astype(np.float32),
        "label_names": LABEL_NAMES,
        "n_label": N_LABEL, "n_hidden": 300, "n_pix": NPIX,
    }, pt_path)
    print("Saved OUR SC 5-label model to %s" % pt_path, flush=True)
    print("  label_min = %s" % final_lmin, flush=True)
    print("  label_max = %s" % final_lmax, flush=True)

    # Stage 4: build + save OUR un-normalization continuum table from the RETAINED
    # stars (active set), so the SED-shape table reflects the clean single-star fit.
    act = np.array(active)
    teff_nodes, cont_shapes = build_continuum_table(teffs[act], conts[act])
    np.savez(os.path.join(MODELS_DIR, SC_CONT_OUT),
             teff_nodes=teff_nodes, cont_shapes=cont_shapes)
    print("Saved Stage-4 un-normalization continuum table "
          "(%d Teff nodes %.0f-%.0f K) to models/%s"
          % (len(teff_nodes), teff_nodes[0], teff_nodes[-1], SC_CONT_OUT), flush=True)

    # Validation on the held-out dwarfs.
    net.eval()
    lmin_t = torch.from_numpy(final_lmin).to(DEVICE)
    lmax_t = torch.from_numpy(final_lmax).to(DEVICE)
    with torch.no_grad():
        Lv = torch.from_numpy(labels[val_idx]).to(DEVICE)
        Xn = (Lv - lmin_t) / (lmax_t - lmin_t) - 0.5
        pred = net(Xn).cpu().numpy()
    fv = flux[val_idx]
    abs_res = np.abs(pred - fv)
    good = np.isfinite(fv) & (fv > 0.1)
    med_abs = float(np.median(abs_res[good]))
    frac_res = abs_res / np.maximum(np.abs(fv), 1e-3)
    med_frac = float(np.median(frac_res[good]))
    print("\nVALIDATION (%d held-out dwarfs): median |model-flux| = %.4f  "
          "(median fractional = %.4f)" % (len(val_idx), med_abs, med_frac),
          flush=True)

    ev = err[val_idx]
    g = np.isfinite(fv) & np.isfinite(ev) & (ev > 0)
    r = np.where(g, (pred - fv) / ev, 0.0)
    ex_chi2 = np.sum(r * r, axis=1) / np.maximum(np.count_nonzero(g, axis=1), 1)
    print("Example net reduced_chi2 (first 5 val dwarfs): %s"
          % [round(float(x), 2) for x in ex_chi2[:5]], flush=True)

    # Save the held-out validation FLUXES + errors (not just ids/labels) so the
    # net's reduced chi2 on truly-unseen stars is reproducible without re-fetching
    # the raw cloud spectra (the previous schema stored ids/labels only).
    np.savez(os.path.join(MODELS_DIR, VAL_OUT),
             ids=ids[val_idx], labels=labels[val_idx],
             fluxes=flux[val_idx], errors=err[val_idx])
    print("Saved validation ids/labels/fluxes/errors to models/%s" % VAL_OUT,
          flush=True)
    print("\nStage 3 total runtime: %.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
