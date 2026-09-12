#!/usr/bin/env python3
# =========================================================================== #
# run_census_stage1.py
#
# DR19 dwarf SB2 census -- STAGE-1 COMBINED-SPECTRUM DETECTOR.
#
# This is the production coadd runner, but the current catalog is PROVISIONAL:
# the 2026-06-23 binspec fair test showed a detector sensitivity gap on the same
# DR19 coadds (binspec 69.6% recovery vs SC 32.8%; q=1/search fix lifts SC only
# the production detector (binspec, --detector binspec). Run this to (re)generate
# the paper/catalog numbers. It writes stage1_shards/.
#
# PER STAR, the pipeline is:
#   1. download the star's mwmStar COADD RAW flux + ivar
#      (download_dr19_raw.fetch_one -> a <sdss_id>.npz of {wl, flux_raw, ivar});
#   2. run physics.dr19_sc_single_vs_binary(flux_raw, ivar, seed=catalog_labels);
#   3. write ONE catalog row recording the verdict + statistics. Columns:
#        sdss_id, teff_seed, logg_seed, feh_seed       (catalog 5-label seed)
#        prefers_binary, verdict                        (BINARY / SINGLE)
#        delta_chi2, f_imp, min_fimp_required
#        best_q, best_rv1, best_rv2                      (binary-fit params)
#        teff_single, logg_single, feh_single           (single-model labels)
#        error                                          (any per-star exception)
#   NEVER drop a star: a download failure or a fit exception is RECORDED in the
#   `error` field and the star still gets a row.
#
# RESUMABLE. The per-shard catalog (catalog_shard_<i>.csv) is the checkpoint: a
# star already in it (with a verdict or a recorded error) is skipped on a
# re-boot. COADD npzs are idempotent (a non-trivial existing npz is skipped), so
# a re-boot resumes both the downloads and the fits. The startup script's
# background loop copies this CSV to GCS every ~2 min.
#
# PARALLELISM. COADD downloads run with the project's threaded downloader
# (download_dr19_raw.fetch_many) in batches. The SC fits run across PROCESSES
# under 'spawn' with torch / BLAS capped to ONE thread per worker BEFORE
# importing physics (torch is not fork-safe; its intra-op pool DEADLOCKS under
# 'spawn' when many workers each spin one up). We interleave by BATCH so the
# network and the CPU overlap.
#
# CLI:
#   python run_census_stage1.py \
#       --manifest manifest_shard_<i>.csv \
#       --raw-dir /mnt/coadds \
#       --out catalog_shard_<i>.csv \
#       --workers <N> \
#       [--batch-size 512]
# =========================================================================== #

import os

# Cap BLAS / OpenMP to one thread PER PROCESS BEFORE numpy / torch import. We
# parallelize across processes, so per-worker multithreaded BLAS only
# oversubscribes; torch's intra-op pool also DEADLOCKS under 'spawn' when many
# workers each spin one up.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
import time
import warnings
from multiprocessing import cpu_count, get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# --------------------------------------------------------------------------- #
# 62%-detector patch: run binspec's PUBLIC fitting layer (normalization + Eq.B1)
# but with OUR MLP line net (AB-8 curated, binspec normalization) as the
# normalized-spectrum emulator, so NO binspec trained line weights are used.
# This is the openly reproduced 62% detector (matched-purity benchmark). Applied
# once per worker; idempotent guard. Net path via AB_MLP_NET (default curated).
# --------------------------------------------------------------------------- #
_MLP_PATCHED = False

def _patch_binspec_mlp():
    global _MLP_PATCHED
    if _MLP_PATCHED:
        return
    import torch
    pt = os.environ.get("AB_MLP_NET", os.path.join(_ROOT, "models", "payne_dr19_curated.pt"))
    c = torch.load(pt, map_location="cpu", weights_only=False)
    s = c["state_dict"]
    W0 = s["net.0.weight"].numpy().astype(np.float64); b0 = s["net.0.bias"].numpy().astype(np.float64)
    W1 = s["net.2.weight"].numpy().astype(np.float64); b1 = s["net.2.bias"].numpy().astype(np.float64)
    W2 = s["net.4.weight"].numpy().astype(np.float64); b2 = s["net.4.bias"].numpy().astype(np.float64)
    lmin = np.asarray(c["label_min"], np.float64); lmax = np.asarray(c["label_max"], np.float64)
    span = lmax - lmin
    lr = lambda z: np.where(z > 0, z, 0.01 * z)
    def fwd(labels):
        lab = np.clip(np.asarray(labels, np.float64), lmin, lmax)
        xs = (lab - lmin) / span - 0.5
        h = lr(W0 @ xs + b0); h = lr(W1 @ h + b1); return W2 @ h + b2
    import vendor.binspec.spectral_model as sm
    orig = sm.get_spectrum_from_neural_net
    def patched(labels, NN_coeffs, normalized=False):
        if normalized:
            return fwd(labels)
        return orig(labels, NN_coeffs, normalized)
    sm.get_spectrum_from_neural_net = patched
    _MLP_PATCHED = True

# --------------------------------------------------------------------------- #
# Output schema. One row per star. Kept stable so a resumed run reads the same
# header. This matches the Stage-1 detector return-dict keys.
# --------------------------------------------------------------------------- #
FIELDS = [
    "sdss_id",
    "teff_seed", "logg_seed", "feh_seed",            # catalog 5-label seed (3 shown)
    "verdict", "prefers_binary",                      # BINARY / SINGLE
    "delta_chi2", "f_imp", "min_fimp_required",
    "best_q", "best_rv1", "best_rv2",                 # binary-fit params
    "teff_single", "logg_single", "feh_single",
    "vmacro_single", "vmacro_1", "vmacro_2",          # fitted broadening, single and components
    "labels_single", "labels_binary",                 # full best-fit label vectors
    "error",                                          # any per-star exception
]

DEFAULT_SEED = (5500.0, 4.5, 0.0, 0.0, 5.0)          # solar-dwarf fallback seed


def _clean_seed(seed):
    """NaN / None / missing seed labels -> the dwarf default (stay in the SC box)."""
    if seed is None:
        return DEFAULT_SEED
    out = []
    for i in range(5):
        try:
            v = float(seed[i])
        except (TypeError, ValueError, IndexError):
            v = float("nan")
        out.append(DEFAULT_SEED[i] if not np.isfinite(v) else v)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Worker (spawn). One task = (sdss_id, seed_5tuple, raw_dir). The worker LOADS
# the COADD npz from raw_dir itself and runs the SELF-CONSISTENT combined-
# spectrum detector dr19_sc_single_vs_binary on the RAW (flux_raw, ivar). Imports
# physics INSIDE the child (each loads the SC net once in its own address space)
# and caps torch to one thread BEFORE that import. Never raises: a bad star
# records an `error` and the batch continues.
# --------------------------------------------------------------------------- #
def _fit_one(task):
    # task = (sdss_id, seed, raw_dir, detector). detector in {"binspec","sc"};
    # "binspec" is the PRODUCTION detector (faithful El-Badry reproduction: synthetic
    # net + Eq.B1 f_imp + Table B1). "sc" is OUR all-real-data net (capped ~49%
    # matched-purity; kept as a documented method comparison, see METHODS.md).
    sdss_id, seed, raw_dir, detector = task
    base = {k: "" for k in FIELDS}
    base["sdss_id"] = sdss_id
    s = _clean_seed(seed)
    base["teff_seed"], base["logg_seed"], base["feh_seed"] = s[0], s[1], s[2]
    try:
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import physics

            npz_path = os.path.join(raw_dir, "%d.npz" % int(sdss_id))
            if not (os.path.exists(npz_path) and os.path.getsize(npz_path) > 1000):
                base["error"] = "no_coadd_file"
                return base
            d = np.load(npz_path)
            flux = np.asarray(d["flux_raw"], float)
            ivar = np.asarray(d["ivar"], float)

            # binspec draws its extra binary starts with np.random, so seed per star
            # to make each fit reproducible.
            np.random.seed(int(sdss_id) % (2**32 - 1))
            if detector in ("binspec", "binspec_mlp"):
                # binspec's PUBLIC fitting layer (normalization/masking + Eq.B1). For
                # binspec_mlp we first swap in OUR curated MLP line net (62% detector,
                # no binspec trained line weights); plain binspec uses its own net.
                if detector == "binspec_mlp":
                    _patch_binspec_mlp()
                err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), 0.0)
                res = physics.binspec_single_vs_binary(physics.WAVELENGTH, flux, err)
            else:
                res = physics.dr19_sc_single_vs_binary(flux, ivar, seed=s)

        base.update({
            "verdict": "BINARY" if res["prefers_binary"] else "SINGLE",
            "prefers_binary": bool(res["prefers_binary"]),
            "delta_chi2": float(res["delta_chi2"]),
            "f_imp": float(res["f_imp"]),
            "min_fimp_required": ("" if res.get("min_fimp_required") is None
                                  else float(res["min_fimp_required"])),
            "best_q": float(res["best_q"]),
            "best_rv1": float(res["best_rv1"]),
            "best_rv2": float(res["best_rv2"]),
            "teff_single": float(res["teff_single"]),
            "logg_single": float(res["logg_single"]),
            "feh_single": float(res["feh_single"]),
            "vmacro_single": res.get("vmacro_single", ""),
            "vmacro_1": res.get("vmacro_1", ""),
            "vmacro_2": res.get("vmacro_2", ""),
            "labels_single": ";".join("%.6g" % x for x in res.get("labels_single", [])),
            "labels_binary": ";".join("%.6g" % x for x in res.get("labels_binary", [])),
        })
        return base
    except Exception as exc:  # one bad star must not kill the batch
        base["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:140])
        return base


# --------------------------------------------------------------------------- #
# Resume / checkpoint I/O.
# --------------------------------------------------------------------------- #
def _load_existing(out_path):
    """Read the existing per-shard catalog into sdss_id -> row dict (for resume)."""
    rows = {}
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    rows[int(r["sdss_id"])] = r
                except (KeyError, ValueError):
                    pass
    return rows


def _write_all(out_path, rows_by_id):
    """Rewrite the whole catalog (atomic-ish via .tmp + replace)."""
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for sid in sorted(rows_by_id):
            w.writerow({k: rows_by_id[sid].get(k, "") for k in FIELDS})
    os.replace(tmp, out_path)


def _done(row):
    """A resumed row is complete iff it has a verdict OR a recorded error."""
    return (str(row.get("verdict", "")).strip() != ""
            or str(row.get("error", "")).strip() != "")


def _load_manifest_seeds(manifest_path):
    """Read sdss_id -> 5-label seed tuple from a shard manifest CSV (ordered)."""
    ids = []
    seeds = {}
    with open(manifest_path) as f:
        for r in csv.DictReader(f):
            try:
                sid = int(r["sdss_id"])
            except (KeyError, ValueError):
                continue
            # sdss_id == 0 is the bad placeholder row in the census manifest; skip.
            if sid == 0:
                continue
            ids.append(sid)
            try:
                seeds[sid] = (float(r["teff"]), float(r["logg"]), float(r["fe_h"]),
                              float(r["mg_h"]), float(r["v_macro"]))
            except (KeyError, ValueError):
                seeds[sid] = None
    return ids, seeds


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="DR19 dwarf SB2 census -- Stage-1 combined-spectrum detector "
                    "on EVERY star with the binspec production detector.")
    ap.add_argument("--manifest", required=True,
                    help="shard manifest CSV (sdss_id + 5 labels)")
    ap.add_argument("--raw-dir", required=True,
                    help="dir for <sdss_id>.npz mwmStar COADD spectra")
    ap.add_argument("--out", required=True,
                    help="per-shard catalog CSV (append / resume)")
    ap.add_argument("--detector", choices=["binspec", "sc", "binspec_mlp"], default="binspec",
                    help="binspec = PRODUCTION (El-Badry reproduction; default); "
                         "sc = our all-real-data net (method comparison, ~49% cap)")
    ap.add_argument("--no-download", action="store_true",
                    help="skip the network download phase (coadds already on disk; "
                         "required on internet-less compute nodes)")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    ap.add_argument("--batch-size", type=int, default=512,
                    help="stars per download+fit batch (overlaps net and CPU)")
    ap.add_argument("--checkpoint-every", type=int, default=200,
                    help="rewrite the catalog after this many new fits")
    args = ap.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)

    ids, seeds = _load_manifest_seeds(args.manifest)
    rows_by_id = _load_existing(args.out)

    todo = [i for i in ids
            if i not in rows_by_id or not _done(rows_by_id[i])]
    print("STAGE-1 CENSUS (combined-spectrum, binspec production detector): manifest stars=%d  "
          "done=%d  todo=%d  workers=%d  batch=%d"
          % (len(ids), len(ids) - len(todo), len(todo), args.workers,
             args.batch_size), flush=True)
    if not todo:
        print("nothing to do; catalog already complete.", flush=True)
        return

    # The downloader pulls in `requests`; only import it when we will actually
    # fetch. With --no-download (offline compute node, pre-staged coadds) the
    # network path is never taken, so it must not be a hard dependency.
    dlr = None
    if not args.no_download:
        import download_dr19_raw as dlr

    ctx = get_context("spawn")
    t0 = time.time()
    done_n = 0
    n_err = 0
    n_bin = sum(1 for r in rows_by_id.values()
                if str(r.get("verdict", "")) == "BINARY")
    # Live sanity band for the running flag rate: the 63% real-data run flagged
    # 15.4% raw, so a healthy gap-detector run should sit in roughly [8, 22]%.
    # Outside that band (after enough stars), or an error rate above 5%, prints a
    # greppable WARN so a monitor can flag a bad run before it burns all shards.
    FLAG_LO, FLAG_HI, ERR_MAX, WARN_AFTER = 0.08, 0.22, 0.05, 2000

    # One persistent spawn pool for all batches (re-creating it per batch pays the
    # physics import N times). Download each batch's COADDs with the threaded
    # downloader, then SC-fit it on the pool while we checkpoint.
    with ctx.Pool(args.workers) as pool:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]

            # 1. Get this batch's COADDs present on disk. On a compute node with no
            #    internet, --no-download skips fetching and uses whatever is already
            #    there (pre-staged on the login node).
            if args.no_download:
                present = set(sid for sid in batch if os.path.exists(
                    os.path.join(args.raw_dir, "%d.npz" % sid))
                    and os.path.getsize(os.path.join(args.raw_dir, "%d.npz" % sid)) > 1000)
            else:
                id_label_rows = [(sid, seeds.get(sid) or {}) for sid in batch]
                present = dlr.fetch_many(
                    id_label_rows, out_dir=args.raw_dir,
                    label="batch%d" % (start // args.batch_size))

            # Stars with no COADD at all get a recorded note (still a row).
            for sid in batch:
                if sid not in present:
                    row = {k: "" for k in FIELDS}
                    row["sdss_id"] = sid
                    s = _clean_seed(seeds.get(sid))
                    row["teff_seed"], row["logg_seed"], row["feh_seed"] = (
                        s[0], s[1], s[2])
                    row["error"] = "no_coadd_file"
                    rows_by_id[sid] = row

            # 2. SC-fit the stars whose COADD is present, on the pool.
            tasks = [(sid, seeds.get(sid), args.raw_dir, args.detector)
                     for sid in batch if sid in present]
            for res in pool.imap_unordered(_fit_one, tasks, chunksize=1):
                sid = int(res["sdss_id"])
                rows_by_id[sid] = res
                done_n += 1
                if res.get("verdict") == "BINARY":
                    n_bin += 1
                if res.get("error"):
                    n_err += 1
                if done_n % args.checkpoint_every == 0:
                    _write_all(args.out, rows_by_id)
                    rate = done_n / (time.time() - t0)
                    eta = (len(todo) - done_n) / rate / 3600.0 if rate else 0
                    n_fit = done_n - n_err
                    frac = (n_bin / n_fit) if n_fit else 0.0
                    err_rate = (n_err / done_n) if done_n else 0.0
                    print("  %d/%d  %.2f star/s  ETA %.2f h  BINARY=%d "
                          "(%.1f%% of fit)  err=%d (%.1f%%)"
                          % (done_n, len(todo), rate, eta, n_bin, 100 * frac,
                             n_err, 100 * err_rate), flush=True)
                    if done_n >= WARN_AFTER and (
                            frac < FLAG_LO or frac > FLAG_HI
                            or err_rate > ERR_MAX):
                        print("  WARN shard %s looks off: flag=%.1f%% "
                              "(band %.0f-%.0f%%) err=%.1f%% -- inspect before "
                              "trusting the full run"
                              % (os.environ.get("SLURM_ARRAY_TASK_ID", "?"),
                                 100 * frac, 100 * FLAG_LO, 100 * FLAG_HI,
                                 100 * err_rate), flush=True)

            # Checkpoint at every batch boundary too (covers the no-coadd rows).
            _write_all(args.out, rows_by_id)

    _write_all(args.out, rows_by_id)

    # ----------------------------------------------------------------------- #
    # Final summary.
    # ----------------------------------------------------------------------- #
    n_tot = len(rows_by_id)
    n_bin = sum(1 for r in rows_by_id.values()
                if str(r.get("verdict", "")) == "BINARY")
    n_single = sum(1 for r in rows_by_id.values()
                   if str(r.get("verdict", "")) == "SINGLE")
    n_err = sum(1 for r in rows_by_id.values()
                if str(r.get("error", "")).strip() != "")
    frac = (n_bin / (n_bin + n_single)) if (n_bin + n_single) else 0.0
    print("=" * 64, flush=True)
    print("STAGE-1 CENSUS SHARD COMPLETE (combined-spectrum, binspec production detector)",
          flush=True)
    print("  rows in catalog:   %d" % n_tot, flush=True)
    print("  verdict BINARY:    %d" % n_bin, flush=True)
    print("  verdict SINGLE:    %d" % n_single, flush=True)
    print("  binary fraction:   %.1f%% (of stars with a verdict)" % (100 * frac),
          flush=True)
    print("  errors / no-coadd: %d" % n_err, flush=True)
    print("  catalog -> %s" % args.out, flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
