# Eccentricity chain: rerun on the corrected 16-visit deep table

The twin vs non-twin eccentricity numbers (paper2 Section `sec:res-ecc`, Appendix
`app:orbit`) were computed from wrong per-visit velocities (frame bug, and
velocities in S/N order matched to epochs in time order). The fitter now writes
`mjd_per_visit` in the order of `v1_per_visit`, and every script in the chain
reads a system's epochs from its own row of the deep table through
`scripts/deep_table.py`. Nothing reads `resources/census/stage2_mjds.csv` any
more: that file belongs to the main run (8-visit cap, other stars).

## Method: two-velocity fit to the untied pairs (default)

Steps 3-6 now fit the untied per-visit component velocities
(`rv1_untied_per_visit`, `rv2_untied_per_visit`) with a two-velocity Keplerian
model, summing over the label assignment at each visit, instead of the primary
velocity `v1_per_visit` and its tie inversion. The v1-only model could fold the
curve about gamma at q near 1 and returned too small a K1 for the twins. The
method, the mass-ratio choice and the tests are in `docs/ecc_twovelocity_method.md`.

Every script in steps 3-6 takes `--method twovel` (default) or `--method v1`
(the previous method, unchanged, for reproducing the earlier numbers).
`orbit_deep_posteriors.csv` and `ecc_marginal_real.csv` carry a `method` column;
`ecc_real_marginal.py` warns if the orbit file came from another method, and
`ecc_null_full.py` refuses to run if `ecc_marginal_real.csv` came from another
method (a file without the column counts as `v1`). A full rerun with the new
method repeats steps 3-9 in order; step 1 (deep table) and step 2 are unchanged.
To reproduce the old chain, pass `--method v1` to steps 3-6 (the sbatch files pass
arguments through only for step 6; for steps 3-5 add the flag to the python line).

Environment variables

| variable | used by | default |
|---|---|---|
| `A4B_DEEP_TABLE` | every script below that reads epochs | `resources/census/stage2_deep.csv` (relative paths are taken from the repo root) |
| `WORKERS` | `ecc_null_full.py` | 8 (the sbatch sets 40) |

`sbatch` passes the submitting shell's environment to the job, so export
`A4B_DEEP_TABLE` before submitting if you use a non-default table.

## 0. Move the old outputs aside (in both checkouts, laptop and Pitzer)

```bash
mkdir -p resources/census/pre_epochfix
for f in stage2_deep.csv orbit_deep_posteriors.csv ecc_marginal_real.csv ecc_validation.csv \
         ecc_null_full.csv ecc_null_dalpha.csv ecc_variants.csv ecc_headline.json; do
  [ -e resources/census/$f ] && mv resources/census/$f resources/census/pre_epochfix/
done
rm -rf resources/census/orbit_deep_shards resources/census/ecc_real_shards resources/census/ecc_val_shards
```

Four of these are tracked in git (`ecc_marginal_real.csv`, `ecc_validation.csv`,
`ecc_null_dalpha.csv`, `ecc_variants.csv`). They show as deleted until the rerun
writes them back to the same paths, after which they show as modified. Moving
them rather than overwriting in place means a step cannot quietly read a stale file.

Keep `orbit_run_params.csv` (reused), the catalogs, and `stage2_mjds.csv` (still
the right epochs for `stage2_catalog_full.csv` and the `supereccentric/` scripts).
The old `ecc_null_full.csv` has no `.inputs.json` sidecar, so `ecc_null_full.py`
refuses to append to it anyway. The old null file also holds 39 repeated rows
from an earlier resume, which the new resume key (alpha, rep, sdss_id) prevents.

## 1. Deep table (Pitzer login node, or locally after copying the shards)

The 16-visit shards come from `scripts/stage2_census_shard.py --max-visits 16`
(that run is set up separately). With the shard directory in `$DEEP_SHARDS`:

```bash
python scripts/build_stage2_deep.py --shards $DEEP_SHARDS
# -> resources/census/stage2_deep.csv ; prints rows read, drops by reason, kept, twins
```

Several directories may be given (for example a merged main run as well); the
`n_visits >= 8` cut picks out the deep stars, and a star in more than one shard
keeps its row with the most visits. The old deep table had 4,225 rows.

## 2. Orbit parameters

`resources/census/orbit_run_params.csv` is reused unchanged.

## 3. Orbit posteriors (Pitzer, `scripts/orbit_deep.sbatch`, new)

`orbit_deep_shard.py` runs `orbit_sampler.sample_twovel_refined` on each row's
untied pairs (visits with a missing or non-finite pair dropped; fewer than 6
usable visits gives `status=too_few_epochs`). Output columns as before (`n_ep` is
the number of usable visits, `s_jit` the jitter applied to both components), plus
`method` = `twovel`.

```bash
mkdir -p logs
sbatch --array=0 scripts/orbit_deep.sbatch        # one-task preflight: read CPU time from sacct
sbatch --array=1-99 scripts/orbit_deep.sbatch     # the rest, after checking the cost
python scripts/concat_shards.py --shards resources/census/orbit_deep_shards --expect 100 \
    --key sdss_id --check-deep --out resources/census/orbit_deep_posteriors.csv
```

## 4. Eccentricity likelihoods (Pitzer, `scripts/ecc_real_marginal.sbatch`, new)

Needs steps 1 and 3. Uses `ecc_marginal.system_loglike_on_egrid_twovel` on the
untied pairs (at least 8 usable visits), same e grid, same 60k draws per grid
point, same sigma (1.5 km/s, now on each component). The sample is still
`status=ok` and 6 <= P50 < 400 from step 3, so it changes with the new orbits.
Output columns as before plus `method`.

```bash
sbatch --array=0 scripts/ecc_real_marginal.sbatch
sbatch --array=1-39 scripts/ecc_real_marginal.sbatch
python scripts/concat_shards.py --shards resources/census/ecc_real_shards --expect 40 \
    --key sdss_id --out resources/census/ecc_marginal_real.csv
```

## 5. Injection validation (Pitzer, `scripts/ecc_val_suite.sbatch`, new)

Needs step 1 only (cadences, and the catalog `best_q` for the q pools), so it can
run alongside steps 3 and 4. With `--method twovel` each mock is a pair of
component velocities with q drawn from the real twin (or non-twin) q values,
exchanged at each visit with probability 0.5, and fitted with the same
two-velocity likelihood as the data. Output format unchanged.

```bash
sbatch --array=0 scripts/ecc_val_suite.sbatch
sbatch --array=1-15 scripts/ecc_val_suite.sbatch
python scripts/concat_shards.py --shards resources/census/ecc_val_shards --expect 16 \
    --key da_true,rep --rows 16 --out resources/census/ecc_validation.csv
```

## 6. Injection null (Pitzer, `scripts/ecc_null_full.sbatch`, existing, 40 CPUs x 1 h)

Needs step 4. Arguments pass through to the script. With `--method twovel`
(default) each mock inherits the real system's usable-pair epochs, K1, P50 and
q, and is built and fitted as in step 5. The method is recorded in the sidecar,
so a resume with the other method is refused. Output format unchanged.

```bash
sbatch scripts/ecc_null_full.sbatch --fresh     # first submission: start a new file
sbatch scripts/ecc_null_full.sbatch             # resubmit WITHOUT --fresh if it timed out
```

The script writes `resources/census/ecc_null_full.csv` and a sidecar
`ecc_null_full.csv.inputs.json` (row count and md5 of the deep table and of
`ecc_marginal_real.csv`, plus the run settings). A resume against different
inputs is refused. `--out FILE` writes elsewhere.

## 7-9. Local steps (seconds to a few minutes)

Copy back from Pitzer: `stage2_deep.csv`, `orbit_deep_posteriors.csv`,
`ecc_marginal_real.csv`, `ecc_validation.csv`, `ecc_null_full.csv` and
`ecc_null_full.csv.inputs.json` (all under `resources/census/`).

```bash
python scripts/ecc_null_dalpha.py        # -> ecc_null_dalpha.csv (per-realization dAlpha, ecc_solve logic)
python scripts/ecc_bootstrap.py          # K1-cut table (prints only)
python scripts/ecc_solve.py              # measured, bootstrap, permutation, pooled null bias (prints only)
python scripts/ecc_final_table.py        # -> ecc_variants.csv, then ecc_headline.json
python scripts/ecc_robust.py             # optional cross-check (prints only)
python paper2/figures/fig_ecc_validation.py   # reads ecc_validation.csv, ecc_null_dalpha.csv, ecc_headline.json
```

`ecc_final_table.py` and `ecc_robust.py` read the `method` column of
`ecc_marginal_real.csv`: for `twovel` the observed-only variant uses
`dvmax` = max_i |a_i - b_i| of the untied pairs instead of the range of
`v1_per_visit` (bins and the `dvmax > 25` cut unchanged). The other local steps
read only the loglike grids, K1 and n_ep, and need no change.

`ecc_headline.py` can also be run on its own. It refuses to run if
`ecc_variants.csv` was made from a different `ecc_marginal_real.csv` (the K1 > 12
twin and non-twin counts must agree).

## Pitzer cost

None of the new array jobs has a measured cost. The two-velocity sampler takes
about the same time per system as the v1-only one (about 5 s at the default
200k draws on a laptop core); the two-velocity e-grid likelihood does a little
more array work per draw than the v1-only one, so preflight step 4 again. The time limits above are
ceilings, not estimates: orbit_deep 100 x 3 h, ecc_real 40 x 3 h, ecc_val 16 x 6 h,
ecc_null 40 CPU x 1 h per submission. Run the one-task preflights, scale the
sacct CPU time, and ask before any submission above 50 CPU-h.

## Paper numbers to update from `resources/census/ecc_headline.json`

| paper text (current) | JSON key |
|---|---|
| 4,225 systems with eight or more visits | `n_deep_rows` |
| 2,009 systems with P in 6-400 d | `n_systems` (twins: `n_twins`) |
| median K1 11.8 vs 8.8 km/s (twins vs non-twins) | `median_K1_twin`, `median_K1_nontwin` |
| 896 systems with K1 > 12, 495 twins | `n_k1gt12`, `n_k1gt12_twins` |
| alpha_twin = +0.15, alpha_non-twin = +0.41 | `alpha_twin`, `alpha_nontwin` |
| -0.24 +/- 0.16 (text, caption, figure band) | `combined`, `error` |
| null -0.02 +/- 0.03 over 24 realizations | `null_bias`, `null_err`, `null_n_reps` |
| moves by less than 0.1 across choices | `max_variant_shift` |
| mean e 0.51 and 0.56 | `mean_e_twin`, `mean_e_nontwin` |
| reaches zero at 1.5 sigma | `significance` |
| +0.15 outside two sigma for every choice | `variants[*].upper_2sigma` |
| slope near 0.9 | `validation_slope` |
