# Column descriptions for the released tables

## dr19_sb2_catalog_open.csv — the DR19 SB2 catalog (41,466 rows)

| column | unit | description |
|---|---|---|
| sdss_id | — | SDSS-V identifier |
| gaia_dr3_source_id | — | Gaia DR3 identifier, taken as an exact string from the census input table and checked against Gaia DR3 through VizieR (I/355/gaiadr3): the source exists and lies within 2 arcsec of the APOGEE position. Blank for the 601 stars that fail the check. The first-revision release stored this column through a floating-point conversion that rounds identifiers above 2^53, so about half of those values pointed to no Gaia source |
| ra, dec | deg | J2000 coordinates (ICRS) from the APOGEE input table; blank for 176 stars |
| teff_seed, logg_seed, feh_seed | K, dex, dex | pipeline labels used to seed the fit |
| verdict | — | classifier verdict string |
| prefers_binary | bool | two-component fit beats the single-star fit |
| delta_chi2 | — | chi2(single) − chi2(binary) |
| f_imp | — | improvement fraction (El-Badry et al. 2018, Eq. B1) |
| min_fimp_required | — | the Table B1 rung the star had to clear |
| best_q | — | mass ratio of the two-component model fitted to the combined spectrum. It is a parameter of the composite model (the flux ratio mapped onto mass through one 4 Gyr isochrone), not a measured mass ratio; where the visits allow, use q_dyn or q_wilson of the multi-epoch supplement |
| best_rv1, best_rv2 | km/s | component velocities in the combined spectrum, relative to the DR19 pipeline rest frame of the coadd |
| teff_single, logg_single, feh_single | K, dex, dex | best single-star fit labels |
| error | — | non-empty if the fit failed |
| sb2 | bool | accepted as SB2 at the recalibrated gate (all rows in this file) |
| teff_floor | flag | 1 if the seed temperature sits at the 4200 K lower edge of the model grid, where the single-star labels are unreliable; cut on this for a clean sample |
| feh_metalpoor | flag | 1 if the pipeline [Fe/H] < -0.9, where the control false-positive rate is 38% against 14% for the rest |
| coadd_only | flag | 1 if the coadd shows a split the visits do not: two or more fitted visits in the multi-epoch supplement, abs(best_rv1 - best_rv2) > 15 km/s, DR19 pipeline velocity standard deviation < 1 km/s, and a visit fit that prefers one star. A pair that far apart would show two sets of lines in every visit, and the visit fit prefers two stars for 92-95% of the published SB2 at every coadd separation. 2,236 rows (8.5% of the 26,239 with two or more fitted visits; 1.4% of the published SB2); 941 of them are metal-poor. Replaces the second-revision coadd_symmetric flag (abs(best_rv1 + best_rv2) <= 2 km/s), which also marked real twins |
| vmacro_single | km/s | macroturbulent velocity of our single-star fit, from the seeded refit (`vmacro_refit_*.csv`) with the catalog's fitting layer and network |
| vmacro_1, vmacro_2 | km/s | macroturbulent velocity of each component in our two-component fit |
| vmacro_at_edge | flag | 1 if vmacro_single >= 30 km/s, the edge of the network's training range; above it the model no longer changes, so the value is a lower bound |
| v_macro_pipeline | km/s | the DR19 pipeline's own single-star v_macro (The Payne). The first-revision release called this column v_macro and described it as ours, which it was not |

## stage2_catalog_full.csv — the multi-epoch supplement (50,187 rows, one per candidate; rows with a non-empty error field have no usable per-visit solution)

Second revision: the per-visit velocities are now in the barycentric frame. DR19
mwmVisit spectra arrive shifted to each star's rest frame with the removed
velocity stored as v_rad, and the fit adds it back. The first-revision table
applied the barycentric correction to spectra that were already corrected, so
its per-visit velocities equalled minus the barycentric correction, and it
listed them in signal-to-noise order without their dates. That version is in the
git history of this file (commit 550000d) for reference only. Visits without a
pipeline velocity cannot be put in the barycentric frame and are left out of the
fit (141 stars had such a visit in the first second-revision run and were refit).

In the second revision only the 26,239 catalog SB2 with two or more usable visits
were refit, since they carry every multi-epoch result in the paper. The other rows
keep their Gaia identifier, position and DR19 pipeline velocities.

| column | unit | description |
|---|---|---|
| sdss_id | — | SDSS-V identifier |
| gaia_dr3_source_id, ra, dec | —, deg | Gaia DR3 identifier and position (epoch 2016.0), verified as for the catalog; blank for the 482 stars that fail the check |
| in_sb2_catalog | bool | the star is in dr19_sb2_catalog_open.csv |
| n_visits | — | number of APOGEE visits fit (at most eight, the highest signal-to-noise ones) |
| delta_chi2 | — | joint per-visit chi2(single) − chi2(binary), summed over visits |
| f_imp | — | improvement fraction of the joint fit |
| prefers_binary | bool | joint fit prefers the two-component model |
| q_spec | — | mass-ratio parameter of the two-component model in the joint visit fit (a model parameter like best_q, not a dynamical mass ratio). Where the joint fit falls back to the single-star solution (delta_chi2 = 0, 4,046 rows) it is exactly 1 |
| q_dyn | — | dynamical mass ratio from the per-visit velocity amplitudes; blank unless the joint fit prefers two components, there are three or more visits, and the primary velocity changes by at least 1 km/s (the 1,920 confirmed systems whose primary does not move have no velocity amplitude to take a ratio of). A fit that falls back to the single-star solution sets q to one, which is why the first-revision table had a large group at exactly 1 |
| q_dyn_at_bound | bool | q_dyn lies within 0.05 of an edge of the allowed range 0.1-1.5 (below 0.15 or above 1.45), where values pile up against the edge and are not constrained (586 of 6,490, 9.0%) |
| v_single | km/s | barycentric velocity of the single-star model, one value for all visits |
| gamma | km/s | systemic velocity |
| v1_range | km/s | maximum primary velocity change across visits |
| v1_per_visit, v2_per_visit | km/s | semicolon-separated per-visit component velocities, barycentric |
| mjd_per_visit, visit_index_per_visit | day, — | the date and mwmVisit row of each velocity, in the same order. The date is the integer MJD of the observing night carried by mwmVisit, not the mid-exposure time; for periods of a few days and longer the difference is a small fraction of the phase |
| rv1_untied_per_visit, rv2_untied_per_visit | km/s | component velocities from each visit's own two-component fit, with no momentum tie; the two may be exchanged from visit to visit |
| q_wilson | — | mass ratio from the slope of the untied velocities against each other (Wilson plot), given only where q_dyn is given and only between 0.05 and 3 (3,231 rows); the untied primary velocities must span more than 10 km/s. At each visit the untied pair is oriented so that the component closer to the fitted primary velocity (v1_per_visit) is the primary, and q_wilson is -1/slope of the oriented secondary against the oriented primary; the span is that of the oriented primary. Outside 0.05-3 the slope is not constrained (values reached several hundred) |
| refit | bool | refit in the second revision; rows with refit false keep their identifiers, positions and pipeline velocities but no fit values |
| n_visits_rv, dv_rad_max_astra | —, km/s | number of visits with a DR19 pipeline velocity, and the largest change among them |
| dv_rad_max_astra_ok | km/s | largest change among the pipeline velocities at most 400 km/s in absolute value; larger values are failures of the pipeline's velocity fit (dv_rad_max_astra reaches about 2,000 km/s). Used for sb1 |
| n_visits_rv_ok | — | number of pipeline velocities at most 400 km/s in absolute value; sb1 needs three or more |
| v_rad_median_astra, v_rad_std_astra | km/s | median and standard deviation of the DR19 pipeline per-visit velocities (single-star fits); filled for every row with a pipeline velocity (blank for 101 rows without one), including those not refit |
| sb1 | bool | single-lined velocity variable, defined below |
| v1_span_gt20 | bool | in the SB2 catalog, confirmed by the visit fit (prefers_binary), three or more visits, primary velocity spanning more than 20 km/s (3,115 rows); called orbit_ready before the second revision |
| error | — | non-empty if the fit failed |

A star is a single-lined velocity variable (SB1 candidate) when it is not in the
SB2 catalog and its DR19 pipeline velocities change by more than 10 km/s and at most
480 km/s over three or more visits, leaving out visits whose pipeline velocity is above
400 km/s in absolute value (3,943 rows). Those visits are failures of the pipeline's
velocity fit; 480 km/s is about the largest primary velocity change a main-sequence
pair gives (two 1.4 Msun stars on a 6 hr orbit). Before this cut the flag held 4,089
rows, 180 of them with changes above 500 km/s. The first-revision definition used v1_range, which is
zero for a single-star fit and was driven by the frame error above.

## stage2_controls.csv and stage2_singles.csv — the multi-epoch fit on single-star samples

The joint visit fit of the supplement, run with the same fitter and settings on two
single-star samples, to measure its false-positive rate.

stage2_controls.csv (1,643 rows, 1,293 with two or more usable visits): the benchmark
controls with two or more visits. The single-star network of the visit fit
(models/payne_dr19_sc.pt) was trained on the benchmark controls, so this is not a
held-out test of that network; `in_sc_training` is 0 for the stars that the training
rejected. `split` says whether the star was held out of the catalog network's
training (`heldout`), in its training half (`training`), or in neither list.

stage2_singles.csv (3,000 rows, 1,619 with two or more usable visits): dwarfs drawn at
random (seed 20260916) from the 238,205 searched, outside the SB2 catalog, outside the
network's training set, outside the sigma_v-triaged candidate list and outside the
benchmark. This is the independent test. Rows with fewer than two usable visits
carry no fit.

Both carry the fit columns of stage2_catalog_full.csv plus `v_rad_std_pipeline`
(standard deviation of the star's DR19 pipeline per-visit velocities) and, where the
Gaia cross-match exists, `ruwe` and `gaia_nss`. The rates in the paper: 7.8% of the
independent sample, 6.8% of its velocity-stable stars; 14.9% of the controls, 7.9% of
the velocity-stable ones; 0.8% of the independent sample after requiring three or
more epochs and a primary velocity change above 10 km/s (`scripts/stage2_fpr.py`).

## stage2_singles_astra_rv.csv — pipeline velocities of the random single-star sample (3,000 rows)

The SB1 selection of the supplement applied to the 3,000 random dwarfs of
stage2_singles.csv, using only the pipeline per-visit velocities
(`scripts/stage2_singles_pipeline_rv.py`). Of the 1,387 with three or more visits,
13 (0.9%) pass the cut (visits above 400 km/s left out, change above 10 and at most
480 km/s), which is the false-positive rate of the SB1 flag; 16 (1.2%) pass the
change cut on all visits.

| column | unit | description |
|---|---|---|
| sdss_id | — | SDSS-V identifier |
| n_visits_rv | — | number of usable visits (same visit rule as the visit fit) |
| dv_rad_max | km/s | largest minus smallest pipeline velocity over the visits |
| v_rad_std | km/s | standard deviation of the pipeline velocities |
| mjd_span | d | time between the first and last usable visit |
| n_visits_rv_ok, dv_rad_max_ok | —, km/s | the same count and largest change over the visits with pipeline velocity at most 400 km/s in absolute value |
| error | — | no_visit_file, no_usable_visit or fetch_failed; blank otherwise |

## sb2_component_teff_open.csv

Per-catalog-star component temperatures: primary teff1 from the single-star fit,
secondary teff2 implied by best_q through the isochrone tie.

## sb2_skycoords_open.csv

Per-catalog-star ra, dec (J2000 degrees), for the sky-distribution figure.

## ../gaia_dr19_dwarfs.parquet — Gaia DR3 cross-match

Per-dwarf Gaia photometry and astrometry (G, BP−RP, parallax, ruwe,
non-single-star flag), joinable to the catalog on sdss_id.

## dr19_sb2_orbit_posteriors.csv — joint Keplerian posterior summaries (1,389 rows, confirmed catalog SB2 with 8+ usable visits)

Second revision: recomputed from the corrected deep refit (stage2_deep.csv, up
to 16 visits per system, barycentric velocities with their dates) with the
two-velocity sampler. Each visit contributes the unordered pair of component
velocities from its own two-component fit, the two label assignments are summed
over, and the mass ratio is the dynamical q of the joint fit (the combined-spectrum
q where that is missing or at a bound). See docs/ecc_twovelocity_method.md. The
first-revision file (4,225 rows) used the primary velocity alone, from velocities
that carried the frame error and were paired with the wrong dates, and is
superseded.

| column | unit | description |
|---|---|---|
| sdss_id | — | SDSS-V identifier |
| status | — | 'ok' if the sampler kept enough posterior draws; 'unconstrained' otherwise |
| n_ep | — | number of epochs fit |
| n_kept | — | posterior samples kept by the rejection sampler |
| P16, P50, P84 | days | period posterior percentiles |
| e16, e50, e84 | — | eccentricity posterior percentiles |
| K1 | km/s | median primary velocity semi-amplitude |
| v_sys | km/s | median systemic velocity |
| method | — | 'twovel', the two-velocity sampler of the second revision |

## stage2_deep.csv — the deep refit behind the orbits (1,389 rows)

The 3,983 catalog SB2 with eight or more DR19 pipeline visits were refit with up
to 16 visits each (the highest signal-to-noise ones), by the same fitter as the
multi-epoch supplement. This table keeps the 1,389 that the refit confirms and
that have at least eight usable visits. Its columns are the fit columns of
stage2_catalog_full.csv (n_visits, delta_chi2, f_imp, prefers_binary, q_spec,
q_dyn, q_dyn_at_bound, v_single, gamma, v1_range, v1_per_visit, v2_per_visit,
visit_index_per_visit, mjd_per_visit, rv1_untied_per_visit, rv2_untied_per_visit,
error), with the same meanings, except that q_dyn_at_bound there marks only values at the edges themselves (0.1 or 1.5).

## Eccentricity comparison products

| file | description |
|---|---|
| ecc_marginal_real.csv | per-system eccentricity likelihood on the 36-point grid, for the 520 confirmed SB2 (216 twins) with a constrained orbit and median period 6 to 400 days (`loglike` is semicolon-separated, one value per grid point; `method` is 'twovel') |
| ecc_validation.csv | injection test: recovered against injected index difference, four inputs by four realizations |
| ecc_null_dalpha.csv | injection null: recovered difference when the same index is injected into both classes at the real twin and non-twin sampling, 24 realizations |
| ecc_variants.csv | the difference under each sample and matching choice, with bootstrap errors and sample sizes |
| ecc_variants_qdyn.csv | the six variants of ecc_variants.csv with twins defined by the dynamical mass ratio of stage2_deep.csv, min(q_dyn, 1/q_dyn) above 0.95, instead of best_q; systems with q_dyn missing or at a fit bound are left out (`scripts/ecc_relabel_qdyn.py`) |
| ecc_variants_qdyn090.csv | the same with the threshold at 0.90 |
| ecc_marginal_real_qdyn.csv | ecc_marginal_real.csv restricted to the 476 systems with a usable q_dyn, with `twin` set by min(q_dyn, 1/q_dyn) above 0.95 (written by `scripts/ecc_relabel_qdyn.py`) |
| ecc_validation_qdyn.csv | injection test as in ecc_validation.csv, with the mock twin and non-twin mass ratios drawn from the q_dyn-labelled classes (`scripts/ecc_val_suite.py --twin-label qdyn`); ecc_val_qdyn_shards/ holds the 16 per-task outputs it was concatenated from |
| ecc_null_full_qdyn.csv | the mocks of ecc_null_full.csv with `twin` replaced by the q_dyn label, systems without a usable q_dyn dropped (`scripts/ecc_null_relabel_qdyn.py`) |
| ecc_null_dalpha_qdyn.csv | injection null as in ecc_null_dalpha.csv, scored from ecc_null_full_qdyn.csv |
| ecc_headline_qdyn.json | the fields of ecc_headline.json for the q_dyn labels, from `scripts/ecc_headline.py` with ecc_variants_qdyn.csv, ecc_null_full_qdyn.csv, ecc_marginal_real_qdyn.csv and ecc_validation_qdyn.csv as inputs |
| ecc_headline.json | the numbers quoted in the paper: fiducial difference, null bias and its error, matching spread, combined value and error, sample counts, and the 2-sigma upper bound of each variant |

The estimator is `scripts/ecc_marginal.py` (per-system likelihood on a grid in
eccentricity; `system_loglike_on_egrid_twovel` in the second revision), driven on
the real data by `scripts/ecc_real_marginal.py`. The method and its tests are in
docs/ecc_twovelocity_method.md, and `scripts/ecc_headline.py` writes
ecc_headline.json. The
population step, the matching, and the bootstrap are in
`scripts/ecc_final_table.py`, and the injection null is `scripts/ecc_null_full.py`.
The population density is normalized on [0, EMAX] with EMAX = 0.95; the
EMAX^(1+alpha) term depends on alpha and must not be dropped.

## benchmark_with_gaia.csv — the benchmark set with cross-match keys (10,210 rows)

Released so the SB2 and control samples can be cross-matched against external
catalogs (Gaia NSS, Kounkel et al. 2021, Kovalev et al. 2022/2024, and others).

| column | description |
|---|---|
| sdss_id, gaia_dr3_source_id | identifiers; the Gaia id is the positional nearest neighbour |
| is_sb2_benchmark | 1 for the injected/known SB2 benchmark, 0 for the control (single-star) set |
| verdict, prefers_binary, delta_chi2, f_imp, min_fimp_required | classifier outputs and the gate the star had to clear |
| best_q, best_rv1, best_rv2 | composite-model mass ratio and component velocities |
| teff_single, logg_single, feh_single | best single-star fit labels |
| teff_seed, logg_seed, feh_seed, v_macro, snr | pipeline seed labels and visit S/N |
| ruwe, parallax, phot_g_mean_mag, bp_rp | Gaia DR3 quantities |
| gaia_nss | 1 if Gaia reports a non-single-star solution |
| ruwe_high | 1 if RUWE > 1.4 |

Controls carrying gaia_nss or ruwe_high are detected more often than quiet ones
(14.4% and 13.0% against 7.7%), so the control set retains some genuine binaries
and the quoted control false-positive rate is an upper bound.

## external_sb2_crossmatch.csv — benchmark controls found in published SB2 catalogs (125 rows)

Produced by `scripts/crossmatch_external_sb2.py`, which pulls the catalogs live
from the VizieR TAP service. Added for the OJA referee round, which asked that
the single-star training sample be cleaned against published SB2 catalogs.

Catalogs matched:

| catalog | VizieR table | matched on | hits |
|---|---|---|---|
| Kovalev, Chen & Han (2022), MNRAS 517, 356 | J/MNRAS/517/356/tablec1 | Gaia source_id | 10 |
| Kovalev, Chen & Han (2024), MNRAS 527, 521 | J/MNRAS/527/521/tableb1 | Gaia source_id | 31 |
| Kounkel et al. (2021), AJ 162, 184 | J/AJ/162/184/table1 | position, 2 arcsec | 103 |

| column | description |
|---|---|
| sdss_id, gaia_dr3_source_id | identifiers, keyed to benchmark_with_gaia.csv |
| in_kovalev22, in_kovalev24, in_kounkel21 | 1 if the star appears in that catalog |
| kounkel_sep_arcsec, kounkel_id | separation and APOGEE id of the positional match |
| removed_by_our_preliminary_pass | 1 if our own binary pass had already dropped it |
| in_training_half | 1 if it entered the released network's training set |
| in_heldout_controls | 1 if it is one of the held-out controls |

Of the 7,866 controls, 125 (1.59%) appear in one of these catalogs. Our own
preliminary binary pass had already removed 62 of them. Of the rest, 28 entered
the training set of the catalog network (0.78% of it) and 35 fall in the
held-out controls (0.97%), where dropping them moves the false-positive rate
from 8.1% to about 7.8%.

## kounkel_overlap.csv — the catalog against Kounkel et al. (2021) (41,290 rows)

One row per catalog star with a position (`scripts/kounkel_overlap.py`). Kounkel et
al. (2021, J/AJ/162/184) searched the APOGEE DR16/17 spectra, so the fair
denominator is the catalog stars that DR17 observed. Of the 29,233 such stars,
3,244 (11%) are in the Kounkel SB2 table.

| column | unit | description |
|---|---|---|
| sdss_id | — | SDSS-V identifier |
| ra, dec | deg | catalog position |
| in_dr17 | bool | matched to the APOGEE DR17 allStar table (III/286) within 3 arcsec |
| in_kounkel21 | bool | matched to Kounkel et al. (2021) table1 within 3 arcsec |
| kounkel_sbn | — | number of components Kounkel lists (2, 3 or 4); blank without a match |
| sep_kms | km/s | coadd separation abs(best_rv1 - best_rv2) from our fit |
