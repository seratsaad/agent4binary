# Column descriptions for the released tables

## dr19_sb2_catalog_open.csv — the DR19 SB2 catalog (41,466 rows)

| column | unit | description |
|---|---|---|
| sdss_id | — | SDSS-V identifier |
| gaia_dr3_source_id | — | Gaia DR3 identifier (positional nearest neighbour within 2 arcsec; 0 if unmatched) |
| ra, dec | deg | J2000 coordinates (ICRS) |
| teff_seed, logg_seed, feh_seed | K, dex, dex | pipeline labels used to seed the fit |
| verdict | — | classifier verdict string |
| prefers_binary | bool | two-component fit beats the single-star fit |
| delta_chi2 | — | chi2(single) − chi2(binary) |
| f_imp | — | improvement fraction (El-Badry et al. 2018, Eq. B1) |
| min_fimp_required | — | the Table B1 rung the star had to clear |
| best_q | — | recovered mass ratio |
| best_rv1, best_rv2 | km/s | component velocities in the combined spectrum |
| teff_single, logg_single, feh_single | K, dex, dex | best single-star fit labels |
| error | — | non-empty if the fit failed |
| sb2 | bool | accepted as SB2 at the recalibrated gate (all rows in this file) |
| teff_floor | flag | 1 if the seed temperature sits at the 4200 K lower edge of the model grid, where the single-star labels are unreliable; cut on this for a clean sample |

## stage2_catalog_full.csv — the multi-epoch supplement (50,187 rows, one per candidate; rows with a non-empty error field have no usable per-visit solution)

| column | unit | description |
|---|---|---|
| sdss_id | — | SDSS-V identifier |
| n_visits | — | number of APOGEE visits fit |
| delta_chi2 | — | joint per-visit chi2(single) − chi2(binary), summed over visits |
| f_imp | — | improvement fraction of the joint fit |
| prefers_binary | bool | joint fit prefers the two-component model |
| q_spec | — | mass ratio from the joint spectral fit |
| q_dyn | — | dynamical mass ratio from the per-visit velocity amplitudes |
| gamma | km/s | systemic velocity |
| v1_range | km/s | maximum primary velocity change across visits |
| v1_per_visit, v2_per_visit | km/s | semicolon-separated per-visit component velocities |
| error | — | non-empty if the fit failed |

A star is a single-lined velocity variable (SB1 candidate) when it is not in the
SB2 catalog, has three or more visits, and v1_range > 10 km/s.

## sb2_component_teff_open.csv

Per-catalog-star component temperatures: primary teff1 from the single-star fit,
secondary teff2 implied by best_q through the isochrone tie.

## sb2_skycoords_open.csv

Per-catalog-star ra, dec (J2000 degrees), for the sky-distribution figure.

## ../gaia_dr19_dwarfs.parquet — Gaia DR3 cross-match

Per-dwarf Gaia photometry and astrometry (G, BP−RP, parallax, ruwe,
non-single-star flag), joinable to the catalog on sdss_id.

## dr19_sb2_orbit_posteriors.csv — joint Keplerian posterior summaries (4,225 rows, systems with 8+ visits)

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

## Eccentricity comparison products

| file | description |
|---|---|
| ecc_marginal_real.csv | per-system eccentricity likelihood on the 36-point grid, for the 2,009 multi-epoch SB2 with median period 6 to 400 days (`loglike` is semicolon-separated, one value per grid point) |
| ecc_validation.csv | injection test: recovered against injected index difference, four inputs by four realizations |
| ecc_null_dalpha.csv | injection null: recovered difference when the same index is injected into both classes at the real twin and non-twin sampling, 24 realizations |
| ecc_variants.csv | the difference under each sample and matching choice, with bootstrap errors and sample sizes |

The estimator is `scripts/ecc_marginal.py` (per-system likelihood on a grid in
eccentricity), driven on the real data by `scripts/ecc_real_marginal.py`. The
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
| best_q, best_rv1, best_rv2 | recovered mass ratio and component velocities |
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
