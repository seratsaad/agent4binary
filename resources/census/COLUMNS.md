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

## dr19_sb3_triples.csv — triple-lined (SB3) candidates (8,112 rows, one per candidate)

| column | unit | description |
|---|---|---|
| sdss_id | — | SDSS-V identifier (multi-epoch systems with 3+ epochs) |
| n_visits | — | number of APOGEE visits fit |
| delta_chi2_23 | — | chi2(binary) − chi2(triple), summed over visits |
| f_imp_23 | — | improvement fraction of the binary-to-triple step |
| q2_triple, q3_triple | — | secondary and tertiary mass ratios (relative to the primary) |
| outer | — | index (0/1/2) of the near-constant outer component |
| v1_triple, v2_triple, v3_triple | km/s | per-visit component velocities, ';'-separated |
| inner_corr | — | Pearson correlation of the two inner velocity tracks |
| q_inner_dyn | — | inner dynamical mass ratio from the anti-phase slope (NaN if undefined) |
| sb3 | bool | accepted by the vision inspection (1,022 rows) |
| vision_confidence, vision_reason | — | the inspection's confidence and one-line reason |
| dv_outer_com | km/s | median per-epoch offset between the outer velocity and the inner pair's systemic velocity |
| bound | bool | dv_outer_com < 10 km/s (the El-Badry et al. 2018 Figure 7 check) |
| sb3_final | bool | sb3 AND bound: the SB3 census (347 rows) |

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
