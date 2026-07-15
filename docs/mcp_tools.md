# MCP tool catalog + DR-portability (the agent-callable SB2 method)

The deliverable is the SB2 method packaged as MCP servers + a Skill so an agent can re-run it
on any SDSS release. This documents the tools, and exactly what is DR-agnostic vs what must
change to point at a new data release (DR20, DR21, ...). SKILL.md is the operating procedure;
this is the tool/interface reference.

## Servers and tools (9 servers, 25 tools; FastMCP over stdio)

- apogee_data_server: list_spectra, load_spectrum, load_dr19_spectrum, get_window.
- binary_model_server (THE detector): ccf_rv_scan, compose_sb2, chi2_single_vs_binary
  (model = dr19_sc [production self-consistent], dr19 [survey-norm], binspec [oracle]).
- payne_server: predict_spectrum, fit_labels.
- isochrone_server: star_from_mass, secondary_labels, h_band_flux_ratio, flux_ratio_from_q.
- doppler_server: doppler_shift, wavelength_to_velocity.
- broadening_server: thermal_broadening, rotational_broadening, instrumental_broadening.
- gaia_sql_server: adql_query, crossmatch_2mass, binarity_flags (+ *_local variants).
- gaia_xp_server: get_xp_spectrum.
- vision_server: inspect_spectrum_fit (renders the fit; VLM false-positive cull).

The full single-vs-binary science pipeline is covered: get spectra (apogee_data), labels
(payne), binary composite + decision (binary_model), isochrone geometry (isochrone), RV/
broadening kinematics (doppler, broadening), and Gaia validation (gaia_sql, gaia_xp, vision).

## DR-agnostic vs DR-specific (what a new-release swap requires)

DR-AGNOSTIC already (release-independent physics; no change needed):
- binary_model_server (chi2_single_vs_binary, compose_sb2, ccf_rv_scan): operates on arrays
  via physics.py; the El-Badry Eq. B1 / Table B1 decision and the composite are physical.
- payne (predict_spectrum, fit_labels), isochrone, doppler, broadening: pure physics on the
  fixed APOGEE H-band grid (physics.WAVELENGTH); release-independent.
- gaia_sql / gaia_xp: Gaia DR3 archive, independent of the APOGEE release.

DR-SPECIFIC (the only things to change to target DR20/DR21):
1. Spectrum path + product version. apogee_data_server load_dr19_spectrum resolves the
   mwmStar SAS path .../astra/0.6.0/spectra/star/XX/YY/mwmStar-0.6.0-<sdss_id>.fits and the
   local _DR19_RAW_DIRS. For a new DR, change the astra VERSION (0.6.0) and the product tag.
   RECOMMENDED: add a data_release / astra_version parameter (default 19 / 0.6.0) to the
   loader + download_dr19_raw.py so a new release is one argument. (Multi-visit: same for
   mwmVisit in download_dr19_visits.py.)
2. Label source (CAS table). The training-label pull is the DR19 ASTRA the_payne_apogee_star
   table (download_dr19_raw.py SQL). For a new DR, point at that release's ASTRA table.
   NOTE (AB-14): prefer the best VALIDATED data-driven labels per release -- for DR19,
   the_payne beat APOGEE-Net on the benchmark; validate any label source before using it.
3. Wavelength grid. If a release changes the APOGEE H-band pixel grid, refresh
   physics.WAVELENGTH / the 8575-pixel arrays; the binspec 7214 fit-pixel mask
   (cannon_cont_pixels) is grid-tied. (Stable across DR13-DR19.)
4. Thresholds. The Table B1 (Delta-chi2, f_imp) decision is calibrated on the semi-empirical
   injection set; it transfers across releases, but recalibrating on a per-release injection
   set is the rigorous option (AB-15 found Table B1 already near-optimal for our net).

The physics is release-agnostic; only the DATA ADDRESSING (path/version, CAS table, grid) is
release-specific, and it is localized to apogee_data_server + the two download_* scripts.
Parametrizing those by release makes the package point-at-a-new-DR portable.
