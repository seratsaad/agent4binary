# MCP servers

Nine MCP servers (FastMCP over stdio) expose the science in `src/physics.py` and the external archives as tools. The LangGraph agent (`src/graph/agent.py`) binds these and lets gemini-3.5-flash call them. Each server is a thin wrapper; a new ability is a tool plus a graph node. Run one directly to check it imports: `python src/mcp_servers/<name>_server.py`.

## Tool catalog

### apogee_data_server (APOGEE Data)
- `list_spectra` — list available APOGEE spectrum ids.
- `load_spectrum` — load an APOGEE spectrum and cache its arrays to disk.
- `get_window` — return a downsampled flux excerpt over a wavelength window.

### binary_model_server (Binary Model) — the detector
- `ccf_rv_scan` — cross-correlate an observed spectrum against a template to find RV peaks.
- `compose_sb2` — build a physical SB2 composite (El-Badry Eq. 2) and cache it.
- `chi2_single_vs_binary` — decide single vs binary for an observed spectrum (default `model="dr19_sc"`, the production self-consistent path; `model="dr19"` survey-normalized comparison; `model="binspec"` the oracle).

### payne_server (Payne)
- `predict_spectrum` — predict a rest-frame normalized APOGEE spectrum from stellar labels.
- `fit_labels` — fit (Teff, logg, [Fe/H]) of an observed APOGEE spectrum with the Payne model.

### isochrone_server (Isochrone)
- `star_from_mass` — interpolate the MIST single-star table for one star.
- `secondary_labels` — derive the secondary's labels from the primary and mass ratio.
- `h_band_flux_ratio` — compute the secondary/primary flux ratio f2/f1 in the APOGEE H band.
- `flux_ratio_from_q` — secondary labels plus H-band flux ratio in one call.

### doppler_server (Doppler Tools)
- `doppler_shift` — observed wavelength from rest wavelength and radial velocity.
- `wavelength_to_velocity` — radial velocity from an observed wavelength shift.

### broadening_server (Broadening Tools)
- `thermal_broadening` — thermal (Doppler) broadening of a spectral line.
- `rotational_broadening` — line broadening from stellar rotation.
- `instrumental_broadening` — instrumental (LSF) line broadening of a spectrograph.

### gaia_sql_server (Gaia SQL)
- `adql_query` — run an ADQL query against the Gaia DR3 archive.
- `crossmatch_2mass` — resolve a 2MASS designation to its Gaia DR3 source and astrometry.
- `binarity_flags` — return Gaia DR3 binarity-relevant flags (e.g. RUWE) for one source.

### gaia_xp_server (Gaia XP)
- `get_xp_spectrum` — retrieve the Gaia DR3 BP/RP (XP) spectrum for a source via datalink.

### vision_server (Vision)
- `inspect_spectrum_fit` — render the single-vs-binary fit and have gemini-3.5-flash judge it (fit quality, RV split, artifacts, recommendation).

## Notes

`agent.py` auto-wires the first eight servers (apogee_data, binary_model, doppler, broadening, gaia_sql, gaia_xp, payne, isochrone). The vision server is a standalone tool invoked on demand. `payne_server` loads `models/payne.pt` (the 3-label net); `binary_model_server` reaches the production net `models/payne_dr19_sc.pt` through `physics`.
