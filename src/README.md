# src module map

- `physics.py` — the science module (import-only, no CLI). Payne forward models (3-label `payne_predict`, 5-label survey-normalized `payne_predict5`, 5-label self-consistent `payne_predict5_sc`), `continuum_normalize` (per-chip sigma-clipped Chebyshev from raw flux), the El-Badry Eq. 2 binary composite, `chi2`, `f_imp` (Eq. B1), Table B1, and the detectors. `dr19_sc_single_vs_binary` is the production path; `dr19_single_vs_binary` is the survey-normalized comparison; `binspec_single_vs_binary` is the vendored oracle.
- `isochrone_mist.py` — our MIST v1.2 isochrone: maps mass ratio q to secondary labels and both radii (`secondary_from_q_ours`), the production q-to-flux-ratio relation. Loads `models/mist_ms_grid.npz` (built once from `data/mist_v1.2/`).
- `train_payne_dr19_sc.py` — trains the production self-consistent 5-label Payne (`models/payne_dr19_sc.pt`), the held-out validation set (`models/val_set_dr19_sc.npz`), and the Teff-binned un-normalization continuum table (`models/dr19_sc_continuum.npz`).
- `download_dr19_raw.py` — fetches RAW mwmStar flux + ivar into `data/dr19_raw/` (training pool), `data/dr19_raw_sb2/` (SB2 test set), and `data/dr19_raw_controls/` (single controls).
- `verify_detection.py` — runs the production detector on N SB2 + N controls and reports completeness / contamination.
- `mcp_servers/` — the nine MCP tool servers (see `mcp_servers/README.md`).
- `graph/agent.py` — the LangGraph + gemini-3.5-flash layer that drives the MCP tools.
- `vendor/binspec/` — the vendored binspec package (the oracle; do not modify).
