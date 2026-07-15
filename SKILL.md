---
name: agent4binary
description: Detect and disentangle double-lined spectroscopic binaries (SB2) from APOGEE dwarf spectra (El-Badry et al. 2018a/b method, ported to SDSS DR19) with our own Payne and agents (LangGraph + gemini-3.5-flash) that call every tool over MCP. Single source of truth for the operating procedure; results live in README.md.
---

# agent4binary

This Skill is the operating procedure for SB2 detection on APOGEE dwarf spectra. It states the tricks that decide whether a star is a binary: which single-star model to fit, how to normalize, how to build the binary composite, what acceptance gate to apply, how to cull false positives, and how to read completeness. Adopt these rules to run the method on a new data release.

## Scope
- Classify an APOGEE spectrum as single vs SB2.
- For a binary, recover each component's RV, labels (Teff, log g, [Fe/H], [Mg/H], v_macro), and mass ratio q; cross-check Gaia.
- Dwarfs only. Preselect on catalog logg (4–5): giants lack SB2 detectability and the method is calibrated on the main sequence.

## Single-star model
1. Fit a 5-label Payne (Teff, log g, [Fe/H], [Mg/H], v_macro), not 3. The extra two labels lower the single-star residual, so a true single gains nothing from the binary model (control f_imp ≈ 0). Three labels underfit and a single scores a spurious binary.
2. Use the RAW flux + ivar on the 8575-pixel APOGEE log-λ grid (15100.8–16999.8 Å). Never read the survey continuum column.

## Continuum (self-consistent)
3. Re-derive the continuum per chip (gaps at pixels 3430/6212) as a Chebyshev of degree 4, fit with iterative sigma-clipping that is asymmetric (lower ≈1.5σ, upper ≈4σ) so absorption lines are clipped but the upper envelope is kept.
4. Apply the same continuum operator to BOTH data and model. Self-consistent normalization makes continuum error cancel in Δχ². If the detection net was trained under one normalization, do not change the normalization at detection time — self-consistency breaks and f_imp collapses.

## Binary composite
5. Build the composite as the flux sum of two components, each weighted by its MIST H-band luminosity 10^(−0.4 M_H) (per-component surface brightness in the APOGEE H band), then re-normalize. Do NOT weight by a mean-normalized continuum: that makes the secondary too bright and biases the recovered q low.
6. Tie the flux ratio to the mass ratio q through the MIST v1.2 isochrone (q → Teff2, log g2, R2; R1 from the primary). The composite's flux ratio is a function of q, not a free parameter.

## Mass-ratio fit
7. Fit q with a multi-start optimizer and nest only the q→1, equal-RV limit. An equal-mass binary with rv1 ≠ rv2 is still a line-doubled spectrum and must not be collapsed to one shifted component. The q=1/equal-RV model reduces exactly to the single star, so Δχ² ≥ 0 remains guaranteed by the floor check.
8. Forward model in numpy + scipy least_squares (trf) on (obs−model)/σ; multi-start over q. torch is training-only.

## Acceptance gate
9. Compute inverse-variance χ², Δχ² = χ²_single − χ²_binary, and the improvement fraction f_imp (El-Badry Eq. B1) = Σ[(|f_single−f|−|f_binary−f|)/σ] / Σ[|f_single−f_binary|/σ].
10. Accept on the El-Badry Table B1 sliding scale (Δχ² -> minimum f_imp, the full 9-bin ladder, not a 4-point sketch): Δχ²≥3000 -> 0.0; ≥2500 -> 0.05; ≥2000 -> 0.075; ≥1500 -> 0.10; ≥1000 -> 0.125; ≥750 -> 0.15; ≥600 -> 0.175; ≥450 -> 0.20; ≥300 -> 0.225; <300 inconclusive. This exact ladder is physics.passes_table_b1 (and the TB table in scripts/ab_bench_aggregate.py); reimplement it verbatim.
11. Re-derive the Table B1 rung VALUES for the classifier in use, at a fixed held-out-control false-positive-rate target: the released DR19 census gate scales both rung coordinates by 0.90 (a star passes when Δχ²/0.90 and f_imp/0.90 clear the published ladder), set on one half of the held-out controls and validated on the other (8.1% validated FPR). A global f_imp floor (a positive floor enforced at all Δχ²) is the knob for sweeping operating points; when one is imposed, waive it only above Δχ² = 1e5, where a tiny f_imp still reflects a pervasive real mismatch. Rung values calibrated for one classifier (e.g., El-Badry's own network) do not transfer to another.

## False-positive control
12. Mask sky and chip-gap artifacts: normalized flux outside [0.1, 1.2] and err≤0 → err=inf. Cut to SNR > 60 (cap S/N at 200; at very high S/N the cap leaves residual structure, so sample controls across the real SNR distribution).
13. Chip-edge guard: the velocity-shifted secondary template can step at the chip gaps. Mask a small guard at each chip boundary so an edge discontinuity does not pile onto a few pixels and inflate Δχ²/f_imp.
14. Vision check (gemini-3.5-flash) as the automated false-positive cull. Feed the rendered single-vs-binary fit PLUS the numbers (Δχ², χ²_single, χ²_binary, f_imp, q, dv). Have it flag artifacts: chip-edge drops, telluric residuals, hot-rotator broadening. Do NOT require visible line-doubling — real SB2 are often sub-resolution / two-temperature, so the vision check is a contamination cull, not a completeness check.
15. q reliable band: trust q in ≈0.2–0.95. q railed at the bounds (q ≤ 0.12 or ≥ 0.97) and hot rotators (Teff > 6500) are false-positive modes (broadening, chip-edge), not real detections.

## Multi-visit (RV-variable systems)
16. Treat multi-visit as Stage 2, the extension downstream of the coadd census. A coadd-undetectable RV-variable SB2 is never flagged by the coadd, so a Stage-2 design should not pre-filter only on coadd detections; Stage 2 supplies the orbital confirmation the single coadd cannot.
17. Fit the individual mwmVisit spectra jointly with shared labels + q, one per-visit primary velocity v1_i, and the secondary tied by momentum conservation v2_i = γ + (γ − v1_i)/q_dyn; joint inverse-variance χ² over visits; require N ≥ 2 for a binary verdict. Barycentric-correct each visit (essential). q_dyn falls out of the v2-vs-v1 slope and cross-checks the spectroscopic q.

## Completeness
18. Completeness is q- and Teff-dependent. Report completeness vs q with a Teff-dependent q_min floor; do not quote a single number.
19. Characterize completeness with semi-empirical injections — composites built from sums of real spectra — so the recovery curve reflects real noise and line structure.

## Architecture
- gemini-3.5-flash (langchain-google-genai) orchestrates via LangGraph and reaches every capability over MCP (langchain-mcp-adapters, stdio).
- Science lives in src/physics.py; the MCP servers are thin wrappers — a new ability = a tool + a node.
- MCP servers (src/mcp_servers/): apogee_data, binary_model (the detector), payne, isochrone, doppler, broadening, gaia_sql, gaia_xp, vision (renders a fit plot for gemini-3.5-flash).
- Full tool catalog (9 servers, 25 tools) and DR-agnostic-vs-DR-specific audit: docs/mcp_tools.md. The physics is release-independent; only the data addressing (apogee_data path + Astra version, the CAS label table, the H-band grid) is per-DR, localized to apogee_data_server + download_dr19_*.py.
- binary_model chi2_single_vs_binary exposes model=binspec_mlp (THE OPEN DETECTOR, paper headline: our 5-label net through El-Badry's PUBLIC fitting layer; the net is selected by AB_MLP_NET -- gap_distill.pt = 74% synthetic-target DELIVERED net, payne_dr19_curated.pt = 63% real-data fallback -- on the enlarged 2,344-SB2 benchmark), model=dr19_sc (our self-consistent all-real net, lower), model=dr19 (survey-norm), and model=binspec (the reference net run as oracle, 75.6%). The census and paper headline use binspec_mlp; binspec is the oracle upper bound only, and no binspec LINE weights ever enter the binspec_mlp path.

## Data
- DR19 mwmStar (Astra 0.6.0), APOGEE arm. Path: /sas/dr19/spectro/astra/0.6.0/spectra/star/{(id//100)%100}/{id%100}/mwmStar-0.6.0-{sdss_id}.fits.
- Individual visits: mwmVisit (per-visit flux/ivar/v_rad/bc, same grid) under .../spectra/visit/.../mwmVisit-0.6.0-{sdss_id}.fits.
- Dwarf selection from the DR19 CAS table the_payne_apogee_star (logg 4–5, Teff 4000–7000, SNR>60, flag_bad=0).
- SB2 test set: El-Badry et al. 2018b SB2 with logg>4, cross-matched to DR19.

## Training TARGET -- synthetic grid is the DELIVERED method (74%, do this first)
The single-star net's TRAINING TARGET, not the label count, sets recovery. A controlled
distillation isolates it: the SAME 5-label architecture trained on self-consistent SYNTHETIC
single-star spectra recovers 74.5% (near the 75.6% oracle), versus 63% trained on REAL spectra.
Real spectra are not a single-valued function of 5 labels (unmodeled C/N/O + noise blur the
H-band molecular/metal cores), so a net trained on them blurs the line cores the SB2 decision
depends on; synthetic spectra are sharp and self-consistent.

RECIPE (synthesize -> train -> detect; the reusable, release-agnostic method):
1. SYNTHESIZE a noise-free single-star grid over the dwarf range (Teff 4000-7000, logg 4-5,
   [Fe/H], [Mg/H], v_macro), broadened to APOGEE resolution + each star's v_macro, on the
   release wavelength grid. Source: an open ATLAS12/SYNTHE synthesis (pykurucz), OR a
   Kurucz-based emulator used PURELY as a generator (scripts/gap_distill_gen.py uses the
   reference single-star net as an oracle). HARD RULE: never load the reference LINE weights
   into the detector; the oracle only makes targets, the detector is trained from scratch.
2. TRAIN the identical 5-label net on the synthetic flux (scripts/gap_train_distill.py ->
   models/gap_distill.pt). No dependence on real-label quality.
3. DETECT with model=binspec_mlp, AB_MLP_NET=models/gap_distill.pt. Verified 74.5% recovery at
   ~13% control FPR on the 2,344-SB2 benchmark, +10 pts over the real-data net on the same stars.
For a NEW DR: regenerate the grid on that release's wavelength grid and retrain. Fidelity
matters (a low-fidelity MARCS grid, 7-10% off real spectra, did NOT help).

FALLBACK -- real-data detector (63%). Closing 63->74 by better real-data CURATION is possible
in principle but is the tacit expert step that does not transfer; the synthetic target removes
that dependence. If you must train on real spectra instead, get labels as below.

## Labels (per release) -- for the REAL-DATA fallback detector (63%)
The real-data detector trains a 5-label net on REAL spectra, so its recovery depends on the
LABEL source. Choose labels in this order, and ALWAYS validate the choice on the 250 SB2
+ 146 control benchmark (do not assume a catalog is better):
1. BEST: published fitted single-star labels from the method authors (El-Badry / Ting) if
   available for your stars by APOGEE_ID / 2MASS. Confound-free and directly comparable.
2. A vetted data-driven catalog (Cannon-class). YST notes ASTRA's Payne labels are weak and the
   Cannon does better in general -- BUT verify coverage AND benchmark it: for DR19 there is no
   ASTRA Cannon/AstroNN/SLAM table, external DR14/17 catalogs cover only ~57% of our stars
   (cross-match confound), and the one DR19-native data-driven alternative, APOGEE-Net
   (apogee_net_apogee_star, Teff/logg/[Fe/H]), REGRESSED our detector 62 to 54% (AB-14). So for
   DR19 dwarfs, ASTRA the_payne (the_payne_apogee_star) is the best FULL-COVERAGE label source
   we found -- use it unless a genuinely better, well-covered set is validated to beat it.
3. LAST RESORT for LABELS: synthesize an ab-initio grid (pykurucz ATLAS12+SYNTHE, ~3
   min/spectrum) and fit labels. Only if no data-driven labels exist for the release.

(The delivered detector uses the SYNTHETIC target above, 74%; the label priority here applies
only to the real-data fallback.)
Per-DR portability: file format + CAS table names change per release (mwmStar path, the CAS
pipeline tables); the 5 physical labels are release-agnostic. If the chosen source lacks
v_macro, keep the release's broadening label (vsini/v_macro) for that axis and swap only
Teff/logg/[Fe/H]/[Mg/H]. Definitions differ ([Mg/H] vs [Mg/Fe]) -- match them explicitly.

## Reproduce
- Environment pin matters: torch 2.2.2 cannot interop with numpy>=2 (import fails with
  "RuntimeError: Numpy is not available"). Use numpy<2:
  `python3 -m venv .venv && . .venv/bin/activate && pip install "numpy<2" scipy astropy pandas torch`.
- Pick the detector net via AB_MLP_NET: 74% delivered = `models/gap_distill.pt` (synthetic
  target); 63% real-data fallback = `models/payne_dr19_curated.pt`.
- One-command detector on a staged spectrum (raw flux+ivar npz in data/dr19_raw_sb2/):
  `export AB_MLP_NET=models/gap_distill.pt` then run the binspec_mlp path of
  src/run_census_stage1.py (`--detector binspec_mlp --no-download --manifest <shard> --raw-dir data/dr19_raw_sb2 --out <csv>`),
  or call binary_model.chi2_single_vs_binary(spec_id, model="binspec_mlp") over MCP.
- Recovery at matched 13% control FPR: aggregate with scripts/ab_bench_aggregate.py
  (open vs oracle). A 30-SB2/30-control smoke gives ~74% recovery (gap_distill) or ~63%
  (real-data net) at ~10-13% FPR in ~3 min.

## Repo
- SKILL.md (this) · README.md (results + usage) · METHODS.md (handoff).
- src/physics.py · src/mcp_servers/ · src/graph/agent.py · src/isochrone_mist.py · src/train_payne_dr19_sc.py · src/download_dr19_raw.py.
- models/ (gap_distill.pt = the DELIVERED synthetic-target detector net, 74%, used by binspec_mlp / AB_MLP_NET; payne_dr19_curated.pt = real-data fallback net, 63%; payne_dr19_sc.pt = self-consistent net; mist_ms_grid.npz; dr19_sc_continuum.npz) · data/dr19_raw* · resources/ (manifests, El-Badry catalog).
