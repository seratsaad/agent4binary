# agent4binary

Double-lined spectroscopic binary (SB2) detection on APOGEE DR19, following the
El-Badry, Ting & Rix (2018) method. Each dwarf spectrum is fit as a single star
and as a mass-ratio-tied binary; a star is accepted as SB2 when the binary fit
wins by the Table B1 criteria. The method is packaged so it can be re-run: the
science steps are MCP tool servers, and the operating decisions are written down
in `SKILL.md`. A language-model agent drives the loop on the benchmark and worked
examples; the full catalog run is the same deterministic core under a driver script.

The method and the catalog are described in the accompanying paper (Saad & Ting 2026).

## Results

- **Catalog.** 41,466 SB2 out of 238,205 DR19 main-sequence dwarfs (17.4%), at an
  8.1% control false-positive rate measured on held-out stars. Median mass ratio
  q = 0.91. El-Badry et al. found 2,645 in the 12x smaller DR13 sample.
  Catalog: `resources/census/dr19_sb2_catalog_open.csv`.

- **Benchmark.** On 2,344 known SB2 and 7,866 controls, the classifier recovers
  74.4% at the catalog operating point and 82.6% at 12.3%, against 75.6% at 13.3%
  for the original (non-public) network run as a reference. Per-star results:
  `resources/fair_tests/bench_holdout_all.csv`.

- **No non-public inputs.** The classifier is trained on public DR19 spectra and
  labels only. The training recipe that closes the gap to the reference is one
  self-consistent label refit: train a first network on pipeline labels, refit the
  training labels once with that network, retrain. Weights:
  `models/payne_dr19_sfbig_holdA.pt`; training manifest and split:
  `resources/dr19_sfbig_holdA_manifest.csv`, `resources/sfbig_holdA_keep_ids.txt`,
  `resources/sfbig_holdB_ids.txt`.

- **Multi-epoch.** Fitting the individual visits confirms 68.5% of the
  multiply-visited SB2 by velocity change, finds 519 single-lined velocity
  variables, and leaves 8,981 systems with enough coverage for an orbit.
  Table: `resources/census/stage2_catalog_full.csv`.

## Quick start

`torch 2.2.2` does not work with `numpy >= 2`, so pin numpy:

```
python3 -m venv .venv && . .venv/bin/activate
pip install "numpy<2" scipy astropy pandas torch          # core reproduction
pip install pyarrow requests matplotlib \
    langgraph langchain-google-genai langchain-mcp-adapters mcp   # census + agent
```

Reproduce the benchmark with the released classifier (prints recovery and
false-positive rate; spectra are downloaded on first run):

```
AB_MLP_NET=models/payne_dr19_sfbig_holdA.pt \
python src/run_census_stage1.py --detector binspec_mlp
```

Run the full census the same way, pointing it at the parent sample:

```
AB_MLP_NET=models/payne_dr19_sfbig_holdA.pt \
python src/run_census_stage1.py --detector binspec_mlp \
    --manifest resources/dr19_census_full_manifest.csv \
    --raw-dir data/dr19_census_raw --out my_catalog.csv
```

Drive it with the agent: `src/graph/agent.py`, tool catalog in `docs/mcp_tools.md`.

## What is where

```
SKILL.md                     the operating procedure (the know-how)
src/physics.py               forward models, normalization, detection statistic, acceptance gate
src/run_census_stage1.py     benchmark and census runner
src/mcp_servers/             the nine MCP tool servers
src/graph/agent.py           the language-model agent that drives the tools
scripts/                     census download, shard, merge, and the Gaia reliability chain
models/                      the released classifier (payne_dr19_sfbig_holdA.pt) and its inputs
resources/census/            the catalog (dr19_sb2_catalog_open.csv) and the multi-epoch table
resources/                   training manifest, train/hold-out split, benchmark star lists
docs/mcp_tools.md            one-page tool catalog
```

## Provenance and citation

Method: El-Badry, Rix et al. 2018a (MNRAS 473, 5043) and El-Badry, Ting, Rix et
al. 2018b (MNRAS 476, 528; binspec). MIST v1.2 (Choi et al. 2016); The Payne
(Ting et al. 2019); APOGEE / SDSS DR19; Gaia DR3. The El-Badry network weights
are used only as a benchmark reference and are not redistributed; nothing in the
catalog path depends on them.
