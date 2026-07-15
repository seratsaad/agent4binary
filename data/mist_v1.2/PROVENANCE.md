# MIST v1.2 isochrones (provenance)

Source: MIST website https://waps.cfa.harvard.edu/MIST/ (redirects to https://mist.science/).

Package downloaded: `MIST_v1.2_vvcrit0.4_basic_isos.txz` (~211 MB), the MIST v1.2 vvcrit0.4 basic isochrone set (theoretical columns, no synthetic photometry). MIST version 1.2, MESA revision 7503.

We extracted the four per-[Fe/H] `.iso` files used by the production isochrone interpolation:

- `MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.4_basic.iso` ([Fe/H] = -0.50)
- `MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_basic.iso` ([Fe/H] = 0.00)
- `MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.4_basic.iso` ([Fe/H] = +0.25)
- `MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.4_basic.iso` ([Fe/H] = +0.50)

Each file holds 107 isochrones (log10 age 5.0 to 10.3) with 25 columns. The columns read by `src/isochrone_mist.py` are `log10_isochrone_age_yr`, `initial_mass`, `log_L`, `log_Teff`, `log_g`, and `phase`.

After the first import, `src/isochrone_mist.py` parses these into a compact main-sequence grid at `models/mist_ms_grid.npz` (~2.5 MB, phase == 0 dwarf rows only). The raw `.iso` files and the 211 MB tarball were removed to keep the repo small; the compact grid is sufficient at runtime. To rebuild from scratch, re-download the tarball, extract the four `.iso` files into this directory, delete `models/mist_ms_grid.npz`, and re-import the module.
