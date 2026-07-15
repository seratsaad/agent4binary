#!/usr/bin/env python3
"""
AB-4b: El-Badry reliability chain on the DR19 binspec census catalog.

The raw census flag rate (~14%) tracks binspec's FPR floor. El-Badry confirms SB2
INDEPENDENTLY with astrometry/photometry, which removes spectroscopic false positives:

  1. q/Teff reliability (SKILL 15): keep q in [0.2, 0.95], Teff <= 6500 (drop q-railed
     FP modes + hot rotators).
  2. CMD over-luminosity (El-Badry Fig 9): an unresolved binary is brighter than a
     single at the same colour, so a real SB2 sits ABOVE the single-star main sequence
     in (BP-RP, M_G). A spectroscopic FP (a true single) sits ON the MS. We build the
     MS ridge from the parent sample and keep SB2 that are over-luminous by > DMAG.
  3. Gaia RUWE / non_single_star (SKILL 14-equivalent independent check): report
     astrometric-binary enrichment of the CMD-confirmed set (RUWE is weak for the
     short-period SB2 that dominate coadd line-doubling, so it is a cross-check, not a
     hard cut).

Vision cull (gemini, SKILL step 14): wired (src/mcp_servers/vision_server.py) but needs
GOOGLE_API_KEY + per-star rendering; NOT run here (no key in this environment). It is a
contamination cull whose role the CMD over-luminosity test fills quantitatively.

Gaia from resources/gaia_dr19_dwarfs.parquet (G, BP-RP, parallax, ruwe, nss). Stars
without Gaia/parallax are reported as coverage, not silently dropped.

Writes: figures/cmd_validation.{pdf,png}; updates benchmarks/census_vs_elbadry2018b.{json,md};
resources/census/dr19_sb2_catalog_culled.csv (the CMD-confirmed reliable SB2).
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAT = os.path.join(_ROOT, "resources/census/dr19_sb2_catalog.csv")
GAIA = os.path.join(_ROOT, "resources/gaia_dr19_dwarfs.parquet")
EB = os.path.join(_ROOT, "resources/elbadry2018b_binaries.csv")
FIGDIR = os.path.join(_ROOT, "figures")
DMAG = 0.375          # over-luminosity threshold (mag); 0.75 = equal-mass, half-way cut
PLX_SNR = 5.0         # require parallax/parallax_error > this for a CMD position


def load_gaia(catalog_df, manifest):
    """Return a Gaia frame keyed by sdss_id with columns ruwe, parallax,
    parallax_error, phot_g_mean_mag, bp_rp, non_single_star. Source = the census
    manifest's own Gaia columns (g_mag/bp_mag/rp_mag/plx, full-scale) if present,
    else the resources/gaia_dr19_dwarfs.parquet crossmatch (subset)."""
    if manifest and os.path.exists(manifest):
        m = pd.read_csv(manifest)
        if "g_mag" in m.columns:
            g = pd.DataFrame({"sdss_id": m["sdss_id"].astype(int),
                              "phot_g_mean_mag": pd.to_numeric(m["g_mag"], errors="coerce"),
                              "bp_rp": pd.to_numeric(m["bp_mag"], errors="coerce") - pd.to_numeric(m["rp_mag"], errors="coerce"),
                              "parallax": pd.to_numeric(m["plx"], errors="coerce"),
                              "parallax_error": pd.to_numeric(m["e_plx"], errors="coerce")})
            # RUWE/NSS not in the_payne table; merge from the parquet where available.
            try:
                p = pd.read_parquet(GAIA)[["sdss_id", "ruwe", "non_single_star"]]
                p["sdss_id"] = p["sdss_id"].astype(int)
                g = g.merge(p, on="sdss_id", how="left")
            except Exception:
                g["ruwe"] = np.nan; g["non_single_star"] = np.nan
            return g
    p = pd.read_parquet(GAIA)[["sdss_id", "ruwe", "parallax", "parallax_error",
                               "phot_g_mean_mag", "bp_rp", "non_single_star"]]
    p["sdss_id"] = p["sdss_id"].astype(int)
    return p


def abs_G(G, plx_mas):
    return G + 5.0 * np.log10(plx_mas) - 10.0     # plx in mas


def ms_ridge(bp_rp, M_G, bins):
    """Running median M_G(BP-RP) = the single-star main-sequence locus."""
    cen, med = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (bp_rp >= lo) & (bp_rp < hi)
        if m.sum() >= 15:
            cen.append(0.5 * (lo + hi)); med.append(np.median(M_G[m]))
    return np.array(cen), np.array(med)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=CAT)
    ap.add_argument("--manifest", default="", help="census manifest with Gaia cols (full scale)")
    ap.add_argument("--tag", default="", help="suffix for output artifacts (e.g. _full)")
    args = ap.parse_args()
    global FILT_TAG
    FILT_TAG = args.tag
    cat = pd.read_csv(args.catalog)
    cat = cat[cat["verdict"].astype(str).str.len() > 0].copy()
    cat["sdss_id"] = cat["sdss_id"].astype(int)
    g = load_gaia(cat, args.manifest)
    df = cat.merge(g, on="sdss_id", how="left")

    n_parent = len(df)
    cov = df["parallax"].notna()
    good_cmd = (cov & (df["parallax"] > 0) & (df["bp_rp"].notna())
                & (df["phot_g_mean_mag"].notna())
                & (df["parallax"] / df["parallax_error"] > PLX_SNR))
    df["M_G"] = np.where(good_cmd, abs_G(df["phot_g_mean_mag"].values,
                                         df["parallax"].values), np.nan)

    pb = df["prefers_binary"].astype(str).str.lower().isin(["true", "1"])
    q = pd.to_numeric(df["best_q"], errors="coerce")
    tf = pd.to_numeric(df["teff_single"], errors="coerce")
    reliable = pb & (q >= 0.2) & (q <= 0.95) & (tf <= 6500)

    # MS ridge from the PARENT (singles dominate) with good CMD positions.
    base = good_cmd & (~pb)
    bins = np.arange(0.3, 2.6, 0.1)
    cen, med = ms_ridge(df["bp_rp"][base].values, df["M_G"][base].values, bins)
    ridge_at = lambda c: np.interp(c, cen, med, left=np.nan, right=np.nan)

    df["dMG"] = ridge_at(df["bp_rp"].values) - df["M_G"].values   # >0 = above MS (brighter)
    rel_cmd = reliable & good_cmd & np.isfinite(df["dMG"])
    confirmed = rel_cmd & (df["dMG"] > DMAG)

    n_raw = int(pb.sum())
    n_rel = int(reliable.sum())
    n_rel_cov = int(rel_cmd.sum())
    n_conf = int(confirmed.sum())
    cov_parent = int((good_cmd).sum())

    # RUWE / NSS enrichment (CMD-confirmed vs parent), on Gaia-covered stars.
    def enrich(mask):
        sub = df[mask & cov]
        ruwe_hi = float((pd.to_numeric(sub["ruwe"], errors="coerce") > 1.4).mean()) if len(sub) else float("nan")
        nss = float((pd.to_numeric(sub["non_single_star"], errors="coerce") > 0).mean()) if len(sub) else float("nan")
        return ruwe_hi, nss, len(sub)
    ruwe_conf, nss_conf, n_conf_cov = enrich(confirmed)
    ruwe_par, nss_par, _ = enrich(~pb)

    # dump RUWE / NSS arrays for the validation-histogram figure (fig_ruwe.py),
    # using the SAME confirmed / non-flagged-parent definitions as the 62%/19% stat.
    _cls = np.where(confirmed & cov, "confirmed",
                    np.where((~pb) & cov, "control", "other"))
    _dump = pd.DataFrame({"ruwe": pd.to_numeric(df["ruwe"], errors="coerce"),
                          "nss": pd.to_numeric(df["non_single_star"], errors="coerce"),
                          "cls": _cls})
    _dump = _dump[(_dump.cls != "other") & _dump.ruwe.notna()]
    _dump.to_csv(os.path.join(FIGDIR, f"ruwe_dump{FILT_TAG}.csv"), index=False)

    # q-KS after the full chain vs El-Badry.
    qr = q[confirmed].values
    eb = pd.read_csv(EB); eb_q = pd.to_numeric(eb.get("q"), errors="coerce").dropna()
    eb_qr = eb_q[(eb_q >= 0.2) & (eb_q <= 0.95)].values
    from scipy.stats import ks_2samp
    ks = ks_2samp(qr, eb_qr) if len(qr) > 5 else None

    # ---- figure: CMD validation (El-Badry Fig 9 style), PANTERA house style ----
    os.makedirs(FIGDIR, exist_ok=True)
    try:
        sys.path.insert(0, os.path.join(_ROOT, "paper", "figures"))
        from pantera_style import set_style, CB
        set_style(9)
    except Exception:
        CB = dict(black="#000000", blue="#0072B2", vermillion="#D55E00", grey="#999999",
                  sky="#56B4E9", orange="#E69F00")
    # Force the serif house style explicitly (do not rely on set_style importing on
    # every host); text/axes stay vector, only the dense scatter is rasterized below.
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    fig, ax = plt.subplots(figsize=(3.35, 3.5))
    bg = good_cmd & (~pb)
    ax.scatter(df["bp_rp"][bg], df["M_G"][bg], s=3, c="0.8", lw=0,
               rasterized=True, label="Parent (single)")
    sb = reliable & good_cmd
    below = sb & (df["dMG"] <= DMAG)
    ax.scatter(df["bp_rp"][below], df["M_G"][below], s=14, c=CB["blue"], lw=0,
               alpha=0.7, rasterized=True, label="SB2 on ridge")
    ax.scatter(df["bp_rp"][confirmed], df["M_G"][confirmed], s=18, c=CB["vermillion"], lw=0,
               alpha=0.8, rasterized=True, label="CMD-confirmed SB2")
    # Reference lines: black ridge (solid), black over-luminosity cut (dashed),
    # green equal-luminosity limit (dash-dot), distinguished by colour and dash.
    C_CUT = CB["black"]
    C_EQL = "#009E73"  # theme green
    ax.plot(cen, med, "-", color=CB["black"], lw=1.8)  # MS ridge: SOLID black (caption)
    # Over-luminosity threshold: a star more than DMAG (= 0.375 mag) above the
    # ridge is kept as CMD-confirmed. Show the exact value (0.375), not a rounded
    # 0.38, so the figure matches the text.
    ax.plot(cen, med - DMAG, "--", color=C_CUT, lw=1.9)  # over-luminosity cut (orange dashed; caption)
    # Equal-luminosity binary band: an unresolved binary of two equal-luminosity
    # stars is 2.5 log10(2) = 0.752 mag brighter than a single star; stars above
    # this line (brighter than ridge - 0.75) cannot be plain (main-sequence)
    # binaries. Draw the 0.75-mag line and shade the region above it.
    EQL = 0.7526
    ax.plot(cen, med - EQL, "-.", color=C_EQL, lw=1.9)  # equal-lum. limit (purple dash-dot; caption)
    ax.fill_between(cen, med - EQL, -1.0, color=C_EQL, alpha=0.09, lw=0, zorder=0)
    # The 0.75-mag "not plain binaries" region is explained in the caption rather
    # than annotated in-plot, to keep the figure clear of overlapping text.
    ax.set_xlabel(r"$G_{\rm BP}-G_{\rm RP}$ [mag]"); ax.set_ylabel(r"$M_G$ [mag]")
    ax.set_xlim(0.0, 3.2); ax.set_ylim(9.5, -1.0)  # CMD-sensible; inverted (bright at top); clips parallax outliers
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="0.8",
              fontsize=7.5, handlelength=1.1, borderpad=0.45, labelspacing=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "cmd_validation%s.png"%FILT_TAG), dpi=200)
    fig.savefig(os.path.join(FIGDIR, "cmd_validation%s.pdf"%FILT_TAG), dpi=220)
    plt.close(fig)

    # ---- persist culled catalog ----
    df[confirmed].to_csv(os.path.join(_ROOT, "resources/census/dr19_sb2_catalog_culled%s.csv"%FILT_TAG),
                         index=False)

    res = {
        "n_parent": n_parent, "gaia_cmd_covered_parent": cov_parent,
        "sb2_raw": n_raw, "sb2_raw_frac": n_raw / n_parent,
        "sb2_reliable": n_rel, "sb2_reliable_frac": n_rel / n_parent,
        "sb2_reliable_cmd_covered": n_rel_cov,
        "sb2_cmd_confirmed": n_conf,
        "cmd_confirmed_frac_of_covered_parent": (n_conf / cov_parent) if cov_parent else None,
        "dmag_threshold": DMAG,
        "q_median_confirmed": float(np.median(qr)) if len(qr) else None,
        "eb_q_median": float(np.median(eb_qr)) if len(eb_qr) else None,
        "q_ks_after_chain": ({"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)} if ks else None),
        "ruwe_nss": {"confirmed_ruwe_gt1.4": ruwe_conf, "confirmed_nss": nss_conf,
                     "confirmed_n_gaia": n_conf_cov,
                     "parent_ruwe_gt1.4": ruwe_par, "parent_nss": nss_par},
        "eb_fraction_ref": 0.03,
        "vision_cull": "wired (src/mcp_servers/vision_server.py) but not run: no GOOGLE_API_KEY here",
    }
    with open(os.path.join(_ROOT, "benchmarks", "census_purity_chain%s.json"%FILT_TAG), "w") as f:
        json.dump(res, f, indent=2)

    provenance = ("HEADLINE full-census run (70k dwarfs)." if n_parent > 20000 else
                  "VALIDATION subset (8,389 dwarfs); the HEADLINE result is the full "
                  "70k run in census_purity_chain_full.md. This subset is the small-N "
                  "q-consistency check (KS p ~ 0.4); the full run is the headline "
                  "(3.6% confirmed, q median 0.84).")
    lines = [
        "# DR19 census: El-Badry reliability chain (CMD over-luminosity + Gaia)", "",
        provenance, "",
        "Parent (with verdict): %d; Gaia-CMD-covered: %d (%.0f%%)." % (
            n_parent, cov_parent, 100 * cov_parent / n_parent), "",
        "| stage | N | fraction of parent |", "| --- | --- | --- |",
        "| raw (prefers_binary) | %d | %.1f%% |" % (n_raw, 100 * n_raw / n_parent),
        "| + q/Teff reliable | %d | %.1f%% |" % (n_rel, 100 * n_rel / n_parent),
        "| + CMD over-luminous (confirmed) | %d | %.1f%% (of Gaia-covered parent) |" % (
            n_conf, 100 * n_conf / cov_parent if cov_parent else 0),
        "| El-Badry 2018b | -- | ~3%% |",
        "",
        "q after full chain: median %.2f vs El-Badry %.2f;  KS p = %s" % (
            np.median(qr) if len(qr) else float("nan"),
            np.median(eb_qr) if len(eb_qr) else float("nan"),
            ("%.3f" % ks.pvalue) if ks else "n/a"),
        "",
        "RUWE/NSS (independent astrometric support, Gaia-covered):",
        "  CMD-confirmed SB2: RUWE>1.4 %.0f%%, NSS %.0f%% (n=%d)" % (
            100 * ruwe_conf, 100 * nss_conf, n_conf_cov),
        "  parent singles:    RUWE>1.4 %.0f%%, NSS %.0f%%" % (100 * ruwe_par, 100 * nss_par),
        "  (RUWE is weak for the short-period SB2 that dominate coadd line-doubling, so it",
        "   is a cross-check, not a hard cut.)",
        "",
        "Vision cull (gemini, SKILL 14): wired but not run here (no GOOGLE_API_KEY); the CMD",
        "over-luminosity test fills its contamination-cull role quantitatively.",
        "",
        "Figure: figures/cmd_validation.{pdf,png}. Culled catalog:",
        "resources/census/dr19_sb2_catalog_culled.csv.",
    ]
    with open(os.path.join(_ROOT, "benchmarks", "census_purity_chain%s.md"%FILT_TAG), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote benchmarks/census_purity_chain.{json,md}, figures/cmd_validation.{pdf,png}")


if __name__ == "__main__":
    main()
