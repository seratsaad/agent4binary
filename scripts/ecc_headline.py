#!/usr/bin/env python3
"""Headline eccentricity numbers, exactly as the paper combines them, printed
and written to resources/census/ecc_headline.json.

  fiducial_diff   dAlpha of the fiducial variant (K1 > 12, matched on K1 + epochs)
                  from ecc_variants.csv (written by scripts/ecc_final_table.py)
  bootstrap_err   its bootstrap error (same file)
  null_bias       pooled injection-null bias, ecc_solve.injection_bias + pooled_bias
  null_err        its error (mean per-alpha spread / sqrt(realizations))
  matching_spread sd (ddof=1) of dAlpha over the first five variants
  combined        fiducial_diff - null_bias
  error           sqrt(bootstrap_err^2 + null_err^2 + matching_spread^2)

plus the sample counts the paper quotes. paper2/figures/fig_ecc_validation.py
reads combined and error from the JSON. Run by ecc_final_table.py at its end;
can also be run on its own:

  python scripts/ecc_headline.py [--variants F] [--null F] [--real F] [--out F]
"""
import os, sys, json, hashlib, argparse
import numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import ecc_solve as S
import deep_table as DT

R = lambda p: p if os.path.isabs(p) else os.path.join(_ROOT, p)
KCUT = 12.0
N_SPREAD = 5        # variants entering the matching spread (as ecc_final_table prints)


def _md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def build(V, null_path, real_path, val_path=None):
    fid = V.iloc[0]
    assert "fiducial" in str(fid["label"]), "first row of the variants is not the fiducial one"

    bias = S.injection_bias(null_path)
    if not bias:
        raise SystemExit("no injection null in %s (run scripts/ecc_null_full.py)" % null_path)
    null_bias, null_err = S.pooled_bias(bias)

    da5 = V["da"].values[:N_SPREAD].astype(float)
    spread = float(np.std(da5, ddof=1))
    boot = float(fid["boot"])
    combined = float(fid["da"]) - null_bias
    err = float(np.sqrt(boot**2 + null_err**2 + spread**2))

    d = pd.read_csv(real_path)
    n_file = len(d)
    d = d[(d.P50 >= 6) & (d.P50 <= 400)]
    m = d[d.K1 > KCUT]
    tw = d.twin.astype(bool); mtw = m.twin.astype(bool)
    # the fiducial variant must come from this ecc_marginal_real.csv
    assert (int(fid["n_t"]), int(fid["n_n"])) == (int(mtw.sum()), int((~mtw).sum())), (
        "ecc_variants.csv (n_t=%d, n_n=%d) does not match %s (K1>12: %d twins, %d non-twins); "
        "rerun scripts/ecc_final_table.py" % (fid["n_t"], fid["n_n"], real_path,
                                              mtw.sum(), (~mtw).sum()))

    out = dict(
        fiducial_label=str(fid["label"]),
        fiducial_diff=float(fid["da"]), bootstrap_err=boot,
        null_bias=float(null_bias), null_err=float(null_err),
        null_bias_per_alpha={"%.2f" % al: dict(mean=float(mu), sd=float(sd), n_reps=int(n))
                             for al, (mu, sd, n) in sorted(bias.items())},
        null_n_reps=int(sum(n for _, _, n in bias.values())),
        matching_spread=spread, matching_spread_variants=list(V["label"].values[:N_SPREAD]),
        combined=combined, error=err, significance=abs(combined) / err,
        alpha_twin=float(fid["a_t"]), alpha_nontwin=float(fid["a_n"]),
        mean_e_twin=float(fid["e_t"]), mean_e_nontwin=float(fid["e_n"]),
        ess_nontwin=float(fid["ess"]),
        n_systems_file=int(n_file),
        n_systems=int(len(d)), n_twins=int(tw.sum()),
        median_K1_twin=float(d[tw].K1.median()), median_K1_nontwin=float(d[~tw].K1.median()),
        n_k1gt12=int(len(m)), n_k1gt12_twins=int(mtw.sum()),
        median_K1_twin_k1gt12=float(m[mtw].K1.median()),
        median_K1_nontwin_k1gt12=float(m[~mtw].K1.median()),
        max_variant_shift=float(np.max(np.abs(V["da"].values[1:].astype(float) - float(fid["da"])))),
        variants=[dict(label=str(v["label"]), da=float(v["da"]), boot=float(v["boot"]),
                       n_t=int(v["n_t"]), n_n=int(v["n_n"]),
                       corrected=float(v["da"]) - null_bias,
                       upper_2sigma=float(v["da"]) - null_bias
                       + 2 * float(np.sqrt(float(v["boot"])**2 + null_err**2 + spread**2)))
                  for _, v in V.iterrows()],
    )
    dp = DT.deep_path()
    if os.path.exists(dp):
        out["n_deep_rows"] = int(len(pd.read_csv(dp, usecols=["sdss_id"])))
    if val_path and os.path.exists(val_path):
        val = pd.read_csv(val_path)
        sl, ic = np.polyfit(val.da_true.values, val.da_rec.values, 1)
        out["validation_slope"], out["validation_intercept"] = float(sl), float(ic)
    out["inputs"] = {k: dict(path=p, md5=_md5(p)) for k, p in
                     [("variants", None), ("null", null_path), ("real", real_path),
                      ("validation", val_path), ("deep", dp)] if p}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="resources/census/ecc_variants.csv")
    ap.add_argument("--null", default="resources/census/ecc_null_full.csv")
    ap.add_argument("--real", default="resources/census/ecc_marginal_real.csv")
    ap.add_argument("--validation", default="resources/census/ecc_validation.csv")
    ap.add_argument("--out", default="resources/census/ecc_headline.json")
    a = ap.parse_args(argv)
    V = pd.read_csv(R(a.variants))
    H = build(V, R(a.null), R(a.real), R(a.validation))
    H["inputs"]["variants"] = dict(path=R(a.variants), md5=_md5(R(a.variants)))
    json.dump(H, open(R(a.out), "w"), indent=1)

    print("HEADLINE (%s)" % H["fiducial_label"])
    print("  fiducial dAlpha   %+.3f  (twin %+.3f, non-twin %+.3f)"
          % (H["fiducial_diff"], H["alpha_twin"], H["alpha_nontwin"]))
    print("  null bias         %+.3f +/- %.3f  (%d realizations)"
          % (H["null_bias"], H["null_err"], H["null_n_reps"]))
    print("  matching spread   %.3f   bootstrap %.3f" % (H["matching_spread"], H["bootstrap_err"]))
    print("  combined          %+.3f +/- %.3f  (%.1f sigma)"
          % (H["combined"], H["error"], H["significance"]))
    print("  systems %d (twins %d); K1>12 %d (twins %d); median K1 twin %.1f non-twin %.1f"
          % (H["n_systems"], H["n_twins"], H["n_k1gt12"], H["n_k1gt12_twins"],
             H["median_K1_twin"], H["median_K1_nontwin"]))
    print("  <e> twin %.2f non-twin %.2f; largest variant shift %.3f"
          % (H["mean_e_twin"], H["mean_e_nontwin"], H["max_variant_shift"]))
    for v in H["variants"]:
        print("    %-34s corrected %+.3f  2-sigma upper %+.3f" % (v["label"], v["corrected"],
                                                                  v["upper_2sigma"]))
    if "validation_slope" in H:
        print("  validation slope %.2f" % H["validation_slope"])
    print("wrote", R(a.out))
    return H


if __name__ == "__main__":
    main()
