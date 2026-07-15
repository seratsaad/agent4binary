#!/usr/bin/env python3
# =========================================================================== #
# analyze_binspec_overlap.py
#
# Read resources/fair_tests/binspec_overlap/results.csv (the star-for-star
# binspec-ORACLE vs OUR Stage-1 SC detector comparison on a random ~250-star
# El-Badry SB2 / DR19 overlap) and write summary.md with the headline numbers:
#
#   - overall recovery (fraction flagged binary) for binspec and for OURS;
#   - recovery by q-bin (El-Badry q);
#   - recovery split by q_dyn-presence (combined-only vs has-dynamical-q);
#   - star-for-star agreement (confusion: who flags what);
#   - the interpretation of the SC sensitivity gap.
# =========================================================================== #

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
OUTDIR = os.path.join(_ROOT, "resources/fair_tests/binspec_overlap")


def _b(x):
    """Parse a CSV bool/blank cell to a python bool (blank/NaN -> False)."""
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("true", "1")


def pct(n, d):
    return (100.0 * n / d) if d else float("nan")


def main():
    res = pd.read_csv(os.path.join(OUTDIR, "results.csv"))
    n_total = len(res)
    err = res["error"].astype(str).str.strip()
    ok = res[(err == "") | (err == "nan")].copy()
    n_err = n_total - len(ok)

    ok["bs"] = ok["bs_prefers_binary"].apply(_b)
    ok["sc"] = ok["sc_prefers_binary"].apply(_b)
    ok["has_qdyn"] = ok["q_dyn"].notna() & (ok["q_dyn"].astype(str).str.strip() != "")

    n = len(ok)
    bs_rec = ok["bs"].sum()
    sc_rec = ok["sc"].sum()

    # star-for-star confusion
    both = int((ok["bs"] & ok["sc"]).sum())
    bs_only = int((ok["bs"] & ~ok["sc"]).sum())
    sc_only = int((~ok["bs"] & ok["sc"]).sum())
    neither = int((~ok["bs"] & ~ok["sc"]).sum())
    agree = both + neither

    # q-bins
    qedges = [0.4, 0.6, 0.7, 0.8, 0.9, 1.001]
    qlabels = ["0.4-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    ok["qbin"] = pd.cut(ok["q"], bins=qedges, labels=qlabels, right=False, include_lowest=True)

    L = []
    L.append("# binspec ORACLE vs OUR Stage-1 SB2 recovery on the SAME DR19 coadds")
    L.append("")
    L.append("Decisive consistency test. Random sample of the El-Badry 2018b SB2 / DR19 "
             "overlap (resources/elbadry2018b_binaries.csv type==\"SB2\", mapped "
             "apogee_id<->sdss_id via resources/dr19_sb2_manifest.csv). For each star "
             "we run BOTH single-vs-binary detectors on the IDENTICAL DR19 mwmStar "
             "COADD: the binspec ORACLE (El-Badry's own vendored flux net + Eq. B1 / "
             "Table B1 decision, physics.binspec_single_vs_binary) and OUR Stage-1 "
             "self-consistent detector (physics.dr19_sc_single_vs_binary). The current "
             "question is how large the SC detector sensitivity gap remains.")
    L.append("")
    L.append("These are all El-Badry SB2, so every star is a true binary; \"recovery\" = "
             "fraction the detector flags binary. q_dyn-present rows had a multi-epoch "
             "dynamical mass ratio in El-Badry (their V_scatter route); q_dyn-absent rows "
             "are El-Badry's COMBINED-SPECTRUM-only SB2.")
    L.append("")
    L.append("## Sample")
    L.append("")
    L.append(f"- stars attempted: {n_total}")
    L.append(f"- usable (no per-star error): {n}")
    if n_err:
        L.append(f"- per-star errors: {n_err}")
    L.append(f"- with El-Badry dynamical q (q_dyn present): {int(ok['has_qdyn'].sum())}")
    L.append(f"- combined-spectrum-only (q_dyn absent): {int((~ok['has_qdyn']).sum())}")
    L.append("")
    L.append("## HEADLINE: overall recovery on the same stars")
    L.append("")
    L.append("| detector | flagged binary | recovery |")
    L.append("|---|---|---|")
    L.append(f"| binspec ORACLE | {int(bs_rec)}/{n} | {pct(bs_rec, n):.1f}% |")
    L.append(f"| OUR Stage-1 SC | {int(sc_rec)}/{n} | {pct(sc_rec, n):.1f}% |")
    L.append("")
    L.append("## Star-for-star agreement")
    L.append("")
    L.append(f"- both flag binary: {both}")
    L.append(f"- binspec-only (binspec binary, ours single): {bs_only}")
    L.append(f"- ours-only (ours binary, binspec single): {sc_only}")
    L.append(f"- neither (both call single): {neither}")
    L.append(f"- agreement: {agree}/{n} = {pct(agree, n):.1f}%")
    L.append("")
    L.append("## Recovery by q-bin (El-Badry q)")
    L.append("")
    L.append("| q-bin | n | binspec | ours |")
    L.append("|---|---|---|---|")
    for lab in qlabels:
        sub = ok[ok["qbin"] == lab]
        if len(sub) == 0:
            continue
        L.append(f"| {lab} | {len(sub)} | {pct(sub['bs'].sum(), len(sub)):.0f}% "
                 f"({int(sub['bs'].sum())}/{len(sub)}) | "
                 f"{pct(sub['sc'].sum(), len(sub)):.0f}% ({int(sub['sc'].sum())}/{len(sub)}) |")
    L.append("")
    L.append("## Recovery split by q_dyn-presence")
    L.append("")
    L.append("| subset | n | binspec | ours |")
    L.append("|---|---|---|---|")
    for name, mask in [("q_dyn present (dynamical)", ok["has_qdyn"]),
                       ("combined-only (no q_dyn)", ~ok["has_qdyn"])]:
        sub = ok[mask]
        if len(sub) == 0:
            continue
        L.append(f"| {name} | {len(sub)} | {pct(sub['bs'].sum(), len(sub)):.0f}% "
                 f"({int(sub['bs'].sum())}/{len(sub)}) | "
                 f"{pct(sub['sc'].sum(), len(sub)):.0f}% ({int(sub['sc'].sum())}/{len(sub)}) |")
    L.append("")

    # Interpretation, driven by the data.
    bs_p = pct(bs_rec, n)
    sc_p = pct(sc_rec, n)
    gap = bs_p - sc_p
    L.append("## Interpretation")
    L.append("")
    L.append(f"binspec recovers {bs_p:.1f}% of these SB2 from the DR19 coadd; ours "
             f"recovers {sc_p:.1f}% on the SAME stars (gap = {gap:+.1f} points, "
             f"binspec-only={bs_only}, ours-only={sc_only}).")
    L.append("")
    if abs(gap) <= 7.0:
        L.append("**binspec ~= ours.** The SC detector now matches the vendored oracle "
                 "on this coadd fair test. Re-run controls before promoting the census.")
    elif gap > 7.0:
        L.append("**binspec >> ours -> our SC detector has a real SENSITIVITY GAP.** "
                 "El-Badry's net flags substantially more of these SB2 from the SAME "
                 "coadds than we do, so the signal IS present in the DR19 combined "
                 "spectrum and our detector is leaving it on the table. Fix the SC "
                 "detector's sensitivity before relying on Stage 2 to close the gap. "
                 "Inspect the binspec-only stars (binspec flags, we miss) for the "
                 "common failure mode.")
    else:
        L.append("**ours >> binspec.** We flag substantially more than El-Badry's own "
                 "net on the same coadds. Check the ours-only stars for false positives "
                 "/ over-flagging before claiming a true sensitivity advantage.")
    L.append("")
    L.append("(Reference: the binspec production detector recovers ~68% at 14% FPR; "
             "see benchmarks/binspec_production_purity.md.)")
    L.append("")

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "summary.md"), "w") as f:
        f.write("\n".join(L))

    # Echo the headline to stdout.
    print("\n".join(L[:30]))
    print("...")
    print("INTERPRETATION:", L[-3] if len(L) >= 3 else "")


if __name__ == "__main__":
    main()
