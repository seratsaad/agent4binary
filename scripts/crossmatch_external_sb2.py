#!/usr/bin/env python3
"""Cross-match the benchmark control sample against published SB2 catalogs.

Answers OJA referee point 2, which asks that the single-star training set be
cleaned against GaiaNSS, Kounkel+21 and Kovalev+22/24 before training.

Catalogs pulled live from the VizieR TAP service:
  Kovalev, Chen & Han (2022), MNRAS 517, 356   J/MNRAS/517/356/tablec1  (2,460 SB2)
  Kovalev, Chen & Han (2024), MNRAS 527, 521   J/MNRAS/527/521/tableb1  (12,426 SB2)
  Kounkel et al. (2021), AJ 162, 184           J/AJ/162/184/table1      (8,105 SB2+)

The Kovalev tables carry Gaia source_ids, so those match exactly. Kounkel is
keyed on APOGEE identifiers, so that one is matched positionally against Gaia
positions for our stars. The separation histogram plateaus by 1 arcsec, and we
quote the 2 arcsec count as the conservative figure.

Writes resources/census/external_sb2_crossmatch.csv.
"""
import csv, io, json, math, os, sys, urllib.parse, urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(_ROOT, "resources", "census", "benchmark_with_gaia.csv")
OUT = os.path.join(_ROOT, "resources", "census", "external_sb2_crossmatch.csv")

VIZIER = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
GAIA = "https://gea.esac.esa.int/tap-server/tap/sync"
MATCH_RADIUS = 2.0  # arcsec, for the positional Kounkel match


def tap(endpoint, query, timeout=300):
    data = urllib.parse.urlencode({
        "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query,
    }).encode()
    req = urllib.request.Request(endpoint, data=data)
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def sep_arcsec(ra1, de1, ra2, de2):
    dra = math.radians(ra1 - ra2) * math.cos(math.radians((de1 + de2) / 2.0))
    dde = math.radians(de1 - de2)
    return math.degrees(math.hypot(dra, dde)) * 3600.0


def gaia_positions(source_ids, chunk=600):
    pos = {}
    for i in range(0, len(source_ids), chunk):
        ids = ",".join(source_ids[i:i + chunk])
        q = "SELECT source_id, ra, dec FROM gaiadr3.gaia_source WHERE source_id IN (%s)" % ids
        for r in rows(tap(GAIA, q)):
            pos[r["source_id"]] = (float(r["ra"]), float(r["dec"]))
        sys.stderr.write("  positions %d/%d\n" % (min(i + chunk, len(source_ids)), len(source_ids)))
    return pos


def main():
    bench = list(csv.DictReader(open(BENCH)))
    ctl = [r for r in bench if r["is_sb2_benchmark"].strip() in ("0", "0.0")]
    gid_to_sid, sids = {}, []
    for r in ctl:
        g = r["gaia_dr3_source_id"].split(".")[0].strip()
        if g:
            gid_to_sid[g] = r["sdss_id"].split(".")[0]
            sids.append(g)
    sys.stderr.write("control stars with a Gaia id: %d\n" % len(sids))

    k22 = {r["gid"].split(".")[0] for r in rows(tap(VIZIER,
        'SELECT "GaiaEDR3" AS gid FROM "J/MNRAS/517/356/tablec1" WHERE "GaiaEDR3" IS NOT NULL'))}
    k24 = {r["gid"].split(".")[0] for r in rows(tap(VIZIER,
        'SELECT "GaiaDR3" AS gid FROM "J/MNRAS/527/521/tableb1" WHERE "GaiaDR3" IS NOT NULL'))}
    kounkel = [r for r in rows(tap(VIZIER,
        'SELECT "ID", "RAJ2000", "DEJ2000", "SBn" FROM "J/AJ/162/184/table1"'))
        if r["RAJ2000"].strip()]
    sys.stderr.write("Kovalev+22 %d, Kovalev+24 %d, Kounkel+21 %d\n"
                     % (len(k22), len(k24), len(kounkel)))

    pos = gaia_positions(sids)

    cell = 0.05
    grid = {}
    for g, (ra, de) in pos.items():
        grid.setdefault((int(ra / cell), int(de / cell)), []).append((g, ra, de))

    kmatch = {}
    for r in kounkel:
        ra, de = float(r["RAJ2000"]), float(r["DEJ2000"])
        ci, cj = int(ra / cell), int(de / cell)
        for i in (ci - 1, ci, ci + 1):
            for j in (cj - 1, cj, cj + 1):
                for g, sra, sde in grid.get((i, j), ()):
                    s = sep_arcsec(ra, de, sra, sde)
                    if s <= MATCH_RADIUS and (g not in kmatch or s < kmatch[g][0]):
                        kmatch[g] = (s, r["ID"], r["SBn"])

    ours = set(pos)
    hits = (ours & k22) | (ours & k24) | set(kmatch)
    ids = lambda p: {l.strip() for l in open(os.path.join(_ROOT, "resources", p)) if l.strip()}
    pool = ids("selffit_big_keep_ids.txt")
    holdA = ids("sfbig_holdA_keep_ids.txt")
    holdB = ids("sfbig_holdB_ids.txt")

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sdss_id", "gaia_dr3_source_id", "in_kovalev22", "in_kovalev24",
                    "in_kounkel21", "kounkel_sep_arcsec", "kounkel_id",
                    "removed_by_our_preliminary_pass", "in_training_half", "in_heldout_controls"])
        for g in sorted(hits):
            s = gid_to_sid[g]
            km = kmatch.get(g)
            w.writerow([s, g, int(g in k22), int(g in k24), int(g in kmatch),
                        "%.3f" % km[0] if km else "", km[1] if km else "",
                        int(s not in pool), int(s in holdA), int(s in holdB)])

    hs = {gid_to_sid[g] for g in hits}
    print("external SB2 among the %d controls : %d (%.2f%%)"
          % (len(ours), len(hits), 100.0 * len(hits) / len(ours)))
    print("  Kovalev+22 %d, Kovalev+24 %d, Kounkel+21 %d"
          % (len(ours & k22), len(ours & k24), len(kmatch)))
    print("already removed by our own pass    : %d (%.0f%%)"
          % (len(hs - pool), 100.0 * len(hs - pool) / max(len(hs), 1)))
    print("entered the released training set   : %d of %d (%.2f%%)"
          % (len(hs & holdA), len(holdA), 100.0 * len(hs & holdA) / len(holdA)))
    print("in the held-out controls            : %d of %d (%.2f%%)"
          % (len(hs & holdB), len(holdB), 100.0 * len(hs & holdB) / len(holdB)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
