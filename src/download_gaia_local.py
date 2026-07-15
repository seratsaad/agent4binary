#!/usr/bin/env python3
"""
Build resources/gaia_dr19_dwarfs.parquet: a LOCAL Gaia DR3 subset for the DR19
dwarf sample, fetched WITHOUT the Gaia TAP archive (which is network-blocked on
this machine).

Motivation. The runtime Gaia cross-check in src/mcp_servers/gaia_sql_server.py
hits gea.esac.esa.int (Gaia TAP/ADQL). That host is blocked here ("remote end
closed connection"). Two other channels ARE reachable and together cover every
field we need, so we pre-fetch the subset once into a local Parquet table that
the MCP server reads with no network call.

Two-channel fetch (both verified reachable from this machine):

  Channel A -- SDSS CAS SkyServer SQL (astroquery.sdss, data_release=19).
    Table the_payne_apogee_star carries the sdss_id -> gaia_dr3_source_id map and
    Gaia-DR3-derived astrometry/photometry: plx (parallax), e_plx, g_mag, bp_mag,
    rp_mag. It does NOT carry ruwe, non_single_star, or astrometric_excess_noise
    (the only Gaia tables hosted on SDSS CAS DR19 are Gaia DR2, e.g.
    mos_gaia_dr2_source / mos_gaia_dr2_ruwe -- wrong data release). So Channel A
    supplies the source_id map plus parallax and magnitudes.

  Channel B -- VizieR Gaia DR3 main catalog I/355/gaiadr3 (astroquery.vizier).
    VizieR runs on a different server than the blocked Gaia TAP and IS reachable.
    Queried by Source (= Gaia DR3 source_id) it returns the fields SDSS CAS lacks:
    RUWE, NSS (non_single_star), epsi (astrometric_excess_noise), sepsi
    (astrometric_excess_noise_sig), and also Plx/e_Plx/Gmag/BP-RP for cross-check.

Output. resources/gaia_dr19_dwarfs.parquet, one row per unique sdss_id, columns:

    sdss_id                         int64   (key)
    gaia_dr3_source_id              int64   (Gaia DR3 source_id; 0 if unmapped)
    ruwe                            float   (renormalised unit weight error)
    parallax                        float   (mas)
    parallax_error                  float   (mas)
    phot_g_mean_mag                 float   (Gaia G)
    bp_rp                           float   (BP - RP colour)
    non_single_star                 int     (Gaia NSS bitmask; !=0 => NSS solution)
    astrometric_excess_noise        float   (mas)
    astrometric_excess_noise_sig    float
    has_astrometry                  bool    (True if RUWE etc. were obtained)
    source                          str     ("sdss_cas+vizier" / "sdss_cas_only")

The MCP server tool binarity_local(sdss_id) reads this file and returns the row,
flagging RUWE > 1.4 as a possible unresolved binary.

Idempotent-ish: each run rebuilds the table from the live channels. Both queries
retry on transient SSL/network drops (SDSS CAS occasionally resets a connection).

Run: python src/download_gaia_local.py
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_RES = os.path.join(_ROOT, "resources")

# Manifests that define the DR19 dwarf sample (key: sdss_id).
MANIFESTS = [
    os.path.join(_RES, "dr19_raw_train_manifest.csv"),
    os.path.join(_RES, "dr19_sb2_manifest.csv"),
    os.path.join(_RES, "dr19_raw_controls_manifest.csv"),
]

OUT_PARQUET = os.path.join(_RES, "gaia_dr19_dwarfs.parquet")
# Cache of the slow Channel-A (SDSS CAS) map, so a re-run that only needs to
# re-fetch Channel B (VizieR) does not re-query SDSS. Delete it to force a
# fresh SDSS pull.
_SDSS_CACHE = os.path.join(_RES, ".gaia_sdss_map_cache.parquet")

# Batch sizes keep each query (a GET URL) short enough for each service.
# SDSS CAS errors out on very large IN-clauses; 100 ids per batch is reliable.
# VizieR encodes the Source list in the URL; a 150-id "=a,b,c" constraint is
# ~2850 chars and works, while ~400 ids (~7800 chars) exceeds the URL limit and
# silently returns 0 rows. So keep VIZIER_BATCH at 150.
SDSS_BATCH = 100       # sdss_ids per the_payne_apogee_star IN-clause query
VIZIER_BATCH = 150     # source_ids per VizieR I/355/gaiadr3 constraint query
RETRIES = 5            # per-batch retry count for transient network drops
RETRY_SLEEP = 4.0      # seconds between retries


# --------------------------------------------------------------------------- #
# Sample
# --------------------------------------------------------------------------- #
def load_sample_ids():
    """Collect the unique, valid sdss_ids across the three manifests.

    Returns a sorted list of int sdss_ids. sdss_id == 0 is a placeholder/missing
    value in the manifests and is dropped (it has no Gaia counterpart).
    """
    ids = set()
    for path in MANIFESTS:
        if not os.path.exists(path):
            print(f"  WARN: missing manifest {path}", flush=True)
            continue
        df = pd.read_csv(path, usecols=["sdss_id"])
        ids.update(int(x) for x in df["sdss_id"].dropna().astype("int64"))
    ids.discard(0)  # placeholder id, not a real star
    return sorted(ids)


def _chunks(seq, n):
    """Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --------------------------------------------------------------------------- #
# Channel A: SDSS CAS -> sdss_id -> gaia_dr3_source_id + parallax + magnitudes
# --------------------------------------------------------------------------- #
def fetch_sdss_cas(sdss_ids):
    """Map sdss_id -> Gaia DR3 source_id (+ plx/e_plx/g/bp/rp) via SDSS CAS DR19.

    Queries the_payne_apogee_star in batches. That table can hold more than one
    spectrum row per sdss_id, so we de-duplicate by sdss_id (keep first). Returns
    a DataFrame keyed by sdss_id.
    """
    from astroquery.sdss import SDSS

    cols = ("sdss_id, gaia_dr3_source_id, plx, e_plx, g_mag, bp_mag, rp_mag")
    frames = []
    batches = list(_chunks(sdss_ids, SDSS_BATCH))
    for bi, batch in enumerate(batches):
        idlist = ",".join(str(i) for i in batch)
        q = (f"SELECT {cols} FROM the_payne_apogee_star "
             f"WHERE sdss_id IN ({idlist})")
        tbl = None
        for attempt in range(RETRIES):
            try:
                tbl = SDSS.query_sql(q, data_release=19)
                break
            except Exception as e:
                # SDSS CAS occasionally resets the TLS connection; just retry.
                print(f"    SDSS batch {bi+1}/{len(batches)} attempt "
                      f"{attempt+1} failed: {type(e).__name__}", flush=True)
                time.sleep(RETRY_SLEEP)
        if tbl is None:
            print(f"    SDSS batch {bi+1}/{len(batches)} GAVE UP", flush=True)
            continue
        if len(tbl):
            frames.append(tbl.to_pandas())
        print(f"    SDSS batch {bi+1}/{len(batches)}: {len(tbl)} rows "
              f"(cumulative)", flush=True)

    if not frames:
        return pd.DataFrame(columns=["sdss_id", "gaia_dr3_source_id", "plx",
                                     "e_plx", "g_mag", "bp_mag", "rp_mag"])
    df = pd.concat(frames, ignore_index=True)
    # One row per star: keep the first spectrum row per sdss_id.
    df = df.drop_duplicates(subset="sdss_id", keep="first").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Channel B: VizieR Gaia DR3 -> RUWE / NSS / excess noise by source_id
# --------------------------------------------------------------------------- #
def fetch_vizier_dr3(source_ids):
    """Fetch Gaia DR3 binarity fields from VizieR I/355/gaiadr3 by source_id.

    Returns a DataFrame keyed by Source (= gaia_dr3_source_id) with columns
    ruwe, nss, epsi, sepsi, plx_v, e_plx_v, gmag_v, bp_rp_v. The "_v" columns are
    VizieR cross-check copies of parallax/mag/colour (SDSS CAS values are used as
    primary; VizieR fills any gap).
    """
    from astroquery.vizier import Vizier

    # epsi = astrometric_excess_noise, sepsi = its significance, NSS = non_single_star.
    want = ["Source", "RUWE", "NSS", "epsi", "sepsi",
            "Plx", "e_Plx", "Gmag", "BP-RP"]
    rows = []
    src = [int(s) for s in source_ids if s and int(s) != 0]
    batches = list(_chunks(src, VIZIER_BATCH))
    for bi, batch in enumerate(batches):
        # row_limit=-1 -> no truncation; one row is returned per matched Source.
        v = Vizier(columns=want, row_limit=-1)
        constraint = "=" + ",".join(str(i) for i in batch)
        res = None
        for attempt in range(RETRIES):
            try:
                res = v.query_constraints(catalog="I/355/gaiadr3",
                                          Source=constraint)
                break
            except Exception as e:
                print(f"    VizieR batch {bi+1}/{len(batches)} attempt "
                      f"{attempt+1} failed: {type(e).__name__}", flush=True)
                time.sleep(RETRY_SLEEP)
        if res is None or len(res) == 0:
            print(f"    VizieR batch {bi+1}/{len(batches)}: 0 rows", flush=True)
            continue
        t = res[0]
        df = t.to_pandas()
        rows.append(df)
        print(f"    VizieR batch {bi+1}/{len(batches)}: {len(df)} rows",
              flush=True)

    if not rows:
        return pd.DataFrame(columns=["Source", "RUWE", "NSS", "epsi", "sepsi",
                                     "Plx", "e_Plx", "Gmag", "BP-RP"])
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset="Source", keep="first").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Merge + write
# --------------------------------------------------------------------------- #
def build_table():
    """Run both channels, merge to one row per sdss_id, write the Parquet file."""
    ids = load_sample_ids()
    print(f"sample: {len(ids)} unique sdss_ids (placeholder 0 dropped)",
          flush=True)

    print("Channel A: SDSS CAS the_payne_apogee_star ...", flush=True)
    if os.path.exists(_SDSS_CACHE):
        # Re-use the cached map (fast path for a VizieR-only re-run).
        cas = pd.read_parquet(_SDSS_CACHE)
        print(f"  SDSS CAS: loaded {len(cas)} stars from cache "
              f"{os.path.basename(_SDSS_CACHE)}", flush=True)
    else:
        cas = fetch_sdss_cas(ids)
        cas.to_parquet(_SDSS_CACHE, index=False)
        print(f"  SDSS CAS: {len(cas)} stars mapped to gaia_dr3_source_id "
              f"(cached)", flush=True)

    # Source ids to look up on VizieR (drop missing/zero).
    cas["gaia_dr3_source_id"] = (
        pd.to_numeric(cas["gaia_dr3_source_id"], errors="coerce")
        .fillna(0).astype("int64"))
    src_ids = [s for s in cas["gaia_dr3_source_id"].tolist() if s != 0]
    print(f"Channel B: VizieR I/355/gaiadr3 for {len(src_ids)} source_ids ...",
          flush=True)
    viz = fetch_vizier_dr3(src_ids)
    viz = viz.rename(columns={
        "Source": "gaia_dr3_source_id", "RUWE": "ruwe", "NSS": "non_single_star",
        "epsi": "astrometric_excess_noise",
        "sepsi": "astrometric_excess_noise_sig",
        "Plx": "plx_v", "e_Plx": "e_plx_v", "Gmag": "gmag_v", "BP-RP": "bp_rp_v",
    })
    viz["gaia_dr3_source_id"] = (
        pd.to_numeric(viz["gaia_dr3_source_id"], errors="coerce")
        .fillna(0).astype("int64"))
    print(f"  VizieR: {len(viz)} sources with astrometry fields", flush=True)

    # Start from the full sample so id-only stars (no SDSS map) still appear.
    base = pd.DataFrame({"sdss_id": ids})
    df = base.merge(cas, on="sdss_id", how="left")
    df["gaia_dr3_source_id"] = (
        pd.to_numeric(df["gaia_dr3_source_id"], errors="coerce")
        .fillna(0).astype("int64"))
    df = df.merge(viz, on="gaia_dr3_source_id", how="left")

    # Primary astrometry/photometry: prefer SDSS CAS, fall back to VizieR.
    def coalesce(a, b):
        return pd.to_numeric(df[a], errors="coerce").fillna(
            pd.to_numeric(df[b], errors="coerce"))

    parallax = coalesce("plx", "plx_v")
    parallax_error = coalesce("e_plx", "e_plx_v")
    g_mag = coalesce("g_mag", "gmag_v")
    # bp_rp: SDSS CAS has bp_mag/rp_mag; compute the colour, else use VizieR BP-RP.
    bp_rp_cas = (pd.to_numeric(df["bp_mag"], errors="coerce")
                 - pd.to_numeric(df["rp_mag"], errors="coerce"))
    bp_rp = bp_rp_cas.fillna(pd.to_numeric(df["bp_rp_v"], errors="coerce"))

    ruwe = pd.to_numeric(df["ruwe"], errors="coerce")
    has_astro = ruwe.notna()

    out = pd.DataFrame({
        "sdss_id": df["sdss_id"].astype("int64"),
        "gaia_dr3_source_id": df["gaia_dr3_source_id"].astype("int64"),
        "ruwe": ruwe,
        "parallax": parallax,
        "parallax_error": parallax_error,
        "phot_g_mean_mag": g_mag,
        "bp_rp": bp_rp,
        "non_single_star": pd.to_numeric(
            df["non_single_star"], errors="coerce").astype("Int64"),
        "astrometric_excess_noise": pd.to_numeric(
            df["astrometric_excess_noise"], errors="coerce"),
        "astrometric_excess_noise_sig": pd.to_numeric(
            df["astrometric_excess_noise_sig"], errors="coerce"),
        "has_astrometry": has_astro,
    })
    # Provenance per row, for transparency in the report and the MCP return.
    mapped = out["gaia_dr3_source_id"] != 0
    out["source"] = np.where(
        out["has_astrometry"], "sdss_cas+vizier",
        np.where(mapped, "sdss_cas_only", "unmapped"))

    out.to_parquet(OUT_PARQUET, index=False)

    # Summary for the run log.
    n_total = len(out)
    n_mapped = int(mapped.sum())
    n_full = int(out["has_astrometry"].sum())
    print("-" * 60, flush=True)
    print(f"wrote {OUT_PARQUET}", flush=True)
    print(f"  rows                : {n_total}", flush=True)
    print(f"  with source_id      : {n_mapped}", flush=True)
    print(f"  with full Gaia astro: {n_full} (ruwe etc.)", flush=True)
    print(f"  id-only / unmapped  : {n_total - n_full}", flush=True)
    return out


if __name__ == "__main__":
    build_table()
