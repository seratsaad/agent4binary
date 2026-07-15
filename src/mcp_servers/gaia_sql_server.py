#!/usr/bin/env python3
"""
MCP Server for Gaia DR3 SQL / TAP (ADQL) queries.

Thin, demo-safe wrapper around the Gaia archive. Designed for SMALL, fast,
single-source lookups — never bulk crossmatches or long async jobs. All tools
degrade gracefully on network errors (returning {"error": ...} rather than
raising), and results are capped in rows/columns so they never flood the
LLM context.
"""

import os
import warnings
from mcp.server.fastmcp import FastMCP

warnings.filterwarnings("ignore")

mcp = FastMCP("Gaia SQL", log_level="WARNING")

# Hard caps so a tool return can never flood the agent context.
MAX_ROWS_HARD = 200
MAX_COLS_HARD = 40

# --------------------------------------------------------------------------- #
# Local Gaia DR3 subset (no-network path).
#
# The Gaia TAP archive (gea.esac.esa.int) is network-blocked on some machines
# ("remote end closed connection"). For the DR19 dwarf sample we pre-fetch the
# Gaia DR3 fields once into resources/gaia_dr19_dwarfs.parquet (built by
# src/download_gaia_local.py via SDSS CAS + VizieR, which ARE reachable). The
# tools below read that Parquet with no network call, so the binarity check
# works offline. The TAP tools above are unchanged and still used on a host
# where TAP is open.
#
# Parquet schema (one row per sdss_id):
#   sdss_id, gaia_dr3_source_id, ruwe, parallax, parallax_error,
#   phot_g_mean_mag, bp_rp, non_single_star, astrometric_excess_noise,
#   astrometric_excess_noise_sig, has_astrometry, source
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
LOCAL_PARQUET = os.path.join(_ROOT, "resources", "gaia_dr19_dwarfs.parquet")

# Cache the table in-process so repeated lookups do not re-read the file.
_LOCAL_DF = None


def _load_local_table():
    """Lazily load the local Gaia DR3 Parquet. Returns (df, None) or (None, err)."""
    global _LOCAL_DF
    if _LOCAL_DF is not None:
        return _LOCAL_DF, None
    if not os.path.exists(LOCAL_PARQUET):
        return None, (f"local Gaia table not found at {LOCAL_PARQUET}; "
                      f"run src/download_gaia_local.py to build it")
    try:
        import pandas as pd
        _LOCAL_DF = pd.read_parquet(LOCAL_PARQUET)
        return _LOCAL_DF, None
    except Exception as e:  # pragma: no cover
        return None, f"failed to read {LOCAL_PARQUET}: {e}"


def _row_to_local_dict(row):
    """Convert one local-table row (a pandas Series) to a JSON-safe binarity dict.

    Adds a RUWE > 1.4 note (classic unresolved-binary flag) and an NSS note when
    the Gaia non_single_star bitmask is non-zero.
    """
    import numpy as np
    import pandas as pd

    def num(v):
        # NaN / pandas-NA -> None; numpy scalar -> Python scalar.
        try:
            if v is None or pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    nss = num(row.get("non_single_star"))
    out = {
        "sdss_id": num(row.get("sdss_id")),
        "source_id": num(row.get("gaia_dr3_source_id")),
        "ruwe": num(row.get("ruwe")),
        "parallax": num(row.get("parallax")),
        "parallax_error": num(row.get("parallax_error")),
        "phot_g_mean_mag": num(row.get("phot_g_mean_mag")),
        "bp_rp": num(row.get("bp_rp")),
        "non_single_star": int(nss) if nss is not None else None,
        "astrometric_excess_noise": num(row.get("astrometric_excess_noise")),
        "astrometric_excess_noise_sig": num(
            row.get("astrometric_excess_noise_sig")),
        "source": row.get("source"),
    }
    flags = []
    if out["ruwe"] is not None and out["ruwe"] > 1.4:
        flags.append("RUWE>1.4")
        out["ruwe_note"] = ("RUWE > 1.4: possible unresolved binary "
                            "(poor single-star astrometric fit)")
    if out["non_single_star"]:
        flags.append("NSS solution present")
    if flags:
        out["binary_flags"] = flags
    return out


def _local_lookup_by_sdss_id(sdss_id):
    """Look one sdss_id up in the local table. Returns a dict or None (not found)."""
    df, err = _load_local_table()
    if err:
        return {"error": err}
    try:
        sid = int(sdss_id)
    except Exception:
        return {"error": "sdss_id must be an integer"}
    hit = df[df["sdss_id"] == sid]
    if len(hit) == 0:
        return None
    return _row_to_local_dict(hit.iloc[0])


def _local_lookup_by_source_id(source_id):
    """Look one Gaia source_id up in the local table. Returns a dict or None."""
    df, err = _load_local_table()
    if err:
        return {"error": err}
    try:
        sid = int(source_id)
    except Exception:
        return {"error": "source_id must be an integer"}
    hit = df[df["gaia_dr3_source_id"] == sid]
    if len(hit) == 0:
        return None
    return _row_to_local_dict(hit.iloc[0])


# 2MASS designation -> sdss_id, read from the SB2 manifest (the only manifest
# that carries apogee_id = 2MASS). Lets crossmatch_2mass fall back to the local
# table for SB2 stars when TAP is blocked. Cached in-process.
_TWOMASS_TO_SDSS = None


def _load_2mass_map():
    """Build {bare_2MASS_designation: sdss_id} from the SB2 manifest. ({} on miss)."""
    global _TWOMASS_TO_SDSS
    if _TWOMASS_TO_SDSS is not None:
        return _TWOMASS_TO_SDSS
    _TWOMASS_TO_SDSS = {}
    sb2 = os.path.join(_ROOT, "resources", "dr19_sb2_manifest.csv")
    if not os.path.exists(sb2):
        return _TWOMASS_TO_SDSS
    try:
        import pandas as pd
        df = pd.read_csv(sb2, usecols=["apogee_id", "sdss_id"])
        for _, r in df.iterrows():
            desig = _normalize_2mass(str(r["apogee_id"]))
            try:
                _TWOMASS_TO_SDSS[desig] = int(r["sdss_id"])
            except Exception:
                continue
    except Exception:
        pass
    return _TWOMASS_TO_SDSS


def _get_gaia():
    """Lazily import + configure astroquery.gaia. Returns (Gaia, None) or (None, errmsg)."""
    try:
        from astroquery.gaia import Gaia
        Gaia.ROW_LIMIT = MAX_ROWS_HARD
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        return Gaia, None
    except Exception as e:  # pragma: no cover
        return None, f"astroquery.gaia unavailable: {e}"


def _table_to_rows(table, max_rows: int):
    """Convert an astropy Table to a list of JSON-safe dicts, capped."""
    import numpy as np

    n = min(len(table), max_rows, MAX_ROWS_HARD)
    cols = list(table.colnames)[:MAX_COLS_HARD]
    rows = []
    for i in range(n):
        row = {}
        for c in cols:
            v = table[c][i]
            try:
                if v is None or (np.ma.is_masked(v)):
                    row[c] = None
                    continue
            except Exception:
                pass
            if isinstance(v, (np.integer,)):
                row[c] = int(v)
            elif isinstance(v, (np.floating,)):
                fv = float(v)
                row[c] = None if (fv != fv) else fv  # NaN -> None
            elif isinstance(v, (np.bool_, bool)):
                row[c] = bool(v)
            elif isinstance(v, bytes):
                row[c] = v.decode("utf-8", "replace")
            else:
                row[c] = str(v)
        rows.append(row)
    return rows


@mcp.tool()
def adql_query(query: str, max_rows: int = 20) -> dict:
    """
    Run an ADQL query against the Gaia DR3 archive and return up to max_rows rows.

    Use this for SMALL, fast lookups (always include a TOP N clause, N <= 50).
    Do NOT run bulk crossmatches or long jobs. Tables live under the `gaiadr3`
    schema, e.g. `gaiadr3.gaia_source`. The result is returned as a compact list
    of dicts; rows are capped at max_rows (hard cap 200) and columns at 40 to
    avoid flooding context.

    Args:
        query: An ADQL SELECT statement (e.g.
               "SELECT TOP 3 source_id, ra, dec, parallax FROM gaiadr3.gaia_source WHERE parallax > 50").
        max_rows: Maximum rows to return (default 20, hard-capped at 200).

    Returns:
        {"n_rows": int, "columns": [...], "rows": [ {...}, ... ]} or {"error": "..."}.
    """
    if not query or "select" not in query.lower():
        return {"error": "query must be a SELECT/ADQL statement"}
    max_rows = max(1, min(int(max_rows), MAX_ROWS_HARD))

    Gaia, err = _get_gaia()
    if err:
        return {"error": err}
    try:
        job = Gaia.launch_job(query)  # synchronous; small queries only
        table = job.get_results()
    except Exception as e:
        return {"error": f"ADQL query failed: {e}"}

    rows = _table_to_rows(table, max_rows)
    cols = list(rows[0].keys()) if rows else list(table.colnames)[:MAX_COLS_HARD]
    return {"n_rows": len(rows), "columns": cols, "rows": rows}


def _normalize_2mass(twomass_id: str) -> str:
    """Strip a leading '2M'/'J' prefix to get the bare 2MASS designation."""
    s = twomass_id.strip()
    for pre in ("2MASS", "2M", "J"):
        if s.upper().startswith(pre):
            s = s[len(pre):]
            break
    return s.strip()


@mcp.tool()
def crossmatch_2mass(twomass_id: str) -> dict:
    """
    Resolve a 2MASS designation to its Gaia DR3 source and return key astrometry.

    Accepts either "2M17335483-2753043" or the bare "17335483-2753043". Uses the
    Gaia DR3 precomputed neighbour table gaiadr3.tmass_psc_xsc_best_neighbour,
    matching the 2MASS designation via gaiadr3.tmass_psc_xsc_join ->
    gaiadr1.tmass_original_valid.designation, then joins gaiadr3.gaia_source for
    astrometry/photometry.

    NOTE: RUWE > ~1.4 is the classic flag for an unresolved (astrometric) binary —
    a poor single-star astrometric fit. Check the returned `ruwe`.

    Args:
        twomass_id: 2MASS designation, with or without a "2M"/"J" prefix.

    Returns:
        {source_id, ra, dec, parallax, parallax_error, ruwe, phot_g_mean_mag, bp_rp}
        or {"available": false, "twomass_id": ...} if no Gaia match, or {"error": "..."}.
    """
    desig = _normalize_2mass(twomass_id)
    if not desig:
        return {"error": "empty 2MASS designation"}

    # Local fallback used when TAP is unavailable: map 2MASS -> sdss_id (SB2
    # manifest) -> local Parquet row. Returns a dict or None (no local match).
    def _local_2mass():
        sdss_id = _load_2mass_map().get(desig)
        if sdss_id is None:
            return None
        local = _local_lookup_by_sdss_id(sdss_id)
        if isinstance(local, dict) and "error" not in local:
            local["twomass_id"] = twomass_id
            local["via"] = "local_parquet"
            return local
        return None

    Gaia, err = _get_gaia()
    if err:
        return _local_2mass() or {"error": err}

    # Join chain: 2MASS designation -> xjoin -> best_neighbour -> gaia_source.
    adql = f"""
    SELECT TOP 1
        gs.source_id, gs.ra, gs.dec, gs.parallax, gs.parallax_error,
        gs.ruwe, gs.phot_g_mean_mag, gs.bp_rp
    FROM gaiadr3.tmass_psc_xsc_best_neighbour AS bn
    JOIN gaiadr3.tmass_psc_xsc_join AS xj
        ON bn.clean_tmass_psc_xsc_oid = xj.clean_tmass_psc_xsc_oid
    JOIN gaiadr1.tmass_original_valid AS tm
        ON xj.original_psc_source_id = tm.designation
    JOIN gaiadr3.gaia_source AS gs
        ON bn.source_id = gs.source_id
    WHERE tm.designation = '{desig}'
    """
    try:
        job = Gaia.launch_job(adql)
        table = job.get_results()
    except Exception as e:
        # TAP unreachable -> try the local table before reporting the error.
        return _local_2mass() or {"error": f"crossmatch query failed: {e}"}

    if len(table) == 0:
        return {"available": False, "twomass_id": twomass_id}

    rows = _table_to_rows(table, 1)
    r = rows[0]
    out = {
        "source_id": r.get("source_id"),
        "ra": r.get("ra"),
        "dec": r.get("dec"),
        "parallax": r.get("parallax"),
        "parallax_error": r.get("parallax_error"),
        "ruwe": r.get("ruwe"),
        "phot_g_mean_mag": r.get("phot_g_mean_mag"),
        "bp_rp": r.get("bp_rp"),
    }
    if out.get("ruwe") is not None and out["ruwe"] > 1.4:
        out["ruwe_note"] = "RUWE > 1.4: possible unresolved binary (poor single-star astrometric fit)"
    return out


@mcp.tool()
def binarity_flags(source_id: int) -> dict:
    """
    Return Gaia DR3 binarity-relevant flags for one source.

    Pulls RUWE, astrometric excess noise (+ significance), and the
    non_single_star bitmask from gaiadr3.gaia_source. non_single_star != 0 means
    the source has a Gaia non-single-star (NSS) solution (astrometric / spectroscopic
    / eclipsing). RUWE > ~1.4 and large astrometric_excess_noise also flag binarity.

    Args:
        source_id: Gaia DR3 source_id (integer).

    Returns:
        {source_id, ruwe, astrometric_excess_noise, astrometric_excess_noise_sig,
         non_single_star} or {"available": false, ...} / {"error": "..."}.
    """
    try:
        sid = int(source_id)
    except Exception:
        return {"error": "source_id must be an integer"}

    Gaia, err = _get_gaia()
    if err:
        # astroquery.gaia missing -> try the local Parquet before giving up.
        local = _local_lookup_by_source_id(sid)
        if isinstance(local, dict) and "error" not in local:
            local["via"] = "local_parquet"
            return local
        return {"error": err}

    adql = f"""
    SELECT TOP 1 source_id, ruwe, astrometric_excess_noise,
           astrometric_excess_noise_sig, non_single_star
    FROM gaiadr3.gaia_source
    WHERE source_id = {sid}
    """
    try:
        job = Gaia.launch_job(adql)
        table = job.get_results()
    except Exception as e:
        # TAP unreachable (gea.esac.esa.int blocked) -> fall back to local table.
        local = _local_lookup_by_source_id(sid)
        if isinstance(local, dict) and "error" not in local:
            local["via"] = "local_parquet"
            return local
        return {"error": f"binarity query failed (and no local fallback): {e}"}

    if len(table) == 0:
        return {"available": False, "source_id": sid}

    r = _table_to_rows(table, 1)[0]
    out = {
        "source_id": r.get("source_id", sid),
        "ruwe": r.get("ruwe"),
        "astrometric_excess_noise": r.get("astrometric_excess_noise"),
        "astrometric_excess_noise_sig": r.get("astrometric_excess_noise_sig"),
        "non_single_star": r.get("non_single_star"),
    }
    flags = []
    if out.get("ruwe") is not None and out["ruwe"] > 1.4:
        flags.append("RUWE>1.4")
    if out.get("non_single_star"):
        flags.append("NSS solution present")
    if flags:
        out["binary_flags"] = flags
    return out


# --------------------------------------------------------------------------- #
# Local-lookup tools (no TAP, no network) -- read the pre-fetched Parquet.
# --------------------------------------------------------------------------- #
@mcp.tool()
def binarity_local(sdss_id: int) -> dict:
    """
    Return Gaia DR3 binarity fields for one DR19 dwarf from the LOCAL table.

    No network call: reads resources/gaia_dr19_dwarfs.parquet, pre-fetched via
    SDSS CAS + VizieR (see src/download_gaia_local.py). Use this when the Gaia
    TAP archive is unreachable, or for any sdss_id in the DR19 dwarf sample.

    RUWE > ~1.4 flags an unresolved (astrometric) binary -- a poor single-star
    astrometric fit. A non-zero non_single_star bitmask means Gaia fitted a
    non-single-star (astrometric / spectroscopic / eclipsing) solution.

    Args:
        sdss_id: SDSS-V sdss_id (integer) of the star.

    Returns:
        {sdss_id, source_id, ruwe, parallax, parallax_error, phot_g_mean_mag,
         bp_rp, non_single_star, astrometric_excess_noise,
         astrometric_excess_noise_sig, source, [ruwe_note], [binary_flags]}
        or {"available": false, "sdss_id": ...} if the id is not in the table,
        or {"error": "..."} (e.g. the Parquet file has not been built yet).
    """
    res = _local_lookup_by_sdss_id(sdss_id)
    if res is None:
        return {"available": False, "sdss_id": int(sdss_id)}
    return res


@mcp.tool()
def binarity_flags_local(source_id: int) -> dict:
    """
    Like binarity_flags, but read from the LOCAL Parquet by Gaia source_id (no TAP).

    Args:
        source_id: Gaia DR3 source_id (integer).

    Returns:
        Same shape as binarity_local, or {"available": false, "source_id": ...},
        or {"error": "..."}.
    """
    res = _local_lookup_by_source_id(source_id)
    if res is None:
        return {"available": False, "source_id": int(source_id)}
    return res


if __name__ == "__main__":
    mcp.run()
