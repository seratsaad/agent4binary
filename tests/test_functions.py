#!/usr/bin/env python3
"""
agent4binary -- end-to-end functional verification + Vertex cost instrumentation.

Re-runnable. READ/RUN ONLY: this script imports and exercises the project; it
never edits source files. It writes an ignored tests/functional_report.md.

What it checks (each -> PASS / FAIL + the error string on failure):

  A. src.physics imports.
  B. dr19_sc_single_vs_binary on SB2 61434605 (expect prefers_binary True,
     delta_chi2 ~1802, f_imp ~0.247 after the q=1 split-RV fix) and on a
     control (expect single).
  C. Every public physics function on real inputs: continuum_normalize,
     payne_predict5_sc / payne5_single_model_sc, compose_binary5_sc,
     secondary_from_q, fit_single5_sc, fit_binary5_sc, chi2, f_imp, the
     Table B1 helpers (table_b1_min_fimp, passes_table_b1).
  D. src.isochrone_mist.secondary_from_q_ours.
  E. MCP layer: src.graph.agent.load_tools() registers tools; one representative
     tool per server is called (apogee_data.load_dr19_spectrum,
     binary_model.chi2_single_vs_binary, doppler_shift, rotational_broadening,
     gaia_sql.binarity_local [offline], isochrone.secondary_labels,
     payne.predict_spectrum, gaia_xp.get_xp_spectrum, vision.inspect_spectrum_fit).
  F. The agent: build_agent on AGENT_BACKEND=studio AND vertex; one short analyze
     flow; confirm a tool was invoked and a verdict returned.
  G. Web backend: import web.backend.server; FastAPI TestClient over
     /api/resolve, /api/analyze (61434605 -> binary + DR3 ruwe ~7.47), /api/chat.
  H. Notebook: imports parse and every public function it calls exists.
  I. Vertex cost: ONE gemini-3.5-flash text call AND ONE image call, capture
     usage_metadata, then estimate total spend so far.

Run:  python tests/test_functions.py
Flags:
  --fast            skip the two ~27s detector fits (B/G analyze use the on-disk
                    cache instead; the SB2 fit is still verified once).
  --no-net          skip everything that touches Vertex / Gaia TAP / Gaia XP
                    (the offline checks still run).
  --skip-detector   alias for --fast.
"""
import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Expected canonical numbers for SB2 61434605 (from the production detector).
SB2_ID = 61434605
SB2_SEED = (6159.65, 4.2977, -0.2714, -0.3080, 17.4260)   # catalog labels (manifest)
EXP_DELTA_CHI2 = 1381.2   # re-baselined to the C-space detector + loss-fix net (2026-06-23)
EXP_FIMP = 0.143          # catalog SB2_SEED; clears FIMP_FLOOR=0.14 -> prefers binary
EXP_RUWE = 7.47
CONTROL_ID = 55279291   # summary.csv: class=control, prefers_binary=False

# Vertex / gemini-3.5-flash (mirrors the vision MCP tool + graph agent setup).
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "osu-prd-as-tingastroml-ad00")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
GEMINI_MODEL = "gemini-3.5-flash"


# --------------------------------------------------------------------------- #
# Tiny result recorder.
# --------------------------------------------------------------------------- #
RESULTS = []   # list of (component, status, note)


def record(component, status, note=""):
    RESULTS.append((component, status, str(note)))
    tag = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}.get(status, status)
    print(f"[{tag}] {component}" + (f"  -- {note}" if note else ""), flush=True)


def check(component, fn):
    """Run fn(); PASS if it returns (ok, note) with ok True, else FAIL."""
    try:
        ok, note = fn()
        record(component, "PASS" if ok else "FAIL", note)
        return ok
    except Exception as e:
        record(component, "FAIL", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return False


def approx(a, b, rel=0.02, abs_=1e-6):
    return abs(a - b) <= max(abs_, rel * abs(b))


# --------------------------------------------------------------------------- #
# A. physics import
# --------------------------------------------------------------------------- #
def section_physics_import():
    def _f():
        import src.physics as p
        globals()["P"] = p
        return True, f"src.physics imported (NPIX={p.NPIX}, WAVELENGTH[0]={p.WAVELENGTH[0]:.3f})"
    return check("A. import src.physics", _f)


# --------------------------------------------------------------------------- #
# B. detector on SB2 + control
# --------------------------------------------------------------------------- #
def _load_raw_npz(path):
    d = np.load(path)
    fk = "flux_raw" if "flux_raw" in d.files else "flux"
    return np.asarray(d[fk], float), np.asarray(d["ivar"], float)


def section_detector(fast):
    p = globals()["P"]

    # --- SB2 61434605 ---
    def _sb2():
        npz = ROOT / "data" / "cache" / f"{SB2_ID}.npz"
        if not npz.exists():
            npz = ROOT / "data" / "dr19_raw_sb2" / f"{SB2_ID}.npz"
        flux, ivar = _load_raw_npz(npz)
        r = p.dr19_sc_single_vs_binary(flux, ivar, seed=SB2_SEED)
        globals()["_SB2_RESULT"] = r
        ok = (r["prefers_binary"] is True
              and approx(r["delta_chi2"], EXP_DELTA_CHI2, rel=0.01)
              and approx(r["f_imp"], EXP_FIMP, rel=0.03))
        note = (f"prefers_binary={r['prefers_binary']} "
                f"delta_chi2={r['delta_chi2']:.1f} (exp ~{EXP_DELTA_CHI2}) "
                f"f_imp={r['f_imp']:.3f} (exp ~{EXP_FIMP}) "
                f"best_q={r['best_q']:.3f}")
        return ok, note
    check("B1. dr19_sc_single_vs_binary SB2 61434605 (prefers binary)", _sb2)

    # --- control (expect single) ---
    def _ctrl():
        if fast:
            return True, "SKIPPED via --fast (control fit is ~27s; SB2 verified above)"
        npz = ROOT / "data" / "dr19_raw_controls" / f"{CONTROL_ID}.npz"
        if not npz.exists():
            # any control on disk
            cand = sorted((ROOT / "data" / "dr19_raw_controls").glob("*.npz"))
            if not cand:
                return False, "no control npz on disk"
            npz = cand[0]
        flux, ivar = _load_raw_npz(npz)
        r = p.dr19_sc_single_vs_binary(flux, ivar)
        ok = (r["prefers_binary"] is False)
        return ok, (f"id={npz.stem} prefers_binary={r['prefers_binary']} "
                    f"delta_chi2={r['delta_chi2']:.1f} f_imp={r['f_imp']:.3f}")
    if fast:
        record("B2. dr19_sc_single_vs_binary control (expect single)", "SKIP",
               "skipped via --fast")
    else:
        check("B2. dr19_sc_single_vs_binary control (expect single)", _ctrl)


# --------------------------------------------------------------------------- #
# C. every public physics function on real inputs
# --------------------------------------------------------------------------- #
def section_physics_functions():
    p = globals()["P"]
    npz = ROOT / "data" / "cache" / f"{SB2_ID}.npz"
    if not npz.exists():
        npz = ROOT / "data" / "dr19_raw_sb2" / f"{SB2_ID}.npz"
    flux_raw, ivar = _load_raw_npz(npz)

    # continuum_normalize
    def _cn():
        nf, ne = p.continuum_normalize(flux_raw, ivar, return_error=True)
        ok = nf.shape == (p.NPIX,) and np.isfinite(np.nanmedian(nf)) and \
            0.5 < np.nanmedian(nf) < 1.5
        globals()["_OBS_PREP"] = (nf, ne)
        return ok, f"len={nf.size} median_norm_flux={np.nanmedian(nf):.4f}"
    check("C1. continuum_normalize", _cn)

    # payne_predict5_sc
    def _pp():
        f = p.payne_predict5_sc(5800.0, 4.4, -0.2, 0.0, 5.0)
        ok = f.shape == (p.NPIX,) and np.isfinite(f).all() and 0.3 < np.median(f) <= 1.05
        return ok, f"len={f.size} median={np.median(f):.4f} min={f.min():.3f}"
    check("C2. payne_predict5_sc", _pp)

    # payne5_single_model_sc
    def _psm():
        f = p.payne5_single_model_sc(5800.0, 4.4, -0.2, 0.0, 5.0)
        ok = f.shape == (p.NPIX,) and np.isfinite(np.nanmedian(f))
        return ok, f"len={f.size} median={np.nanmedian(f):.4f}"
    check("C3. payne5_single_model_sc", _psm)

    # compose_binary5_sc
    def _cb():
        f = p.compose_binary5_sc(6000.0, 4.3, -0.2, 0.0, 5.0, q=0.5,
                                 rv1_kms=10.0, rv2_kms=-30.0)
        # q=1 identity check
        f1 = p.compose_binary5_sc(6000.0, 4.3, -0.2, 0.0, 5.0, q=1.0,
                                  rv1_kms=0.0, rv2_kms=0.0)
        single = p.payne5_single_model_sc(6000.0, 4.3, -0.2, 0.0, 5.0)
        ident = float(np.nanmax(np.abs(f1 - single)))
        split = p.compose_binary5_sc(6000.0, 4.3, -0.2, 0.0, 5.0, q=1.0,
                                     rv1_kms=-20.0, rv2_kms=20.0)
        one_rv = p.compose_binary5_sc(6000.0, 4.3, -0.2, 0.0, 5.0, q=1.0,
                                      rv1_kms=-20.0, rv2_kms=-20.0)
        split_diff = float(np.nanmax(np.abs(split - one_rv)))
        ok = (f.shape == (p.NPIX,) and np.isfinite(np.nanmedian(f))
              and ident < 1e-6 and split_diff > 1e-4)
        return ok, (f"len={f.size} median={np.nanmedian(f):.4f} "
                    f"q=1-identity_maxabs={ident:.2e} "
                    f"q=1-split_diff={split_diff:.2e}")
    check("C4. compose_binary5_sc (+ q=1 identity)", _cb)

    # secondary_from_q
    def _sq():
        t2, g2, R2, R1 = p.secondary_from_q(6000.0, 4.3, -0.2, 0.5)
        ok = (3000 < t2 < 6000) and (R2 < R1) and np.isfinite([t2, g2, R2, R1]).all()
        return ok, f"teff2={t2:.1f} logg2={g2:.3f} R2={R2:.3f} R1={R1:.3f}"
    check("C5. secondary_from_q", _sq)

    # build a clean obs/err for the fits (same prep the detector uses)
    nf, ne = globals().get("_OBS_PREP", p.continuum_normalize(flux_raw, ivar, return_error=True))
    nf = np.where(np.isfinite(nf), nf, 1.0)
    ne_f = np.where(np.isfinite(ne), ne, 1e6)
    obs, obs_err = p.normalize_like_model(nf, ne_f)
    obs_err = p.mask_bad_pixels(obs, obs_err)
    obs_err = np.where(np.isfinite(obs_err), obs_err, 1e6)
    globals()["_OBS"] = obs
    globals()["_OBS_ERR"] = obs_err

    # fit_single5_sc
    def _fs():
        labels, model, c2 = p.fit_single5_sc(obs, obs_err, *SB2_SEED)
        globals()["_FIT_S"] = (labels, model, c2)
        # params6 = [Teff, logg, feh, mgh, vmacro, dv] (the decoupled single dv).
        ok = len(labels) == 6 and model.shape == (p.NPIX,) and np.isfinite(c2)
        return ok, f"labels={np.round(labels,2).tolist()} chi2={c2:.1f}"
    check("C6. fit_single5_sc", _fs)

    # fit_binary5_sc
    def _fb():
        labels_s = globals()["_FIT_S"][0]
        params, model, c2 = p.fit_binary5_sc(obs, obs_err, labels_s[0], labels_s[1],
                                             labels_s[2], labels_s[3], labels_s[4])
        globals()["_FIT_B"] = (params, model, c2)
        ok = model.shape == (p.NPIX,) and np.isfinite(c2) and "q" in params
        return ok, f"q={params['q']:.3f} rv1={params['rv1']:.1f} rv2={params['rv2']:.1f} chi2={c2:.1f}"
    check("C7. fit_binary5_sc", _fb)

    # chi2
    def _c2():
        model = globals()["_FIT_S"][1]
        val = p.chi2(obs, obs_err, model)
        # identity: chi2(obs, err, obs) over good pixels == 0
        zero = p.chi2(obs, obs_err, obs)
        ok = np.isfinite(val) and val > 0 and abs(zero) < 1e-9
        return ok, f"chi2(single)={val:.1f} chi2(obs,obs)={zero:.2e}"
    check("C8. chi2", _c2)

    # f_imp
    def _fi():
        ms = globals()["_FIT_S"][1]
        mb = globals()["_FIT_B"][1]
        val = p.f_imp(obs, ms, mb, obs_err)
        same = p.f_imp(obs, ms, ms, obs_err)   # identical models -> 0
        ok = np.isfinite(val) and abs(same) < 1e-12
        return ok, f"f_imp(single,binary)={val:.3f} f_imp(same)={same:.2e}"
    check("C9. f_imp", _fi)

    # Table B1 helpers
    def _tb():
        m_lo = p.table_b1_min_fimp(100.0)        # below 300 -> None
        m_mid = p.table_b1_min_fimp(1121.5)      # the SB2 bin
        pass_hi = p.passes_table_b1(1121.5, 0.188)   # should pass
        fail_lo = p.passes_table_b1(100.0, 0.5)      # below 300 -> fail
        ok = (m_lo is None and m_mid is not None and pass_hi is True
              and fail_lo is False)
        return ok, (f"min_fimp(100)={m_lo} min_fimp(1121.5)={m_mid} "
                    f"passes(1121.5,0.188)={pass_hi} passes(100,0.5)={fail_lo}")
    check("C10. table_b1_min_fimp / passes_table_b1", _tb)


# --------------------------------------------------------------------------- #
# D. isochrone_mist.secondary_from_q_ours
# --------------------------------------------------------------------------- #
def section_isochrone_mist():
    def _f():
        import src.isochrone_mist as iso
        t2, g2, R2, R1 = iso.secondary_from_q_ours(6000.0, 4.3, -0.2, 0.5)
        ok = (3000 < t2 < 6000) and (R2 < R1) and np.isfinite([t2, g2, R2, R1]).all()
        # q=1 identity: secondary equals primary, R2 == R1
        t2e, g2e, R2e, R1e = iso.secondary_from_q_ours(6000.0, 4.3, -0.2, 1.0)
        ident = abs(R2e - R1e) < 1e-6 and abs(t2e - 6000.0) < 50.0
        return ok and ident, (f"q=0.5 -> teff2={t2:.1f} R2={R2:.3f} R1={R1:.3f}; "
                              f"q=1 identity ok={ident}")
    check("D. isochrone_mist.secondary_from_q_ours", _f)


# --------------------------------------------------------------------------- #
# E + F. MCP layer + agent (async)
# --------------------------------------------------------------------------- #
def section_mcp_and_agent(no_net, fast):
    import asyncio
    asyncio.run(_mcp_and_agent_async(no_net, fast))


def _tool_map(tools):
    return {t.name: t for t in tools}


def _coerce_tool_output(out):
    """langchain-mcp tools return a str OR a list of content blocks
    ([{'type':'text','text': '<json>'}]). Pull the text and parse JSON if possible.
    """
    text = None
    if isinstance(out, str):
        text = out
    elif isinstance(out, list):
        parts = []
        for b in out:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
        text = "".join(parts)
    elif isinstance(out, dict):
        return out
    if text is None:
        return out
    try:
        return json.loads(text)
    except Exception:
        return text


async def _call(tool, args):
    """Invoke a langchain MCP tool and parse its (usually JSON) text result."""
    out = await tool.ainvoke(args)
    return _coerce_tool_output(out)


async def _mcp_and_agent_async(no_net, fast):
    # --- E0. load_tools registers tools across all servers ---
    tools = None
    try:
        from src.graph.agent import load_tools, make_client, SERVER_NAMES
        t0 = time.time()
        tools = await load_tools()
        names = sorted(t.name for t in tools)
        record("E0. load_tools() registers MCP tools", "PASS",
               f"{len(tools)} tools in {time.time()-t0:.1f}s: {names}")
    except Exception as e:
        record("E0. load_tools() registers MCP tools", "FAIL",
               f"{type(e).__name__}: {e}")
        return

    tm = _tool_map(tools)

    async def _checktool(component, name, args, validate, netonly=False):
        if netonly and no_net:
            record(component, "SKIP", "skipped via --no-net")
            return
        if name not in tm:
            record(component, "FAIL", f"tool '{name}' not registered")
            return
        try:
            res = await _call(tm[name], args)
            ok, note = validate(res)
            record(component, "PASS" if ok else "FAIL", note)
        except Exception as e:
            record(component, "FAIL", f"{type(e).__name__}: {e}")

    # --- E1. apogee_data.load_dr19_spectrum (offline; caches raw arrays) ---
    await _checktool(
        "E1. apogee_data.load_dr19_spectrum",
        "load_dr19_spectrum", {"sdss_id": SB2_ID},
        lambda r: (isinstance(r, dict) and r.get("npix") == 8575 and "error" not in r,
                   f"npix={r.get('npix')} median_snr={r.get('median_snr')} "
                   f"n_good={r.get('n_good')} src={r.get('source')}"))

    # --- E2. binary_model.chi2_single_vs_binary (offline; the detector over MCP) ---
    # Pass the catalog seed (teff1/logg1/feh/mgh/vmacro) so the MCP path reproduces
    # the SAME single-fit starting point as the canonical physics call in B1; then
    # the numbers match exactly. (With no seed the tool uses a solar-dwarf default
    # seed -> a slightly different but still-correct binary verdict, delta_chi2~1164.)
    if fast:
        record("E2. binary_model.chi2_single_vs_binary", "SKIP",
               "skipped via --fast (~27s; physics path verified in B1)")
    else:
        await _checktool(
            "E2. binary_model.chi2_single_vs_binary",
            "chi2_single_vs_binary",
            {"spec_id": str(SB2_ID), "model": "dr19_sc",
             "teff1": SB2_SEED[0], "logg1": SB2_SEED[1], "feh": SB2_SEED[2],
             "mgh": SB2_SEED[3], "vmacro": SB2_SEED[4]},
            lambda r: (isinstance(r, dict) and r.get("prefers_binary") is True
                       and approx(float(r.get("delta_chi2", 0)), EXP_DELTA_CHI2, rel=0.02),
                       f"prefers_binary={r.get('prefers_binary')} "
                       f"delta_chi2={r.get('delta_chi2')} f_imp={r.get('f_imp')} "
                       f"(seeded with catalog labels to match B1)"))

    # --- E3. doppler_shift (offline) ---
    await _checktool(
        "E3. doppler_server.doppler_shift",
        "doppler_shift", {"rest_wavelength_angstroms": 16000.0, "velocity_kms": 30.0},
        lambda r: (isinstance(r, dict) and r.get("observed_wavelength_angstroms", 0) > 16000.0
                   and r.get("direction") == "redshift",
                   f"observed={r.get('observed_wavelength_angstroms')} "
                   f"shift={r.get('shift_angstroms')} dir={r.get('direction')}"))

    # --- E4. broadening.rotational_broadening (offline) ---
    await _checktool(
        "E4. broadening_server.rotational_broadening",
        "rotational_broadening", {"wavelength_angstroms": 16000.0, "vsini_kms": 30.0},
        lambda r: (isinstance(r, dict) and r.get("fwhm_angstroms", 0) > 0,
                   f"fwhm={r.get('fwhm_angstroms')} class={r.get('rotation_class')}"))

    # --- E5. gaia_sql.binarity_local (OFFLINE DR3 from local Parquet) ---
    await _checktool(
        "E5. gaia_sql.binarity_local (offline DR3)",
        "binarity_local", {"sdss_id": SB2_ID},
        lambda r: (isinstance(r, dict) and approx(float(r.get("ruwe", 0)), EXP_RUWE, rel=0.02),
                   f"ruwe={r.get('ruwe')} (exp ~{EXP_RUWE}) "
                   f"non_single_star={r.get('non_single_star')} src={r.get('source')}"))

    # --- E6. isochrone.secondary_labels (offline) ---
    await _checktool(
        "E6. isochrone.secondary_labels",
        "secondary_labels", {"teff1": 6000.0, "logg1": 4.3, "feh": -0.2, "q": 0.5},
        lambda r: (isinstance(r, dict) and "teff2" in r and r["teff2"] < 6000.0,
                   f"teff2={r.get('teff2')} logg2={r.get('logg2')} "
                   f"m1={r.get('mass1_msun')} m2={r.get('mass2_msun')}"))

    # --- E7. payne.predict_spectrum (offline; caches flux) ---
    await _checktool(
        "E7. payne.predict_spectrum",
        "predict_spectrum", {"teff": 5800.0, "logg": 4.4, "feh": -0.2},
        lambda r: (isinstance(r, dict) and r.get("npix") == 8575,
                   f"npix={r.get('npix')} in_range={r.get('label_in_range')} "
                   f"cache={os.path.basename(str(r.get('cache_path')))}"))

    # --- E8. gaia_xp.get_xp_spectrum (NETWORK: Gaia datalink; may be blocked) ---
    await _checktool(
        "E8. gaia_xp.get_xp_spectrum (network)",
        "get_xp_spectrum",
        {"source_id": 1458891268218971776},   # 61434605's Gaia source_id
        lambda r: (isinstance(r, dict) and ("error" not in r or "available" in r),
                   f"keys={list(r.keys())[:6]} "
                   f"{'available=' + str(r.get('available')) if 'available' in r else ('error=' + str(r.get('error'))[:80])}"),
        netonly=True)

    # --- E9. vision.inspect_spectrum_fit (NETWORK: gemini multimodal) ---
    # The vision server is NOT in graph.agent.SERVER_NAMES, so it is exercised
    # directly here. It needs a data/cache/<id>.npz with keys flux/error (the form
    # apogee_data.load_spectrum writes); we synthesize one from the SB2 raw
    # spectrum (writing a DATA cache file, not editing any source).
    if no_net:
        record("E9. vision.inspect_spectrum_fit (gemini multimodal)", "SKIP",
               "skipped via --no-net")
    else:
        try:
            import importlib.util as _u
            spec = _u.spec_from_file_location(
                "vision_server", str(ROOT / "src" / "mcp_servers" / "vision_server.py"))
            vs = _u.module_from_spec(spec)
            spec.loader.exec_module(vs)
            # Build a flux/error cache entry the vision server can load.
            raw = np.load(ROOT / "data" / "cache" / f"{SB2_ID}.npz")
            flux_raw = np.asarray(raw["flux_raw"], float)
            ivar = np.asarray(raw["ivar"], float)
            p = globals()["P"]
            nf, ne = p.continuum_normalize(flux_raw, ivar, return_error=True)
            nf = np.where(np.isfinite(nf), nf, 1.0)
            ne = np.where(np.isfinite(ne), ne, 1e6)
            cid = f"VISIONQA_{SB2_ID}"
            np.savez_compressed(ROOT / "data" / "cache" / f"{cid}.npz",
                                flux=nf.astype(np.float32), error=ne.astype(np.float32))
            fn = getattr(vs.inspect_spectrum_fit, "fn", vs.inspect_spectrum_fit)
            res = fn(cid)
            ok = (isinstance(res, dict) and "assessment" in res and "fit" in res
                  and isinstance(res["assessment"], dict))
            if ok:
                note = (f"fit={res['fit']} assessment_keys={list(res['assessment'].keys())} "
                        f"looks_like_sb2={res['assessment'].get('looks_like_sb2')}")
            else:
                note = f"error={res.get('error') if isinstance(res, dict) else res}"
            record("E9. vision.inspect_spectrum_fit (gemini multimodal)",
                   "PASS" if ok else "FAIL", note)
        except Exception as e:
            record("E9. vision.inspect_spectrum_fit (gemini multimodal)", "FAIL",
                   f"{type(e).__name__}: {e}")

    # ----------------------------------------------------------------------- #
    # F. the agent (build_agent on studio AND vertex; one short analyze flow).
    # ----------------------------------------------------------------------- #
    if no_net:
        record("F1. build_agent backend=studio (analyze flow)", "SKIP", "skipped via --no-net")
        record("F2. build_agent backend=vertex (analyze flow)", "SKIP", "skipped via --no-net")
        return

    for backend in ("studio", "vertex"):
        comp = f"F{'1' if backend=='studio' else '2'}. build_agent backend={backend} (analyze flow)"
        try:
            from src.graph.agent import build_agent
            os.environ["AGENT_BACKEND"] = backend
            agent = await build_agent()
            # Short, tool-forcing prompt: offline binarity check (no Gaia TAP needed).
            prompt = (
                f"For DR19 star sdss_id {SB2_ID}: call gaia_sql binarity_local to get its "
                f"Gaia DR3 RUWE, then state in one sentence whether RUWE>1.4 indicates a "
                f"binary. Use the tool; do not guess.")
            t0 = time.time()
            out = await agent.ainvoke({"messages": [("human", prompt)]})
            msgs = out.get("messages", [])
            # Detect a tool invocation in the message trace.
            tool_called = False
            tool_names = []
            for m in msgs:
                tc = getattr(m, "tool_calls", None) or []
                for c in tc:
                    tool_called = True
                    tool_names.append(c.get("name") if isinstance(c, dict) else getattr(c, "name", "?"))
                if getattr(m, "type", "") == "tool" or m.__class__.__name__ == "ToolMessage":
                    tool_called = True
            # Final reply text.
            reply = ""
            for m in reversed(msgs):
                c = getattr(m, "content", None)
                if isinstance(c, list):
                    c = " ".join(part.get("text", "") for part in c if isinstance(part, dict))
                if c:
                    reply = c
                    break
            ok = tool_called and len(reply) > 0
            note = (f"tool_called={tool_called} tools={tool_names} "
                    f"reply[:90]={reply[:90]!r} ({time.time()-t0:.1f}s)")
            record(comp, "PASS" if ok else "FAIL", note)
        except Exception as e:
            record(comp, "FAIL", f"{type(e).__name__}: {e}")
    os.environ["AGENT_BACKEND"] = "studio"


# --------------------------------------------------------------------------- #
# G. web backend via FastAPI TestClient
# --------------------------------------------------------------------------- #
def section_web(no_net, fast):
    if not (ROOT / "web" / "backend" / "server.py").exists():
        record("G. web backend", "SKIP", "web/backend/server.py not present in this checkout")
        return
    try:
        from fastapi.testclient import TestClient
        # Force offline-friendly defaults; chat will try the agent (network).
        os.environ.setdefault("AGENT_BACKEND", "vertex")
        import web.backend.server as server
        client = TestClient(server.app)
    except Exception as e:
        record("G. web.backend.server import / TestClient", "FAIL", f"{type(e).__name__}: {e}")
        return
    record("G0. import web.backend.server + TestClient", "PASS", "app constructed")

    # /api/resolve
    def _resolve():
        r = client.post("/api/resolve", json={"query": str(SB2_ID)})
        j = r.json()
        ok = r.status_code == 200 and (str(j.get("sdss_id")) == str(SB2_ID)
                                       or str(j.get("id")) == str(SB2_ID))
        return ok, f"status={r.status_code} keys={list(j.keys())[:6]} sdss_id={j.get('sdss_id')}"
    check("G1. POST /api/resolve", _resolve)

    # /api/analyze (uses on-disk web cache if present; else runs the detector)
    def _analyze():
        cache = ROOT / "data" / "cache" / "web" / f"{SB2_ID}.json"
        if fast and not cache.exists():
            return True, "SKIPPED via --fast (no web cache; detector is ~27s)"
        r = client.post("/api/analyze", json={"target_id": str(SB2_ID)})
        j = r.json()
        ruwe = (j.get("gaia") or {}).get("ruwe")
        ok = (r.status_code == 200 and j.get("verdict") == "binary"
              and j.get("prefers_binary") is True
              and ruwe is not None and approx(float(ruwe), EXP_RUWE, rel=0.02))
        return ok, (f"status={r.status_code} verdict={j.get('verdict')} "
                    f"delta_chi2={j.get('delta_chi2')} f_imp={j.get('f_imp')} "
                    f"gaia.ruwe={ruwe} (exp ~{EXP_RUWE})")
    if fast and not (ROOT / "data" / "cache" / "web" / f"{SB2_ID}.json").exists():
        record("G2. POST /api/analyze (61434605 -> binary + ruwe ~7.47)", "SKIP",
               "skipped via --fast (no web cache)")
    else:
        check("G2. POST /api/analyze (61434605 -> binary + ruwe ~7.47)", _analyze)

    # /health
    def _health():
        r = client.get("/health")
        j = r.json()
        return r.status_code == 200 and j.get("ok") is True, f"status={r.status_code} {j}"
    check("G3. GET /health", _health)

    # /api/chat (network: builds agent). The endpoint never 500s by contract;
    # a clear 'Agent unavailable: ...' reply still proves the route works.
    def _chat():
        if no_net:
            return True, "SKIPPED via --no-net (would build the live agent)"
        r = client.post("/api/chat", json={"message": "Say the single word READY.",
                                           "history": []})
        j = r.json()
        ok = r.status_code == 200 and isinstance(j.get("reply"), str) and len(j["reply"]) > 0
        return ok, f"status={r.status_code} reply[:80]={j.get('reply','')[:80]!r}"
    if no_net:
        record("G4. POST /api/chat", "SKIP", "skipped via --no-net")
    else:
        check("G4. POST /api/chat (agent over MCP)", _chat)


# --------------------------------------------------------------------------- #
# H. notebook imports + public functions it calls
# --------------------------------------------------------------------------- #
def section_notebook():
    import re
    p = globals()["P"]
    nb_path = ROOT / "notebooks" / "agent4binary_chapter.ipynb"
    if not nb_path.exists():
        record("H. notebook imports + physics symbols exist", "SKIP",
               "notebooks/agent4binary_chapter.ipynb not present in this checkout")
        return

    def _f():
        nb = json.loads(nb_path.read_text())
        imports, calls = set(), set()
        for c in nb.get("cells", []):
            if c.get("cell_type") != "code":
                continue
            src = "".join(c.get("source", []))
            for ln in src.splitlines():
                s = ln.strip()
                if s.startswith("import ") or s.startswith("from "):
                    imports.add(s)
            for m in re.findall(r"\bphysics\.([A-Za-z_][A-Za-z0-9_]*)", src):
                calls.add(m)
        # Drop the filename false positive: the word "physics.py" in prose/strings
        # matches the pattern but is not a module attribute.
        calls.discard("py")

        # The two project imports the notebook relies on must resolve.
        import importlib
        importlib.import_module("src.physics")
        from src.graph.agent import load_tools, build_agent  # noqa: F401

        # Every physics.<name> the notebook calls must exist on the module.
        missing = sorted(n for n in calls if not hasattr(p, n))
        ok = not missing
        note = (f"code cells={sum(1 for c in nb['cells'] if c['cell_type']=='code')}, "
                f"physics symbols used={len(calls)}, "
                f"missing={missing if missing else 'none'}")
        return ok, note
    check("H. notebook imports + physics symbols exist", _f)


# --------------------------------------------------------------------------- #
# I. Vertex cost: instrument one text + one image gemini-3.5-flash call.
# --------------------------------------------------------------------------- #
# Pricing assumptions (USD per 1M tokens) -- gemini-3.5-flash on Vertex AI.
# These are stated in the report; edit here if the rate sheet changes.
PRICE_IN_PER_M = 0.30     # input (text + image) tokens
PRICE_OUT_PER_M = 2.50    # output tokens (candidates + thinking)


def section_vertex_cost(no_net):
    if no_net:
        record("I. Vertex gemini-3.5-flash cost instrumentation", "SKIP", "skipped via --no-net")
        return None
    usage = {}

    def _text():
        from google import genai
        client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        resp = client.models.generate_content(
            model=GEMINI_MODEL, contents=["Reply with the single word: OK"])
        um = resp.usage_metadata
        usage["text"] = dict(
            prompt=int(um.prompt_token_count or 0),
            candidates=int(um.candidates_token_count or 0),
            thoughts=int(getattr(um, "thoughts_token_count", 0) or 0),
            total=int(um.total_token_count or 0))
        ok = usage["text"]["prompt"] > 0 and usage["text"]["total"] > 0
        return ok, f"text usage_metadata={usage['text']} reply={resp.text!r}"
    text_ok = check("I1. Vertex text call usage_metadata", _text)

    def _image():
        from google import genai
        from google.genai import types
        client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        # Use a real cached QA plot when one is present.
        pngs = sorted((ROOT / "data" / "cache" / "qa").glob("*.png"))
        if not pngs:
            return False, "no QA PNG on disk to send"
        img = pngs[0].read_bytes()
        prompt = ("Return ONLY a JSON object with keys looks_like_sb2 (bool), "
                  "rv_split_visible (bool), unmasked_artifacts (bool), "
                  "fit_quality (string), agrees_with_metric (bool), note (string).")
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Part.from_bytes(data=img, mime_type="image/png"), prompt],
            config=types.GenerateContentConfig(temperature=0.0,
                                               response_mime_type="application/json"))
        um = resp.usage_metadata
        img_tok = 0
        for d in (um.prompt_tokens_details or []):
            if getattr(d.modality, "value", "") == "IMAGE":
                img_tok = int(d.token_count or 0)
        usage["image"] = dict(
            prompt=int(um.prompt_token_count or 0),
            image_tokens=img_tok,
            candidates=int(um.candidates_token_count or 0),
            thoughts=int(getattr(um, "thoughts_token_count", 0) or 0),
            total=int(um.total_token_count or 0),
            png=pngs[0].name, png_bytes=len(img))
        ok = usage["image"]["prompt"] > 0 and usage["image"]["total"] > 0
        return ok, f"image usage_metadata={usage['image']}"
    image_ok = check("I2. Vertex image+text call usage_metadata", _image)

    globals()["_USAGE"] = usage
    return usage


# --------------------------------------------------------------------------- #
# Cost estimate (from instrumented per-call usage + known call counts).
# --------------------------------------------------------------------------- #
def estimate_cost(usage):
    """Return a dict of the spend estimate, or None if usage missing."""
    if not usage or "image" not in usage or "text" not in usage:
        return None
    img = usage["image"]
    txt = usage["text"]

    # Per-call billed tokens. Output billed = candidates + thoughts (thinking).
    img_in = img["prompt"]
    img_out = img["candidates"] + img["thoughts"]
    txt_in = txt["prompt"]
    txt_out = txt["candidates"] + txt["thoughts"]

    def cost(n_in, n_out):
        return (n_in * PRICE_IN_PER_M + n_out * PRICE_OUT_PER_M) / 1e6

    # Known call counts so far (from the task + repo evidence):
    #   - Visual QA: 60 image+text calls (data/cache/qa/summary.csv = 60 rows).
    #   - Continuum QA: a handful of image calls (~10) -- same image-call profile.
    #   - Agent end-to-end runs: a handful of short text/tool turns (~25 turns),
    #     text-call profile (no image). Each ReAct turn is a text generation.
    N_QA_IMAGE = 60
    N_CONT_IMAGE = 10
    N_AGENT_TEXT = 25

    qa_cost = N_QA_IMAGE * cost(img_in, img_out)
    cont_cost = N_CONT_IMAGE * cost(img_in, img_out)
    agent_cost = N_AGENT_TEXT * cost(txt_in, txt_out)
    total = qa_cost + cont_cost + agent_cost

    return dict(
        img_in=img_in, img_out=img_out, txt_in=txt_in, txt_out=txt_out,
        image_tokens=img["image_tokens"],
        per_image_call=cost(img_in, img_out),
        per_text_call=cost(txt_in, txt_out),
        N_QA_IMAGE=N_QA_IMAGE, N_CONT_IMAGE=N_CONT_IMAGE, N_AGENT_TEXT=N_AGENT_TEXT,
        qa_cost=qa_cost, cont_cost=cont_cost, agent_cost=agent_cost, total=total,
        price_in=PRICE_IN_PER_M, price_out=PRICE_OUT_PER_M)


# --------------------------------------------------------------------------- #
# Report writer.
# --------------------------------------------------------------------------- #
def write_report(cost):
    out = ROOT / "tests" / "functional_report.md"
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    n_skip = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    lines = []
    lines.append("# agent4binary -- functional verification report")
    lines.append("")
    lines.append(f"Generated by `tests/test_functions.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}.")
    lines.append("")
    lines.append(f"Summary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP "
                 f"(of {len(RESULTS)} checks).")
    lines.append("")
    lines.append("## Component status")
    lines.append("")
    lines.append("| Component | Status | Note |")
    lines.append("|---|---|---|")
    for comp, status, note in RESULTS:
        note1 = note.splitlines()[0] if note else ""
        note1 = note1.replace("|", "\\|")[:300]
        lines.append(f"| {comp} | {status} | {note1} |")
    lines.append("")

    # Cost section.
    lines.append("## Vertex / gemini-3.5-flash spend estimate")
    lines.append("")
    if cost is None:
        lines.append("Not computed (the Vertex instrumentation was skipped, e.g. `--no-net`).")
    else:
        lines.append("Pricing assumed (Vertex AI gemini-3.5-flash, USD per 1M tokens):")
        lines.append("")
        lines.append(f"- input (text + image): **${cost['price_in']:.2f} / 1M**")
        lines.append(f"- output (candidates + thinking): **${cost['price_out']:.2f} / 1M**")
        lines.append("- image tokenization: a PNG QA plot tokenizes to "
                     f"**{cost['image_tokens']} input tokens** (measured via usage_metadata).")
        lines.append("")
        lines.append("Per-call billed tokens (measured live):")
        lines.append("")
        lines.append("| Call type | input tok | output tok (cand+think) | USD/call |")
        lines.append("|---|---|---|---|")
        lines.append(f"| image+text (QA) | {cost['img_in']} | {cost['img_out']} | "
                     f"${cost['per_image_call']:.6f} |")
        lines.append(f"| text (agent turn) | {cost['txt_in']} | {cost['txt_out']} | "
                     f"${cost['per_text_call']:.6f} |")
        lines.append("")
        lines.append("Known call counts so far and their cost:")
        lines.append("")
        lines.append("| Workload | calls | profile | USD |")
        lines.append("|---|---|---|---|")
        lines.append(f"| Visual QA (summary.csv) | {cost['N_QA_IMAGE']} | image+text | "
                     f"${cost['qa_cost']:.4f} |")
        lines.append(f"| Continuum QA | {cost['N_CONT_IMAGE']} | image+text | "
                     f"${cost['cont_cost']:.4f} |")
        lines.append(f"| Agent end-to-end turns | {cost['N_AGENT_TEXT']} | text | "
                     f"${cost['agent_cost']:.4f} |")
        lines.append(f"| **TOTAL** | | | **${cost['total']:.4f}** |")
        lines.append("")
        lines.append(f"**Estimated total Vertex spend so far: ~${cost['total']:.4f} USD "
                     f"(about {cost['total']*100:.0f} US cents).**")
        lines.append("")
        lines.append("Assumptions / caveats:")
        lines.append("")
        lines.append("- Output billing includes gemini-3.5-flash *thinking* tokens "
                     "(`thoughts_token_count`), which dominate output on the JSON QA "
                     "calls; they are counted at the output rate.")
        lines.append("- Each QA image call sends one ~200 KB PNG (the three-window fit "
                     "plot) -> ~1078 image input tokens plus ~60 text-prompt tokens.")
        lines.append(f"- The total is dominated by the {cost['N_QA_IMAGE']}+"
                     f"{cost['N_CONT_IMAGE']} = {cost['N_QA_IMAGE']+cost['N_CONT_IMAGE']} "
                     f"image calls at ~${cost['per_image_call']:.5f} each "
                     f"(~${(cost['N_QA_IMAGE']+cost['N_CONT_IMAGE'])*cost['per_image_call']:.3f}); "
                     f"the {cost['N_AGENT_TEXT']} text agent turns add only "
                     f"~${cost['agent_cost']:.4f}. So the figure is driven almost entirely "
                     f"by the QA image-call count, which is firmly known "
                     f"({cost['N_QA_IMAGE']} from summary.csv).")
        lines.append("- Continuum-QA call count and agent-turn count are "
                     "order-of-magnitude estimates; even doubling both changes the total "
                     "by only a few US cents, because the text turns are negligible and "
                     "only the image-call count matters.")
        lines.append("- Rates are the assumed Vertex list prices above; substitute the "
                     "current rate sheet by editing PRICE_IN_PER_M / PRICE_OUT_PER_M.")
    lines.append("")
    out.write_text("\n".join(lines))
    print(f"\nWrote {out}")
    return out


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true",
                    help="skip the ~27s detector fits (use on-disk caches)")
    ap.add_argument("--skip-detector", action="store_true", help="alias for --fast")
    ap.add_argument("--no-net", action="store_true",
                    help="skip Vertex / Gaia TAP / XP / agent network calls")
    args = ap.parse_args()
    fast = args.fast or args.skip_detector
    no_net = args.no_net

    print("=" * 70)
    print(f"agent4binary functional verification (fast={fast}, no_net={no_net})")
    print("=" * 70)

    if not section_physics_import():
        print("physics import failed; aborting.")
        cost = None
        write_report(cost)
        sys.exit(1)

    section_detector(fast)
    section_physics_functions()
    section_isochrone_mist()
    section_notebook()
    section_web(no_net, fast)
    # MCP + agent last among the heavy ones (spawns subprocesses).
    section_mcp_and_agent(no_net, fast)
    usage = section_vertex_cost(no_net)
    cost = estimate_cost(usage) if usage else None

    write_report(cost)

    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    n_skip = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")
    if n_fail:
        print("FAILURES:")
        for comp, status, note in RESULTS:
            if status == "FAIL":
                print(f"  - {comp}: {note.splitlines()[0] if note else ''}")
    if cost:
        print(f"VERTEX SPEND ESTIMATE: ~${cost['total']:.4f} USD")
    print("=" * 70)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
