#!/usr/bin/env python3
"""
MCP server: visual inspection of spectral fits with a multimodal LLM.

Motivation. A binary-detection pipeline produces a number (Delta-chi2, f_imp) and
a verdict, but a human astronomer also LOOKS at the fit: are the two components'
lines actually traced, are there unmasked sky spikes, is the radial-velocity split
visible, does the "binary" improvement sit on real features or on noise? This
server gives the agent that same ability. It renders the observed spectrum against
the best-fit single-star and binary models, then asks gemini-3.5-flash (a
multimodal model) to reason over the IMAGE and return a structured judgment. The
agent can use that judgment to accept a detection, to flag masking problems, or to
re-fit. This is the "visual reasoning as a tool" idea: the model sees its own work.

Design notes.
  - The forward model and fits come from src/physics.py (the El-Badry et al. 2018b
    / binspec path). We reproduce the same masking the detector uses so the plot
    the model sees matches what the chi^2 was computed on.
  - Heavy data never leaves through the return value: the rendered PNG is written
    under data/cache/ and only its path plus a COMPACT structured assessment is
    returned.
  - The LLM call degrades gracefully: any API / parse error returns {"error": ...}
    rather than raising, so the agent loop keeps running.
"""
import os
import io
import json
import base64

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt
from mcp.server.fastmcp import FastMCP

# Project root (this file is src/mcp_servers/), so two levels up.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
import sys
sys.path.insert(0, _ROOT)
from src import physics as P  # noqa: E402

CACHE = os.path.join(_ROOT, "data", "cache")
os.makedirs(CACHE, exist_ok=True)

mcp = FastMCP("Vision", log_level="WARNING")

# gemini-3.5-flash: the multimodal model that does the visual reasoning. The key
# lives in ~/.env (GOOGLE_API_KEY); we read it lazily so the module imports even
# without it (the tool then returns an error instead of crashing the server).
VISION_MODEL = "gemini-3.5-flash"


def _load_env_key():
    """Read GOOGLE_API_KEY from the environment or ~/.env (KEY=VALUE lines)."""
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    env_path = os.path.expanduser("~/.env")
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("GOOGLE_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def _load_cached(spec_id):
    """Load wl/flux/error for a cached spectrum; raise if not yet loaded."""
    path = os.path.join(CACHE, f"{spec_id}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{spec_id} not cached; call apogee_data.load_spectrum first")
    d = np.load(path)
    return d["flux"].astype(float), d["error"].astype(float)


def _fit_and_render(spec_id, out_png):
    """Fit single + binary models and render obs vs models to a PNG.

    Returns a small dict of the fit summary (the same quantities the detector
    reports) so the LLM prompt can be given the numbers alongside the picture.
    """
    flux, err = _load_cached(spec_id)
    # Same normalization + masking the detector uses, so the picture matches chi^2.
    obs, nerr = P.normalize_like_model(flux, err)
    nerr = P.mask_bad_pixels(obs, nerr)

    # Single-star fit, then binary fit (q + 2 RVs + primary refine), as in the
    # detector. We reconstruct the model arrays here to plot them.
    p_s, single_model, chi2_s = P.fit_single(obs, nerr, 5200.0, 4.5, 0.0)
    best, binary_model, chi2_b = P.fit_binary(
        obs, nerr, float(p_s[0]), float(p_s[1]), float(p_s[2]),
        age_gyr=5.0, fit_primary=True)
    # Binary-nests-single floor (mirror the detector) so the plot is honest.
    if chi2_b > chi2_s:
        chi2_b = chi2_s
        binary_model = single_model
        best = {"q": 1.0, "rv1": 0.0, "rv2": 0.0}
    delta = float(chi2_s - chi2_b)
    fimp = P.f_imp(obs, single_model, binary_model, nerr)

    wl = P.WAVELENGTH
    # Three H-band windows on strong-line regions plus a residual panel.
    windows = [(15200, 15350), (15700, 15850), (16150, 16300)]
    fig, axes = plt.subplots(4, 1, figsize=(13, 9))
    for ax, (lo, hi) in zip(axes[:3], windows):
        m = (wl >= lo) & (wl <= hi)
        ax.plot(wl[m], obs[m], "k-", lw=1.0, label="observed")
        ax.plot(wl[m], single_model[m], "C0-", lw=1.0, label="single")
        ax.plot(wl[m], binary_model[m], "C3--", lw=1.0, label="binary")
        ax.set_xlim(lo, hi)
        ax.set_ylabel("norm flux")
        ax.legend(fontsize=7, loc="lower left")
    # Residual panel over a window: shows where binary beats single.
    m = (wl >= windows[0][0]) & (wl <= windows[0][1])
    axes[3].axhline(0, color="0.7", lw=0.6)
    axes[3].plot(wl[m], (obs - single_model)[m], "C0-", lw=0.8, label="obs - single")
    axes[3].plot(wl[m], (obs - binary_model)[m], "C3-", lw=0.8, label="obs - binary")
    axes[3].set_xlim(*windows[0]); axes[3].set_ylabel("residual")
    axes[3].set_xlabel("wavelength (Angstrom)"); axes[3].legend(fontsize=7)
    axes[0].set_title(
        f"{spec_id}  Delta-chi2={delta:.0f}  f_imp={fimp:.2f}  "
        f"q={best['q']:.2f}  rv1={best['rv1']:.0f} rv2={best['rv2']:.0f}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return {"delta_chi2": round(delta, 1), "f_imp": round(float(fimp), 3),
            "q": round(float(best["q"]), 3),
            "rv1": round(float(best["rv1"]), 1), "rv2": round(float(best["rv2"]), 1)}


# Structured fields we ask the vision model to fill, so the return is machine-usable.
_SCHEMA_HINT = (
    '{"fit_quality": "good|fair|poor", '
    '"looks_like_sb2": true/false, '
    '"rv_split_visible": true/false, '
    '"unmasked_artifacts": true/false, '
    '"reasoning": "one or two sentences citing what you see", '
    '"recommendation": "accept|reject|remask|refit"}')


@mcp.tool()
def inspect_spectrum_fit(spec_id: str) -> dict:
    """
    Render the single-vs-binary fit for a spectrum and have gemini-3.5-flash judge it.

    The agent calls this to LOOK at a candidate the way an astronomer would. The
    tool fits the single and binary models, plots observed vs both over three
    H-band windows plus a residual panel, then sends the image to gemini-3.5-flash
    and asks whether the binary model is justified, whether a radial-velocity split
    is visible, and whether unmasked artifacts are present.

    Args:
        spec_id: 2MASS id of a spectrum already loaded (apogee_data.load_spectrum).

    Returns:
        {spec_id, fit, assessment, plot_path} where `fit` is the numeric summary
        (delta_chi2, f_imp, q, rv1, rv2) and `assessment` is the model's structured
        judgment (fit_quality, looks_like_sb2, rv_split_visible, unmasked_artifacts,
        reasoning, recommendation). Returns {"error": ...} on failure.
    """
    key = _load_env_key()
    if not key:
        return {"error": "GOOGLE_API_KEY not found in env or ~/.env"}
    out_png = os.path.join(CACHE, f"fit_{spec_id}.png")
    try:
        fit = _fit_and_render(spec_id, out_png)
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:  # rendering / fit failure
        return {"error": f"render failed: {e}"}

    # Send the rendered image to gemini-3.5-flash for visual reasoning.
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        img_bytes = open(out_png, "rb").read()
        prompt = (
            "You are inspecting an APOGEE H-band spectrum fit for a spectroscopic "
            "binary search. Black is the observed normalized spectrum; blue is the "
            "best-fit SINGLE-star model; red dashed is the best-fit BINARY model "
            "(two stars). The bottom panel shows residuals (observed minus model). "
            f"Fit summary: {json.dumps(fit)}. Judge whether the binary model is "
            "justified by real spectral features (line cores the single model "
            "misses, a second set of lines, a velocity split) rather than by noise "
            "or unmasked artifacts (sharp spikes up to ~1.4 or drops to ~0). "
            "Respond with ONLY a JSON object of this exact shape: " + _SCHEMA_HINT)
        resp = client.models.generate_content(
            model=VISION_MODEL,
            contents=[types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                      prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        text = resp.text.strip()
        assessment = json.loads(text)
    except Exception as e:
        return {"error": f"vision call failed: {e}", "plot_path": out_png, "fit": fit}

    return {"spec_id": spec_id, "fit": fit, "assessment": assessment,
            "plot_path": out_png}


if __name__ == "__main__":
    mcp.run()
