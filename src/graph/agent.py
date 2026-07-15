#!/usr/bin/env python3
"""
LangGraph coordination layer for agent4binary.

gemini-3.5-flash (via langchain-google-genai) drives agents that reach every science
capability ONLY through MCP (langchain-mcp-adapters MultiServerMCPClient over stdio).
This module wires the servers, binds the model, and builds a small multi-node graph:

    orchestrator -> data -> single_fit -> binary_fit -> verify -> report

Keep it simple: each node is a ReAct-style agent with the relevant MCP tool subset.
"""
import os, sys, asyncio
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path.home() / ".env")  # GOOGLE_API_KEY

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

ROOT = Path(__file__).resolve().parents[2]
SERVERS = ROOT / "src" / "mcp_servers"
PY = sys.executable

# Backend toggle: which gemini-3.5-flash provider drives the agents.
#   AGENT_BACKEND=studio (default) -> ChatGoogleGenerativeAI (AI Studio API key
#                                     GOOGLE_API_KEY from ~/.env).
#   AGENT_BACKEND=vertex            -> ChatVertexAI on Vertex AI (GCP project
#                                     osu-prd-as-tingastroml-ad00, location global;
#                                     uses Application Default Credentials).
# The MCP tool layer (MultiServerMCPClient over all servers) is identical for both.
AGENT_BACKEND = os.environ.get("AGENT_BACKEND", "studio").strip().lower()
VERTEX_PROJECT = os.environ.get(
    "VERTEX_PROJECT", "osu-prd-as-tingastroml-ad00")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")

# Every science tool is an MCP server. Payne/isochrone load once their files exist.
def _conn(name):
    return {"command": PY, "args": [str(SERVERS / f"{name}.py")], "transport": "stdio"}

SERVER_NAMES = [
    "apogee_data_server", "binary_model_server", "doppler_server",
    "broadening_server", "gaia_sql_server", "gaia_xp_server",
    "payne_server", "isochrone_server",
]

MODEL = "gemini-3.5-flash"


def make_client():
    conns = {}
    for n in SERVER_NAMES:
        if (SERVERS / f"{n}.py").exists():
            conns[n] = _conn(n)
    return MultiServerMCPClient(conns)


def make_model(temperature=0.0, backend=None):
    """Build the gemini-3.5-flash chat model for the selected backend.

    backend=None reads AGENT_BACKEND ("studio" or "vertex"). Imports are local so
    that selecting one backend never requires the other's package to be installed.
    """
    backend = (backend or AGENT_BACKEND).strip().lower()
    if backend == "vertex":
        # Vertex AI path: gemini-3.5-flash served from GCP project osu-prd, using
        # Application Default Credentials (gcloud auth application-default login,
        # or a service-account key via GOOGLE_APPLICATION_CREDENTIALS). No API key.
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(model=MODEL, temperature=temperature,
                            project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    # Default AI Studio path: gemini-3.5-flash via the GOOGLE_API_KEY in ~/.env.
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=MODEL, temperature=temperature,
                                  google_api_key=os.environ["GOOGLE_API_KEY"])


async def load_tools():
    client = make_client()
    tools = await client.get_tools()
    return tools


# --- single-agent demo: gemini + all MCP tools (simplest showcase) ---
async def build_agent():
    tools = await load_tools()
    model = make_model()
    system = (
        "You are an astronomy assistant that disentangles spectroscopic binaries from "
        "APOGEE spectra. Use the MCP tools: load RAW DR19 spectra by sdss_id with "
        "apogee_data.load_dr19_spectrum (this caches flux_raw+ivar for the detector), "
        "load DR17 aspcap spectra with apogee_data.load_spectrum, detect SB2 candidates "
        "with binary_model (ccf_rv_scan, chi2_single_vs_binary; the default dr19_sc model "
        "is the production self-consistent detector and reads the raw cache), model single "
        "stars with payne, convert mass ratio to flux ratio with isochrone, and validate "
        "with Gaia (gaia_sql crossmatch_2mass / binarity_flags for RUWE). For a DR19 star: "
        "(1) load_dr19_spectrum(sdss_id); (2) chi2_single_vs_binary(spec_id=str(sdss_id)); "
        "(3) cross-check Gaia RUWE; (4) report a single-vs-binary verdict. Always load a "
        "spectrum before analyzing it."
    )
    return create_react_agent(model, tools, prompt=system)


if __name__ == "__main__":
    async def main():
        tools = await load_tools()
        print(f"Loaded {len(tools)} MCP tools from {len(make_client().connections)} servers:")
        for t in tools:
            print(f"  - {t.name}")
    asyncio.run(main())
