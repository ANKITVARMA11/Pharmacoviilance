"""Central settings. This is the ONLY module that reads the environment.

Put your settings in the `.env` file in the project folder (copy `.env.example` if it is missing).
A real environment variable with the same name wins over the file, which is handy for one-off runs.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = Path(os.getenv("PV_ENV_FILE") or ROOT / ".env")  # PV_ENV_FILE exists so tests can use a throwaway file


def load_env(path=None) -> bool:
    """Read KEY=VALUE lines from the .env file into the process environment (existing variables are kept)."""
    return load_dotenv(path or ENV_FILE, override=False)


def env(name: str, default=None):
    """An environment value with whitespace stripped. Blank or missing gives `default`."""
    value = (os.getenv(name) or "").strip()
    return value or default


def mask(secret) -> str:
    """Safe-to-print form of a secret."""
    if not secret:
        return "(not set)"
    return f"{secret[:5]}...{secret[-4:]}" if len(secret) > 12 else "set (short value)"


load_env()

# --- where files go ---
DATA_DIR = Path(env("PV_DATA", ROOT / "data"))
SAMPLE_FAERS = ROOT / "data" / "sample_faers.jsonl"
RAW_PATH = DATA_DIR / "faers_raw.jsonl"
PAIRS_PATH = DATA_DIR / "pairs.jsonl"
EXTRACTIONS_PATH = DATA_DIR / "extractions.jsonl"
TERMS_PATH = DATA_DIR / "terms.jsonl"  # reaction vocabulary for the coding step
CODED_PATH = DATA_DIR / "coded.jsonl"
AUDIT_PATH = DATA_DIR / "audit.jsonl"  # every agent input/output of every crew run
REVIEWS_PATH = DATA_DIR / "reviews.jsonl"  # the human reviewer's decisions
SAMPLE_TERMS = ROOT / "data" / "sample_terms.jsonl"

# --- LLM provider: "nim" (NVIDIA NIM, default) or "ollama" (local) ---
PROVIDER = env("PV_PROVIDER", "nim").lower()

NIM_HOSTED_URL = "https://integrate.api.nvidia.com/v1"
NIM_BASE_URL = env("NIM_BASE_URL", NIM_HOSTED_URL)  # point at a self-hosted NIM to stay in-house
NVIDIA_API_KEY = env("NVIDIA_API_KEY")  # free key: build.nvidia.com -> open any model -> "Get API Key"
NIM_MIN_INTERVAL = float(env("NIM_MIN_INTERVAL", "1.5"))  # seconds between calls (~40 requests/min)
NIM_TIMEOUT = float(env("NIM_TIMEOUT", "90"))  # seconds to wait for one answer before the model counts as stalled

OLLAMA_HOST = env("OLLAMA_HOST", "http://localhost:11434")

# Hosted NIM retires models often (HTTP 410) and its model list still shows some retired ones, so no ID is
# trusted. When PV_MODEL is not set, the models below are tried first, then other chat/instruct models on
# your key's list, until one answers a small test call. Pin the winner in .env to skip the search.
# Prefer non-reasoning models: reasoning models can spend the token budget thinking.
NIM_CANDIDATES = [
    "google/gemma-4-31b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-4b-it",
]
NIM_MAX_MODEL_TRIES = int(env("NIM_MAX_MODEL_TRIES", "10"))  # how many models to test before giving up
DEFAULT_MODELS = {"nim": NIM_CANDIDATES[0], "ollama": "llama3.2:3b"}
MODEL = env("PV_MODEL") or DEFAULT_MODELS.get(PROVIDER, NIM_CANDIDATES[0])

# --- crew (Day 3) ---
CREW_MAX_REVISIONS = int(env("CREW_MAX_REVISIONS", "1"))  # how many times the Critic may send the case back to Intake
MONITOR_HOST = "127.0.0.1"  # the monitor window is local only
MONITOR_PORT = int(env("PV_MONITOR_PORT", "8765"))  # next free port is used if busy
MONITOR_ENABLED = env("PV_MONITOR", "on").lower() not in ("off", "0", "false", "no")  # off: no monitor server
MONITOR_AUTO_OPEN = env("PV_MONITOR_OPEN", "on").lower() not in ("off", "0", "false", "no")  # off: do not open a window

# --- open data ---
OPENFDA_URL = "https://api.fda.gov/drug/event.json"
OPENFDA_API_KEY = env("OPENFDA_API_KEY")  # optional, raises openFDA rate limits
