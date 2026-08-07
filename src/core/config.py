"""
Lab 11 — Configuration & API Key Setup

Default provider: OpenRouter (OpenAI-compatible) via LiteLLM / ADK LiteLlm.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Prefer a smaller free model (usually more capacity). Override via LLM_MODEL in .env
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-nano-9b-v2:free"

# LiteLLM/OpenRouter fallbacks when primary is at capacity / TTFT fail
FREE_FALLBACK_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "inclusionai/ling-3.0-tiny:free",
    "openai/gpt-oss-20b:free",
    "poolside/laguna-xs-2.1:free",
    "poolside/laguna-s-2.1:free",
    "openrouter/free",
]


def _repo_root() -> Path:
    # src/core/config.py → repo root
    return Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_repo_root() / ".env", override=False)


def get_openrouter_model_slug() -> str:
    """Model id for OpenRouter's OpenAI-compatible API."""
    _load_dotenv()
    raw = os.environ.get("LLM_MODEL", DEFAULT_OPENROUTER_MODEL).strip().strip("'\"")
    if not raw:
        raw = DEFAULT_OPENROUTER_MODEL
    # Allow accidental LiteLLM-style value: openrouter/<slug>
    if raw.startswith("openrouter/openrouter/"):
        return raw[len("openrouter/") :]
    if raw.startswith("openrouter/") and raw not in {
        "openrouter/free",
    }:
        rest = raw[len("openrouter/") :]
        # litellm id for third-party models looks like openrouter/google/gemma...
        if "/" in rest:
            return rest
    return raw


def get_litellm_model_id() -> str:
    """Model id for ADK LiteLlm / LiteLLM: openrouter/<openrouter-slug>."""
    return f"openrouter/{get_openrouter_model_slug()}"


def get_litellm_fallback_ids() -> list[str]:
    """Alternate free models (as LiteLLM ids), excluding the primary."""
    primary = get_openrouter_model_slug()
    env_raw = os.environ.get("LLM_FALLBACKS", "").strip()
    if env_raw:
        slugs = [s.strip() for s in env_raw.split(",") if s.strip()]
    else:
        slugs = list(FREE_FALLBACK_MODELS)
    out = []
    for slug in slugs:
        if slug == primary:
            continue
        if slug.startswith("openrouter/") and "/" in slug[len("openrouter/") :]:
            # already a litellm id for non-router models
            if slug.startswith("openrouter/openrouter/"):
                out.append(slug)
            elif slug == "openrouter/free":
                out.append("openrouter/openrouter/free")
            else:
                # treat as openrouter model slug already? keep as litellm id
                out.append(slug if slug.count("/") >= 2 else f"openrouter/{slug}")
        else:
            out.append(f"openrouter/{slug}")
    return out


@lru_cache(maxsize=1)
def get_chat_model():
    """Return an ADK model object pointed at OpenRouter (free + fallbacks)."""
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(
        model=get_litellm_model_id(),
        fallbacks=get_litellm_fallback_ids(),
        num_retries=3,
    )


def setup_api_key():
    """Load OpenRouter API key from .env / environment (or prompt)."""
    _load_dotenv()
    get_chat_model.cache_clear()

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or key.startswith("your-"):
        key = input("Enter OpenRouter API Key: ").strip()
        os.environ["OPENROUTER_API_KEY"] = key

    # LiteLLM reads OPENROUTER_API_KEY; also set OpenAI-compat vars for other clients
    os.environ["OPENROUTER_API_KEY"] = key
    os.environ.setdefault("OPENAI_API_KEY", key)
    os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    # Avoid accidental Vertex / Google AI Studio paths
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "0")

    model = get_openrouter_model_slug()
    print(f"API key loaded (OpenRouter). Model: {model}")
    print(f"Fallbacks: {', '.join(get_litellm_fallback_ids()) or '(none)'}")


# Allowed banking topics (used by topic_filter)
ALLOWED_TOPICS = [
    "banking", "account", "transaction", "transfer",
    "loan", "interest", "savings", "credit",
    "deposit", "withdrawal", "balance", "payment",
    "tai khoan", "giao dich", "tiet kiem", "lai suat",
    "chuyen tien", "the tin dung", "so du", "vay",
    "ngan hang", "atm",
]

# Blocked topics (immediate reject)
BLOCKED_TOPICS = [
    "hack", "exploit", "weapon", "drug", "illegal",
    "violence", "gambling", "bomb", "kill", "steal",
]
