"""
Centralized application configuration.

All settings are read from environment variables (12-factor style) with
sensible defaults so the app runs out-of-the-box in development.
Use `.env` for local overrides. No secrets live in source control.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(key: str, default: List[str]) -> List[str]:
    raw = os.getenv(key)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _default_llm_providers(environment: str) -> List[str]:
    """Prefer hosted providers in production; keep local-first development."""
    if environment.strip().lower() in {"production", "prod"}:
        return ["openai", "anthropic"]
    return ["ollama", "ivis", "openai", "anthropic"]


class Settings:
    """
    Application settings resolved from the environment.

    Every field is a plain class attribute so an instance can still be
    constructed with no arguments, while `__init__` accepts keyword
    overrides for tests and for embedding the app with explicit config.
    """

    # --- Application ---
    APP_NAME: str = _env("APP_NAME", "InteriorAI")
    ENVIRONMENT: str = _env("ENVIRONMENT", "development")
    DEBUG: bool = _env_bool("DEBUG", True)
    API_VERSION: str = "1.0.0"

    # --- Server ---
    HOST: str = _env("HOST", "0.0.0.0")
    PORT: int = _env_int("PORT", 8001)
    WORKERS: int = _env_int("WORKERS", 1)
    CORS_ORIGINS: List[str] = _env_list("CORS_ORIGINS", ["*"])

    # --- Security ---
    API_KEY: Optional[str] = _env("API_KEY", "") or None
    RATE_LIMIT_REQUESTS: int = _env_int("RATE_LIMIT_REQUESTS", 60)
    RATE_LIMIT_WINDOW_SECONDS: int = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)

    # --- LLM providers (ordered by preference; first healthy one wins) ---
    LLM_PROVIDERS: List[str] = _env_list(
        "LLM_PROVIDERS", _default_llm_providers(_env("ENVIRONMENT", "development"))
    )

    # The model used for the design pipeline.
    #
    # gemma2:2b was the default because it is fast and always present, but it
    # is too small to hold a design together: it invents SKUs, emits malformed
    # JSON often enough to need a repair layer, and writes pros and cons that
    # are generic to the point of being useless. llama3.1:8b is the point where
    # local generation becomes usable for this workload, so it leads.
    #
    # gemma2:2b remains a valid fast fallback: set LLM_FALLBACK_MODEL to it for
    # a machine with limited RAM, or leave it as the second entry so a failed
    # 8B load degrades instead of erroring.
    DEFAULT_MODEL: str = _env("DEFAULT_MODEL", "llama3.1:8b")
    FALLBACK_MODEL: str = _env("LLM_FALLBACK_MODEL", "gemma2:2b")
    # Models tried in order until one reports healthy. The fallback keeps a
    # machine without an 8B model working rather than failing every request.
    MODEL_CANDIDATES: List[str] = _env_list(
        "MODEL_CANDIDATES",
        [_env("DEFAULT_MODEL", "llama3.1:8b"), _env("LLM_FALLBACK_MODEL", "gemma2:2b")],
    )

    # Ollama keeps the larger model resident between calls. Without this an 8B
    # model is loaded into RAM on every request, which dominates latency.
    OLLAMA_KEEP_ALIVE: str = _env("OLLAMA_KEEP_ALIVE", "10m")

    OLLAMA_BASE_URL: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_TIMEOUT: float = _env_float("OLLAMA_TIMEOUT", 120.0)

    # An OpenAI-compatible gateway. Left blank by default: an internal or
    # third-party endpoint should only be used when explicitly configured.
    IVIS_GATEWAY_BASE_URL: str = _env("IVIS_GATEWAY_BASE_URL", "")
    IVIS_GATEWAY_API_KEY: Optional[str] = _env("IVIS_GATEWAY_API_KEY", "") or None
    IVIS_MODEL: str = _env("IVIS_MODEL", "gemma2:2b")

    OPENAI_API_KEY: Optional[str] = _env("OPENAI_API_KEY", "") or None
    OPENAI_BASE_URL: str = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = _env("OPENAI_MODEL", "gpt-4o-mini")

    ANTHROPIC_API_KEY: Optional[str] = _env("ANTHROPIC_API_KEY", "") or None
    ANTHROPIC_MODEL: str = _env("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

    # --- Generation controls ---
    # Ollama's `num_predict`. gemma2:2b tends to ramble; the agent prompts ask
    # for large JSON documents, and too low a ceiling truncates output
    # mid-object, which is unrecoverable. 3072 leaves headroom for the largest
    # agent payload while still bounding the request.
    MAX_TOKENS: int = _env_int("MAX_TOKENS", 3072)
    TEMPERATURE: float = _env_float("TEMPERATURE", 0.7)
    REQUEST_TIMEOUT: float = _env_float("REQUEST_TIMEOUT", 120.0)
    MAX_REFLECTION_ROUNDS: int = _env_int("MAX_REFLECTION_ROUNDS", 2)
    MAX_CONCURRENT_LLM_CALLS: int = _env_int("MAX_CONCURRENT_LLM_CALLS", 4)

    # --- Embeddings / RAG ---
    EMBEDDING_MODEL: str = _env("EMBEDDING_MODEL", "all-minilm-l6-v2")
    EMBEDDING_DIM: int = _env_int("EMBEDDING_DIM", 384)
    RAG_TOP_K: int = _env_int("RAG_TOP_K", 4)
    # The retriever must return 3-5 passages. One passage at 0.014 similarity
    # is not grounding - the architect was writing from model memory and the
    # citation was decoration. If too few passages clear the filter, the
    # service relaxes the filter until it reaches the floor rather than
    # returning a single weak hit.
    RAG_MIN_PASSAGES: int = _env_int("RAG_MIN_PASSAGES", 3)
    RAG_MAX_PASSAGES: int = _env_int("RAG_MAX_PASSAGES", 5)
    RAG_MIN_SCORE: float = _env_float("RAG_MIN_SCORE", 0.15)
    RERANK_ENABLED: bool = _env_bool("RERANK_ENABLED", True)

    # --- Caching ---
    # NOTE: the threshold is deliberately high. With the hashing embedder a
    # false positive is expensive - serving "spacious living room" to someone
    # who asked for "tiny" is far worse than a cache miss - and near-miss
    # similarity sits around 0.55-0.60. Raise this only when running a
    # sentence-transformers embedder, whose similarity distribution is
    # compressed far higher and much better separated.
    SEMANTIC_CACHE_ENABLED: bool = _env_bool("SEMANTIC_CACHE_ENABLED", True)
    SEMANTIC_CACHE_THRESHOLD: float = _env_float("SEMANTIC_CACHE_THRESHOLD", 0.9)
    CACHE_TTL_SECONDS: int = _env_int("CACHE_TTL_SECONDS", 3600)
    CACHE_MAX_ENTRIES: int = _env_int("CACHE_MAX_ENTRIES", 500)

    # --- Observability ---
    LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")
    LOG_JSON: bool = _env_bool("LOG_JSON", False)
    ENABLE_METRICS: bool = _env_bool("ENABLE_METRICS", True)
    ENABLE_TRACING: bool = _env_bool("ENABLE_TRACING", False)
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = _env(
        "OTEL_EXPORTER_OTLP_ENDPOINT", ""
    ) or None

    # --- Storage ---
    DATA_DIR: str = _env("DATA_DIR", "data")
    VECTOR_STORE_PATH: str = _env("VECTOR_STORE_PATH", "data/vector_store")

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() in {"production", "prod"}

    def __init__(self, **overrides: object) -> None:
        """
        Build a settings instance.

        With no arguments this reads the environment (the normal path). With
        keyword arguments, each one is applied over the environment-derived
        value, which lets tests construct a specific configuration without
        mutating `os.environ`.
        """
        for key, value in overrides.items():
            if not hasattr(type(self), key):
                raise AttributeError(f"Unknown setting: {key}")
            setattr(self, key, value)

    @property
    def pricing_inr(self) -> dict:
        """Approximate cost per 1K tokens in INR, for the cost dashboard."""
        return {
            "ollama": 0.0,  # local inference is free
            "ivis": 0.5,
            "openai": 0.35,
            "anthropic": 0.90,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton)."""
    return Settings()


settings = get_settings()
