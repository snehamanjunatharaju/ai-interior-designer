"""
LLM provider abstraction.

A uniform async interface over heterogeneous model backends (Ollama, IVIS
OpenAI-compatible gateway, OpenAI, Anthropic) with:

- Ordered failover with automatic fallback
- Per-provider circuit breaking (fail fast, half-open recovery)
- Token accounting and INR cost estimation
- Bounded concurrency via a semaphore
- Full tracing and metrics for every call
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from app.core.config import settings
from app.core.observability import get_logger, metrics, tracer

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------


class CircuitBreaker:
    """
    Classic three-state circuit breaker.

    CLOSED -> normal. After `threshold` consecutive failures -> OPEN (calls
    fail fast). After `recovery_timeout` seconds -> HALF_OPEN, where a
    single probe is allowed through. Success closes it; failure re-opens.
    """

    def __init__(self, threshold: int = 3, recovery_timeout: float = 30.0) -> None:
        self.threshold = threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "closed"  # closed | open | half_open
        self.opened_at: float = 0.0
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            if self.state == "closed":
                return True
            if self.state == "open":
                if time.time() - self.opened_at >= self.recovery_timeout:
                    self.state = "half_open"
                    logger.info("circuit_breaker half_open")
                    return True
                return False
            return True  # half_open: allow a single probe

    async def record_success(self) -> None:
        async with self._lock:
            if self.state == "half_open":
                logger.info("circuit_breaker closed")
            self.failures = 0
            self.state = "closed"

    async def record_failure(self) -> None:
        async with self._lock:
            self.failures += 1
            if self.state == "half_open" or self.failures >= self.threshold:
                if self.state != "open":
                    logger.warning("circuit_breaker open failures=%d", self.failures)
                self.state = "open"
                self.opened_at = time.time()

    @property
    def is_available(self) -> bool:
        return self.state != "open"


# --------------------------------------------------------------------------
# Provider interface
# --------------------------------------------------------------------------


class LLMProvider(ABC):
    """Base class every backend implements."""

    name: str = "base"

    def __init__(self) -> None:
        self.breaker = CircuitBreaker()
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_LLM_CALLS)

    @abstractmethod
    async def _complete(
        self, prompt: str, system: Optional[str], **kwargs: Any
    ) -> tuple[str, int, int]:
        """Return (text, prompt_tokens, completion_tokens)."""

    async def available(self) -> bool:
        """Cheap liveness check. Default: assume reachable."""
        return True

    def estimate_cost_inr(self, prompt_tokens: int, completion_tokens: int) -> float:
        rate = settings.pricing_inr.get(self.name, 1.0)
        return ((prompt_tokens + completion_tokens) / 1000.0) * rate

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a completion with breaker, concurrency limiting, and telemetry.
        Raises RuntimeError when the circuit is open or the call fails.
        """
        if not await self.breaker.allow():
            metrics.increment("llm.circuit_open", labels=self.name)
            raise RuntimeError(f"Circuit open for provider '{self.name}'")

        async with self._semaphore:
            start = time.perf_counter()
            with tracer.start("llm.call", provider=self.name):
                try:
                    text, p_tok, c_tok = await self._complete(
                        prompt,
                        system,
                        max_tokens=max_tokens or settings.MAX_TOKENS,
                        temperature=(
                            settings.TEMPERATURE if temperature is None else temperature
                        ),
                        json_mode=json_mode,
                    )
                except Exception as exc:
                    await self.breaker.record_failure()
                    metrics.increment("llm.errors", labels=self.name)
                    metrics.observe("llm.latency_ms", (time.perf_counter() - start) * 1000)
                    logger.error("provider=%s error=%s", self.name, exc)
                    raise RuntimeError(f"{self.name} failed: {exc}") from exc

                latency = (time.perf_counter() - start) * 1000
                await self.breaker.record_success()
                cost = self.estimate_cost_inr(p_tok, c_tok)
                metrics.increment("llm.success", labels=self.name)
                metrics.increment("llm.prompt_tokens", value=p_tok, labels=self.name)
                metrics.increment("llm.completion_tokens", value=c_tok, labels=self.name)
                metrics.observe("llm.latency_ms", latency)
                metrics.increment("llm.cost_inr", value=cost, labels=self.name)

                return {
                    "text": text,
                    "provider": self.name,
                    "model": getattr(self, "model", "unknown"),
                    "latency_ms": latency,
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "cost_inr": cost,
                    "cached": False,
                }

    async def stream(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        """
        Stream tokens. Default implementation performs a single completion and
        yields it in chunks, so every provider supports streaming semantics.
        """
        result = await self.complete(prompt, system, **kwargs)
        for chunk in _chunk_text(result["text"]):
            yield chunk


def _chunk_text(text: str, size: int = 64) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def _rough_token_count(text: str) -> int:
    """Heuristic token estimate (~4 chars/token) when no usage is reported."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# Concrete providers
# --------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    """Local Ollama inference (free, private, no network egress)."""

    name = "ollama"

    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__()
        self.model = model or settings.DEFAULT_MODEL
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        # Local models fail fast rather than hanging the whole request.
        self.breaker = CircuitBreaker(threshold=2, recovery_timeout=10.0)

    async def available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def _complete(
        self, prompt: str, system: Optional[str], **kwargs: Any
    ) -> tuple[str, int, int]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature"),
                "num_predict": kwargs.get("max_tokens"),
            },
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = data.get("response", "")
        if not text:
            raise RuntimeError("empty response from Ollama")
        return text, _rough_token_count(prompt), _rough_token_count(text)


class OpenAICompatibleProvider(LLMProvider):
    """
    Any OpenAI-compatible /v1/chat/completions endpoint.

    Used for both the IVIS gateway and the official OpenAI API; the only
    differences are base URL, credentials, and model id.
    """

    name = "openai"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        if name:
            self.name = name

    async def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return resp.status_code < 500
        except Exception:
            return False

    async def _complete(
        self, prompt: str, system: Optional[str], **kwargs: Any
    ) -> tuple[str, int, int]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens"),
            "temperature": kwargs.get("temperature"),
        }
        if kwargs.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("no choices in response")
        text = choices[0].get("message", {}).get("content", "")
        usage = data.get("usage") or {}
        return (
            text,
            int(usage.get("prompt_tokens", _rough_token_count(prompt))),
            int(usage.get("completion_tokens", _rough_token_count(text))),
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API (Claude family)."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"
        self.version = "2023-06-01"

    async def available(self) -> bool:
        return bool(self.api_key)

    async def _complete(
        self, prompt: str, system: Optional[str], **kwargs: Any
    ) -> tuple[str, int, int]:
        payload = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens") or settings.MAX_TOKENS,
            "temperature": kwargs.get("temperature", settings.TEMPERATURE),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": self.version,
                    "content-type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage") or {}
        return (
            text,
            int(usage.get("input_tokens", _rough_token_count(prompt))),
            int(usage.get("output_tokens", _rough_token_count(text))),
        )


# --------------------------------------------------------------------------
# Registry & router
# --------------------------------------------------------------------------


class LLMRegistry:
    """
    Holds configured providers and routes calls with automatic failover.

    Ordering is controlled by `LLM_PROVIDERS`. Each call attempts
    providers in order, skipping any whose circuit is open, until one
    succeeds. A partial-response policy ensures a failure on the primary
    never surfaces as a user error if a secondary is healthy.
    """

    def __init__(self) -> None:
        self._providers: Dict[str, LLMProvider] = {}
        self._order: List[str] = []
        self._build()

    def _build(self) -> None:
        candidates: Dict[str, LLMProvider] = {}

        if "ollama" in settings.LLM_PROVIDERS:
            candidates["ollama"] = OllamaProvider()

        if "ivis" in settings.LLM_PROVIDERS and settings.IVIS_GATEWAY_API_KEY and settings.IVIS_GATEWAY_BASE_URL:
            candidates["ivis"] = OpenAICompatibleProvider(
                base_url=f"{settings.IVIS_GATEWAY_BASE_URL}/api",
                api_key=settings.IVIS_GATEWAY_API_KEY,
                model=settings.IVIS_MODEL,
                name="ivis",
            )

        if "openai" in settings.LLM_PROVIDERS and settings.OPENAI_API_KEY:
            candidates["openai"] = OpenAICompatibleProvider(
                base_url=settings.OPENAI_BASE_URL,
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_MODEL,
            )

        if "anthropic" in settings.LLM_PROVIDERS and settings.ANTHROPIC_API_KEY:
            candidates["anthropic"] = AnthropicProvider(
                api_key=settings.ANTHROPIC_API_KEY,
                model=settings.ANTHROPIC_MODEL,
            )

        self._providers = candidates
        self._order = [n for n in settings.LLM_PROVIDERS if n in candidates]
        logger.info("llm_registry_initialized providers=%s", self._order)

    def get(self, name: str) -> Optional[LLMProvider]:
        return self._providers.get(name)

    def all(self) -> Dict[str, LLMProvider]:
        return dict(self._providers)

    async def complete(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Try each provider in order until one succeeds.

        Raises RuntimeError only when every configured provider has failed.
        """
        errors: List[str] = []
        with tracer.start("llm.route"):
            for name in self._order:
                provider = self._providers[name]
                try:
                    result = await provider.complete(prompt, system, **kwargs)
                    metrics.increment("llm.fallback_used")
                    if errors:
                        logger.info("fallback_succeeded provider=%s", name)
                    return result
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
                    metrics.increment("llm.fallback_attempted", labels=name)
                    continue

        metrics.increment("llm.all_providers_failed")
        raise RuntimeError(
            "All LLM providers failed: " + " | ".join(errors) if errors
            else "No LLM providers configured"
        )

    async def complete_json(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Complete and coerce the result into valid JSON.

        Small local models frequently wrap JSON in prose or fences, so this
        applies several recovery strategies before giving up.
        """
        kwargs.setdefault("json_mode", True)
        result = await self.complete(prompt, system, **kwargs)
        raw = result["text"].strip()
        result["json"] = extract_json(raw)
        return result

    async def health(self) -> Dict[str, Any]:
        """Probe every provider for reachability and circuit state."""
        status: Dict[str, Any] = {}
        for name, provider in self._providers.items():
            reachable = await provider.available()
            status[name] = {
                "reachable": reachable,
                "circuit": provider.breaker.state,
                "model": getattr(provider, "model", "unknown"),
            }
            metrics.gauge(f"provider.{name}.reachable", 1.0 if reachable else 0.0)
        return status


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from LLM output."""
    if not text:
        raise ValueError("empty LLM response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
        cleaned = cleaned.rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost brace/bracket span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"could not parse JSON from: {text[:200]}")


registry = LLMRegistry()
