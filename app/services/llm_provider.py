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

    async def available_for(self, model: Optional[str] = None) -> bool:
        """
        Is this model actually pulled?

        Ollama reports healthy while a specific model is absent - the daemon is
        up, but `llama3.1:8b` was never downloaded, and the request fails at
        generation time. That is a much worse failure than a health check that
        says so up front, so availability is checked per model.
        """
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code != 200:
                    return False
                names = {
                    m.get("name", "").split(":")[0]
                    for m in (resp.json().get("models") or [])
                }
                wanted = (model or self.model or "").split(":")[0]
                return bool(wanted) and wanted in names
        except Exception:
            return False

    async def _complete(
        self, prompt: str, system: Optional[str], **kwargs: Any
    ) -> tuple[str, int, int]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # Hold the model in RAM between calls. Without this an 8B model is
            # evicted and reloaded on every request, which adds tens of seconds
            # and makes the stronger model slower than the 2B fallback.
            "keep_alive": settings.OLLAMA_KEEP_ALIVE,
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
            # The provider starts on the best model that is actually present.
            # `available_for` is async so this cannot be resolved here, so the
            # candidate list is kept and resolved on first use.
            candidates["ollama"] = OllamaProvider()
            self._ollama_models = list(settings.MODEL_CANDIDATES)

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

    async def resolve_model(self) -> str:
        """
        Pick the best configured model this machine can actually run.

        `DEFAULT_MODEL` is a preference, not a fact: a developer machine may
        not have an 8B model pulled. Ollama reports healthy as long as the
        daemon is up, so the first request against a missing model fails deep
        inside generation. Resolving against the models Ollama actually has
        turns that into a quiet downgrade to the fallback.
        """
        provider = self._providers.get("ollama")
        candidates = getattr(self, "_ollama_models", None) or [
            settings.DEFAULT_MODEL
        ]
        if provider is None or not hasattr(provider, "available_for"):
            return candidates[0]
        for model in candidates:
            if await provider.available_for(model):
                return model
        return candidates[-1]

    async def use_best_available_model(self) -> str:
        """Point the Ollama provider at the best available model. Idempotent."""
        model = await self.resolve_model()
        provider = self._providers.get("ollama")
        if provider is not None and hasattr(provider, "available_for"):
            provider.model = model
        logger.info("llm_model_selected model=%s", model)
        return model

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


def _drop_trailing_comma(out: List[str]) -> None:
    """
    Remove a comma that sits immediately before the end of the output.

    The comma and the bracket that follows it are usually separated by a
    newline, so the previously emitted character is rarely the comma itself.
    """
    k = len(out) - 1
    while k >= 0 and out[k].isspace():
        k -= 1
    if k >= 0 and out[k] == ",":
        del out[k:]


def _followed_by_colon(text: str, i: int) -> bool:
    """True when the string starting at `i` is immediately followed by ':'."""
    j = i + 1
    while j < len(text):
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        if ch == '"':
            k = j + 1
            while k < len(text) and text[k].isspace():
                k += 1
            return k < len(text) and text[k] == ":"
        j += 1
    return False


def _normalize_json_punctuation(text: str) -> str:
    """
    Fix the punctuation slips small models make constantly:

      * a trailing comma before a closing brace/bracket - `{"a": 1,}`
      * a missing comma between two values - `["x" "y"]`
      * a literal newline or tab inside a string value
      * a stray quote mark after a colon - `{"a": "key"}

    Each is structural noise in an otherwise complete document. Left alone it
    costs the entire response, because no JSON parser accepts them and the
    salvage path then has to discard the whole design. Commas inside string
    literals are never touched.
    """
    out: List[str] = []
    in_string = False
    escaped = False
    # What the scanner last consumed outside a string, used to decide whether a
    # separator is missing or redundant:
    #   "open"   - just after { or [      "colon"  - just after :
    #   "comma"  - just after ,            "value"  - a complete value was read
    #   "close"  - just after } or ]
    prev = ""
    key_pos = False
    containers: List[str] = []
    in_object = False
    i = 0
    n = len(text)

    def read_string_at(j: int) -> int:
        """Return the index just past the string literal starting at j."""
        k = j + 1
        while k < n:
            c = text[k]
            if c == "\\":
                k += 2
                continue
            if c == '"':
                return k + 1
            k += 1
        return n  # unterminated; caller treats as a value

    while i < n:
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
                out.append(ch)
                i += 1
                continue
            if ch == "\\":
                escaped = True
                out.append(ch)
                i += 1
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                # A string in key position is a key, not a value: it must not
                # trigger comma insertion before whatever follows.
                prev = "key" if key_pos else "value"
                i += 1
                continue
            elif ch in "\n\r\t":
                # A raw newline or tab inside a string is illegal JSON. The
                # model wrote a multi-line sentence; escape it so the sentence
                # survives instead of the whole response being discarded.
                out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
                i += 1
                continue
            else:
                out.append(ch)
                i += 1
                continue

        if ch.isspace():
            out.append(ch)
            i += 1
            continue

        if ch == '"':
            # An immediately-closing quote right after ":" is a stray pair,
            # e.g. `{"a": "b"}` written as `{"a": "b": "}`. Drop the empty
            # pair so the real value that follows can parse.
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if prev == "colon" and j < n and text[j] == '"':
                i += 1
                continue
            # A string right after a comma or an opening brace is an object key;
            # inside an array it is a value. Only a value needs a separator
            # inserted before whatever follows it.
            is_key_position = (in_object and prev in ("comma", "open")) or (
                # A quoted name followed by a colon can only be an object key.
                # A model that forgot an array's closing bracket writes
                # ["a note", "brands": [...]] - without this the key is read as
                # another array element and the document never parses.
                not in_object and _followed_by_colon(text, i)
            )
            if prev == "value":
                out.append(",")
            in_string = True
            key_pos = is_key_position
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            # A comma is only valid after a complete value or a closed
            # container. Anywhere else it is redundant, so drop it.
            if prev not in ("value", "close"):
                i += 1
                continue
            out.append(ch)
            prev = "comma"
            i += 1
            continue

        if ch in "}]":
            # Drop a comma that is only there because a value follows nothing.
            if prev == "comma":
                _drop_trailing_comma(out)
            out.append(ch)
            if containers:
                containers.pop()
            in_object = bool(containers) and containers[-1] == "{"
            prev = "close"
            i += 1
            continue

        if ch in "{[":
            if prev == "value":
                out.append(",")
            out.append(ch)
            # Push the container kind so a string inside "[" is read as a value
            # rather than a key.
            containers.append(ch)
            in_object = ch == "{"
            prev = "open"
            i += 1
            continue

        if ch == ":":
            if prev != "value":
                # A key with no name: nothing sensible to repair here.
                pass
            out.append(ch)
            prev = "colon"
            i += 1
            continue

        # Bare literal: number, true, false, or null. Read it whole so that a
        # missing comma is only inserted between values, never inside one.
        j = i
        while j < n and text[j] not in ",}]\" \t\r\n:":
            j += 1
        token = text[i:j]
        if token:
            if prev == "value":
                out.append(",")
            out.append(token)
            prev = "value"
        i = j if j > i else i + 1

    return "".join(out)


def _repair_json(text: str) -> Any:
    """
    Recover a usable object from truncated or malformed JSON.

    Small local models frequently hit the token ceiling mid-document, which
    leaves an unterminated string/object. Rather than discarding the whole
    response, close the open structures and drop the incomplete tail. The
    salvage is worth more than a total failure: a design missing its last
    field is still a design.
    """
    # Walk the text tracking string state and open brackets.
    stack: List[str] = []
    in_string = False
    escaped = False
    last_complete = 0

    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
                last_complete = i + 1
        elif ch == "," and stack:
            # A comma outside a string at depth>0 means the preceding value
            # completed cleanly; a safe truncation point.
            last_complete = i

    if not stack:
        return None

    # Prefer the safe cut point (a completed key/value pair). Only if that
    # yields nothing do we close a dangling string - appending a quote next
    # to a bare number (e.g. `5000` -> `5000"`) creates invalid JSON.
    candidates = []
    safe = text[:last_complete].rstrip().rstrip(",")
    if safe.strip():
        candidates.append(safe)
    if in_string:
        # Close the string at the last delimiter that was not inside it.
        candidates.append(text.rstrip().rstrip(",") + '"')

    for base in candidates:
        if not base.strip():
            continue
        for depth_trim in range(0, 40):
            attempt = base[: len(base) - depth_trim].rstrip().rstrip(",") if depth_trim else base
            if not attempt.strip():
                break
            # Recompute the open-bracket stack for this prefix.
            open_stack = _open_brackets(attempt)
            if open_stack is None:
                break
            attempt += "".join("}" if c == "{" else "]" for c in reversed(open_stack))
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
    return None


def _open_brackets(text: str) -> Optional[List[str]]:
    """Return the unclosed bracket stack for `text`, or None if unbalanced."""
    stack: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None  # closing without an opener
            stack.pop()
    return stack


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction and repair from LLM output."""
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

    # Fall back to the outermost span. The opener must be the first
    # structural character in the document, otherwise a nested array could
    # be mistaken for the whole payload.
    first_struct = min(
        (i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1),
        default=-1,
    )
    if first_struct != -1:
        opener = cleaned[first_struct]
        closer = "}" if opener == "{" else "]"
        end = cleaned.rfind(closer)
        if end > first_struct:
            try:
                return json.loads(cleaned[first_struct : end + 1])
            except json.JSONDecodeError:
                pass

    # Salvage a truncated document.
    repaired = _repair_json(cleaned)
    if repaired is not None:
        logger.info("json_repaired truncated_output_recovered")
        return repaired

    # Still nothing: the document is structurally complete but carried trailing
    # commas, which is the most common way a small model emits valid-looking
    # JSON that no parser accepts. Try once more without them.
    de_commad = _normalize_json_punctuation(cleaned)
    if de_commad != cleaned:
        try:
            return json.loads(de_commad)
        except json.JSONDecodeError:
            pass
        repaired = _repair_json(de_commad)
        if repaired is not None:
            logger.info("json_repaired punctuation_normalised")
            return repaired

    # Report both ends: the head shows the shape the model chose, the tail
    # shows where generation stopped, which is what distinguishes a truncated
    # response from prose the model never formatted as JSON at all.
    raise ValueError(
        f"could not parse JSON from: {text[:200]} ...[tail]... {text[-200:]}"
    )


registry = LLMRegistry()
