"""
Security utilities: API key authentication and token-bucket rate limiting.

Both are in-process implementations that are correct for a single-worker
deployment. The interfaces are intentionally simple so they can be swapped
for Redis-backed equivalents when scaling horizontally.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import deque
from typing import Deque, Dict

from fastapi import HTTPException, Request, status

from app.core.config import settings
from app.core.observability import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# API key auth
# --------------------------------------------------------------------------


def verify_api_key(api_key: str | None) -> bool:
    """
    Validate an API key using constant-time comparison.

    Returns True when auth is disabled (no key configured) so local
    development does not require credentials.
    """
    if not settings.API_KEY:
        return True
    if not api_key:
        return False
    return hmac.compare_digest(api_key, settings.API_KEY)


async def require_api_key(request: Request) -> None:
    """FastAPI dependency enforcing API key auth when one is configured."""
    if not settings.API_KEY:
        return
    header_key = request.headers.get("X-API-Key") or request.headers.get(
        "Authorization"
    )
    if header_key and header_key.lower().startswith("bearer "):
        header_key = header_key[7:]

    if not verify_api_key(header_key):
        logger.warning("auth_failed path=%s", request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


class RateLimiter:
    """
    Sliding-window rate limiter keyed by client IP.

    Uses a deque of request timestamps per key. Thread-safe, with periodic
    eviction of idle keys to bound memory.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def check(self, key: str) -> tuple[bool, int]:
        """
        Register a hit for `key`.

        Returns (allowed, remaining_requests). Raises nothing; callers decide
        how to surface rejection.
        """
        now = time.time()
        with self._lock:
            self._maybe_cleanup(now)
            hits = self._hits.setdefault(key, deque())
            cutoff = now - self.window_seconds
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.max_requests:
                retry_after = int(self.window_seconds - (now - hits[0])) + 1
                return False, retry_after

            hits.append(now)
            return True, self.max_requests - len(hits)

    def _maybe_cleanup(self, now: float) -> None:
        """Drop keys that have not been used within a full window."""
        if now - self._last_cleanup < self.window_seconds:
            return
        self._last_cleanup = now
        cutoff = now - self.window_seconds
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
            del self._hits[key]


limiter = RateLimiter(
    max_requests=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
)


async def rate_limit(request: Request) -> None:
    """FastAPI dependency enforcing the configured rate limit."""
    client_ip = request.client.host if request.client else "unknown"
    allowed, info = limiter.check(client_ip)

    if not allowed:
        logger.warning("rate_limited ip=%s", client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry shortly.",
            headers={"Retry-After": str(info)},
        )
