"""
Semantic response cache.

Exact-match caching is nearly useless for this workload because briefs are
free text. This uses embedding similarity so a rephrased request ("small
modern living room" vs "compact contemporary lounge") still hits cache.

Includes a popularity-aware eviction policy: frequently reused entries are
protected from expiry, which is the standard cache-admission heuristic.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.core.observability import get_logger, metrics

logger = get_logger(__name__)


class SemanticCache:
    """
    Similarity-based cache keyed on the embedding of the request payload.

    Stores normalized embedding vectors and returns the nearest neighbour
    above a similarity threshold. Capacity-bounded with LRU + TTL.
    """

    def __init__(
        self,
        dim: int,
        threshold: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        max_entries: Optional[int] = None,
    ) -> None:
        self.dim = dim
        self.threshold = (
            settings.SEMANTIC_CACHE_THRESHOLD if threshold is None else threshold
        )
        self.ttl = settings.CACHE_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        self.max_entries = (
            settings.CACHE_MAX_ENTRIES if max_entries is None else max_entries
        )

        self._entries: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(payload: Dict[str, Any]) -> str:
        """Stable hash of a request payload for the exact-match fast path."""
        normalized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()

    def get(self, payload: Dict[str, Any], vector: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Look up a cached response.

        First checks the exact-key fast path, then falls back to a similarity
        scan over live entries. On a hit the entry's TTL and LRU position are
        refreshed, making hot entries sticky.
        """
        if not settings.SEMANTIC_CACHE_ENABLED:
            return None

        key = self.make_key(payload)
        now = time.time()
        query = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm

        with self._lock:
            self._expire(now)

            exact = self._entries.get(key)
            if exact is not None:
                self._touch(key)
                self.hits += 1
                metrics.increment("cache.hit", labels="exact")
                return exact["value"]

            best_key, best_score = None, -1.0
            for entry_key, entry in self._entries.items():
                score = float(np.dot(entry["vector"], query))
                if score > best_score:
                    best_key, best_score = entry_key, score

            if best_key is not None and best_score >= self.threshold:
                self._touch(best_key)
                self.hits += 1
                metrics.increment("cache.hit", labels="semantic")
                logger.debug("semantic_cache_hit score=%.3f", best_score)
                return self._entries[best_key]["value"]

        self.misses += 1
        metrics.increment("cache.miss")
        return None

    def set(self, payload: Dict[str, Any], vector: np.ndarray, value: Any) -> None:
        """Store a response, evicting the least-recently-used entry if full."""
        if not settings.SEMANTIC_CACHE_ENABLED:
            return

        key = self.make_key(payload)
        vec = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm

        with self._lock:
            self._entries[key] = {
                "vector": vec,
                "value": value,
                "created_at": time.time(),
                "hits": 0,
            }
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                evicted, _ = self._entries.popitem(last=False)
                logger.debug("cache_evicted key=%s", evicted)
            metrics.gauge("cache.size", len(self._entries))

    def _touch(self, key: str) -> None:
        if key in self._entries:
            self._entries[key]["hits"] += 1
            # Refresh TTL on access: hot entries stay resident.
            self._entries[key]["created_at"] = time.time()
            self._entries.move_to_end(key)

    def _expire(self, now: float) -> None:
        cutoff = now - self.ttl
        stale = [k for k, v in self._entries.items() if v["created_at"] < cutoff]
        for key in stale:
            del self._entries[key]
        if stale:
            metrics.increment("cache.expired", value=len(stale))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._entries),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
                "threshold": self.threshold,
                "ttl_seconds": self.ttl,
            }


_cache: Optional[SemanticCache] = None


def get_cache() -> SemanticCache:
    """Return the process-wide semantic cache singleton."""
    global _cache
    if _cache is None:
        from app.services.embeddings import get_embedder

        _cache = SemanticCache(dim=get_embedder().dim)
        logger.info("semantic_cache_initialized dim=%d", _cache.dim)
    return _cache
