"""
Design service: the application's public use-case layer.

Sits between the API and the agent pipeline, adding cross-cutting concerns
that don't belong in either: semantic caching, concurrency control, and
response enrichment.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np

from app.core.config import settings
from app.core.observability import get_logger, metrics, tracer
from app.core.schemas import DesignRequest, DesignResponse, RetrievedContext
from app.agents.orchestrator import design_pipeline
from app.services.embeddings import get_embedder
from app.services.retrieval import retrieval_service
from app.services.semantic_cache import get_cache

logger = get_logger(__name__)

# Guards against a burst of expensive generations exhausting local resources.
_generation_semaphore = asyncio.Semaphore(max(1, settings.MAX_CONCURRENT_LLM_CALLS * 2))


def _cache_payload(request: DesignRequest) -> Dict[str, Any]:
    """
    Fields that materially change the output.

    Deliberately excludes presentation-only fields so that e.g. a different
    `include_details` still hits a compatible cache entry.
    """
    return {
        "room_type": request.room_type.value,
        "style": request.style.value,
        "budget": request.budget.value,
        "budget_amount_inr": request.budget_amount_inr,
        "client_preferences": request.client_preferences.strip().lower(),
        "color_scheme": request.color_scheme,
        "room_dimensions": request.room_dimensions,
        "furniture": sorted(request.furniture),
        "requirements": sorted(request.specific_requirements),
        "num_options": request.num_options,
        "vastu": request.vastu_compliance.value,
    }


class DesignService:
    """
    Orchestrates cached and uncached design generation.
    """

    def __init__(self) -> None:
        self.pipeline = design_pipeline
        self.cache = get_cache()
        self.embedder = get_embedder()

    async def generate(
        self, request: DesignRequest, use_cache: bool = True
    ) -> DesignResponse:
        """Generate a design, returning a cached response when semantically similar."""
        request_id = uuid.uuid4().hex
        start = time.perf_counter()

        payload = _cache_payload(request)
        query_vector = self.embedder.embed(
            f"{request.client_preferences} {request.room_type.value} {request.style.value} "
            f"{request.budget.value} {' '.join(request.specific_requirements)}"
        )

        if use_cache:
            cached = self.cache.get(payload, query_vector)
            if cached is not None:
                response = DesignResponse(**cached)
                response.cached = True
                response.request_id = request_id
                response.latency_ms = (time.perf_counter() - start) * 1000
                logger.info("design_cache_hit request_id=%s", request_id)
                metrics.increment("design.served_from_cache")
                return response

        async with _generation_semaphore:
            response = await self.pipeline.generate(request, request_id=request_id)

        if use_cache:
            try:
                self.cache.set(payload, query_vector, response.model_dump())
            except Exception as exc:  # caching must never break a request
                logger.warning("cache_set_failed error=%s", exc)

        metrics.increment("design.generated")
        return response

    async def stream(
        self, request: DesignRequest, use_cache: bool = True
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream pipeline progress, short-circuiting on a cache hit.
        """
        payload = _cache_payload(request)
        query_vector = self.embedder.embed(
            f"{request.client_preferences} {request.room_type.value} {request.style.value} "
            f"{request.budget.value} {' '.join(request.specific_requirements)}"
        )

        if use_cache:
            cached = self.cache.get(payload, query_vector)
            if cached is not None:
                response = DesignResponse(**cached)
                response.cached = True
                response.request_id = uuid.uuid4().hex
                yield {
                    "type": "step",
                    "data": {
                        "agent": "cache", "status": "ok",
                        "message": "Served from semantic cache (no LLM call)",
                    },
                }
                yield {"type": "complete", "data": response.model_dump()}
                metrics.increment("design.served_from_cache")
                return

        async with _generation_semaphore:
            async for event in self.pipeline.stream(request, request_id=uuid.uuid4().hex):
                if event.get("type") == "complete":
                    try:
                        self.cache.set(payload, query_vector, event["data"])
                    except Exception as exc:
                        logger.warning("cache_set_failed error=%s", exc)
                yield event

    async def retrieve_context(
        self, query: str, top_k: int = 4
    ) -> List[RetrievedContext]:
        """Direct RAG access, exposed through the API for transparency."""
        retrieval_service.initialize()
        return await retrieval_service.retrieve(query, top_k=top_k)


design_service = DesignService()
