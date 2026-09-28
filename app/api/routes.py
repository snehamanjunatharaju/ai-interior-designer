"""
API routes.

Versioned REST surface with streaming, health, metrics, retrieval, and
evaluation endpoints.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse, StreamingResponse

from app.core.config import settings
from app.core.observability import get_logger, metrics, tracer
from app.core.schemas import (
    DesignRequest,
    DesignResponse,
    EvaluationScore,
    HealthStatus,
    RetrievalRequest,
)
from app.core.security import rate_limit, require_api_key
from app.services.design_service import design_service
from app.services.evaluation import evaluation_harness
from app.services.llm_provider import registry
from app.services.retrieval import retrieval_service
from app.services.semantic_cache import get_cache

logger = get_logger(__name__)

router = APIRouter()

_START_TIME = time.time()

# Auth and rate limiting are applied to expensive endpoints only; health and
# metrics stay open so orchestrators can scrape them unauthenticated.
protected = [Depends(require_api_key), Depends(rate_limit)]


# --------------------------------------------------------------------------
# Design generation
# --------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=DesignResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate interior design options",
    description=(
        "Runs the multi-agent design pipeline (retrieval -> planner -> architect "
        "-> budget -> critic with reflection) and returns fully structured, "
        "INR-priced design options."
    ),
    dependencies=protected,
)
async def generate_design(
    payload: DesignRequest,
    use_cache: bool = Query(True, description="Allow semantic cache lookup"),
) -> DesignResponse:
    """
    Generate interior design options for a client brief.

    Returns a structured payload: layout, furniture with INR pricing and brand
    suggestions, colour palette, lighting plan, budget breakdown, Vastu notes,
    and the critic's evaluation.
    """
    with tracer.start("api.generate", room=payload.room_type.value):
        try:
            return await design_service.generate(payload, use_cache=use_cache)
        except RuntimeError as exc:
            # All providers down - this is a capacity problem, not a bad request.
            logger.error("generation_unavailable error=%s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No LLM provider is currently available. Please retry shortly.",
            ) from exc
        except Exception as exc:
            logger.exception("generation_failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Design generation failed: {exc}",
            ) from exc


@router.post(
    "/generate/stream",
    summary="Stream design generation progress (SSE)",
    description=(
        "Server-Sent Events stream reporting each agent as it runs, followed by "
        "the final design payload. Provides real pipeline visibility instead of "
        "an opaque spinner."
    ),
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Pipeline progress events then a final `complete` event.",
        }
    },
    dependencies=protected,
)
async def stream_design(payload: DesignRequest) -> StreamingResponse:
    """
    Stream the design pipeline as Server-Sent Events.

    Event types are `step` (agent progress) and `complete` (final response).
    """
    request_id = uuid.uuid4().hex

    async def event_generator() -> AsyncIterator[bytes]:
        yield f"event: start\ndata: {json.dumps({'request_id': request_id})}\n\n".encode()
        try:
            async for event in design_service.stream(payload):
                event_type = event.get("type", "message")
                yield (
                    f"event: {event_type}\n"
                    f"data: {json.dumps(event.get('data', {}), default=str)}\n\n"
                ).encode()
        except Exception as exc:
            logger.exception("stream_failed request_id=%s", request_id)
            error = json.dumps({"error": str(exc)})
            yield f"event: error\ndata: {error}\n\n".encode()
        finally:
            yield b"event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering.
        },
    )


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


@router.post(
    "/retrieve",
    summary="Retrieve design knowledge",
    description="Semantic search over the interior-design knowledge base (RAG).",
    dependencies=protected,
)
async def retrieve(payload: RetrievalRequest) -> Dict[str, Any]:
    """
    Search the design knowledge base.

    Useful for grounding, debugging retrieval, and demonstrating the RAG
    subsystem independently of generation.
    """
    try:
        contexts = await design_service.retrieve_context(payload.query, top_k=payload.top_k)
        return {
            "query": payload.query,
            "results": [c.model_dump() for c in contexts],
            "count": len(contexts),
        }
    except Exception as exc:
        logger.exception("retrieval_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        ) from exc


# --------------------------------------------------------------------------
# Health & observability
# --------------------------------------------------------------------------


@router.get("/health", response_model=HealthStatus, summary="Liveness and dependency health")
async def health() -> HealthStatus:
    """
    Health check.

    Reports `ok` when at least one LLM provider is reachable, `degraded` when
    the app is up but no provider is available. Fronted by Docker/K8s
    liveness probes.
    """
    providers = await registry.health()
    any_up = any(p["reachable"] for p in providers.values())

    return HealthStatus(
        status="ok" if any_up else "degraded",
        version=settings.API_VERSION,
        environment=settings.ENVIRONMENT,
        providers=providers,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@router.get("/health/ready", summary="Readiness probe")
async def readiness() -> Dict[str, Any]:
    """Readiness probe: verifies the knowledge index is loaded."""
    try:
        retrieval_service.initialize()
        stats = retrieval_service.stats()
        return {"ready": True, "vector_store": stats}
    except Exception as exc:
        return {"ready": False, "error": str(exc)}


@router.get("/metrics", summary="Application metrics (JSON)")
async def get_metrics() -> Dict[str, Any]:
    """
    Full metrics snapshot: LLM calls, latency, cost, cache, RAG, agents.
    """
    return {
        "timestamp": time.time(),
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "metrics": metrics.snapshot(),
        "cache": get_cache().stats(),
        "vector_store": retrieval_service.stats(),
    }


@router.get(
    "/metrics/prometheus",
    summary="Metrics in Prometheus format",
    response_class=PlainTextResponse,
)
async def get_metrics_prometheus() -> str:
    """Prometheus text exposition, scrapeable by a standard Prometheus server."""
    return metrics.render_prometheus()


@router.get("/traces", summary="Recent trace spans")
async def get_traces(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    """Recent spans, for debugging pipeline latency."""
    spans = tracer.recent(limit)
    return {
        "count": len(spans),
        "spans": [
            {
                "name": s.name,
                "trace_id": s.trace_id,
                "duration_ms": round(s.duration_ms, 2),
                "status": s.status,
                "attributes": s.attributes,
            }
            for s in spans
        ],
    }


# --------------------------------------------------------------------------
# Models (kept for backward compatibility with the original API)
# --------------------------------------------------------------------------


@router.get("/models", summary="List available LLM models")
async def get_models() -> Dict[str, Any]:
    """
    List available models.

    Backward-compatible with the original endpoint, now backed by the
    provider registry so the result reflects live availability.
    """
    providers = registry.all()
    if not providers:
        return {"models": [settings.DEFAULT_MODEL]}

    models = [getattr(p, "model", settings.DEFAULT_MODEL) for p in providers.values()]
    fallback = ["gemma2:2b", "qwen2.5:0.5b", "deepseek-r1:1.5b", "deepseek-coder:latest"]
    return {"models": models or fallback}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@router.post(
    "/evaluate/retrieval",
    summary="Evaluate RAG retrieval quality",
    dependencies=protected,
)
async def evaluate_retrieval() -> Dict[str, Any]:
    """
    Run retrieval evaluation over the golden dataset.

    Reports hit rate, MRR, and precision so retrieval changes can be
    regression-tested.
    """
    try:
        return await evaluation_harness.evaluate_retrieval()
    except Exception as exc:
        logger.exception("eval_retrieval_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval evaluation failed: {exc}",
        ) from exc


@router.post(
    "/evaluate/generation",
    summary="Evaluate end-to-end generation quality (LLM-as-judge)",
    dependencies=protected,
)
async def evaluate_generation(
    num_cases: int = Query(2, ge=1, le=5),
    run_generation: bool = Query(True),
) -> Dict[str, Any]:
    """
    Score generated designs with an LLM judge.

    Produces quality scores on a 0-10 rubric plus latency and cost. This is
    the headline quality metric for the system.
    """
    try:
        return await evaluation_harness.evaluate_generation(
            num_cases=num_cases, run_generation=run_generation
        )
    except Exception as exc:
        logger.exception("eval_generation_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation evaluation failed: {exc}",
        ) from exc


# --------------------------------------------------------------------------
# Cache management
# --------------------------------------------------------------------------


@router.post("/cache/clear", summary="Clear the semantic cache", dependencies=protected)
async def clear_cache() -> Dict[str, Any]:
    """Flush the semantic response cache (admin operation)."""
    get_cache().clear()
    logger.info("cache_cleared_by_request")
    return {"status": "cleared"}


@router.get("/cache/stats", summary="Semantic cache statistics")
async def cache_stats() -> Dict[str, Any]:
    """Cache hit rate, size, and configuration."""
    return get_cache().stats()
