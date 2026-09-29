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
    "/layout",
    summary="Compute a 2D floor plan for a brief",
    description=(
        "Places every catalog item in the room using real centimetre "
        "dimensions, honouring 60 cm circulation clearance, door swing, and "
        "collision rules. Returns SVG-ready JSON plus a derived layout score."
    ),
    dependencies=protected,
)
async def compute_layout(payload: DesignRequest) -> Dict[str, Any]:
    """
    Build a floor plan without running the design pipeline.

    Kept separate from /generate so the layout can be recomputed cheaply
    while a client iterates on dimensions - this is a pure function of the
    brief and the catalog, so there is no reason to pay for an LLM call.
    """
    with tracer.start("api.layout", room=payload.room_type.value):
        from app.catalog import catalog
        from app.engine.brief import find_products, parse_brief
        from app.engine.layout import place_furniture
        from app.engine.pricing import price_design

        constraints = parse_brief(
            client_preferences=payload.client_preferences,
            furniture=payload.furniture,
            specific_requirements=payload.specific_requirements,
            occupants=payload.occupants,
            room_type=payload.room_type.value,
            style=payload.style.value,
            color_scheme=payload.color_scheme,
            budget_inr=payload.budget_amount_inr,
            vastu_compliance=payload.vastu_compliance.value,
            room_dimensions=payload.room_dimensions or "",
        )

        dims = constraints.dimensions
        # Without dimensions a plan cannot be drawn honestly, so say so rather
        # than inventing a room size.
        if not dims.is_known():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "Room dimensions are required for a floor plan. "
                    "Supply room_dimensions as e.g. '12ft x 10ft'."
                ),
            )

        # Which items to place: the brief's required items plus anything
        # else the client listed. Catalog provides the real dimensions.
        wanted = dict(constraints.required_items)
        for item in constraints.optional_items:
            wanted.setdefault(item, 1)

        line_items: List[Dict[str, Any]] = []
        for category, qty in wanted.items():
            # Resolved through the brief->catalog mapping so "sofa" finds the
            # seating products rather than reporting the category as empty.
            options = find_products(category)
            if not options:
                continue
            product = options[0]
            line_items.append({
                "sku": product.sku, "name": product.name, "category": category,
                "width_cm": product.width_cm, "depth_cm": product.depth_cm,
                "height_cm": product.height_cm, "quantity": qty,
            })

        # Direction fields on the form are authoritative and arrive outside
        # the brief text, so they are read straight from the request. The
        # brief parser is only the fallback for prose like "the door is north".
        door = payload.door_direction or constraints.door_direction
        window_dir = payload.window_direction or constraints.window_direction

        plan = place_furniture(
            line_items,
            room_width_cm=dims.width_cm,
            room_depth_cm=dims.length_cm,
            ceiling_cm=dims.ceiling_cm or 270,
            door_direction=door,
            window_direction=window_dir,
        )

        costing = price_design(line_items, budget_inr=constraints.budget_inr)
        result = plan.to_dict()
        result["room"]["door_direction"] = door
        result["room"]["window_direction"] = window_dir
        result["cost"] = {
            "total_inr": costing["estimated_cost_inr"],
            "budget_inr": constraints.budget_inr,
            "within_budget": costing["within_budget"],
        }
        result["constraints"] = constraints.to_dict()
        metrics.increment("api.layout")
        return result


@router.post(
    "/proposal.pdf",
    summary="Download a client proposal as PDF",
    description=(
        "Renders the design, floor plan, shopping list, phased budget plan, "
        "Vastu assessment, and trade-offs into a client-facing PDF. All "
        "figures come from the code-computed costing, so the document cannot "
        "disagree with the screen."
    ),
)
async def proposal_pdf(payload: Dict[str, Any]) -> Any:
    """Build the PDF client proposal."""
    with tracer.start("api.proposal_pdf"):
        from fastapi.responses import Response

        from app.services.proposal import build_proposal_pdf

        design = payload.get("design")
        if not isinstance(design, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Send the current design as `design`.",
            )

        pdf = build_proposal_pdf(
            design=design,
            cost=payload.get("cost"),
            layout=payload.get("layout"),
            validation=payload.get("validation"),
            vastu=payload.get("vastu"),
            budget_plan=payload.get("budget_plan"),
            client_name=str(payload.get("client_name", "Client")),
            room_type=str(payload.get("room_type", "")),
        )
        metrics.increment("api.proposal_pdf")
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="interiorai-proposal.pdf"'
            },
        )


@router.post(
    "/refine",
    summary="Refine one part of an existing design",
    description=(
        "Parses a plain-English change request, applies it to the targeted "
        "part only, re-prices in code, and re-validates. Everything outside "
        "the target is returned byte-for-byte unchanged."
    ),
    dependencies=protected,
)
async def refine_design(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Change one thing, leave the rest alone.

    Takes the design dict the client already has, so no LLM call is needed to
    re-run the pipeline. That is the point: a full regeneration would rewrite
    decisions the client already approved.
    """
    with tracer.start("api.refine"):
        from app.engine.pricing import build_phased_plan, price_design
        from app.engine.refine import (
            apply_refinement, classify_refinement, describe_change,
        )

        design = payload.get("design")
        if not isinstance(design, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Send the current design as `design`.",
            )
        request_text = str(payload.get("request", "")).strip()
        if not request_text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Send a `request` describing what to change.",
            )

        intent = classify_refinement(request_text)
        if intent.target == "unknown":
            return {
                "understood": False,
                "message": describe_change(intent),
                "intent": intent.to_dict(),
            }

        before = json.loads(json.dumps(design))
        updated = apply_refinement(dict(design), intent)

        budget = payload.get("budget_inr")
        cost = price_design(updated, budget_inr=budget)
        plan = (
            build_phased_plan(cost["line_items"], budget)
            if budget and not cost["within_budget"] else None
        )

        # Report exactly which fields moved, so the client can see the blast
        # radius of their own request rather than having to diff two blobs.
        changed_fields = [
            k for k in updated
            if k != "_refinement" and updated.get(k) != before.get(k)
        ]

        return {
            "understood": True,
            "message": describe_change(intent),
            "intent": intent.to_dict(),
            "changed_fields": changed_fields,
            "design": updated,
            "cost": cost,
            "budget_plan": plan,
        }


@router.post(
    "/vastu",
    summary="Score a layout against rule-based Vastu",
    description=(
        "Evaluates each Vastu rule against real layout geometry and returns a "
        "weighted score with a plain explanation and remedy per rule."
    ),
    dependencies=protected,
)
async def score_vastu(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based Vastu assessment. No LLM, fully reproducible."""
    with tracer.start("api.vastu"):
        from app.engine.vastu import evaluate_vastu

        items = payload.get("items")
        if not isinstance(items, list):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Send layout `items` (from /api/v1/layout) to assess.",
            )
        report = evaluate_vastu(
            items,
            room_type=str(payload.get("room_type", "")),
            door_direction=payload.get("door_direction"),
            window_direction=payload.get("window_direction"),
            compliance=str(payload.get("compliance", "soft")),
        )
        return report.to_dict()


@router.get(
    "/evaluation",
    summary="Evaluation dashboard data",
    description="Aggregate pass rates, latency, and cache behaviour for the dashboard.",
)
async def evaluation_dashboard() -> Dict[str, Any]:
    """
    Report the system's own behaviour, not a model's opinion of it.

    Constraint pass rate and latency are measured from real traffic, which
    makes them the numbers worth showing: they cannot be talked up by a
    language model.
    """
    with tracer.start("api.evaluation"):
        from app.catalog import catalog

        snap = metrics.snapshot()
        counters = snap.get("counters", {})
        gauges = snap.get("gauges", {})
        hists = snap.get("histograms", {})

        def counter(name: str) -> int:
            return int(counters.get(name, {}).get("value", 0))

        def gauge(name: str):
            return gauges.get(name)

        def hist_mean(name: str):
            h = hists.get(name)
            if not h or not h.get("count"):
                return None
            return round(h["sum"] / h["count"], 1)

        passed = counter("validator.passed_designs")
        failed = counter("validator.failed_designs")
        total = passed + failed
        validation = {
            "checks_run": counter("validator.checks_run"),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total, 3) if total else None,
        }

        return {
            "validation": validation,
            "latency": {
                "pipeline_mean_ms": hist_mean("pipeline.latency_ms"),
                "agents_mean_ms": {
                    agent: hist_mean(f"agent.{agent}.ms")
                    for agent in ("planner", "architect", "budget", "critic")
                },
            },
            "throughput": {
                "pipeline_completed": counter("pipeline.completed"),
                "revisions_requested": counter("critic.revision_requested"),
                "layouts_built": counter("layout.built"),
                "refinements_applied": counter("refinement.applied"),
                "refinements_no_op": counter("refinement.no_op"),
            },
            "quality": {
                "critic_mean_score": hist_mean("critic.overall_score"),
                "layout_score": gauge("layout.score"),
                "vastu_score": gauge("vastu.score"),
            },
            "economics": {
                "catalog_items": catalog.count(),
                "catalog_version": catalog.version(),
                "unpriced_items": counter("pricing.unpriced_items"),
                "inserted_required_items": counter("design.inserted_required_item"),
                "over_budget_designs": counter("pricing.over_budget"),
            },
            "limits": {
                "embedding_model": settings.EMBEDDING_MODEL,
                "rag_min_score": settings.RAG_MIN_SCORE,
                "max_reflection_rounds": settings.MAX_REFLECTION_ROUNDS,
                "llm_providers": settings.LLM_PROVIDERS,
            },
        }


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
