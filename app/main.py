"""
FastAPI application factory.

Builds the app with lifespan-managed resources, middleware, exception
handlers, and the versioned API router. Using a factory (rather than a
module-level singleton) makes the app testable and importable-safe.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.config import settings
from app.core.observability import get_logger, metrics, setup_logging
from app.services.llm_provider import registry
from app.services.retrieval import retrieval_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Startup/shutdown lifecycle.

    Warms the retrieval index and probes providers on boot so the first user
    request does not pay the cold-start cost.
    """
    setup_logging()
    logger.info(
        "starting app=%s env=%s version=%s",
        settings.APP_NAME, settings.ENVIRONMENT, settings.API_VERSION,
    )

    try:
        retrieval_service.initialize()
        logger.info("retrieval_index_ready stats=%s", retrieval_service.stats())
    except Exception as exc:
        # Never block startup on the index; retrieval degrades gracefully.
        logger.error("retrieval_init_failed error=%s", exc)

    # Warm the price catalog here rather than on the first request. Without
    # this the first /layout call pays the seeding cost, and a read-only
    # data directory would turn that into a 500 on the client's first click
    # instead of a clear startup warning.
    try:
        from app.catalog import catalog

        catalog.initialize()
        logger.info(
            "catalog_ready items=%d version=%s", catalog.count(), catalog.version()
        )
    except Exception as exc:
        logger.error("catalog_init_failed error=%s", exc)

    provider_health = await registry.health()
    up = [n for n, h in provider_health.items() if h["reachable"]]
    logger.info("providers_ready reachable=%s all=%s", up, list(provider_health))

    # Resolve which model this machine can actually run. A healthy Ollama says
    # nothing about whether the requested model is pulled, and discovering that
    # on the user's first request turns a configuration preference into a 500.
    # Resolving at startup logs the downgrade instead.
    try:
        chosen = await registry.use_best_available_model()
        if chosen != settings.DEFAULT_MODEL:
            logger.warning(
                "model_downgraded requested=%s using=%s (preferred model not "
                "available locally)", settings.DEFAULT_MODEL, chosen,
            )
    except Exception as exc:
        logger.warning("model_resolution_failed error=%s", exc)

    yield

    logger.info("shutting_down app=%s", settings.APP_NAME)


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    app = FastAPI(
        title=f"{settings.APP_NAME} - AI Interior Design Platform",
        description=(
            "A production-grade, agentic interior design platform. Combines a "
            "multi-agent LLM pipeline, retrieval-augmented generation, semantic "
            "caching, and full observability to produce detailed, budget-accurate "
            "interior design proposals for the Indian market."
        ),
        version=settings.API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # --- Middleware (order matters: CORS outermost) ---
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    # --- Request/response instrumentation ---
    @app.middleware("http")
    async def instrument(request: Request, call_next):
        """Attach a request id, timing, and metrics to every HTTP call."""
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()

        metrics.increment("http.requests", labels=f"{request.method} {request.url.path}")
        metrics.increment("http.requests_by_method", labels=request.method)

        try:
            response = await call_next(request)
        except Exception:
            duration = (time.perf_counter() - start) * 1000
            metrics.increment("http.errors", labels=request.url.path)
            metrics.observe("http.latency_ms", duration)
            logger.exception("request_failed path=%s request_id=%s", request.url.path, request_id)
            raise

        duration = (time.perf_counter() - start) * 1000
        metrics.observe("http.latency_ms", duration)
        metrics.increment(
            "http.responses", labels=f"{response.status_code} {request.url.path}"
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration:.2f}ms"

        logger.info(
            "%s %s -> %d in %.1fms request_id=%s",
            request.method, request.url.path, response.status_code, duration, request_id,
        )
        return response

    # --- Exception handlers ---
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Return structured JSON for any unhandled error, never a stack trace."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("unhandled_exception path=%s request_id=%s", request.url.path, request_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
                "request_id": request_id,
            },
        )

    # --- Routers ---
    app.include_router(router, prefix="/api/v1")
    app.include_router(router, prefix="", include_in_schema=False)  # legacy paths

    # --- Web UI ---
    templates = Jinja2Templates(directory="templates")
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/", include_in_schema=False)
    async def home(request: Request):
        """Serve the web UI."""
        return templates.TemplateResponse(request, "index.html")

    return app


app = create_app()
