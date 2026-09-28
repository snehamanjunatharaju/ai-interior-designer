"""
API-level tests using FastAPI's TestClient.

LLM calls are stubbed so the HTTP contract can be verified offline.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"ok", "degraded"}
        assert "providers" in body
        assert "uptime_seconds" in body

    def test_readiness(self, client):
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True

    def test_root_serves_ui(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "InteriorAI" in response.text

    def test_openapi_available(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "/api/v1/generate" in response.json()["paths"]


class TestObservabilityEndpoints:
    def test_metrics_json(self, client):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        body = response.json()
        assert "metrics" in body and "cache" in body and "vector_store" in body

    def test_prometheus_format(self, client):
        response = client.get("/api/v1/metrics/prometheus")
        assert response.status_code == 200
        assert "# TYPE" in response.text

    def test_request_id_header_returned(self, client):
        response = client.get("/api/v1/health")
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers

    def test_request_id_echoed(self, client):
        response = client.get("/api/v1/health", headers={"X-Request-ID": "abc123"})
        assert response.headers["X-Request-ID"] == "abc123"

    def test_cache_stats(self, client):
        response = client.get("/api/v1/cache/stats")
        assert response.status_code == 200
        assert "hit_rate" in response.json()

    def test_traces_endpoint(self, client):
        response = client.get("/api/v1/traces")
        assert response.status_code == 200
        assert "spans" in response.json()


class TestRetrievalEndpoint:
    def test_retrieve_returns_results(self, client):
        response = client.post(
            "/api/v1/retrieve", json={"query": "small budget living room", "top_k": 3}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["count"] >= 1
        assert "content" in body["results"][0]
        assert "score" in body["results"][0]

    def test_retrieve_rejects_short_query(self, client):
        response = client.post("/api/v1/retrieve", json={"query": "a"})
        assert response.status_code == 422


class TestModelsEndpoint:
    def test_lists_models(self, client):
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        assert isinstance(response.json()["models"], list)
        assert response.json()["models"]


class TestValidation:
    def test_rejects_invalid_room_type(self, client):
        response = client.post(
            "/api/v1/generate", json={"client_preferences": "x", "room_type": "spaceship"}
        )
        assert response.status_code == 422

    def test_rejects_missing_brief(self, client):
        response = client.post("/api/v1/generate", json={})
        assert response.status_code == 422

    def test_rejects_too_many_options(self, client):
        response = client.post(
            "/api/v1/generate", json={"client_preferences": "x", "num_options": 99}
        )
        assert response.status_code == 422


class TestStreamingEndpoint:
    def test_stream_emits_sse(self, client, monkeypatch):
        """
        The streaming contract must hold even when the pipeline is stubbed:
        events are framed correctly and the stream terminates cleanly.
        """
        from app.core.schemas import AgentStep

        async def fake_stream(request, **kwargs):
            yield {
                "type": "step",
                "data": AgentStep(
                    agent="planner", status="ok", message="planned"
                ).model_dump(),
            }
            yield {
                "type": "complete",
                "data": {
                    "request_id": "test",
                    "options": [],
                    "context_sources": [],
                    "total_cost_inr": 0,
                    "latency_ms": 1,
                    "cost_inr": 0,
                    "model_used": "test",
                    "cached": False,
                    "evaluation": None,
                },
            }

        from app.services.design_service import design_service

        monkeypatch.setattr(design_service, "stream", fake_stream)

        with client.stream(
            "POST", "/api/v1/generate/stream", json={"client_preferences": "test room"}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join(response.iter_text())

        assert "event: start" in body
        assert "event: step" in body
        assert "event: complete" in body
        assert "event: done" in body

        # Every data line must be valid JSON.
        for line in body.splitlines():
            if line.startswith("data:"):
                json.loads(line[5:].strip())


class TestAuthEnforcement:
    def test_requires_key_when_configured(self, client, monkeypatch):
        from app.core import security

        monkeypatch.setattr(security.settings, "API_KEY", "test-secret", raising=False)

        # No key supplied -> rejected.
        assert client.get("/api/v1/models").status_code == 200  # not protected
        response = client.post("/api/v1/retrieve", json={"query": "living room"})
        assert response.status_code == 401

        # Correct key supplied -> allowed.
        ok = client.post(
            "/api/v1/retrieve",
            json={"query": "living room"},
            headers={"X-API-Key": "test-secret"},
        )
        assert ok.status_code == 200

    def test_health_is_unprotected(self, client, monkeypatch):
        from app.core import security

        monkeypatch.setattr(security.settings, "API_KEY", "test-secret", raising=False)
        assert client.get("/api/v1/health").status_code == 200
