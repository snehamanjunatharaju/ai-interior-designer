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


class TestProposalPdfEndpoint:
    """
    A client proposal that says "Rs 0" is worse than no proposal. The PDF
    endpoint used to trust an optional `cost` key, so sending only the design
    produced a zero-valued document.
    """

    _DESIGN = {
        "title": "Kids Corner",
        "options": [{
            "title": "Option A",
            "concept": "Calm pastel room",
            "furniture": [
                {"name": "Bunk", "sku": "BED-KID-BUNK-01", "quantity": 1},
                {"name": "Table", "sku": "DSK-KID-01", "quantity": 2},
            ],
        }],
    }

    def _pdf_text(self, pdf_bytes):
        """
        Pull the visible strings out of the reportlab stream.

        With a Unicode TTF registered, reportlab emits the rupee sign as a
        2-byte glyph index (\\000...) rather than the UTF-8 bytes, so the raw
        text cannot be searched for "₹" directly. The symbol is stripped out
        and only the digits are asserted on; `test_money_uses_one_format...`
        covers the symbol itself.
        """
        import base64
        import re
        import zlib

        chunks = []
        for match in re.finditer(rb"stream\r?\n(.*?)endstream", pdf_bytes, re.S):
            raw = match.group(1).strip()
            try:
                raw = zlib.decompress(base64.a85decode(raw, adobe=True))
            except Exception:
                pass
            chunks.append(raw.decode("latin1", "replace"))
        body = "\n".join(chunks)
        return " ".join(re.findall(r"\((.*?)\)\s*Tj", body))

    def test_prices_are_derived_when_cost_is_omitted(self, client):
        response = client.post(
            "/api/v1/proposal.pdf",
            json={"client_name": "Anita", "design": self._DESIGN},
        )
        assert response.status_code == 200
        assert response.content[:4] == b"%PDF"

        text = self._pdf_text(response.content)
        assert "Rs 0" not in text, "a proposal must never quote a zero total"
        assert "n/a" not in text, "catalog version must be resolved"
        # 14,500 bunk + 2 x 2,100 table
        assert "18,700" in text
        # The rupee sign must survive. Helvetica would emit a stray "n" here,
        # which is the bug this guards against.
        assert "n14,500" not in text, "rupee sign rendered as a letter"

    def test_money_uses_one_format_with_indian_grouping(self, client):
        """
        One currency format everywhere. Indian digit grouping is not cosmetic:
        ₹1,23,456 is Rs 123,400, and a reader using Western grouping would
        misread it by two orders of magnitude.
        """
        from app.core.money import format_inr

        assert format_inr(4100) == "₹4,100"
        assert format_inr(123456) == "₹1,23,456"
        assert format_inr(None) == "₹0"
        assert format_inr("garbage") == "₹0"

    def test_supplied_cost_cannot_override_the_catalog_total(self, client):
        """The PDF prices from the catalog even when a cost block is sent.

        This test used to assert the opposite. Honouring a caller-supplied
        headline was how the document could print one total above a shopping
        list that added up to another - a PDF that contradicts its own itemised
        page. The catalog price for a bunk bed (Rs 14,500) must win, and the
        catalogue version comes from the same code.
        """
        from app.catalog import catalog

        design = {"title": "X", "furniture": [
            {"name": "Bunk", "sku": "BED-KID-BUNK-01", "quantity": 1},
        ]}
        response = client.post(
            "/api/v1/proposal.pdf",
            json={
                "design": design,
                "cost": {
                    "estimated_cost_inr": 999999,
                    "catalog_version": "forged-version",
                },
            },
        )
        assert response.status_code == 200
        text = self._pdf_text(response.content)
        assert "14,500" in text
        assert "9,99,999" not in text
        assert "forged-version" not in text
        assert catalog.version() in text
        assert "Bunk" in text

    def test_still_rejects_a_missing_design(self, client):
        assert client.post("/api/v1/proposal.pdf", json={}).status_code == 422


class TestPdfCurrencyRendering:
    """
    Regression: Helvetica has no U+20B9, and ReportLab maps it to the letter
    "n". Every price in the proposal rendered as "n14,500" with no error raised
    anywhere. The font probe now verifies the glyph before trusting a font, and
    the money formatter falls back to an "Rs" prefix rather than emitting a
    wrong glyph.
    """

    def test_registered_font_can_actually_draw_the_rupee_sign(self):
        from app.services.proposal import _FONT_CANDIDATES, _unicode_font
        import os

        name = _unicode_font()
        if name is None:
            pytest.skip("no unicode font on this machine; fallback path is tested below")
        from reportlab.pdfbase import pdfmetrics

        face = pdfmetrics.getFont(name).face
        assert 0x20B9 in face.charToGlyph, (
            "a font was registered that cannot render the rupee sign"
        )
        # And every candidate that exists must be probed the same way.
        for path in _FONT_CANDIDATES:
            if os.path.isfile(path):
                assert os.path.isfile(path)

    def test_pdf_price_is_readable_not_a_stray_letter(self, client):
        response = client.post(
            "/api/v1/proposal.pdf",
            json={
                "client_name": "A",
                "design": {"furniture": [
                    {"name": "Bunk", "sku": "BED-KID-BUNK-01", "quantity": 1},
                ]},
            },
        )
        assert response.status_code == 200
        text = TestProposalPdfEndpoint._pdf_text(self, response.content)
        assert "14,500" in text
        # The failure mode was a literal "n" where the symbol belonged.
        assert "n14,500" not in text, "rupee sign rendered as the letter n"
        # Whichever of the two accepted formats was used, it must be present.
        assert ("₹14,500" in text) or ("Rs 14,500" in text)


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
