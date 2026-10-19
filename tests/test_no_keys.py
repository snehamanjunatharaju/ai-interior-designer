"""
Verify the app is fully usable with no API keys and no LLM at all.

The claim under test: a user with nothing configured still gets a priced,
brief-compliant design with a floor plan and a Vastu score. The provider
registry must report itself unavailable, and the pipeline must not raise.
"""
from __future__ import annotations

import pytest

from app.catalog import catalog
from app.engine.brief import parse_brief
from app.engine.layout import place_furniture
from app.engine.pricing import build_phased_plan, price_design
from app.engine.vastu import evaluate_vastu


@pytest.fixture(scope="module", autouse=True)
def ready():
    catalog.initialize()
    return catalog


class TestNoKeysRequired:
    def test_catalog_needs_no_key_or_network(self):
        """The catalog is seeded from source, so pricing always works."""
        assert catalog.count() >= 40
        assert catalog.get("BED-KID-COT-01").price_inr > 0

    def test_layout_needs_no_llm(self):
        plan = place_furniture(
            [{"sku": "BED-KID-COT-01", "name": "Kids Cot", "category": "bed",
              "width_cm": 90, "depth_cm": 190, "height_cm": 60, "quantity": 2}],
            305, 366, 270, door_direction="south", window_direction="east",
        )
        assert plan.items
        assert not plan.unplaced
        assert plan.to_dict()["items"]

    def test_vastu_needs_no_llm(self):
        report = evaluate_vastu(
            [{"name": "Bed", "category": "bed", "wall": "south",
              "x": 0, "y": 0, "width": 90, "depth": 190, "height": 60}],
            "bedroom", "south", "east", "strict",
        )
        assert 0 <= report.score <= 100

    def test_pipeline_falls_back_without_any_provider(self):
        """
        With every provider unreachable the pipeline must still return a
        design, not a 503. This is the path a keyless user hits whenever
        Ollama is down.
        """
        from app.agents.orchestrator import ArchitectAgent
        from app.engine.validator import validate_design
        from app.core.schemas import DesignRequest as DR

        request = DR(
            client_preferences="Two kids share a room, need 2 beds and a wardrobe.",
            room_type="kids_room",
            color_scheme="vibrant",
            budget_amount_inr=10_000,
            room_dimensions="12 x 10 ft",
            num_options=1,
        )
        brief = parse_brief(
            client_preferences=request.client_preferences,
            room_type=request.room_type.value,
            color_scheme=request.color_scheme,
            budget_inr=request.budget_amount_inr,
            room_dimensions=request.room_dimensions,
        )
        fallback = ArchitectAgent()._fallback_design(request, {}, brief)
        cost = price_design(fallback, budget_inr=brief.budget_inr)
        report = validate_design(fallback, brief, cost)

        assert fallback["furniture"], "a keyless user must still get products"
        assert cost["estimated_cost_inr"] > 0
        assert report.is_valid, [v.to_dict() for v in report.errors]
        assert fallback["cons"], "the fallback must admit its own limits"

    @pytest.mark.asyncio
    async def test_full_generation_returns_a_priced_strict_kitchen_without_provider(
        self, monkeypatch
    ):
        from app.agents.orchestrator import design_pipeline
        from app.core.schemas import DesignRequest as DR
        from app.services.llm_provider import registry

        async def unavailable(*args, **kwargs):
            raise RuntimeError("No LLM providers configured")

        monkeypatch.setattr(registry, "complete_json", unavailable)
        response = await design_pipeline.generate(
            DR(
                client_preferences="A practical family kitchen with storage",
                room_type="kitchen",
                room_dimensions="9 x 11 ft",
                color_scheme="warm",
                vastu_compliance="strict",
                budget_amount_inr=150_000,
                num_options=1,
            )
        )

        option = response.options[0]
        assert option.furniture
        assert all(item.sku for item in option.furniture)
        assert option.estimated_cost_inr > 0
        assert option.layout["room"]["width_ft"] == 9
        cooking = next(
            result for result in option.vastu_results
            if result["code"] == "cook_south_east"
        )
        assert cooking["passed"]
        assert "south-east" in cooking["explanation"]
        assert "east" in cooking["explanation"]

    def test_budget_negotiator_needs_no_llm(self):
        cost = price_design(
            {"furniture": [
                {"sku": "BED-KID-COT-01", "category": "bed", "quantity": 2},
                {"sku": "WRD-2D-01", "category": "wardrobe", "quantity": 1},
            ]},
            budget_inr=5_000,
        )
        plan = build_phased_plan(cost["line_items"], 5_000)
        assert plan["phase1"]["total_inr"] <= 5_000
        assert plan["headline"]

    def test_pdf_needs_no_key(self):
        from app.services.proposal import build_proposal_pdf
        design = {
            "title": "Keyless", "concept": "Works offline.",
            "furniture": [{"name": "Kids Cot", "category": "bed",
                           "sku": "BED-KID-COT-01", "quantity": 1}],
        }
        pdf = build_proposal_pdf(design=design, cost=price_design(design))
        assert pdf[:4] == b"%PDF"
