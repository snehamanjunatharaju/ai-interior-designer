"""
Tests for the architect's fallback path.

When the 2B model fails or returns unusable JSON, the pipeline must still
return a usable design. The original stub returned an empty `furniture` list
and the literal string "Layout derived from the design plan", which meant a
90-second request ended in a Rs 0 design that the validator then rejected -
the user got nothing at all.

These tests pin the contract: a fallback is priced, satisfies the brief, and
contains no template sentinel.
"""

from __future__ import annotations

import pytest

from app.agents.orchestrator import ArchitectAgent
from app.catalog import catalog
from app.core.schemas import DesignRequest
from app.engine.brief import parse_brief
from app.engine.pricing import price_design
from app.engine.validator import validate_design


@pytest.fixture(scope="module", autouse=True)
def ready():
    catalog.initialize()
    return catalog


def _run(**kw) -> tuple:
    request = DesignRequest(**kw)
    brief = parse_brief(
        client_preferences=request.client_preferences,
        furniture=request.furniture,
        specific_requirements=request.specific_requirements,
        occupants=request.occupants,
        room_type=request.room_type.value,
        color_scheme=request.color_scheme,
        budget_inr=request.budget_amount_inr,
        room_dimensions=request.room_dimensions or "",
    )
    design = ArchitectAgent()._fallback_design(request, {}, brief)
    cost = price_design(design, budget_inr=brief.budget_inr)
    return design, cost, validate_design(design, brief, cost), brief


class TestFallback:
    def test_reported_bug_brief_produces_a_priced_valid_design(self):
        """The exact brief from the bug report must not come back at Rs 0."""
        design, cost, report, brief = _run(
            client_preferences=(
                "Two kids share a room, need 2 beds, a study desk and a wardrobe."
            ),
            room_type="kids_room", color_scheme="vibrant", budget_amount_inr=10000,
        )
        assert design["furniture"], "fallback must select real catalog items"
        assert cost["estimated_cost_inr"] > 0
        assert report.is_valid, [v.to_dict() for v in report.errors]

    def test_satisfies_requested_item_counts(self):
        _, _, report, brief = _run(
            client_preferences="Two kids share a room, need 2 beds and a wardrobe.",
            room_type="kids_room", color_scheme="vibrant", budget_amount_inr=10000,
        )
        assert brief.required_items.get("bed") == 2
        assert "missing_items" not in {v.code for v in report.violations}

    def test_vague_brief_still_returns_furniture(self):
        design, cost, _, _ = _run(
            client_preferences="A calm bedroom to unwind in.",
            room_type="bedroom",
        )
        assert design["furniture"]
        assert cost["estimated_cost_inr"] > 0

    def test_every_item_has_a_real_sku(self):
        design, _, _, _ = _run(
            client_preferences="A bright living room with a sofa and a rug.",
            room_type="living_room", color_scheme="vibrant",
            budget_amount_inr=50000,
        )
        for item in design["furniture"]:
            assert item.get("sku")
            assert catalog.get(item["sku"]) is not None

    def test_no_placeholder_sentinel(self):
        """The original bug: the stub shipped a template string to clients."""
        design, _, report, _ = _run(
            client_preferences="A kids room for two.",
            room_type="kids_room", color_scheme="vibrant", budget_amount_inr=10000,
        )
        blob = " ".join([
            design["title"], design["concept"], design["layout_description"],
        ]).lower()
        assert "derived from the design plan" not in blob
        assert "placeholder_text" not in {v.code for v in report.violations}

    def test_vibrant_brief_gets_a_vibrant_palette(self):
        """The fallback must satisfy the same colour check as a written one."""
        for scheme in ("vibrant", "cool", "earthy", "pastel", "monochrome", "neutral"):
            design, _, report, _ = _run(
                client_preferences="A comfortable room for the family.",
                room_type="living_room", color_scheme=scheme, budget_amount_inr=60000,
            )
            codes = {v.code for v in report.violations}
            assert "color_scheme_mismatch" not in codes, f"{scheme} palette rejected"

    def test_excluded_items_are_omitted(self):
        design, _, _, brief = _run(
            client_preferences="One bed and a wardrobe, no mirror, avoid beige.",
            room_type="bedroom", color_scheme="vibrant", budget_amount_inr=50000,
        )
        assert "mirror" in brief.excluded_items
        categories = {i.get("category") for i in design["furniture"]}
        assert "mirror" not in categories

    def test_over_budget_is_flagged_but_still_valid(self):
        """Honest about the overage without failing outright."""
        _, cost, report, _ = _run(
            client_preferences="2 beds, a wardrobe, a desk and a TV unit.",
            room_type="kids_room", color_scheme="vibrant", budget_amount_inr=2000,
        )
        assert cost["within_budget"] is False
        assert report.is_valid

    def test_is_honest_about_being_generic(self):
        """The user should know the prose came from the fallback."""
        design, _, _, _ = _run(
            client_preferences="A kids room for two.",
            room_type="kids_room", color_scheme="vibrant", budget_amount_inr=10000,
        )
        assert design["cons"], "a fallback design must state its own limitation"

    def test_deterministic(self):
        a, _, _, _ = _run(
            client_preferences="Two kids share a room, need 2 beds.",
            room_type="kids_room", color_scheme="vibrant", budget_amount_inr=10000,
        )
        b, _, _, _ = _run(
            client_preferences="Two kids share a room, need 2 beds.",
            room_type="kids_room", color_scheme="vibrant", budget_amount_inr=10000,
        )
        assert a == b
