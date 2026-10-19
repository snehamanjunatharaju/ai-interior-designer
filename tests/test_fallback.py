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
    def test_reported_bug_brief_produces_a_priced_design(self):
        """
        The exact brief from the bug report must not come back at Rs 0.

        Validity is deliberately not asserted: with a Rs 10,000 cap and a
        bunk bed at Rs 14,500 this brief is genuinely unaffordable, and the
        correct outcome is an honest shortfall rather than a silent overspend.
        """
        design, cost, report, brief = _run(
            client_preferences=(
                "Two kids share a room, need 2 beds, a study desk and a wardrobe."
            ),
            room_type="kids_room", color_scheme="vibrant", budget_amount_inr=10000,
        )
        assert design["furniture"], "fallback must select real catalog items"
        assert cost["estimated_cost_inr"] > 0
        assert cost["costing_complete"] is True
        assert all(item.get("sku") for item in design["furniture"])

    def test_satisfies_requested_item_counts(self):
        """
        A bunk bed sleeps two, so one unit satisfies a two-child brief. Buying
        it once is both correct and what keeps this inside Rs 10,000.
        """
        _, _, report, brief = _run(
            client_preferences="Two kids share a room, need 2 beds and a wardrobe.",
            room_type="kids_room", color_scheme="vibrant", budget_amount_inr=10000,
        )
        assert brief.required_items.get("bed") == 2
        codes = {v.code for v in report.violations}
        assert "missing_items" not in codes, codes

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

    def test_over_budget_is_now_a_failure_not_a_warning(self):
        """
        This used to assert `is_valid is True` alongside an over-budget cost -
        which is precisely the bug: a green "Pass 7/7" badge rendered directly
        above a budget warning. An over-budget design must now FAIL, so the
        badge and the warning cannot disagree.

        Driven through `validate_design` directly, because the fallback is now
        budget-aware and will no longer produce an over-budget design on its
        own - which is the point of the fix.
        """
        from app.engine.pricing import price_design
        from app.engine.validator import validate_design

        _, _, _, brief = _run(
            client_preferences="2 beds, a wardrobe, a desk and a TV unit.",
            room_type="kids_room", color_scheme="vibrant", budget_amount_inr=2000,
        )
        over_budget_design = {"furniture": [
            {"name": "Kids Bunk Bed", "sku": "BED-KID-BUNK-01",
             "category": "bed", "quantity": 2},
            {"name": "Wardrobe", "sku": "WRD-2D-01",
             "category": "wardrobe", "quantity": 1},
        ]}
        cost = price_design(over_budget_design, budget_inr=brief.budget_inr)
        report = validate_design(over_budget_design, brief, cost)
        over = [v for v in report.violations if v.code == "over_budget"]
        assert over, "an over-budget design must be flagged"
        assert over[0].severity == "error", "over-budget is an error, not a warning"
        assert report.is_valid is False, "the badge must not be able to read green"

    def test_unaffordable_fallback_respects_the_budget(self):
        """
        The fallback used to ignore the budget and ship the full wish list,
        overage included, so the design a user saw by default was always
        invalid. It now buys only what the budget actually covers.

        At Rs 2,000 no real catalog product in this brief is affordable, so the
        correct result is an empty room plus an explicit infeasibility record -
        not furniture the client cannot buy.
        """
        design, cost, _, _ = _run(
            client_preferences="2 beds, a wardrobe, a desk and a TV unit.",
            room_type="kids_room", color_scheme="vibrant", budget_amount_inr=2000,
        )
        assert cost["estimated_cost_inr"] <= 2000
        assert cost["within_budget"] is True
        assert design["furniture"] == [], "nothing is affordable at Rs 2,000"
        infeasible = design.get("_budget_infeasible")
        assert infeasible, "the brief is unaffordable and must say so"
        assert infeasible["budget_inr"] == 2000
        assert infeasible["shortfall_inr"] > 0

    def test_partial_budget_buys_the_essentials_it_can_afford(self):
        """
        The useful middle case: enough budget for a cot, not enough for the
        bunk bed the brief asked for. Buy what fits, report the gap.
        """
        design, cost, report, _ = _run(
            client_preferences="A kids bed and a wardrobe.",
            room_type="kids_room", color_scheme="vibrant", budget_amount_inr=9000,
        )
        assert cost["within_budget"] is True
        assert design["furniture"], "Rs 9,000 does buy a bed"
        assert cost["estimated_cost_inr"] <= 9000
        codes = {v.code for v in report.violations}
        assert "missing_items" not in codes, codes

    def test_does_not_leak_internal_implementation_notes(self):
        """Fallback internals belong in logs, not client-facing copy."""
        design, _, _, _ = _run(
            client_preferences="A kids room for two.",
            room_type="kids_room", color_scheme="vibrant", budget_amount_inr=10000,
        )
        client_copy = " ".join(
            design.get(key, "") if isinstance(design.get(key), str)
            else " ".join(design.get(key, []))
            for key in ("concept", "layout_description", "cons", "trade_offs")
        ).lower()
        assert "assembled by code" not in client_copy
        assert "fallback path" not in client_copy

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
