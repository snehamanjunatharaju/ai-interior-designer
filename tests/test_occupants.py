"""
Tests for B1 and B2: who the room is for, and what that forbids.

The failure these pin down was concrete: a brief that said "adults" was served
a Kids Single Cot and a Kids Chair, and nothing objected. Two things were
missing rather than one - selection filtered on audience, but the validator did
not, so a leak had nowhere to be caught.

Every test here asserts the *whole path*: the catalog lookup, the optimizer, and
the validator. Checking only one of them reproduces the original bug, because
the original bug was that the other two were silent.
"""

from __future__ import annotations

import pytest

from app.catalog import catalog
from app.engine.brief import parse_brief, products_satisfying
from app.engine.budget_optimizer import optimize_to_budget
from app.engine.occupants import (
    ADULT,
    ANY,
    KIDS,
    disallowed_products,
    filter_products,
    infer_audience,
    product_is_allowed,
    required_categories_for,
    requires_wfh_kit,
)
from app.engine.pricing import price_design
from app.engine.validator import validate_design

KIDS_SKUS = {
    "DSK-KID-01", "CHR-KID-01", "BED-KID-COT-01", "BED-KID-BUNK-01",
    "MAT-KID-01", "RUG-KID-01", "QUIL-KID-01", "CUR-KID-01",
    "CHV-STO-01", "DEC-DSL-01",
}


@pytest.fixture(scope="module", autouse=True)
def ready():
    catalog.initialize()
    return catalog


def _adult_brief(**kw):
    return parse_brief(
        client_preferences=kw.pop(
            "text", "A master bedroom for two adults"
        ),
        occupants=["adults", "adults"],
        room_type=kw.pop("room_type", "bedroom"),
        **kw,
    )


def _kids_brief(**kw):
    return parse_brief(
        client_preferences=kw.pop("text", "A room for two children"),
        occupants=["child", "child"],
        room_type=kw.pop("room_type", "kids_room"),
        **kw,
    )


class TestInferAudience:
    def test_explicit_adults_read_as_adult(self):
        assert infer_audience(["adults"]) == ADULT

    def test_explicit_children_read_as_kids(self):
        assert infer_audience(["child"]) == KIDS

    def test_free_text_is_read_too(self):
        assert infer_audience([], "a master bedroom for a couple") == ADULT
        assert infer_audience([], "a nursery for a toddler") == KIDS

    def test_child_in_mixed_occupants_makes_child_safe_items_eligible(self):
        assert infer_audience(["2 adults", "10-year-old child"]) == KIDS
        assert infer_audience(["adults", "child"], "shared family room") == KIDS

    def test_unstated_allows_everything(self):
        """No guess, no filtering - a missing occupant list must not reject."""
        assert infer_audience([], "") == ANY


class TestCatalogFiltering:
    def test_adults_never_see_kids_desks(self):
        found = products_satisfying("desk", audience=ADULT)
        assert found, "an adult desk must exist"
        assert all(p.audience != KIDS for p in found), [p.sku for p in found]

    def test_kids_rooms_can_see_kids_desks(self):
        found = products_satisfying("desk", audience=KIDS)
        assert any(p.sku == "DSK-KID-01" for p in found)

    def test_unstated_audience_filters_nothing(self):
        assert len(products_satisfying("desk", audience=ANY)) >= len(
            products_satisfying("desk", audience=KIDS)
        )

    def test_filter_and_disallowed_are_complements(self):
        pool = catalog.all()
        kept = filter_products(pool, ADULT)
        dropped = disallowed_products(pool, ADULT)
        assert len(kept) + len(dropped) == len(pool)
        assert not ({p.sku for p in kept} & {p.sku for p in dropped})

    def test_any_product_is_allowed_everywhere(self):
        assert product_is_allowed(catalog.get("WRD-2D-01"), ADULT)
        assert product_is_allowed(catalog.get("WRD-2D-01"), KIDS)


class TestOptimizerRespectsAudience:
    def test_no_kids_item_in_an_adult_selection(self):
        result = optimize_to_budget(
            required={"bed": 2, "desk": 1, "chair": 1, "storage": 1},
            budget_inr=400_000,
            style_tier="medium",
            audience=ADULT,
        )
        leaked = {s["sku"] for s in result["selections"]} & KIDS_SKUS
        assert not leaked, f"kids' furniture leaked into an adult room: {leaked}"

    def test_kids_selection_may_use_kids_items(self):
        result = optimize_to_budget(
            required={"bed": 2, "desk": 1},
            budget_inr=200_000,
            style_tier="essential",
            audience=KIDS,
        )
        assert result["selections"]


class TestValidatorRejectsAudienceLeak:
    """
    The missing half of the original bug. Selection filtered, but nothing
    checked, so a leak could still reach the client.
    """

    def test_kids_furniture_in_an_adult_room_fails(self):
        brief = _adult_brief()
        assert brief.audience == ADULT
        design = {"furniture": [
            {"name": "Kids Cot", "sku": "BED-KID-COT-01",
             "category": "bed", "quantity": 1},
            {"name": "Kids Chair", "sku": "CHR-KID-01",
             "category": "chair", "quantity": 1},
        ]}
        report = validate_design(
            design, brief, price_design(design, budget_inr=brief.budget_inr)
        )
        codes = {v.code for v in report.violations}
        assert "audience_mismatch" in codes, codes
        violation = next(v for v in report.violations if v.code == "audience_mismatch")
        assert violation.severity == "error"
        assert report.is_valid is False
        # The message must name the offending product.
        assert "Kids" in violation.message

    def test_adult_furniture_in_an_adult_room_is_fine(self):
        brief = _adult_brief()
        design = {"furniture": [
            {"name": "Queen Bed", "sku": "BED-DBL-QUEEN-01",
             "category": "bed", "quantity": 1},
        ]}
        report = validate_design(
            design, brief, price_design(design, budget_inr=brief.budget_inr)
        )
        assert "audience_mismatch" not in {v.code for v in report.violations}

    def test_kids_furniture_in_a_kids_room_is_fine(self):
        brief = _kids_brief()
        design = {"furniture": [
            {"name": "Kids Cot", "sku": "BED-KID-COT-01",
             "category": "bed", "quantity": 2},
        ]}
        report = validate_design(
            design, brief, price_design(design, budget_inr=brief.budget_inr)
        )
        assert "audience_mismatch" not in {v.code for v in report.violations}

    def test_unstated_audience_raises_no_audience_violation(self):
        brief = parse_brief(client_preferences="A flexible spare room")
        design = {"furniture": [
            {"name": "Kids Cot", "sku": "BED-KID-COT-01",
             "category": "bed", "quantity": 1},
        ]}
        report = validate_design(
            design, brief, price_design(design, budget_inr=brief.budget_inr)
        )
        assert "audience_mismatch" not in {v.code for v in report.violations}


class TestAudienceIsPreventedNotJustDetected:
    """
    Detection alone was not enough. An adults-only room still reached the
    client carrying a Kids Cot: the prompt listed children's products, the
    validator flagged it, and the design was served anyway.

    These assert the two preventive layers, because either alone leaves the
    other failure mode open.
    """

    def test_catalog_block_hides_forbidden_products(self):
        """
        Do not offer the model a choice it should not have. A 2B model will
        pick from a list it was given, so the list is the real control.
        """
        from app.agents.orchestrator import _catalog_block

        adult_block = _catalog_block(_adult_brief())
        assert "DSK-KID-01" not in adult_block, "kids desk offered to an adult room"
        assert "BED-KID-COT-01" not in adult_block
        assert "CHR-KID-01" not in adult_block
        # ...while the adult equivalents remain available.
        assert "DSK-STD-01" in adult_block
        assert "WRD-2D-01" in adult_block

    def test_kids_block_still_offers_kids_products(self):
        from app.agents.orchestrator import _catalog_block

        kids_block = _catalog_block(_kids_brief())
        assert "DSK-KID-01" in kids_block

    def test_normalizer_swaps_a_leaked_kids_product(self):
        """Even if one slips through, code repairs it before validation."""
        from app.agents.orchestrator import DesignPipeline

        brief = _adult_brief()
        design = {"furniture": [
            {"name": "Kids Study Table 90cm", "sku": "DSK-KID-01",
             "category": "desk", "quantity": 1},
            {"name": "Kids Cot", "sku": "BED-KID-COT-01",
             "category": "bed", "quantity": 1},
        ]}
        out = DesignPipeline()._normalize_design(design, brief)
        leaked = {i["sku"] for i in out["furniture"]} & KIDS_SKUS
        assert not leaked, f"normalizer left kids' furniture in place: {leaked}"

    def test_normalizer_leaves_a_kids_room_alone(self):
        from app.agents.orchestrator import DesignPipeline

        brief = _kids_brief()
        design = {"furniture": [
            {"name": "Kids Study Table 90cm", "sku": "DSK-KID-01",
             "category": "desk", "quantity": 1},
        ]}
        out = DesignPipeline()._normalize_design(design, brief)
        assert out["furniture"][0]["sku"] == "DSK-KID-01"


class TestWorkFromHomeKit:
    def test_wfh_is_detected_from_free_text(self):
        assert requires_wfh_kit([], "I work from home most days") is True

    def test_home_office_room_type_implies_wfh(self):
        assert requires_wfh_kit([], "", room_type="home_office") is True

    def test_plain_brief_is_not_wfh(self):
        assert requires_wfh_kit([], "a calm bedroom", room_type="bedroom") is False

    def test_wfh_creates_a_desk_and_chair_requirement(self):
        assert required_categories_for(ADULT, True) == {"desk": 1, "chair": 1}

    def test_wfh_brief_records_the_requirement(self):
        brief = parse_brief(
            client_preferences="A home office where I work from home",
            room_type="home_office",
        )
        assert brief.requires_workstation is True
        assert brief.required_items.get("desk", 0) >= 1
        assert brief.required_items.get("chair", 0) >= 1

    def test_desk_without_a_chair_fails(self):
        """A desk with nowhere to sit is a shelf."""
        brief = parse_brief(
            client_preferences="A home office where I work from home",
            room_type="home_office",
        )
        design = {"furniture": [
            {"name": "Study Desk", "sku": "DSK-STD-01",
             "category": "desk", "quantity": 1},
        ]}
        report = validate_design(
            design, brief, price_design(design, budget_inr=brief.budget_inr)
        )
        codes = {v.code for v in report.violations}
        assert "wfh_kit_incomplete" in codes, codes

    def test_desk_with_a_chair_passes(self):
        brief = parse_brief(
            client_preferences="A home office where I work from home",
            room_type="home_office",
        )
        design = {"furniture": [
            {"name": "Study Desk", "sku": "DSK-STD-01",
             "category": "desk", "quantity": 1},
            {"name": "Office Chair", "sku": "CHR-OFF-01",
             "category": "chair", "quantity": 1},
        ]}
        report = validate_design(
            design, brief, price_design(design, budget_inr=brief.budget_inr)
        )
        assert "wfh_kit_incomplete" not in {v.code for v in report.violations}

    def test_chair_without_a_desk_also_fails(self):
        brief = parse_brief(
            client_preferences="A home office where I work from home",
            room_type="home_office",
        )
        design = {"furniture": [
            {"name": "Office Chair", "sku": "CHR-OFF-01",
             "category": "chair", "quantity": 1},
        ]}
        report = validate_design(
            design, brief, price_design(design, budget_inr=brief.budget_inr)
        )
        assert "wfh_kit_incomplete" in {v.code for v in report.violations}

    def test_non_wfh_brief_is_not_penalised_for_a_missing_desk(self):
        brief = _adult_brief()
        design = {"furniture": [
            {"name": "Queen Bed", "sku": "BED-DBL-QUEEN-01",
             "category": "bed", "quantity": 1},
        ]}
        report = validate_design(
            design, brief, price_design(design, budget_inr=brief.budget_inr)
        )
        assert "wfh_kit_incomplete" not in {v.code for v in report.violations}