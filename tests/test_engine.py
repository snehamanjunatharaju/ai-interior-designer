"""
Tests for the deterministic engine: catalog, brief parsing, pricing, and
the validator.

These are the tests that matter most in the project. Everything else depends
on the LLM; this layer does not, so it can assert hard facts: a total is the
sum of its parts, a two-bed brief yields two beds, a vibrant palette is
actually saturated.

The rule throughout: a test must fail if the corresponding bug from the
original build is reintroduced.
"""

from __future__ import annotations

import pytest

from app.catalog import TIER_ESSENTIAL, TIER_UPGRADE, catalog
from app.engine.brief import (
    parse_brief,
    parse_dimensions,
    parse_money,
    parse_quantity,
)
from app.engine.pricing import build_phased_plan, price_design, required_units_for
from app.engine.validator import validate_design


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def ready_catalog():
    catalog.initialize()
    return catalog


@pytest.fixture
def kids_brief():
    """The brief from the bug report: two beds, vibrant, Rs 10,000, strict Vastu."""
    return parse_brief(
        "Rainbow room for two children",
        room_type="kids_room",
        style="modern",
        color_scheme="vibrant",
        budget_inr=10000,
        vastu_compliance="strict",
        room_dimensions="12 x 10 ft",
    )


def _design(**overrides):
    """A minimal valid kids-room design; override to induce a failure."""
    design = {
        "title": "Sunny Duo",
        "concept": "Two sleeping bays under one window wall.",
        "layout_description": "Beds on the south wall, desk under the east window.",
        "furniture": [
            {"name": "Kids Single Cot 90cm", "category": "bed", "quantity": 2,
             "sku": "BED-KID-COT-01"},
            {"name": "Kids Study Table 90cm", "category": "desk", "quantity": 1,
             "sku": "DSK-KID-01"},
        ],
        "color_palette": {
            "primary": "coral", "secondary": "teal", "accent": "marigold",
            "neutrals": ["white"], "hex_codes": ["#FF6B5B", "#17A2A2", "#F2A93B"],
        },
    }
    design.update(overrides)
    return design


def _codes(report):
    return {v.code for v in report.violations}


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------


class TestCatalog:
    def test_seeded_with_real_items(self):
        assert catalog.count() >= 40, "catalog should hold a usable product range"

    def test_every_item_has_price_and_dimensions(self):
        for item in catalog.all():
            assert item.price_inr > 0, f"{item.sku} has no price"
            assert item.width_cm > 0 and item.depth_cm > 0, f"{item.sku} has no footprint"

    def test_kids_room_essentials_exist(self):
        """The categories a kids room brief references must actually be stocked."""
        for category in ("bed", "desk", "chair", "wardrobe", "storage", "tv_unit"):
            assert catalog.by_category(category), f"no items in {category}"

    def test_prices_are_plausible(self):
        """Guards against a decimal-point or unit error in the seed data."""
        for item in catalog.all():
            assert 200 <= item.price_inr <= 500_000, f"{item.sku} price looks wrong"

    def test_skus_unique(self):
        skus = [i.sku for i in catalog.all()]
        assert len(skus) == len(set(skus))

    def test_tiers_assigned(self):
        tiers = {i.tier for i in catalog.all()}
        assert TIER_ESSENTIAL in tiers and TIER_UPGRADE in tiers


# --------------------------------------------------------------------------
# Brief parsing
# --------------------------------------------------------------------------


class TestParseMoney:
    @pytest.mark.parametrize("text,expected", [
        ("budget 10000", 10000),
        ("Rs 10,000", 10000),
        ("Rs. 1,50,000", 150000),
        ("10k", 10000),
        ("1.5 lakh", 150000),
        ("2 lakhs", 200000),
        ("1 crore", 10000000),
        ("₹25000", 25000),
    ])
    def test_formats(self, text, expected):
        assert parse_money(text) == expected

    def test_no_money_returns_none(self):
        assert parse_money("a cosy corner for reading") is None

    def test_ignores_non_money_k(self):
        """'10 km' must not be read as Rs 10,000."""
        assert parse_money("10 km from the station") != 10000


class TestParseQuantity:
    @pytest.mark.parametrize("text,item,expected", [
        ("2 beds", "bed", 2),
        ("two beds", "bed", 2),
        ("need a bed", "bed", 1),
        ("a pair of chairs", "chair", 2),
        ("3 chairs", "chair", 3),
        ("single bed", "bed", 1),
    ])
    def test_counts(self, text, item, expected):
        assert parse_quantity(text, item) == expected

    def test_count_does_not_leak_across_conjunction(self):
        """'two beds and a desk' is 2 beds and 1 desk, not 2 of each."""
        brief = parse_brief("two beds and a desk", room_type="kids_room")
        assert brief.required_items.get("bed") == 2
        assert brief.required_items.get("desk") == 1

    def test_absent_item_returns_none(self):
        assert parse_quantity("a quiet reading nook", "wardrobe") is None


class TestParseDimensions:
    def test_feet_pair(self):
        d = parse_dimensions("12 x 10 ft")
        assert (d.length_ft, d.width_ft) == (12.0, 10.0)
        assert d.area_sqft == 120.0

    def test_longer_side_is_length(self):
        d = parse_dimensions("10ft x 12ft")
        assert d.length_ft == 12.0

    def test_metric_conversion(self):
        d = parse_dimensions("3.6m x 3m")
        assert d.length_ft and abs(d.length_ft - 11.8) < 0.2

    def test_ceiling_height(self):
        d = parse_dimensions("12ft x 10ft, ceiling 9ft")
        assert d.ceiling_ft == 9.0

    def test_unparseable(self):
        assert parse_dimensions("a small room").is_known() is False


class TestParseBrief:
    def test_reproduces_the_reported_bug_brief(self, kids_brief):
        """The exact brief from the screenshots must yield 2 beds."""
        assert kids_brief.required_items["bed"] == 2
        assert kids_brief.budget_inr == 10000
        assert kids_brief.color_scheme == "vibrant"
        assert kids_brief.dimensions.area_sqft == 120.0

    def test_infers_beds_from_occupants(self):
        brief = parse_brief("room for two children", room_type="kids_room")
        assert brief.required_items["bed"] == 2
        assert brief.occupant_count == 2

    def test_explicit_count_beats_inference(self):
        brief = parse_brief("3 beds for two children", room_type="kids_room")
        assert brief.required_items["bed"] == 3

    def test_no_beds_inferred_for_study(self):
        brief = parse_brief("a study for one person", room_type="home_office")
        assert "bed" not in brief.required_items

    def test_negation_becomes_exclusion(self):
        brief = parse_brief("no mirror in the room", room_type="kids_room")
        assert "mirror" in brief.excluded_items

    def test_explicit_field_overrides_inferred_colour(self):
        brief = parse_brief("keep it beige", color_scheme="vibrant",
                            room_type="kids_room")
        assert brief.color_scheme == "vibrant"

    def test_directions(self):
        brief = parse_brief("window faces east, door on the north side",
                            room_type="kids_room")
        assert brief.window_direction == "east"
        assert brief.door_direction == "north"

    def test_intercardinal_not_read_as_cardinal(self):
        brief = parse_brief("window faces northeast", room_type="kids_room")
        assert brief.window_direction == "northeast"


# --------------------------------------------------------------------------
# Pricing - the arithmetic the LLM must never do
# --------------------------------------------------------------------------


class TestPricing:
    def test_total_is_sum_of_lines(self):
        design = _design()
        cost = price_design(design, budget_inr=10_000_000)
        line_sum = sum(i["line_total_inr"] for i in cost["line_items"])
        assert cost["estimated_cost_inr"] == line_sum

    def test_breakdown_closes(self):
        cost = price_design(_design(), budget_inr=10_000_000)
        breakdown_sum = sum(b["amount_inr"] for b in cost["budget_breakdown"])
        assert breakdown_sum == cost["estimated_cost_inr"]

    def test_model_price_is_ignored(self):
        """A hallucinated price must never enter the total."""
        design = _design(furniture=[{
            "name": "Kids Cot", "category": "bed", "quantity": 1,
            "sku": "BED-KID-COT-01", "estimated_price_inr": 999_999,
        }])
        cost = price_design(design)
        assert cost["estimated_cost_inr"] == 3200  # catalog price, not 999,999

    def test_quantity_multiplies(self):
        design = _design(furniture=[{
            "name": "Kids Cot", "category": "bed", "quantity": 2,
            "sku": "BED-KID-COT-01",
        }])
        assert price_design(design)["estimated_cost_inr"] == 6400

    def test_unknown_sku_is_reported_not_guessed(self):
        """The original bug: an item with no price contributed Rs 0 silently."""
        design = _design(furniture=[{"name": "Magic bed", "category": "bed",
                                     "quantity": 1, "sku": "FAKE-001"}])
        cost = price_design(design)
        assert cost["costing_complete"] is False
        assert cost["unpriced_items"]
        assert cost["estimated_cost_inr"] == 0

    def test_brand_comes_from_catalog_not_model(self):
        design = _design(furniture=[{"name": "Cot", "category": "bed",
                                     "quantity": 1, "sku": "BED-KID-COT-01",
                                     "brand": "Bison Kids"}])
        cost = price_design(design)
        assert cost["line_items"][0]["brand"] == "Sleepwell"

    def test_over_budget_detected(self):
        design = _design(furniture=[
            {"name": "Wardrobe", "category": "wardrobe", "quantity": 1,
             "sku": "WRD-2D-01"},
        ])
        cost = price_design(design, budget_inr=1000)
        assert cost["within_budget"] is False
        assert cost["overage_inr"] == 5500
        assert cost["cost_savings"]


class TestPhasedPlan:
    def _over_budget_items(self):
        return price_design({"furniture": [
            {"sku": "BED-KID-COT-01", "category": "bed", "quantity": 2},
            {"sku": "DSK-KID-01", "category": "desk", "quantity": 1},
            {"sku": "WRD-2D-01", "category": "wardrobe", "quantity": 1},
            {"sku": "TVU-WAL-01", "category": "tv_unit", "quantity": 1},
        ]}, budget_inr=10_000)

    def test_phase1_respects_budget(self):
        plan = build_phased_plan(self._over_budget_items()["line_items"], 10_000)
        assert plan["phase1"]["total_inr"] <= 10_000

    def test_phases_preserve_all_items(self):
        """Nothing may be silently dropped when splitting the plan."""
        items = self._over_budget_items()["line_items"]
        plan = build_phased_plan(items, 10_000)
        split = plan["phase1"]["line_items"] + plan["phase2"]["line_items"]
        assert len(split) == len(items)
        assert sum(i["line_total_inr"] for i in split) == sum(
            i["line_total_inr"] for i in items
        )

    def test_beds_bought_before_wardrobe(self):
        """Essentials first: the children sleep before the wardrobe is bought."""
        plan = build_phased_plan(self._over_budget_items()["line_items"], 10_000)
        phase1_skus = {i["sku"] for i in plan["phase1"]["line_items"]}
        assert "BED-KID-COT-01" in phase1_skus
        assert "WRD-2D-01" not in phase1_skus

    def test_infeasible_budget_is_declared(self):
        plan = build_phased_plan(self._over_budget_items()["line_items"], 1_000)
        assert plan["feasible"] is False
        assert "two-phase" in plan["headline"]

    def test_affordable_budget_needs_one_phase(self):
        cheap = price_design({"furniture": [
            {"sku": "DEC-PLT-01", "category": "decor", "quantity": 1}]})
        plan = build_phased_plan(cheap["line_items"], 10_000)
        assert plan["feasible"] is True
        assert not plan["phase2"]["line_items"]


# --------------------------------------------------------------------------
# Validator - the gate that makes the reported bugs unshippable
# --------------------------------------------------------------------------


class TestValidator:
    def test_valid_design_passes(self, kids_brief):
        report = validate_design(_design(), kids_brief, price_design(_design(), 10_000))
        assert report.is_valid, [v.to_dict() for v in report.errors]

    def test_too_few_beds_is_an_error(self, kids_brief):
        """THE headline bug: asked for 2 beds, design has 1."""
        design = _design(furniture=[{
            "name": "Kids Cot", "category": "bed", "quantity": 1,
            "sku": "BED-KID-COT-01"}])
        report = validate_design(design, kids_brief, price_design(design, 10_000))
        assert not report.is_valid
        assert "missing_items" in _codes(report)

    def test_correct_bed_count_passes(self, kids_brief):
        design = _design(furniture=[{
            "name": "Kids Cot", "category": "bed", "quantity": 2,
            "sku": "BED-KID-COT-01"}])
        report = validate_design(design, kids_brief, price_design(design, 10_000))
        assert "missing_items" not in _codes(report)

    def test_bunk_bed_satisfies_two_bed_request(self, kids_brief):
        """
        One bunk sleeps two, so it satisfies a 2-bed brief at quantity 1.

        Counting units instead of sleeping places would report a false
        shortfall here, and "fixing" it would buy two bunks and double the
        price.
        """
        design = _design(furniture=[{
            "name": "Kids Bunk Bed", "category": "bed", "quantity": 1,
            "sku": "BED-KID-BUNK-01"}])
        report = validate_design(design, kids_brief, price_design(design, 10_000))
        assert "missing_items" not in _codes(report), [v.to_dict() for v in report.errors]

    def test_single_bunk_not_double_charged(self):
        """The capacity rule must not inflate the price of a bunk."""
        cost = price_design({"furniture": [
            {"sku": "BED-KID-BUNK-01", "category": "bed", "quantity": 1}]})
        assert cost["estimated_cost_inr"] == 14500
        assert cost["line_items"][0]["sleeps_per_unit"] == 2

    def test_required_units_accounts_for_capacity(self):
        priced = [{"category": "bed", "sleeps_per_unit": 2}]
        assert required_units_for("bed", 2, priced) == 1
        assert required_units_for("bed", 3, priced) == 2

    def test_single_bed_still_needs_two_units(self):
        cost = price_design({"furniture": [
            {"sku": "BED-KID-COT-01", "category": "bed", "quantity": 2}]})
        assert cost["line_items"][0]["sleeps_per_unit"] == 1
        assert cost["estimated_cost_inr"] == 6400

    def test_muted_palette_rejected_for_vibrant_brief(self, kids_brief):
        design = _design(color_palette={
            "primary": "beige", "secondary": "terracotta", "accent": "warm neutral",
            "hex_codes": ["#D9CFC0", "#C0704A"],
        })
        report = validate_design(design, kids_brief, price_design(design, 10_000))
        assert "color_scheme_mismatch" in _codes(report)

    def test_vibrant_palette_accepted(self, kids_brief):
        report = validate_design(_design(), kids_brief, price_design(_design(), 10_000))
        assert "color_scheme_mismatch" not in _codes(report)

    def test_single_warm_accent_is_not_vibrant(self, kids_brief):
        """Terracotta alone must not rescue an otherwise beige palette."""
        design = _design(color_palette={
            "primary": "beige", "secondary": "beige", "accent": "terracotta",
            "hex_codes": ["#D9CFC0", "#C0704A"],
        })
        report = validate_design(design, kids_brief, price_design(design, 10_000))
        assert "color_scheme_mismatch" in _codes(report)

    def test_placeholder_text_rejected(self, kids_brief):
        design = _design(layout_description="Layout derived from the design plan.")
        report = validate_design(design, kids_brief, price_design(design, 10_000))
        assert "placeholder_text" in _codes(report)

    def test_real_design_prose_is_not_a_placeholder(self, kids_brief):
        """
        Ordinary design language must not trip the placeholder check.

        "Insert the wardrobe" and hex codes both contain substrings that a
        naive `in` test mistakes for template sentinels. Since this check
        blocks a design, a false positive here rejects good work.
        """
        design = _design(layout_description=(
            "Insert the wardrobe along the north wall; the cot sits under the "
            "window. Accent wall painted #FF6B5B, trim #17A2A2."
        ))
        report = validate_design(design, kids_brief, price_design(design, 10_000))
        assert "placeholder_text" not in _codes(report), [v.message for v in report.violations]

    def test_bracket_placeholder_still_caught(self, kids_brief):
        design = _design(layout_description="Storage: [insert wardrobe size]")
        report = validate_design(design, kids_brief, price_design(design, 10_000))
        assert "placeholder_text" in _codes(report)

    def test_invented_brand_rejected(self, kids_brief):
        design = _design(furniture=[{
            "name": "Bison Kids Bed", "category": "bed", "quantity": 2,
            "sku": "NOT-A-REAL-SKU"}])
        report = validate_design(design, kids_brief, price_design(design, 10_000))
        assert "uncatalogued_item" in _codes(report)

    def test_excluded_item_present_is_an_error(self):
        brief = parse_brief("no mirror, one bed", room_type="bedroom")
        design = _design(furniture=[
            {"name": "Kids Cot", "category": "bed", "quantity": 1, "sku": "BED-KID-COT-01"},
            # The catalog stores a mirror under `decor`, and the model may
            # report either label, so the check resolves the product by SKU.
            {"name": "Wall Mirror", "category": "decor", "quantity": 1, "sku": "DEC-MIR-01"},
        ])
        report = validate_design(design, brief, price_design(design, 10_000))
        assert "excluded_item_present" in _codes(report)

    def test_zero_total_is_an_error(self, kids_brief):
        """An unpriced design must never read as 'within budget'."""
        design = _design(furniture=[{"name": "Mystery", "category": "bed", "quantity": 2}])
        cost = price_design(design, 10_000)
        report = validate_design(design, kids_brief, cost)
        assert not report.is_valid

    def test_missing_costing_is_an_error(self, kids_brief):
        report = validate_design(_design(), kids_brief, cost=None)
        assert "no_costing" in _codes(report)

    def test_over_budget_is_a_warning_not_a_block(self, kids_brief):
        """The negotiator handles overage; it should not fail the design."""
        # Beds are included so this design is otherwise valid, and the only
        # thing wrong with it is that it costs more than the budget.
        design = _design(furniture=[
            {"name": "Kids Cot", "category": "bed", "quantity": 2,
             "sku": "BED-KID-COT-01"},
            {"name": "Wardrobe", "category": "wardrobe", "quantity": 1,
             "sku": "WRD-2D-01"},
        ])
        report = validate_design(design, kids_brief, price_design(design, 1_000))
        assert "over_budget" in _codes(report)
        assert report.is_valid

    def test_pass_rate_reported(self, kids_brief):
        report = validate_design(_design(), kids_brief, price_design(_design(), 10_000))
        assert 0.0 <= report.pass_rate <= 1.0
        assert report.checks_passed <= report.checks_run

    def test_critique_lines_are_actionable(self, kids_brief):
        design = _design(furniture=[{
            "name": "Kids Cot", "category": "bed", "quantity": 1,
            "sku": "BED-KID-COT-01"}])
        report = validate_design(design, kids_brief, price_design(design, 10_000))
        lines = report.critique_lines()
        assert lines and all("[missing_items]" in l or l.strip() for l in lines)

    def test_unpriceable_brief_skips_budget_check(self):
        """No budget in the brief means no budget check, not a failure."""
        brief = parse_brief("cosy reading nook", room_type="living_room")
        report = validate_design(_design(), brief, cost=None)
        assert "no_costing" not in _codes(report)
