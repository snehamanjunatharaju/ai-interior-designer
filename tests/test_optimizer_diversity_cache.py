"""
Tests for the deterministic pieces of A2, D1 and F3.

These three are grouped because they share a property: each replaces a judgement
the language model was previously trusted with. The optimizer replaces "does this
fit the budget", the diversity filter replaces "are these actually different",
and the cache key replaces "is this stored answer still true". Each is therefore
asserted on behaviour rather than on shape.
"""

from __future__ import annotations

import pytest

from app.catalog import catalog
from app.core.config import settings
from app.engine.budget_optimizer import (
    CATEGORY_PRIORITY,
    ESSENTIAL_CATEGORIES,
    optimize_to_budget,
)
from app.engine.diversity import (
    DUPLICATE_THRESHOLD,
    fingerprint,
    select_distinct,
    similarity,
)
from app.services.semantic_cache import SemanticCache


@pytest.fixture(scope="module", autouse=True)
def ready():
    catalog.initialize()
    return catalog


# ---------------------------------------------------------------------------
# A2 - budget optimizer
# ---------------------------------------------------------------------------


class TestBudgetOptimizer:
    def test_total_is_exactly_the_sum_of_its_lines(self):
        """The invariant that matters: no total that disagrees with its items."""
        result = optimize_to_budget(
            required={"bed": 1, "mattress": 1, "wardrobe": 1},
            budget_inr=250_000, style_tier="high",
        )
        assert result["total_inr"] == sum(
            s["line_total_inr"] for s in result["selections"]
        )

    def test_every_selected_sku_is_real(self):
        result = optimize_to_budget(
            required={"bed": 1, "desk": 1, "chair": 1},
            budget_inr=300_000, style_tier="medium",
        )
        for selection in result["selections"]:
            assert catalog.get(selection["sku"]) is not None

    def test_respects_the_budget(self):
        result = optimize_to_budget(
            required={"bed": 2, "wardrobe": 2, "desk": 1, "chair": 1},
            budget_inr=80_000, style_tier="medium",
        )
        assert result["total_inr"] <= 80_000

    def test_picks_the_cheapest_first_when_tight(self):
        """A tight budget must get the most room it can afford."""
        result = optimize_to_budget(
            required={"bed": 1}, budget_inr=20_000, style_tier="essential",
        )
        if result["feasible"]:
            cheapest = min(
                (p for p in catalog.by_category("bed")),
                key=lambda p: p.price_inr,
            )
            assert result["selections"][0]["unit_price_inr"] == cheapest.price_inr

    def test_upgrades_when_there_is_headroom(self):
        cheap = optimize_to_budget(
            required={"bed": 1}, budget_inr=20_000, style_tier="essential",
        )
        rich = optimize_to_budget(
            required={"bed": 1}, budget_inr=200_000, style_tier="upgrade",
        )
        assert rich["total_inr"] >= cheap["total_inr"]
        assert any(s.get("upgraded") for s in rich["selections"])

    def test_an_upgrade_never_breaks_the_budget(self):
        result = optimize_to_budget(
            required={"bed": 1, "wardrobe": 1},
            budget_inr=30_000, style_tier="upgrade",
        )
        assert result["total_inr"] <= 30_000

    def test_bunk_bed_satisfies_two_children_at_one_price(self):
        """
        A bunk sleeps two, so a two-child brief costs one bunk rather than two
        cots. Counting units instead of sleeping places doubles the price.
        """
        result = optimize_to_budget(
            required={"bed": 2}, budget_inr=200_000, style_tier="standard",
        )
        assert result["feasible"]
        bunks = [s for s in result["selections"] if s["sku"] == "BED-KID-BUNK-01"]
        if bunks:
            assert bunks[0]["quantity"] == 1

    def test_one_child_does_not_get_a_bunk_bed(self):
        """
        Sorting by `sleeps` first gave a single child the Rs 14,500 bunk instead
        of the Rs 3,200 cot: 4x the price for one occupant.
        """
        result = optimize_to_budget(
            required={"bed": 1}, budget_inr=20_000, style_tier="essential",
        )
        assert result["feasible"]
        assert result["selections"][0]["sku"] != "BED-KID-BUNK-01"

    def test_two_children_prefer_one_bunk_over_two_singles(self):
        """
        The other side of the same rule: one bunk (Rs 14,500) satisfies two
        children more cheaply than two singles (Rs 19,000).
        """
        result = optimize_to_budget(
            required={"bed": 2}, budget_inr=30_000, style_tier="essential",
        )
        assert result["feasible"]
        bunks = [s for s in result["selections"] if s["sku"] == "BED-KID-BUNK-01"]
        assert len(bunks) == 1, "one bunk covers two children"
        assert len(result["selections"]) == 1

    def test_impossible_budget_reports_the_shortfall(self):
        result = optimize_to_budget(
            required={"bed": 2, "wardrobe": 2},
            budget_inr=5_000, style_tier="luxury",
        )
        assert result["feasible"] is False
        assert result["shortfall_inr"] > 0
        assert "cannot cover" in result["infeasible"]

    def test_impossible_budget_offers_three_ways_out(self):
        """
        "No" is not an answer. Each route must name a concrete change.
        """
        result = optimize_to_budget(
            required={"bed": 2, "wardrobe": 2},
            budget_inr=5_000, style_tier="luxury",
        )
        kinds = {o["kind"] for o in result["options"]}
        assert kinds == {"raise_budget", "reduce_scope", "change_tier"}
        for route in result["options"]:
            assert route["label"] and route["detail"]

    def test_raise_budget_route_quotes_the_real_minimum(self):
        result = optimize_to_budget(
            required={"bed": 1, "wardrobe": 1},
            budget_inr=3_000, style_tier="luxury",
        )
        route = next(o for o in result["options"] if o["kind"] == "raise_budget")
        minimum = int(route["label"].split()[-1].replace(",", ""))
        assert minimum == result["total_inr"] + result["shortfall_inr"]

    def test_never_overspends_even_when_impossible(self):
        """An impossible brief still must not produce an over-budget list."""
        result = optimize_to_budget(
            required={"bed": 2, "wardrobe": 2, "desk": 1, "chair": 1},
            budget_inr=2_000, style_tier="luxury",
        )
        assert result["total_inr"] <= 2_000

    def test_no_budget_selects_the_cheapest_per_category(self):
        result = optimize_to_budget(
            required={"bed": 1}, budget_inr=None, style_tier="essential",
        )
        assert result["feasible"]
        assert result["total_inr"] > 0

    def test_unserved_names_the_category_and_count(self):
        result = optimize_to_budget(
            required={"bed": 3}, budget_inr=7_000, style_tier="essential",
        )
        if not result["feasible"]:
            assert result["unserved"]
            entry = result["unserved"][0]
            assert entry["category"] == "bed"
            assert entry["missing"] >= 1

    def test_essentials_are_all_known_categories(self):
        """Every declared essential must actually be a priority-ordered term."""
        for category in ESSENTIAL_CATEGORIES:
            assert category in CATEGORY_PRIORITY

    def test_essentials_outrank_decor(self):
        assert CATEGORY_PRIORITY["bed"] < CATEGORY_PRIORITY["decor"]
        assert CATEGORY_PRIORITY["mattress"] < CATEGORY_PRIORITY["decor"]
        assert CATEGORY_PRIORITY["desk"] < CATEGORY_PRIORITY["decor"]

    def test_is_deterministic(self):
        first = optimize_to_budget(
            required={"bed": 1, "wardrobe": 1}, budget_inr=200_000,
            style_tier="medium",
        )
        second = optimize_to_budget(
            required={"bed": 1, "wardrobe": 1}, budget_inr=200_000,
            style_tier="medium",
        )
        assert [s["sku"] for s in first["selections"]] == \
               [s["sku"] for s in second["selections"]]


# ---------------------------------------------------------------------------
# D1 - option diversity
# ---------------------------------------------------------------------------


def _option(title, skus, hexes, zones, cost):
    return {
        "title": title,
        "furniture": [
            {"sku": s, "category": s.split("-")[0]} for s in skus
        ],
        "color_palette": {"hex_codes": hexes},
        "zones": zones,
        "estimated_cost_inr": cost,
    }


class TestOptionDiversity:
    def test_identical_options_score_as_duplicates(self):
        a = _option("A", ["BED-KID-COT-01", "DSK-KID-01"],
                    ["#E8F1F5"], ["sleeping"], 50_000)
        b = _option("B", ["BED-KID-COT-01", "DSK-KID-01"],
                    ["#E8F1F5"], ["sleeping"], 50_000)
        assert similarity(fingerprint(a), fingerprint(b)) >= DUPLICATE_THRESHOLD

    def test_renaming_does_not_make_an_option_distinct(self):
        """'Option 2' is not a different room."""
        a = _option("Modern Calm Bedroom", ["BED-KID-COT-01"],
                    ["#E8F1F5"], ["sleeping"], 50_000)
        b = _option("Option 2 - Modern Calm Bedroom", ["BED-KID-COT-01"],
                    ["#E8F1F5"], ["sleeping"], 50_000)
        assert similarity(fingerprint(a), fingerprint(b)) >= DUPLICATE_THRESHOLD

    def test_genuinely_different_options_are_kept(self):
        a = _option("Sleep Focus", ["BED-KID-COT-01", "MAT-KID-01"],
                    ["#E8F1F5"], ["sleeping"], 50_000)
        b = _option("Reading Nook", ["ARM-01", "TBL-SID-01", "LMP-FLO-01"],
                    ["#1B3B6F"], ["reading"], 140_000)
        assert similarity(fingerprint(a), fingerprint(b)) < DUPLICATE_THRESHOLD

    def test_select_distinct_drops_the_copy(self):
        a = _option("A", ["BED-KID-COT-01"], ["#E8F1F5"], ["sleeping"], 50_000)
        b = _option("A v2", ["BED-KID-COT-01"], ["#E8F1F5"], ["sleeping"], 50_000)
        kept, rejected = select_distinct([a, b])
        assert len(kept) == 1
        assert len(rejected) == 1
        assert rejected[0]["duplicate_of"] == "A"

    def test_rejection_explains_itself(self):
        from app.engine.diversity import describe_rejection

        a = _option("A", ["BED-KID-COT-01"], ["#E8F1F5"], ["sleeping"], 50_000)
        b = _option("A v2", ["BED-KID-COT-01"], ["#E8F1F5"], ["sleeping"], 50_000)
        _, rejected = select_distinct([a, b])
        reason = rejected[0]
        assert reason["similarity"] >= DUPLICATE_THRESHOLD
        message = describe_rejection(reason)
        assert "duplicate" in message.lower()
        assert "A" in message

    def test_three_distinct_options_all_survive(self):
        a = _option("Sleep", ["BED-KID-COT-01"], ["#E8F1F5"], ["sleeping"], 50_000)
        b = _option("Read", ["ARM-01", "TBL-SID-01"], ["#1B3B6F"], ["reading"], 90_000)
        c = _option("Work", ["DSK-STD-01", "CHR-OFF-01"], ["#5C6B73"],
                    ["study"], 120_000)
        kept, rejected = select_distinct([a, b, c])
        assert len(kept) == 3
        assert rejected == []

    def test_preserves_order(self):
        a = _option("First", ["BED-KID-COT-01"], ["#E8F1F5"], ["sleeping"], 50_000)
        b = _option("Second", ["ARM-01", "TBL-SID-01"], ["#1B3B6F"],
                    ["reading"], 90_000)
        kept, _ = select_distinct([a, b])
        assert [k["title"] for k in kept] == ["First", "Second"]

    def test_is_deterministic(self):
        options = [
            _option("A", ["BED-KID-COT-01"], ["#E8F1F5"], ["s"], 50_000),
            _option("A2", ["BED-KID-COT-01"], ["#E8F1F5"], ["s"], 50_000),
        ]
        first = select_distinct(options)[0]
        second = select_distinct(options)[0]
        assert [o["title"] for o in first] == [o["title"] for o in second]

    def test_empty_input_is_safe(self):
        kept, rejected = select_distinct([])
        assert kept == [] and rejected == []

    def test_zone_dicts_do_not_crash(self):
        """Zones arrive as dicts from some model responses."""
        a = _option("A", ["BED-KID-COT-01"], ["#E8F1F5"],
                    [{"name": "sleeping", "purpose": "rest"}], 50_000)
        b = _option("B", ["ARM-01"], ["#1B3B6F"], [{"name": "reading"}], 90_000)
        kept, _ = select_distinct([a, b])
        assert len(kept) == 2


class TestDistinctTitles:
    def test_collision_gets_a_variant_suffix(self):
        from app.agents.orchestrator import DesignPipeline

        options = [{"title": "Calm Bedroom"}, {"title": "Calm Bedroom"}]
        DesignPipeline._ensure_distinct_titles(options)
        assert options[0]["title"] != options[1]["title"]

    def test_no_collision_leaves_titles_alone(self):
        from app.agents.orchestrator import DesignPipeline

        options = [{"title": "Calm Bedroom"}, {"title": "Bright Nook"}]
        DesignPipeline._ensure_distinct_titles(options)
        assert [o["title"] for o in options] == ["Calm Bedroom", "Bright Nook"]


# ---------------------------------------------------------------------------
# F3 - cache key
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_same_payload_gives_same_key(self):
        payload = {"brief": "a bedroom", "budget": 100_000}
        assert SemanticCache.make_key(payload) == \
               SemanticCache.make_key(dict(payload))

    def test_different_payload_gives_different_key(self):
        assert SemanticCache.make_key({"brief": "a bedroom"}) != \
               SemanticCache.make_key({"brief": "a kitchen"})

    def test_key_depends_on_catalog_version(self, monkeypatch):
        """
        A repriced catalog must invalidate every stored answer. Without this a
        client is shown figures the database no longer supports.
        """
        payload = {"brief": "a bedroom"}
        before = SemanticCache.make_key(payload)

        real_version = catalog.version
        monkeypatch.setattr(catalog, "version", lambda: "9999-99-99")
        assert SemanticCache.make_key(payload) != before
        monkeypatch.setattr(catalog, "version", real_version)

    def test_key_depends_on_model(self, monkeypatch):
        payload = {"brief": "a bedroom"}
        before = SemanticCache.make_key(payload)
        monkeypatch.setattr(
            settings, "DEFAULT_MODEL", "some-other-model:70b"
        )
        assert SemanticCache.make_key(payload) != before

    def test_key_is_order_independent(self):
        a = {"brief": "x", "budget": 1, "style": "modern"}
        b = {"style": "modern", "budget": 1, "brief": "x"}
        assert SemanticCache.make_key(a) == SemanticCache.make_key(b)

    def test_field_the_caller_forgot_is_still_separated(self):
        """Two briefs differing only in an unpassed field must not collide."""
        a = {"brief": "a bedroom"}
        b = {"brief": "a bedroom", "occupants": ["adults"]}
        assert SemanticCache.make_key(a) != SemanticCache.make_key(b)


class TestMoneyFormatting:
    def test_indian_grouping(self):
        from app.core.money import format_inr

        assert format_inr(4100) == "₹4,100"
        assert format_inr(123456) == "₹1,23,456"
        assert format_inr(10000000) == "₹1,00,00,000"

    def test_zero_and_garbage(self):
        from app.core.money import format_inr

        assert format_inr(0) == "₹0"
        assert format_inr(None) == "₹0"
        assert format_inr("abc") == "₹0"

    def test_negative(self):
        from app.core.money import format_inr

        assert format_inr(-250) == "-₹250"