"""
Tests that the response agrees with itself.

Each test here corresponds to something the client could see and could not
reconcile: a colour chip that did not match its label, a rug praised in the pros
but absent from the shopping list, a requested bookshelf that quietly vanished,
a Strict-Vastu compliance score with no bearing behind it.
"""

from __future__ import annotations

import pytest

from app.catalog import catalog
from app.engine.assumptions import (
    assess_assumptions,
    has_bearings,
    missing_requirements,
    normalise_direction,
    unsupported_claims,
)
from app.engine.consistency import (
    audit_option,
    normalise_hex,
    palette_problems,
)


@pytest.fixture(scope="module", autouse=True)
def ready():
    catalog.initialize()
    return catalog


# ---------------------------------------------------------------------------
# D3 - colour chips must match the palette labels
# ---------------------------------------------------------------------------


class TestPaletteConsistency:
    def test_a_label_that_contradicts_the_chips_is_flagged(self):
        """'Coral' beside a navy chip is the bug that was observed."""
        problems = palette_problems({
            "primary": "Coral",
            "secondary": "Sage",
            "hex_codes": ["#1B3B6F", "#2F4858"],
        })
        codes = {p["code"] for p in problems}
        assert "palette_name_hex_mismatch" in codes

    def test_a_matching_palette_is_not_flagged(self):
        problems = palette_problems({
            "primary": "Coral",
            "secondary": "Sage",
            "accent": "Mustard",
            "hex_codes": ["#FF6B5B", "#A5B478", "#E0A93B"],
        })
        assert problems == [], problems

    def test_malformed_hex_is_an_error(self):
        """A non-hex renders as a blank or wrong-coloured chip."""
        problems = palette_problems({
            "primary": "Blue", "hex_codes": ["#GGG", "not-a-colour"],
        })
        codes = {p["code"] for p in problems}
        assert "invalid_hex" in codes
        assert any(p["severity"] == "error" for p in problems)

    def test_names_without_hexes_are_an_error(self):
        problems = palette_problems({"primary": "Coral", "hex_codes": []})
        codes = {p["code"] for p in problems}
        assert "palette_without_hex" in codes

    def test_neutral_palettes_are_accepted(self):
        """Neutral names match on low saturation, not on hue."""
        problems = palette_problems({
            "primary": "White", "secondary": "Soft Grey",
            "hex_codes": ["#F2F2F2", "#8C8C8C"],
        })
        assert problems == [], problems

    def test_shorthand_hex_is_normalised(self):
        assert normalise_hex("#abc") == "#AABBCC"
        assert normalise_hex("FF6B5B") == "#FF6B5B"
        assert normalise_hex("nonsense") is None

    def test_no_palette_is_not_a_problem(self):
        assert palette_problems(None) == []


# ---------------------------------------------------------------------------
# B5 - every requested item must appear in the shopping list
# ---------------------------------------------------------------------------


class TestRequestedItemCoverage:
    def test_a_missing_requested_item_is_an_error(self):
        result = audit_option(
            {"furniture": [
                {"sku": "BED-KID-COT-01", "name": "Kids Cot", "category": "bed"},
            ]},
            requested=["bed", "wardrobe"],
        )
        codes = {p["code"] for p in result["problems"]}
        assert "requested_item_missing" in codes
        assert result["valid"] is False

    def test_a_complete_list_passes(self):
        result = audit_option(
            {"furniture": [
                {"sku": "BED-KID-COT-01", "name": "Kids Cot", "category": "bed"},
                {"sku": "WRD-2D-01", "name": "Wardrobe", "category": "wardrobe"},
            ]},
            requested=["bed", "wardrobe"],
        )
        assert result["valid"] is True, result["problems"]

    def test_the_prose_alone_does_not_satisfy_a_request(self):
        """
        Describing the bookshelf beautifully is not the same as buying one. This
        is the whole point of checking the list rather than the narrative.
        """
        option = {
            "concept": "A study with a large bookshelf for the books.",
            "layout_description": "The bookshelf anchors the north wall.",
            "furniture": [
                {"sku": "DSK-STD-01", "name": "Study Desk", "category": "desk"},
            ],
        }
        gaps = missing_requirements(option, ["desk", "storage"])
        items = {g["item"] for g in gaps if g.get("fatal") == "true"}
        assert "storage" in items

    def test_quantity_is_honoured(self):
        result = audit_option(
            {"furniture": [
                {"sku": "BED-KID-COT-01", "name": "Kids Cot",
                 "category": "bed", "quantity": 2},
            ]},
            requested=["bed"],
        )
        assert result["valid"] is True

    def test_an_unservable_request_is_not_fatal(self):
        """A catalog gap is not the design's fault and must not fail it."""
        result = audit_option(
            {"furniture": [
                {"sku": "BED-KID-COT-01", "name": "Kids Cot", "category": "bed"},
            ]},
            requested=["bed", "grandfather-clock"],
        )
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# B4 / C3 - no claim without evidence
# ---------------------------------------------------------------------------


class TestNoClaimWithoutEvidence:
    def test_natural_light_claim_without_bearings_is_flagged(self):
        problems = unsupported_claims({
            "concept": "A bright room with plenty of natural light.",
            "door_direction": None,
            "window_direction": None,
        })
        codes = {p["code"] for p in problems}
        assert "unverified_bearing_claim" in codes

    def test_the_same_claim_with_bearings_is_accepted(self):
        problems = unsupported_claims({
            "concept": "A bright room with plenty of natural light.",
            "door_direction": "south",
            "window_direction": "east",
        })
        codes = {p["code"] for p in problems}
        assert "unverified_bearing_claim" not in codes

    def test_a_vastu_claim_without_bearings_is_flagged(self):
        problems = unsupported_claims({
            "vastu_notes": ["This layout is Vastu compliant."],
        })
        codes = {p["code"] for p in problems}
        assert "unverified_bearing_claim" in codes

    def test_mentioning_an_item_that_was_not_bought_is_flagged(self):
        """Prose describing a mirror the design never buys."""
        problems = unsupported_claims({
            "concept": "A calm room.",
            "lighting_plan": "Layered lighting with a large wall mirror.",
            "furniture": [
                {"sku": "BED-KID-COT-01", "name": "Kids Cot", "category": "bed"},
            ],
        })
        codes = {p["code"] for p in problems}
        assert "mentioned_not_bought" in codes

    def test_mentioning_an_item_that_was_bought_is_accepted(self):
        problems = unsupported_claims({
            "concept": "A calm room.",
            "lighting_plan": "Layered lighting.",
            "furniture": [
                {"sku": "DEC-MIR-01", "name": "Wall Mirror", "category": "decor"},
                {"sku": "LMP-CEL-01", "name": "Ceiling Light", "category": "lighting"},
            ],
        })
        codes = {p["code"] for p in problems}
        assert "mentioned_not_bought" not in codes

    def test_strict_vastu_without_bearings_fails_the_option(self):
        result = audit_option(
            {"furniture": []}, strict_vastu=True, has_bearings=False,
        )
        codes = {p["code"] for p in result["problems"]}
        assert "vastu_claim_without_bearing" in codes
        assert result["valid"] is False

    def test_strict_vastu_with_bearings_passes(self):
        result = audit_option(
            {"furniture": []}, strict_vastu=True, has_bearings=True,
        )
        assert "vastu_claim_without_bearing" not in {
            p["code"] for p in result["problems"]
        }

    def test_soft_vastu_without_bearings_is_not_penalised(self):
        """Only a Strict brief is a promise of compliance."""
        result = audit_option(
            {"furniture": []}, strict_vastu=False, has_bearings=False,
        )
        assert "vastu_claim_without_bearing" not in {
            p["code"] for p in result["problems"]
        }


class TestAssumptionsAreReported:
    def test_missing_bearings_are_listed_as_assumptions(self):
        report = assess_assumptions({"furniture": []}, strict_vastu=True)
        fields = {a["field"] for a in report["assumptions"]}
        assert "door_direction" in fields
        assert "window_direction" in fields

    def test_vastu_is_unverifiable_without_bearings(self):
        report = assess_assumptions({"furniture": []}, strict_vastu=True)
        assert report["vastu_verifiable"] is False
        assert report["vastu_withheld"] is True

    def test_vastu_is_verifiable_with_bearings(self):
        report = assess_assumptions(
            {"furniture": [], "door_direction": "south",
             "window_direction": "east"},
            strict_vastu=True,
        )
        assert report["vastu_verifiable"] is True
        assert report["vastu_withheld"] is False

    def test_known_bearings_are_not_listed_as_assumptions(self):
        report = assess_assumptions(
            {"furniture": [], "door_direction": "north",
             "window_direction": "west"},
        )
        fields = {a["field"] for a in report["assumptions"]}
        assert "door_direction" not in fields

    def test_direction_normalisation(self):
        assert normalise_direction("N") == "north"
        assert normalise_direction("North-East") == "north-east"
        assert normalise_direction("sideways") is None
        assert normalise_direction(None) is None
        assert has_bearings("N", "E") is True
        assert has_bearings("N", None) is False


# ---------------------------------------------------------------------------
# D4 - pros must be grounded
# ---------------------------------------------------------------------------


class TestGroundedPros:
    def test_a_pro_about_an_unbought_item_is_flagged(self):
        from app.engine.consistency import ungrounded_claims

        problems = ungrounded_claims({
            "pros": ["The large area rug anchors the seating"],
            "furniture": [
                {"sku": "SOF-2S-01", "name": "2-Seater Sofa",
                 "category": "seating"},
            ],
        })
        assert problems, "a rug that was never bought cannot be praised"
        assert problems[0]["code"] == "ungrounded_pro"

    def test_a_pro_about_a_bought_item_is_accepted(self):
        from app.engine.consistency import ungrounded_claims

        problems = ungrounded_claims({
            "pros": ["The sofa anchors the seating"],
            "furniture": [
                {"sku": "SOF-2S-01", "name": "2-Seater Sofa",
                 "category": "seating"},
            ],
        })
        assert problems == [], problems

    def test_warnings_alone_do_not_fail_an_option(self):
        """A design with a soft problem is still a usable design."""
        result = audit_option(
            {
                "furniture": [
                    {"sku": "BED-KID-COT-01", "name": "Kids Cot",
                     "category": "bed"},
                ],
                "pros": ["A warm rug softens the room"],
            },
            requested=["bed"],
        )
        assert result["warning_count"] >= 1
        assert result["valid"] is True, "warnings must not fail a design"