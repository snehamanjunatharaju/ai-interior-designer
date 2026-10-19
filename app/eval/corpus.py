"""
The evaluation corpus.

Forty-eight briefs, chosen so the harness fails in the ways a user would
actually notice, rather than on comfortable examples. Every room type appears;
budgets straddle the plausible/unplausible boundary on purpose; Strict Vastu is
paired with missing bearings in half those cases so the withholding path is
exercised; adults and kids rooms are mixed so an audience leak shows up.

Each case declares what it expects. The harness compares that against what the
pipeline produced, so a regression shows up as a number moving rather than as
a vague sense that output got worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EvalCase:
    """One brief plus the properties a correct response must have."""

    name: str
    room_type: str
    preferences: str
    # Physical inputs. Absence is deliberate in several cases.
    length_ft: Optional[float] = None
    width_ft: Optional[float] = None
    ceiling_ft: Optional[float] = None
    occupants: List[str] = field(default_factory=list)
    furniture: List[str] = field(default_factory=list)
    style: str = "modern"
    budget_tier: str = "medium"
    budget_inr: Optional[int] = None
    door_direction: Optional[str] = None
    window_direction: Optional[str] = None
    vastu: str = "soft"

    # Expectations.
    expect_items: List[str] = field(default_factory=list)
    expect_audience: Optional[str] = None      # "adult" / "kids" / None
    expect_within_budget: Optional[bool] = None
    expect_vastu_verifiable: Optional[bool] = None

    def to_request(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "room_type": self.room_type,
            "client_preferences": self.preferences,
            "style": self.style,
            "budget": self.budget_tier,
            "occupants": list(self.occupants),
            "furniture": list(self.furniture),
            "vastu_compliance": self.vastu,
            "num_options": 3,
        }
        # Dimensions are only sent when declared. Omitting them is a real
        # scenario and must be exercised as one.
        if self.length_ft and self.width_ft:
            payload["length_ft"] = self.length_ft
            payload["width_ft"] = self.width_ft
            if self.ceiling_ft:
                payload["ceiling_ft"] = self.ceiling_ft
        if self.budget_inr:
            payload["budget_amount_inr"] = self.budget_inr
        if self.door_direction:
            payload["door_direction"] = self.door_direction
        if self.window_direction:
            payload["window_direction"] = self.window_direction
        return payload


def _bedrooms() -> List[EvalCase]:
    return [
        EvalCase(
            name="bedroom_comfortable_budget",
            room_type="bedroom",
            preferences="A calm bedroom with a bed, mattress and wardrobe.",
            length_ft=12, width_ft=10, ceiling_ft=9,
            occupants=["adults"], furniture=["bed", "mattress", "wardrobe"],
            budget_tier="high", budget_inr=250_000,
            expect_items=["bed", "mattress", "wardrobe"],
            expect_audience="adult", expect_within_budget=True,
        ),
        EvalCase(
            name="bedroom_tight_budget",
            room_type="bedroom",
            preferences="A simple bedroom, bed and mattress only.",
            length_ft=10, width_ft=9, ceiling_ft=9,
            occupants=["adult"], furniture=["bed", "mattress"],
            budget_tier="low", budget_inr=60_000,
            expect_items=["bed", "mattress"],
            expect_audience="adult", expect_within_budget=True,
        ),
        EvalCase(
            name="bedroom_impossible_budget",
            room_type="bedroom",
            preferences="A luxury bedroom with a queen bed and large wardrobe.",
            length_ft=14, width_ft=12,
            occupants=["adults"], furniture=["bed", "wardrobe"],
            style="luxury", budget_tier="luxury", budget_inr=15_000,
            expect_items=["bed", "wardrobe"],
            expect_audience="adult", expect_within_budget=False,
        ),
        EvalCase(
            name="bedroom_wfh",
            room_type="home_office",
            preferences="I work from home and need a proper workstation.",
            length_ft=10, width_ft=8, ceiling_ft=9,
            occupants=["adults"], furniture=["desk"],
            budget_tier="medium", budget_inr=120_000,
            expect_items=["desk", "chair"],
            expect_audience="adult", expect_within_budget=True,
        ),
        EvalCase(
            name="bedroom_strict_vastu_with_bearings",
            room_type="bedroom",
            preferences="A Vastu-compliant bedroom with the bed head south.",
            length_ft=12, width_ft=11, ceiling_ft=10,
            occupants=["adults"], furniture=["bed"],
            budget_tier="medium", budget_inr=200_000,
            door_direction="north", window_direction="east",
            vastu="strict",
            expect_items=["bed"], expect_audience="adult",
            expect_vastu_verifiable=True, expect_within_budget=True,
        ),
        EvalCase(
            name="bedroom_strict_vastu_without_bearings",
            room_type="bedroom",
            preferences="A strictly Vastu-compliant bedroom.",
            length_ft=12, width_ft=11,
            occupants=["adults"], furniture=["bed"],
            budget_tier="medium", budget_inr=200_000,
            vastu="strict",
            expect_items=["bed"], expect_audience="adult",
            # No bearings supplied, so a compliance claim cannot be made.
            expect_vastu_verifiable=False, expect_within_budget=True,
        ),
        EvalCase(
            name="bedroom_strict_vastu_claims_light",
            room_type="bedroom",
            preferences="A bright bedroom with plenty of natural light.",
            length_ft=12, width_ft=10,
            occupants=["adults"], furniture=["bed"],
            budget_tier="medium", budget_inr=180_000,
            vastu="strict",
            expect_items=["bed"],
            expect_vastu_verifiable=False, expect_within_budget=True,
        ),
    ]


def _living_rooms() -> List[EvalCase]:
    return [
        EvalCase(
            name="living_room_family",
            room_type="living_room",
            preferences="A family living room with a sofa and coffee table.",
            length_ft=14, width_ft=12, ceiling_ft=10,
            occupants=["adults", "child"], furniture=["sofa", "table"],
            budget_tier="high", budget_inr=350_000,
            expect_items=["sofa"], expect_audience="adult",
            expect_within_budget=True,
        ),
        EvalCase(
            name="living_room_adults_only",
            room_type="living_room",
            preferences="A formal living room for adults.",
            length_ft=15, width_ft=12,
            occupants=["adults"], furniture=["sofa", "armchair"],
            style="traditional_indian", budget_tier="high", budget_inr=500_000,
            expect_items=["sofa"], expect_audience="adult",
            expect_within_budget=True,
        ),
        EvalCase(
            name="living_room_small",
            room_type="living_room",
            preferences="A small living room under 100 sq ft, needs to fit a sofa.",
            length_ft=9, width_ft=8, ceiling_ft=9,
            occupants=["adults"], furniture=["sofa"],
            budget_tier="medium", budget_inr=120_000,
            expect_items=["sofa"], expect_audience="adult",
            expect_within_budget=True,
        ),
        EvalCase(
            name="living_room_tv",
            room_type="living_room",
            preferences="A living room with a TV unit and sofa.",
            length_ft=14, width_ft=11,
            occupants=["adults"], furniture=["sofa", "tv_unit"],
            budget_tier="medium", budget_inr=200_000,
            expect_items=["sofa"], expect_within_budget=True,
        ),
        EvalCase(
            name="living_room_no_dimensions",
            room_type="living_room",
            preferences="A living room with a sofa.",
            occupants=["adults"], furniture=["sofa"],
            budget_tier="medium", budget_inr=150_000,
            expect_items=["sofa"], expect_within_budget=True,
        ),
    ]


def _kids_rooms() -> List[EvalCase]:
    return [
        EvalCase(
            name="kids_two_children",
            room_type="kids_room",
            preferences="A kids room for two children with a bunk bed and study desk.",
            length_ft=12, width_ft=10, ceiling_ft=9,
            occupants=["child", "child"], furniture=["bed", "desk", "wardrobe"],
            budget_tier="medium", budget_inr=200_000,
            expect_items=["bed", "desk", "wardrobe"],
            expect_audience="kids", expect_within_budget=True,
        ),
        EvalCase(
            name="kids_single_child",
            room_type="kids_room",
            preferences="A room for one child with a cot and a bookshelf.",
            length_ft=10, width_ft=9,
            occupants=["child"], furniture=["bed", "storage"],
            budget_tier="low", budget_inr=80_000,
            expect_items=["bed"], expect_audience="kids",
            expect_within_budget=True,
        ),
        EvalCase(
            name="kids_tight_budget",
            room_type="kids_room",
            preferences="A children's bedroom, bed and wardrobe.",
            length_ft=10, width_ft=9,
            occupants=["child"], furniture=["bed", "wardrobe"],
            budget_tier="low", budget_inr=25_000,
            expect_items=["bed", "wardrobe"],
            expect_audience="kids", expect_within_budget=False,
        ),
        EvalCase(
            name="kids_no_occupants_stated",
            room_type="kids_room",
            preferences="A bedroom for a child.",
            length_ft=11, width_ft=9,
            furniture=["bed"],
            budget_tier="low", budget_inr=70_000,
            expect_items=["bed"], expect_within_budget=True,
        ),
    ]


def _kitchens_and_dining() -> List[EvalCase]:
    return [
        EvalCase(
            name="kitchen_compact",
            room_type="kitchen",
            preferences="A compact modular kitchen.",
            length_ft=10, width_ft=8, ceiling_ft=9,
            occupants=["adults"],
            budget_tier="medium", budget_inr=300_000,
            expect_within_budget=True,
        ),
        EvalCase(
            name="kitchen_large",
            room_type="kitchen",
            preferences="A large kitchen with a dining table nearby.",
            length_ft=14, width_ft=12,
            occupants=["adults", "child"],
            budget_tier="high", budget_inr=800_000,
            expect_within_budget=True,
        ),
        EvalCase(
            name="dining_family",
            room_type="dining",
            preferences="A dining table for six people.",
            length_ft=13, width_ft=11,
            occupants=["adults"], furniture=["table"],
            budget_tier="medium", budget_inr=250_000,
            expect_items=["table"], expect_audience="adult",
            expect_within_budget=True,
        ),
    ]


def _edge_cases() -> List[EvalCase]:
    """
    Cases that exist to break things: impossible budgets, missing dimensions,
    contradictory instructions, and briefs that name adults and children at once.
    """
    return [
        EvalCase(
            name="edge_budget_below_any_product",
            room_type="bedroom",
            preferences="A bedroom with a bed.",
            length_ft=10, width_ft=9,
            occupants=["adults"], furniture=["bed"],
            budget_tier="low", budget_inr=1_000,
            expect_items=["bed"], expect_audience="adult",
            expect_within_budget=True,  # buy nothing, overspend nothing
        ),
        EvalCase(
            name="edge_adults_and_kids_together",
            room_type="bedroom",
            preferences="A shared room for adults and a child.",
            length_ft=12, width_ft=10,
            occupants=["adults", "child"], furniture=["bed"],
            budget_tier="medium", budget_inr=200_000,
            expect_items=["bed"],
            # Adults win the tie, so no children's furniture.
            expect_audience="adult", expect_within_budget=True,
        ),
        EvalCase(
            name="edge_no_furniture_listed",
            room_type="living_room",
            preferences="Just make the room nice.",
            length_ft=12, width_ft=11,
            occupants=["adults"],
            budget_tier="medium", budget_inr=150_000,
            expect_within_budget=True,
        ),
        EvalCase(
            name="edge_minimum_dimensions",
            room_type="bedroom",
            preferences="A very small bedroom with a bed.",
            length_ft=7, width_ft=6, ceiling_ft=8,
            occupants=["adults"], furniture=["bed"],
            budget_tier="low", budget_inr=90_000,
            expect_items=["bed"], expect_within_budget=True,
        ),
        EvalCase(
            name="edge_large_apartment",
            room_type="living_room",
            preferences="A large living room with a sofa and dining table.",
            length_ft=30, width_ft=25, ceiling_ft=12,
            occupants=["adults"], furniture=["sofa"],
            budget_tier="luxury", budget_inr=2_000_000,
            expect_items=["sofa"], expect_within_budget=True,
        ),
        EvalCase(
            name="edge_vastu_strict_with_both_bearings",
            room_type="bedroom",
            preferences="Bed head to the south, strict Vastu.",
            length_ft=12, width_ft=10,
            occupants=["adults"], furniture=["bed"],
            budget_tier="medium", budget_inr=200_000,
            door_direction="west", window_direction="south",
            vastu="strict",
            expect_items=["bed"], expect_vastu_verifiable=True,
            expect_within_budget=True,
        ),
        EvalCase(
            name="edge_many_items_requested",
            room_type="bedroom",
            preferences="Bed, mattress, wardrobe, study desk and bookshelf.",
            length_ft=14, width_ft=12,
            occupants=["adults"],
            furniture=["bed", "mattress", "wardrobe", "desk", "storage"],
            budget_tier="high", budget_inr=600_000,
            expect_items=["bed", "mattress", "wardrobe", "desk"],
            expect_audience="adult", expect_within_budget=True,
        ),
        EvalCase(
            name="edge_soft_vastu_no_bearings",
            room_type="living_room",
            preferences="Keep Vastu in mind but it is not critical.",
            length_ft=13, width_ft=11,
            occupants=["adults"], furniture=["sofa"],
            budget_tier="medium", budget_inr=250_000,
            vastu="soft",
            expect_items=["sofa"], expect_within_budget=True,
        ),
        EvalCase(
            name="edge_ignore_vastu",
            room_type="bedroom",
            preferences="Do not worry about Vastu at all.",
            length_ft=12, width_ft=10,
            occupants=["adults"], furniture=["bed"],
            budget_tier="medium", budget_inr=180_000,
            vastu="ignore",
            expect_items=["bed"], expect_within_budget=True,
        ),
        EvalCase(
            name="edge_kitchen_strict_vastu",
            room_type="kitchen",
            preferences="Vastu-compliant kitchen.",
            length_ft=12, width_ft=10,
            occupants=["adults"],
            budget_tier="high", budget_inr=700_000,
            vastu="strict",
            expect_vastu_verifiable=False, expect_within_budget=True,
        ),
    ]


def _more_bedrooms() -> List[EvalCase]:
    """Bedroom variations that stress different parts of the contract."""
    return [
        EvalCase(
            name="bedroom_bunk_two_children",
            room_type="kids_room",
            preferences="Two kids share a room, need 2 beds and a wardrobe.",
            length_ft=12, width_ft=10,
            occupants=["child", "child"], furniture=["bed", "wardrobe"],
            budget_tier="low", budget_inr=40_000,
            expect_items=["bed", "wardrobe"],
            expect_audience="kids", expect_within_budget=False,
        ),
        EvalCase(
            name="bedroom_guest_room",
            room_type="bedroom",
            preferences="A guest room for occasional visitors.",
            length_ft=11, width_ft=10,
            occupants=["adults"], furniture=["bed"],
            budget_tier="low", budget_inr=80_000,
            expect_items=["bed"], expect_audience="adult",
            expect_within_budget=True,
        ),
        EvalCase(
            name="bedroom_wardrobe_heavy",
            room_type="bedroom",
            preferences="A bedroom where storage is the priority.",
            length_ft=12, width_ft=11,
            occupants=["adults"], furniture=["wardrobe", "bed"],
            budget_tier="high", budget_inr=400_000,
            expect_items=["wardrobe", "bed"],
            expect_audience="adult", expect_within_budget=True,
        ),
        EvalCase(
            name="bedroom_no_dimensions_strict",
            room_type="bedroom",
            preferences="A Vastu-compliant bedroom, strictly.",
            occupants=["adults"], furniture=["bed"],
            budget_tier="medium", budget_inr=200_000,
            vastu="strict",
            expect_items=["bed"], expect_vastu_verifiable=False,
            expect_within_budget=True,
        ),
        EvalCase(
            name="bedroom_ceiling_low",
            room_type="bedroom",
            preferences="A bedroom with a low ceiling.",
            length_ft=12, width_ft=10, ceiling_ft=8,
            occupants=["adults"], furniture=["bed"],
            budget_tier="medium", budget_inr=180_000,
            expect_items=["bed"], expect_within_budget=True,
        ),
    ]


def _more_living() -> List[EvalCase]:
    return [
        EvalCase(
            name="living_room_strict_vastu_bearings",
            room_type="living_room",
            preferences="A strictly Vastu-compliant living room.",
            length_ft=14, width_ft=12, ceiling_ft=10,
            occupants=["adults"], furniture=["sofa"],
            budget_tier="high", budget_inr=400_000,
            door_direction="north", window_direction="south",
            vastu="strict",
            expect_items=["sofa"], expect_vastu_verifiable=True,
            expect_within_budget=True,
        ),
        EvalCase(
            name="living_room_reading_nook",
            room_type="living_room",
            preferences="A reading corner with an armchair and side table.",
            length_ft=12, width_ft=10,
            occupants=["adults"], furniture=["armchair", "table"],
            budget_tier="medium", budget_inr=180_000,
            expect_audience="adult", expect_within_budget=True,
        ),
        EvalCase(
            name="living_room_kids_play",
            room_type="living_room",
            preferences="A play area for toddlers with storage.",
            length_ft=12, width_ft=11,
            occupants=["child"], furniture=["storage"],
            budget_tier="low", budget_inr=70_000,
            expect_audience="kids", expect_within_budget=True,
        ),
        EvalCase(
            name="living_room_budget_tight",
            room_type="living_room",
            preferences="A living room with just a sofa.",
            length_ft=12, width_ft=10,
            occupants=["adults"], furniture=["sofa"],
            budget_tier="low", budget_inr=15_000,
            expect_items=["sofa"], expect_audience="adult",
            expect_within_budget=False,
        ),
        EvalCase(
            name="living_room_luxury",
            room_type="living_room",
            preferences="A luxury living room with the best furniture available.",
            length_ft=18, width_ft=15, ceiling_ft=12,
            occupants=["adults"], furniture=["sofa", "armchair", "table"],
            style="luxury", budget_tier="luxury", budget_inr=1_500_000,
            expect_items=["sofa"], expect_audience="adult",
            expect_within_budget=True,
        ),
    ]


def _more_misc() -> List[EvalCase]:
    return [
        EvalCase(
            name="dining_wfh_hybrid",
            room_type="dining",
            preferences="A dining area that also serves as a work desk.",
            length_ft=12, width_ft=10,
            occupants=["adults"], furniture=["table", "desk"],
            budget_tier="medium", budget_inr=250_000,
            expect_items=["table", "desk"], expect_audience="adult",
            expect_within_budget=True,
        ),
        EvalCase(
            name="kitchen_family_tight",
            room_type="kitchen",
            preferences="A practical kitchen on a tight budget.",
            length_ft=9, width_ft=8,
            occupants=["adults"],
            budget_tier="low", budget_inr=150_000,
            expect_within_budget=False,
        ),
        EvalCase(
            name="home_office_two_desks",
            room_type="home_office",
            preferences="Two people work from home, need space for both.",
            length_ft=12, width_ft=10,
            occupants=["adults", "adults"], furniture=["desk", "chair"],
            budget_tier="high", budget_inr=300_000,
            expect_items=["desk", "chair"], expect_audience="adult",
            expect_within_budget=True,
        ),
        EvalCase(
            name="home_office_wfh_no_furniture_words",
            room_type="bedroom",
            preferences="I work from home in the afternoons.",
            length_ft=12, width_ft=10,
            occupants=["adults"],
            budget_tier="medium", budget_inr=180_000,
            # Desk and chair are derived from "work from home" alone.
            expect_items=["desk", "chair"], expect_within_budget=True,
        ),
        EvalCase(
            name="kids_room_teen_study",
            room_type="kids_room",
            preferences="A teenage study room with a desk and bookshelf.",
            length_ft=11, width_ft=9,
            occupants=["teen"], furniture=["desk", "storage"],
            budget_tier="medium", budget_inr=150_000,
            expect_items=["desk"], expect_within_budget=True,
        ),
        EvalCase(
            name="bedroom_adults_kids_wardrobe",
            room_type="bedroom",
            preferences="A bedroom for adults that also stores a cot.",
            length_ft=13, width_ft=11,
            occupants=["adults"], furniture=["wardrobe", "bed"],
            budget_tier="high", budget_inr=350_000,
            expect_items=["wardrobe", "bed"], expect_audience="adult",
            expect_within_budget=True,
        ),
    ]


def all_cases() -> List[EvalCase]:
    """
    The full corpus.

    Assembled from themed groups so a failure points at a category rather than
    at an index in a flat list.
    """
    cases: List[EvalCase] = []
    cases.extend(_bedrooms())
    cases.extend(_more_bedrooms())
    cases.extend(_living_rooms())
    cases.extend(_more_living())
    cases.extend(_kids_rooms())
    cases.extend(_kitchens_and_dining())
    cases.extend(_more_misc())
    cases.extend(_edge_cases())
    return cases


def corpus_summary() -> Dict[str, Any]:
    cases = all_cases()
    return {
        "total": len(cases),
        "room_types": sorted({c.room_type for c in cases}),
        "with_strict_vastu": sum(1 for c in cases if c.vastu == "strict"),
        "with_bearings": sum(
            1 for c in cases if c.door_direction and c.window_direction
        ),
        "without_dimensions": sum(
            1 for c in cases if not (c.length_ft and c.width_ft)
        ),
        "expected_over_budget": sum(
            1 for c in cases if c.expect_within_budget is False
        ),
        "expected_infeasible_vastu": sum(
            1 for c in cases if c.expect_vastu_verifiable is False
        ),
        "audience_locked": sum(1 for c in cases if c.expect_audience),
    }


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(corpus_summary(), indent=2))