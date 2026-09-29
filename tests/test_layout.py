"""
Tests for the layout engine.

The layout engine is the one part of the pipeline that must be *geometrically*
correct, so these tests assert arithmetic relationships rather than checking
that a call did not raise: no footprint may leave the room, no two may
overlap, and every pair must respect the clearance rule.

They are fully deterministic - no LLM, no randomness - so a failure here
always means a real regression in placement logic.
"""

from __future__ import annotations

import math

import pytest

from app.engine.layout import (
    MIN_CLEARANCE_CM,
    PlacedItem,
    door_zone,
    place_furniture,
    rectangles_overlap,
    score_layout,
)


# A 12ft x 10ft Indian kids' room, the case from the bug report.
ROOM_W = 305.0   # 10 ft
ROOM_D = 366.0   # 12 ft


def item(sku, name, category, w, d, h, qty=1):
    return {
        "sku": sku, "name": name, "category": category,
        "width_cm": w, "depth_cm": d, "height_cm": h, "quantity": qty,
    }


BED = item("BED-1", "Kids Cot", "bed", 90, 190, 60, 2)
DESK = item("DSK-1", "Study Table", "desk", 90, 55, 72)
WARDROBE = item("WRD-1", "Wardrobe", "wardrobe", 90, 55, 180)


# --------------------------------------------------------------------------
# Pure geometry
# --------------------------------------------------------------------------


class TestGeometry:
    def test_overlap_detected(self):
        a = PlacedItem("a", "A", "bed", "north", 0, 0, 100, 100, 0)
        b = PlacedItem("b", "B", "bed", "north", 50, 50, 100, 100, 0)
        assert rectangles_overlap(a, b)

    def test_touching_edges_do_not_overlap(self):
        """Edge-to-edge is a valid, snug arrangement - not a collision."""
        a = PlacedItem("a", "A", "bed", "north", 0, 0, 100, 100, 0)
        b = PlacedItem("b", "B", "bed", "north", 100, 0, 100, 100, 0)
        assert not rectangles_overlap(a, b)

    def test_fully_contained_overlaps(self):
        a = PlacedItem("a", "A", "bed", "north", 0, 0, 300, 300, 0)
        b = PlacedItem("b", "B", "bed", "north", 10, 10, 20, 20, 0)
        assert rectangles_overlap(a, b)

    def test_door_zone_on_each_wall(self):
        assert door_zone("north", ROOM_W, ROOM_D)[1] == 0.0
        assert door_zone("south", ROOM_W, ROOM_D)[1] == ROOM_D - 90
        assert door_zone("west", ROOM_W, ROOM_D)[0] == 0.0
        assert door_zone("east", ROOM_W, ROOM_D)[0] == ROOM_W - 90

    def test_no_door_no_zone(self):
        assert door_zone(None, ROOM_W, ROOM_D) is None


# --------------------------------------------------------------------------
# The invariants that must always hold
# --------------------------------------------------------------------------


class TestInvariants:
    """Properties checked against every layout the engine produces."""

    @pytest.fixture
    def plan(self):
        return place_furniture(
            [BED, DESK, WARDROBE], ROOM_W, ROOM_D, 270,
            door_direction="south", window_direction="east",
        )

    def test_nothing_leaves_the_room(self, plan):
        for i in plan.items:
            assert i.x >= -0.1, f"{i.name} starts outside on x"
            assert i.y >= -0.1, f"{i.name} starts outside on y"
            assert i.right <= ROOM_W + 0.1, f"{i.name} overflows the right wall"
            assert i.bottom <= ROOM_D + 0.1, f"{i.name} overflows the bottom wall"

    def test_no_two_items_overlap(self, plan):
        for idx, a in enumerate(plan.items):
            for b in plan.items[idx + 1:]:
                assert not rectangles_overlap(a, b), f"{a.name} overlaps {b.name}"

    def test_clearance_is_respected(self, plan):
        for idx, a in enumerate(plan.items):
            for b in plan.items[idx + 1:]:
                dx = max(b.x - a.right, a.x - b.right, 0.0)
                dy = max(b.y - a.bottom, a.y - b.bottom, 0.0)
                gap = dy if dx == 0 else dx if dy == 0 else math.hypot(dx, dy)
                assert gap >= MIN_CLEARANCE_CM - 0.5, (
                    f"{a.name} and {b.name} are only {gap:.0f} cm apart"
                )

    def test_door_swing_is_never_blocked(self, plan):
        zone = door_zone("south", ROOM_W, ROOM_D)
        zx, zy, zs = zone
        box = PlacedItem("door", "Door", "opening", "door", zx, zy, zs, zs, 0)
        for i in plan.items:
            if i.height > 30:
                assert not rectangles_overlap(i, box), f"{i.name} blocks the door"

    def test_wall_items_are_flush_to_a_wall(self, plan):
        for i in plan.items:
            if i.wall == "north":
                assert i.y == 0.0
            elif i.wall == "south":
                assert abs(i.bottom - ROOM_D) < 0.1
            elif i.wall == "west":
                assert i.x == 0.0
            elif i.wall == "east":
                assert abs(i.right - ROOM_W) < 0.1

    def test_all_items_requested_are_placed(self, plan):
        assert not plan.unplaced, [u["name"] for u in plan.unplaced]

    def test_score_is_perfect_for_a_clean_layout(self, plan):
        assert score_layout(plan)["overall"] == 100


# --------------------------------------------------------------------------
# Behaviour under pressure
# --------------------------------------------------------------------------


class TestPlacement:
    def test_tall_items_avoid_the_window(self):
        """A wardrobe must not be placed in front of the window."""
        plan = place_furniture(
            [item("WRD", "Wardrobe", "wardrobe", 90, 55, 180, 3)],
            ROOM_W, ROOM_D, 270, window_direction="east",
        )
        for i in plan.items:
            if i.wall == "east" and i.height > 90:
                # On the east wall it must sit clear of the window band.
                assert i.y >= 30 or (i.y + i.depth) <= 120

    def test_oversized_item_is_reported_not_dropped_silently(self):
        """A wardrobe wider than the room cannot be placed, and must say so."""
        plan = place_furniture(
            [item("HUGE", "Huge Unit", "wardrobe", 400, 55, 180)],
            ROOM_W, ROOM_D,
        )
        assert plan.unplaced
        assert plan.unplaced[0]["sku"] == "HUGE"
        assert score_layout(plan)["overall"] < 100

    def test_oversized_but_rotatable_item_fits(self):
        """
        A 340x30 item is wider than the 305 cm wall but shallower than the
        366 cm depth, so it only fits rotated. Without the rotated pass, a
        plausible narrow item would be wrongly reported unplaceable.
        """
        plan = place_furniture(
            [item("SLIM", "Slim Shelf", "storage", 340, 30, 80)],
            ROOM_W, ROOM_D,
        )
        assert plan.items, "a 340x30 shelf fits when rotated"
        assert plan.items[0].rotated is True
        assert plan.items[0].width == 30 and plan.items[0].depth == 340

    def test_item_that_fits_unrotated_is_not_needlessly_rotated(self):
        """Rotation is a fallback, not a preference."""
        plan = place_furniture(
            [item("SLIM", "Slim Shelf", "storage", 300, 20, 80)],
            ROOM_W, ROOM_D,
        )
        assert plan.items[0].rotated is False

    def test_item_without_dimensions_is_reported(self):
        plan = place_furniture(
            [item("NO-D", "Mystery", "bed", 0, 0, 0)], ROOM_W, ROOM_D
        )
        assert plan.unplaced[0]["sku"] == "NO-D"
        assert "dimension" in plan.unplaced[0]["reason"]

    def test_deterministic(self):
        """Same inputs must give byte-identical output, or scores are noise."""
        a = place_furniture([BED, DESK, WARDROBE], ROOM_W, ROOM_D, 270)
        b = place_furniture([BED, DESK, WARDROBE], ROOM_W, ROOM_D, 270)
        assert [i.to_dict() for i in a.items] == [i.to_dict() for i in b.items]

    def test_large_item_placed_first(self):
        """Bulky items claim the good walls, so ordering must be by footprint."""
        plan = place_furniture(
            [item("RUG", "Rug", "soft", 150, 150, 2),
             item("WRD", "Wardrobe", "wardrobe", 90, 55, 180)],
            ROOM_W, ROOM_D,
        )
        # The wardrobe (4950 cm2) should be placed before the rug (22500 cm2)?
        # No - the rug is bigger, so the rug goes first. The point is the
        # order is by descending footprint, which makes the fit predictable.
        assert plan.items[0].name in {"Rug", "Wardrobe"}
        assert plan.items[0].name == "Rug"   # 22500 > 4950

    def test_crowded_room_still_reports_honestly(self):
        """Too much furniture must produce unplaced items, not silent overlap."""
        plan = place_furniture(
            [BED, DESK, WARDROBE,
             item("SHV", "Shelf", "storage", 60, 30, 90, 4),
             item("TBL", "Table", "table", 40, 40, 45, 3)],
            ROOM_W, ROOM_D, 270, door_direction="south",
        )
        for idx, a in enumerate(plan.items):
            for b in plan.items[idx + 1:]:
                assert not rectangles_overlap(a, b)
        # Either it fits, or it is reported. Never both "placed" and overlapping.
        assert plan.unplaced or len(plan.items) == 2 + 1 + 1 + 4 + 3


# --------------------------------------------------------------------------
# Safety and scoring
# --------------------------------------------------------------------------


class TestSafety:
    def test_low_heavy_storage_is_flagged_for_children(self):
        plan = place_furniture(
            [item("CHV", "Toy Chest", "storage", 70, 40, 45)],
            ROOM_W, ROOM_D,
        )
        codes = [i.code for i in plan.issues]
        assert "low_heavy_storage" in codes

    def test_tall_free_standing_item_flagged(self):
        plan = place_furniture(
            [item("LMP", "Floor Lamp", "lighting", 40, 40, 150)],
            ROOM_W, ROOM_D,
        )
        codes = [i.code for i in plan.issues]
        assert "tall_item_free_standing" in codes

    def test_issue_serialises(self):
        plan = place_furniture(
            [item("CHV", "Toy Chest", "storage", 70, 40, 45)], ROOM_W, ROOM_D,
        )
        d = plan.issues[0].to_dict()
        assert set(d) >= {"code", "severity", "message"}


class TestScoring:
    def test_empty_room_scores_full(self):
        plan = place_furniture([], ROOM_W, ROOM_D)
        assert score_layout(plan)["overall"] == 100

    def test_unplaced_reduces_score(self):
        full = place_furniture([BED], ROOM_W, ROOM_D)
        partial = place_furniture(
            [BED, item("HUGE", "Huge", "wardrobe", 400, 55, 180)], ROOM_W, ROOM_D
        )
        assert score_layout(partial)["overall"] < score_layout(full)["overall"]

    def test_score_never_negative(self):
        plan = place_furniture(
            [item(f"BIG{i}", f"Big{i}", "wardrobe", 200, 100, 200) for i in range(9)],
            ROOM_W, ROOM_D,
        )
        assert score_layout(plan)["overall"] >= 0

    def test_to_dict_is_json_shaped(self):
        import json
        plan = place_furniture([BED, DESK], ROOM_W, ROOM_D, 270,
                               door_direction="north", window_direction="west")
        d = plan.to_dict()
        json.dumps(d)  # must not raise
        assert set(d) >= {"room", "items", "issues", "unplaced", "score"}
        assert d["room"]["door_direction"] == "north"
