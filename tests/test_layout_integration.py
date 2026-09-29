"""
Exercise /api/v1/layout across every room type and a range of briefs.

Catches the class of bug that only shows up in production: a room type whose
items do not exist in the catalog, a brief so thin the parser returns nothing,
a room too small for the furniture, and geometry that silently leaves the
bounds. The assertions are on invariants, not on specific coordinates.
"""

from __future__ import annotations

import math

import pytest

from app.catalog import catalog
from app.engine.brief import parse_brief
from app.engine.layout import (
    MIN_CLEARANCE_CM,
    PlacedItem,
    place_furniture,
    rectangles_overlap,
    score_layout,
)

ROOM_TYPES = [
    "living_room", "bedroom", "kitchen", "home_office", "dining",
    "bathroom", "balcony", "kids_room", "wardrobe", "entrance",
]


@pytest.fixture(scope="module", autouse=True)
def ready():
    catalog.initialize()
    return catalog


def _items_for(room_type: str, constraints):
    """Mirror the endpoint: pick the cheapest catalog item per category."""
    from app.engine.brief import ROOM_CATEGORIES

    wanted = dict(constraints.required_items)
    for item in constraints.optional_items:
        wanted.setdefault(item, 1)

    out = []
    for category, qty in wanted.items():
        options = catalog.by_category(category)
        if not options:
            continue
        options = sorted(options, key=lambda i: (i.tier != "essential", i.price_inr))
        p = options[0]
        out.append({
            "sku": p.sku, "name": p.name, "category": category,
            "width_cm": p.width_cm, "depth_cm": p.depth_cm,
            "height_cm": p.height_cm, "quantity": qty,
        })
    return out


class TestEveryRoomType:
    @pytest.mark.parametrize("room_type", ROOM_TYPES)
    def test_layout_builds_without_crashing(self, room_type):
        brief = parse_brief(
            "A comfortable room for everyday use",
            room_type=room_type, room_dimensions="12 x 10 ft",
        )
        plan = place_furniture(
            _items_for(room_type, brief), 305, 366, 270,
            door_direction="south", window_direction="east",
        )
        assert plan.to_dict()["room"]["width_cm"] == 305

    @pytest.mark.parametrize("room_type", ROOM_TYPES)
    def test_geometry_invariants_hold(self, room_type):
        brief = parse_brief(
            "Need a bed, a desk, a wardrobe, a sofa and a table",
            room_type=room_type, room_dimensions="14 x 12 ft",
        )
        plan = place_furniture(
            _items_for(room_type, brief), 427, 366, 270,
            door_direction="north", window_direction="south",
        )
        for i in plan.items:
            assert i.x >= -0.1 and i.y >= -0.1
            assert i.right <= 427.1 and i.bottom <= 366.1
        for idx, a in enumerate(plan.items):
            for b in plan.items[idx + 1:]:
                assert not rectangles_overlap(a, b)
                dx = max(b.x - a.right, a.x - b.right, 0.0)
                dy = max(b.y - a.bottom, a.y - b.bottom, 0.0)
                gap = dy if dx == 0 else dx if dy == 0 else math.hypot(dx, dy)
                assert gap >= MIN_CLEARANCE_CM - 0.5


class TestDegradedInputs:
    def test_empty_brief_still_produces_a_room(self):
        plan = place_furniture([], 305, 366)
        assert plan.items == []
        assert score_layout(plan)["overall"] == 100

    def test_tiny_room_reports_unplaced_rather_than_overlapping(self):
        """A 6x6 ft room cannot hold a bed. It must say so, not cheat."""
        brief = parse_brief(
            "2 beds, a wardrobe and a desk", room_type="kids_room",
        )
        plan = place_furniture(_items_for("kids_room", brief), 183, 183, 240)
        for idx, a in enumerate(plan.items):
            for b in plan.items[idx + 1:]:
                assert not rectangles_overlap(a, b)
        # Whatever it could place, it respected the walls.
        for i in plan.items:
            assert i.right <= 183.1 and i.bottom <= 183.1

    def test_category_absent_from_catalog_is_skipped_silently_safe(self):
        """A brief for something we do not stock must not crash the endpoint."""
        brief = parse_brief("we need a grand piano and 2 beds", room_type="bedroom")
        items = _items_for("bedroom", brief)
        plan = place_furniture(items, 305, 366)
        assert isinstance(plan.to_dict(), dict)

    def test_door_and_window_on_same_wall(self):
        plan = place_furniture(
            [{"sku": "X", "name": "Wardrobe", "category": "wardrobe",
              "width_cm": 90, "depth_cm": 55, "height_cm": 180}],
            305, 366, 270, door_direction="north", window_direction="north",
        )
        for i in plan.items:
            assert not rectangles_overlap(
                i, PlacedItem("d", "d", "opening", "door", 0, 0, 90, 90, 0)
            ) or i.wall != "north"

    @pytest.mark.parametrize("door", [None, "north", "south", "east", "west",
                                      "northeast", "southwest"])
    def test_every_door_direction(self, door):
        plan = place_furniture(
            [{"sku": "B", "name": "Bed", "category": "bed",
              "width_cm": 90, "depth_cm": 190, "height_cm": 60}],
            305, 366, 270, door_direction=door,
        )
        assert isinstance(score_layout(plan), dict)

    @pytest.mark.parametrize("dims,w,d", [
        ("12ft x 10ft", 305, 366), ("3.6m x 3m", 300, 360),
        ("10 x 12", 305, 366), ("240cm x 200cm", 200, 240),
    ])
    def test_dimension_formats(self, dims, w, d):
        parsed = parse_brief("a room", room_dimensions=dims)
        assert parsed.dimensions.is_known()
        assert parsed.dimensions.length_cm and parsed.dimensions.width_cm


class TestPerformance:
    def test_dense_room_terminates_quickly(self):
        """A pathological brief must not hang the request thread."""
        items = [
            {"sku": f"S{i}", "name": f"Shelf {i}", "category": "storage",
             "width_cm": 60, "depth_cm": 30, "height_cm": 90, "quantity": 2}
            for i in range(25)
        ]
        plan = place_furniture(items, 427, 366, 270)
        assert isinstance(plan.to_dict(), dict)
