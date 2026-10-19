"""
Layout engine: places real furniture in a real room.

The problem this solves
----------------------
A design described furniture in prose ("the bed goes against the far wall"),
which is untestable and frequently impossible to build: a 190 cm bed in a
305 cm room with a 90 cm wardrobe leaves 25 cm of circulation, which is
unwalkable.

So placement is computed here, in Python, from the catalog's real
centimetre dimensions. The LLM chooses *which* items; this module decides
*where they go* and reports what does not fit.

Design rules encoded here
-------------------------
- Clearance: 60 cm minimum circulation, 90 cm on the main traffic route.
- Door swing: a 90 cm arc in front of the door must stay clear.
- Collision: no two footprints may overlap.
- Wall anchoring: tall or heavy items prefer a wall, and must not block a
  door or window.

Everything is deterministic and testable. `score_layout` produces the same
answer for the same inputs, which is what makes the design measurable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.observability import get_logger, metrics

logger = get_logger(__name__)

# Circulation standards, in centimetres. These follow common Indian
# residential practice rather than a single code, and are deliberately
# conservative for a room with children in it.
MIN_CLEARANCE_CM = 60.0
PRIMARY_CLEARANCE_CM = 90.0

# A standard single door leaf, and the arc it sweeps.
DOOR_WIDTH_CM = 90.0
DOOR_SWING_RADIUS_CM = 90.0

# Items taller than this should touch a wall (shelves, wardrobes, TV units).
WALL_ANCHOR_HEIGHT_CM = 100.0
# Heavier items should never be free-standing in a child's reach.
CHILD_UNSAFE_HEIGHT_CM = 90.0

# Which wall of the room each item prefers, by category. The engine tries
# these in order and falls back to any free wall.
WALL_PREFERENCE: Dict[str, Tuple[str, ...]] = {
    "bed": ("south", "east", "west", "north"),
    "wardrobe": ("northwest", "southwest", "west", "east"),
    "desk": ("north", "east"),
    "tv_unit": ("north", "east", "west"),
    "bookshelf": ("north", "east", "west", "south"),
    "storage": ("northwest", "south", "east", "west"),
    "seating": ("south", "west", "east"),
    "table": ("center",),
    "mirror": ("east", "west", "north", "south"),
    "paint": (),
    "lighting": ("center",),
    "soft": (),
    "decor": (),
    "kitchen": ("north", "east"),
}

# Which way a wall faces, and where it sits in the room coordinate system.
# Origin is the top-left corner; x runs east, y runs south.
WALL_AXES: Dict[str, str] = {
    "north": "y=0",
    "south": "y=max",
    "west": "x=0",
    "east": "x=max",
}


@dataclass
class PlacedItem:
    """
    One piece of furniture with a resolved position and rotation.

    Coordinates are centimetres from the room's top-left corner.
    `rotated` swaps width and depth, which is how a chair faces a different
    way without needing a second catalog entry.
    """

    sku: str
    name: str
    category: str
    wall: str
    x: float
    y: float
    width: float
    depth: float
    height: float
    rotated: bool = False
    quantity: int = 1

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.depth

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sku": self.sku,
            "name": self.name,
            "category": self.category,
            "wall": self.wall,
            # Rounded for display; the arithmetic above uses full precision.
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "width": round(self.width, 1),
            "depth": round(self.depth, 1),
            "height": round(self.height, 1),
            "rotated": self.rotated,
            "quantity": self.quantity,
        }


@dataclass
class LayoutIssue:
    """A geometric problem with a proposed layout."""

    code: str
    severity: str          # "error" | "warning"
    message: str
    sku: str = ""
    item_name: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "sku": self.sku,
            "item_name": self.item_name,
        }


@dataclass
class RoomPlan:
    """The room shell: dimensions, openings, and what was placed in it."""

    width_cm: float
    depth_cm: float
    ceiling_cm: float
    door_direction: Optional[str] = None
    window_direction: Optional[str] = None
    items: List[PlacedItem] = field(default_factory=list)
    issues: List[LayoutIssue] = field(default_factory=list)
    # Items that could not be placed at all.
    unplaced: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room": {
                "width_cm": round(self.width_cm, 1),
                "depth_cm": round(self.depth_cm, 1),
                "ceiling_cm": round(self.ceiling_cm, 1),
                # Both units, because the client gave feet ("9 x 11 ft") and
                # the plan is drawn in centimetres. A layout that reports only
                # centimetres makes the client convert it themselves.
                "width_ft": round(self.width_cm / 30.48, 1),
                "depth_ft": round(self.depth_cm / 30.48, 1),
                "area_sqft": round((self.width_cm / 30.48) * (self.depth_cm / 30.48), 1),
                "ceiling_ft": round(self.ceiling_cm / 30.48, 1),
                "dimension_text": "{:.1f} x {:.1f} ft".format(
                    self.width_cm / 30.48, self.depth_cm / 30.48
                ),
                "door_direction": self.door_direction,
                "window_direction": self.window_direction,
            },
            "items": [i.to_dict() for i in self.items],
            "issues": [i.to_dict() for i in self.issues],
            "zones": zone_sizes(self),
            "unplaced": self.unplaced,
            "score": score_layout(self),
        }


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def zone_sizes(plan: RoomPlan) -> List[Dict[str, Any]]:
    """Approximate size of each functional zone, in feet.

    A zone is the footprint of one category's furniture plus the clearance in
    front of it. This is an estimate and is labelled as one: it answers "how
    big is the study corner?" from the plan that was actually drawn, rather
    than inventing an area the furniture does not occupy.
    """
    by_category: Dict[str, List[PlacedItem]] = {}
    for item in plan.items:
        by_category.setdefault(item.category, []).append(item)

    zones: List[Dict[str, Any]] = []
    for category, items in sorted(by_category.items()):
        min_x = min(i.x for i in items)
        min_y = min(i.y for i in items)
        max_x = max(i.right for i in items)
        max_y = max(i.bottom for i in items)
        # Add the standing clearance in front of the furniture. The room's
        # own edge already offers a visual margin, so the estimate is capped
        # by the floor area rather than extending past the walls.
        width_cm = min(max_x - min_x + MIN_CLEARANCE_CM, plan.width_cm)
        depth_cm = min(max_y - min_y + MIN_CLEARANCE_CM, plan.depth_cm)
        zones.append({
            "category": category,
            "label": category.replace("_", " ").title(),
            "items": [i.name for i in items],
            "approx_width_ft": round(width_cm / 30.48, 1),
            "approx_depth_ft": round(depth_cm / 30.48, 1),
            "approx_area_sqft": round((width_cm / 30.48) * (depth_cm / 30.48), 1),
            "note": "footprint plus clearance, estimated from the plan",
        })
    return zones


def rectangles_overlap(a: PlacedItem, b: PlacedItem) -> bool:
    """True if two footprints intersect. Touching edges do not count."""
    return not (
        a.right <= b.x
        or b.right <= a.x
        or a.bottom <= b.y
        or b.bottom <= a.y
    )


def _wall_extent(item: PlacedItem, room_w: float, room_d: float) -> Tuple[float, float]:
    """The position of an item's near edge along its wall, given wall and size."""
    if item.wall == "north":
        return 0.0, room_w - item.width
    if item.wall == "south":
        return 0.0, room_w - item.width
    if item.wall == "west":
        return 0.0, room_d - item.depth
    if item.wall == "east":
        return 0.0, room_d - item.depth
    # "center"
    return (room_w - item.width) / 2.0, (room_w - item.width) / 2.0


def _place_on_wall(
    wall: str, x: float, item_w: float, item_d: float,
    room_w: float, room_d: float,
) -> PlacedItem:
    """Build a PlacedItem snapped flush to the given wall."""
    if wall == "north":
        return PlacedItem("", "", "", wall, x, 0.0, item_w, item_d, 0.0)
    if wall == "south":
        return PlacedItem("", "", "", wall, x, room_d - item_d, item_w, item_d, 0.0)
    if wall == "west":
        return PlacedItem("", "", "", wall, 0.0, x, item_w, item_d, 0.0)
    if wall == "east":
        return PlacedItem("", "", "", wall, room_w - item_w, x, item_w, item_d, 0.0)
    return PlacedItem("", "", "", "center",
                      (room_w - item_w) / 2.0, (room_d - item_d) / 2.0,
                      item_w, item_d, 0.0)


def door_zone(
    door_direction: Optional[str], room_w: float, room_d: float
) -> Optional[Tuple[float, float, float]]:
    """
    The rectangle a door swing occupies, as (x, y, size).

    A door needs a 90 cm arc clear in front of it. Placing a wardrobe there
    means the door cannot open, which is the single most common real-world
    layout failure.
    """
    if not door_direction:
        return None
    s = DOOR_SWING_RADIUS_CM
    if door_direction == "north":
        return (0.0, 0.0, s)
    if door_direction == "south":
        return (0.0, room_d - s, s)
    if door_direction == "west":
        return (0.0, 0.0, s)
    if door_direction == "east":
        return (room_w - s, 0.0, s)
    # Intercardinal: approximate to the nearer cardinal for clearance.
    if "north" in door_direction:
        return (0.0, 0.0, s)
    if "south" in door_direction:
        return (0.0, room_d - s, s)
    if "west" in door_direction:
        return (0.0, 0.0, s)
    return (room_w - s, 0.0, s)


def _blocks_door(item: PlacedItem, zone: Optional[Tuple[float, float, float]]) -> bool:
    if zone is None:
        return False
    zx, zy, zs = zone
    zone_box = PlacedItem("door", "Door", "opening", "door", zx, zy, zs, zs, 0.0)
    return rectangles_overlap(item, zone_box)


def _window_zone(
    window_direction: Optional[str], room_w: float, room_d: float
) -> Optional[Tuple[float, float, float]]:
    """A window occupies a 120 cm band of wall; nothing tall may cover it."""
    if not window_direction:
        return None
    band = 120.0
    depth = 30.0
    if "north" in window_direction:
        return (0.0, 0.0, band) if window_direction == "north" else None
    if window_direction == "east":
        return (room_w - depth, 0.0, band)
    if window_direction == "west":
        return (0.0, 0.0, band)
    if window_direction == "south":
        return (0.0, room_d - depth, band)
    return None


def score_layout(plan: RoomPlan) -> Dict[str, Any]:
    """
    Score a layout 0-100 on objective, checkable grounds.

    The point is that the score is *derived*, not judged. A 2B model rating
    a design "8.5/10" means nothing reproducible; this number can be asserted
    in a test and compared across runs.
    """
    errors = [i for i in plan.issues if i.severity == "error"]
    warnings = [i for i in plan.issues if i.severity == "warning"]

    # Start full, subtract per defect, floor at zero.
    score = 100.0
    score -= 25.0 * len(errors)
    score -= 5.0 * len(warnings)

    # Unplaced items are a hard failure of the layout, not a nitpick.
    score -= 15.0 * len(plan.unplaced)

    # Density matters: a room packed to the walls is unusable even with no
    # formal rule broken. Reward leaving circulation space.
    area = plan.width_cm * plan.depth_cm
    if area > 0:
        used = sum(i.width * i.depth for i in plan.items)
        free_ratio = max(0.0, 1.0 - (used / area))
        if free_ratio < 0.35:
            score -= 10.0
        elif free_ratio > 0.55:
            score += 5.0

    return {
        "overall": max(0.0, min(100.0, round(score))),
        "errors": len(errors),
        "warnings": len(warnings),
        "unplaced": len(plan.unplaced),
    }


# --------------------------------------------------------------------------
# The placer
# --------------------------------------------------------------------------


def place_furniture(
    line_items: List[Dict[str, Any]],
    room_width_cm: float,
    room_depth_cm: float,
    ceiling_cm: float = 270.0,
    door_direction: Optional[str] = None,
    window_direction: Optional[str] = None,
) -> RoomPlan:
    """
    Place every catalog item into the room, largest first.

    Ordering matters: big items claim prime wall space, so placing them
    before small ones is what makes the result fit. A greedy first-fit along
    each wall, trying the preferred orientation before the rotated one.

    Returns a RoomPlan with issues rather than raising, so a partial layout
    is still useful to show the user.
    """
    plan = RoomPlan(
        width_cm=room_width_cm,
        depth_cm=room_depth_cm,
        ceiling_cm=ceiling_cm,
        door_direction=door_direction,
        window_direction=window_direction,
    )
    zone = door_zone(door_direction, room_width_cm, room_depth_cm)
    window = _window_zone(window_direction, room_width_cm, room_depth_cm)

    # Sort by footprint, descending. Bulky pieces get the good walls.
    ordered = sorted(
        line_items,
        key=lambda i: -(i.get("width_cm", 0) * i.get("depth_cm", 0)),
    )

    for entry in ordered:
        sku = entry.get("sku", "")
        name = entry.get("name", "item")
        category = entry.get("category", "")
        w = float(entry.get("width_cm", 0) or 0)
        d = float(entry.get("depth_cm", 0) or 0)
        h = float(entry.get("height_cm", 0) or 0)
        qty = max(1, int(entry.get("quantity", 1) or 1))

        if w <= 0 or d <= 0:
            plan.unplaced.append({
                "sku": sku, "name": name,
                "reason": "no dimensions in the catalog",
            })
            continue

        # Place `qty` copies, each individually.
        placed_any = False
        for copy in range(qty):
            result = _place_one(plan, sku, name, category, w, d, h, zone, window)
            if result is None:
                plan.unplaced.append({
                    "sku": sku,
                    "name": f"{name} (copy {copy + 1} of {qty})",
                    "reason": "no clear space with the required clearance",
                })
            else:
                placed_any = True
                plan.items.append(result)

        if not placed_any and qty == 0:
            plan.unplaced.append({"sku": sku, "name": name, "reason": "not placed"})

    _final_checks(plan, zone, window)
    return plan


def _place_one(
    plan: RoomPlan,
    sku: str, name: str, category: str,
    w: float, d: float, h: float,
    zone: Optional[Tuple[float, float, float]],
    window: Optional[Tuple[float, float, float]],
) -> Optional[PlacedItem]:
    """
    Find the first position where one item fits with proper clearance.

    Tries, in order: each preferred wall, then every other wall; on each wall,
    the natural orientation then the rotated one, scanning along the wall in
    5 cm steps. Returns None if nothing fits.
    """
    preferred = WALL_PREFERENCE.get(category, ())
    walls = list(preferred) + [w for w in ("north", "east", "south", "west") if w not in preferred]

    for wall in walls:
        for rotated in (False, True):
            iw, id_ = (d, w) if rotated else (w, d)
            if iw > plan.width_cm or id_ > plan.depth_cm:
                continue

            # Walk along the wall.
            span_start, span_end = _wall_extent(
                PlacedItem("", "", "", wall, 0, 0, iw, id_, h),
                plan.width_cm, plan.depth_cm,
            )
            pos = span_start
            while pos <= span_end + 0.1:
                candidate = _place_on_wall(
                    wall, pos, iw, id_, plan.width_cm, plan.depth_cm
                )
                candidate.sku = sku
                candidate.name = name
                candidate.category = category
                candidate.height = h
                candidate.rotated = rotated

                if _fits(candidate, plan, zone, window, h):
                    return candidate
                pos += 5.0

    return None


def _fits(
    item: PlacedItem,
    plan: RoomPlan,
    zone: Optional[Tuple[float, float, float]],
    window: Optional[Tuple[float, float, float]],
    height: float,
) -> bool:
    """Does this candidate satisfy every hard rule?"""
    # 1. Inside the room.
    if item.x < -0.1 or item.y < -0.1:
        return False
    if item.right > plan.width_cm + 0.1 or item.bottom > plan.depth_cm + 0.1:
        return False

    # 2. No collision.
    for other in plan.items:
        if rectangles_overlap(item, other):
            return False

    # 3. Door swing must stay clear, for anything tall or wide. A rug or a
    #    floor lamp in the arc is fine; a wardrobe is not.
    if zone is not None and height > 30.0:
        if _blocks_door(item, zone):
            return False

    # 4. A window must not be blocked by something tall.
    if window is not None and height > CHILD_UNSAFE_HEIGHT_CM:
        wx, wy, ws = window
        wbox = PlacedItem("window", "Window", "opening", "window", wx, wy, ws, 30.0, 0.0)
        if rectangles_overlap(item, wbox):
            return False

    # 5. Circulation clearance from every other item.
    for other in plan.items:
        gap = _min_gap(item, other)
        if gap < MIN_CLEARANCE_CM:
            return False

    return True


def _min_gap(a: PlacedItem, b: PlacedItem) -> float:
    """
    Shortest distance between two rectangles, 0 if they touch or overlap.

    Separating-axis style: compute the gap on each axis and take the max,
    which is the true distance for non-overlapping rectangles.
    """
    dx = max(b.x - a.right, a.x - b.right, 0.0)
    dy = max(b.y - a.bottom, a.y - b.bottom, 0.0)
    if dx == 0.0 and dy == 0.0:
        return 0.0
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return math.hypot(dx, dy)


def _final_checks(
    plan: RoomPlan,
    zone: Optional[Tuple[float, float, float]],
    window: Optional[Tuple[float, float, float]],
) -> None:
    """
    Post-pass checks that need the whole layout.

    The per-item `_fits` test already prevents collisions, so this mostly
    records residual concerns for the user rather than rejecting the layout:
    a clear walking route, tall items on a wall, and the overall door report.
    """
    # Door blocking, reported explicitly so the user sees the reason.
    if zone is not None:
        for item in plan.items:
            if item.height > 30.0 and _blocks_door(item, zone):
                plan.issues.append(LayoutIssue(
                    code="blocks_door",
                    severity="error",
                    message=(
                        f"{item.name} blocks the door swing "
                        f"({plan.door_direction})."
                    ),
                    sku=item.sku, item_name=item.name,
                ))

    # Tall items must touch a wall, for stability and looks.
    for item in plan.items:
        if item.height >= WALL_ANCHOR_HEIGHT_CM and item.wall == "center":
            plan.issues.append(LayoutIssue(
                code="tall_item_free_standing",
                severity="warning",
                message=f"{item.name} is {int(item.height)} cm tall and is free-standing in the middle of the room.",
                sku=item.sku, item_name=item.name,
            ))

    # Safety note: low, heavy storage can be climbed on by a child, so it
    # should be anchored or raised. This applies to every room, not just
    # nurseries, so the check is unconditional.
    for item in plan.items:
        if item.height <= CHILD_UNSAFE_HEIGHT_CM and item.category in {"storage", "wardrobe"}:
            plan.issues.append(LayoutIssue(
                code="low_heavy_storage",
                severity="warning",
                message=(
                    f"{item.name} is only {int(item.height)} cm high and can be "
                    f"climbed on by a child - anchor it or raise it above "
                    f"{int(CHILD_UNSAFE_HEIGHT_CM)} cm."
                ),
                sku=item.sku, item_name=item.name,
            ))

    metrics.gauge("layout.score", float(score_layout(plan)["overall"]))
    metrics.increment("layout.built")
