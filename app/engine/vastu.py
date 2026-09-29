"""
Rule-based Vastu engine.

Why rules in code and not a prompt
----------------------------------
The original system asked a 2B model to "apply Vastu" and got back generic
sentences like "position the bed in the southwest for stability". That advice
was unfalsifiable: no direction was ever mentioned, so no one could check it,
and it read identically whether the room faced north or south.

Vastu is a system of directional rules. That makes it a natural fit for code:
each rule states a condition on the geometry we already have and returns a
pass or fail with a plain explanation. The score is therefore reproducible and
assertable, rather than a number the model invented.

The rules encoded here are the widely-practised residential subset, and each
one names the tradition it draws on. This is a design tool, not a religious
authority; `app.engine.vastu` is deliberately explicit about severity so a
client who does not follow Vastu is not blocked by it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.observability import get_logger, metrics

logger = get_logger(__name__)

# Compass sectors used for scoring. Eight sectors, because that is the
# granularity Vastu remedies actually distinguish.
DIRECTIONS_8 = (
    "north", "northeast", "east", "southeast",
    "south", "southwest", "west", "northwest",
)


@dataclass
class VastuRule:
    """One checkable Vastu principle."""

    code: str
    title: str
    principle: str
    # "error" = strong recommendation, "advice" = discretionary.
    severity: str
    # Which room types this applies to. Empty means all.
    rooms: tuple = ()
    # The tradition this draws on, for transparency.
    source: str = "widely-practised residential Vastu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code, "title": self.title, "principle": self.principle,
            "severity": self.severity, "source": self.source,
        }


@dataclass
class VastuResult:
    """The outcome for one item, or one global observation."""

    rule: VastuRule
    passed: bool
    explanation: str
    remedy: str = ""
    item_name: str = ""
    item_category: str = ""

    @property
    def severity(self) -> str:
        return self.rule.severity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.rule.code,
            "title": self.rule.title,
            "principle": self.rule.principle,
            "severity": self.severity,
            "passed": self.passed,
            "explanation": self.explanation,
            "remedy": self.remedy,
            "item_name": self.item_name,
            "item_category": self.item_category,
            "source": self.rule.source,
        }


@dataclass
class VastuReport:
    """The full Vastu assessment for a design."""

    score: int
    grade: str
    results: List[VastuResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    # Set when directions were not supplied, so the UI can ask for them.
    missing_inputs: List[str] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> List[VastuResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "grade": self.grade,
            "passed": self.passed_count,
            "total": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "notes": self.notes,
            "missing_inputs": self.missing_inputs,
        }


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

RULES: Dict[str, VastuRule] = {
    "bed_head_south": VastuRule(
        code="bed_head_south",
        title="Bed head towards the south",
        principle="The head of the bed should rest against the south or "
                  "south-west wall, so the sleeper faces north.",
        severity="error",
    ),
    "desk_east_or_north": VastuRule(
        code="desk_east_or_north",
        title="Study desk faces east or north",
        principle="A study desk placed on an east or north wall gives the "
                  "sitter morning light and a clear, undistracting forward view.",
        severity="error",
    ),
    "wardrobe_southwest": VastuRule(
        code="wardrobe_southwest",
        title="Wardrobe on the south-west",
        principle="Heavy storage is traditionally placed in the south-west, "
                  "the most stable direction, and against a wall.",
        severity="advice",
    ),
    "mirror_not_opposite_bed": VastuRule(
        code="mirror_not_opposite_bed",
        title="Mirror not directly opposite a bed",
        principle="A mirror facing a bed is held to reflect energy onto the "
                  "sleeper; it is moved to a side wall instead.",
        severity="error",
    ),
    "cook_south_east": VastuRule(
        code="cook_south_east",
        title="Cooking on the south-east",
        principle="The cooking platform faces south-east so the cook works "
                  "with the sun to their left.",
        severity="error",
        rooms=("kitchen",),
    ),
    "sink_north_east": VastuRule(
        code="sink_north_east",
        title="Water inlet in the north-east",
        principle="The main water inlet or sink sits to the north-east.",
        severity="advice",
        rooms=("kitchen", "bathroom"),
    ),
    "bed_clear_north": VastuRule(
        code="bed_clear_north",
        title="Space above the bed head is clear",
        principle="Nothing heavy or low-hanging is placed directly above the "
                  "bed head, where a sleeping person would bump their head.",
        severity="advice",
    ),
    "entrance_clear": VastuRule(
        code="entrance_clear",
        title="Entrance is not blocked",
        principle="No tall unit stands directly in the door swing or "
                  "immediately inside the entrance.",
        severity="error",
    ),
    "tv_west_or_north": VastuRule(
        code="tv_west_or_north",
        title="TV unit on the west or north",
        principle="A viewing screen is placed on the west or north wall so "
                  "seating faces east or south.",
        severity="advice",
        rooms=("living_room",),
    ),
    "light_from_north_east": VastuRule(
        code="light_from_north_east",
        title="Best light enters from the north-east",
        principle="A north-east window is the preferred source of daylight "
                  "for a study or a main seating area.",
        severity="advice",
    ),
}

# Which directions count as satisfying each directional rule.
_SATISFYING: Dict[str, tuple] = {
    "bed_head_south": ("south", "southwest"),
    "desk_east_or_north": ("east", "northeast", "north"),
    "wardrobe_southwest": ("southwest", "south", "west"),
    "cook_south_east": ("southeast", "south"),
    "sink_north_east": ("northeast", "north", "east"),
    "tv_west_or_north": ("west", "northwest", "north"),
}

# Remedy text, keyed by rule.
_REMEDY: Dict[str, str] = {
    "bed_head_south": "Move the bed so its head is against the south or "
                      "south-west wall.",
    "desk_east_or_north": "Rotate the desk to sit against the east or north "
                          "wall, ideally near a window.",
    "wardrobe_southwest": "Relocate the wardrobe to the south-west wall if "
                          "the plan allows.",
    "mirror_not_opposite_bed": "Move the mirror to a side wall so it no longer "
                               "faces the bed directly.",
    "cook_south_east": "Face the cooking platform south-east.",
    "sink_north_east": "Move the water inlet or sink towards the north-east.",
    "bed_clear_north": "Clear any shelf or cabinet directly above the bed head.",
    "entrance_clear": "Relocate whatever is standing in the door swing.",
    "tv_west_or_north": "Move the TV unit to the west or north wall.",
    "light_from_north_east": "If possible, place the study or main seating near "
                             "the north-east window.",
}


def _grade(score: int) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Needs work"


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


def evaluate_vastu(
    layout_items: List[Dict[str, Any]],
    room_type: str = "",
    door_direction: Optional[str] = None,
    window_direction: Optional[str] = None,
    compliance: str = "soft",
) -> VastuReport:
    """
    Score a design against the Vastu rules, in code.

    Takes the layout JSON (which already has real positions and walls) so each
    rule is checked against geometry rather than prose. A direction of "east"
    is only meaningful when we know where the item actually sits, which is
    exactly what the layout engine supplies.

    `compliance` is "strict" or "soft". A soft score is informational; a
    strict one is expected to be shown prominently. The score itself is the
    same either way - the input describes the client's preference, not a
    different standard of truth.
    """
    missing: List[str] = []
    if not door_direction:
        missing.append("door_direction")
    if not window_direction:
        missing.append("window_direction")

    results: List[VastuResult] = []

    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for item in layout_items or []:
        if not isinstance(item, dict):
            continue
        by_category.setdefault(
            str(item.get("category", "")).lower(), []
        ).append(item)

    def wall_of(cat: str) -> Optional[str]:
        items = by_category.get(cat) or []
        return items[0].get("wall") if items else None

    # --- bed placement -------------------------------------------------
    beds = by_category.get("bed", [])
    rule = RULES["bed_head_south"]
    if beds:
        walls = [b.get("wall") for b in beds]
        if all(w in _SATISFYING["bed_head_south"] for w in walls):
            results.append(VastuResult(
                rule=rule, passed=True,
                explanation=f"Bed heads are against the {', '.join(sorted(set(walls)))} wall, "
                            f"so the sleeper faces north.",
                item_name=", ".join(b.get("name", "bed") for b in beds),
                item_category="bed",
            ))
        else:
            bad = [w for w in walls if w not in _SATISFYING["bed_head_south"]]
            results.append(VastuResult(
                rule=rule, passed=False,
                explanation=f"The bed head is on the {', '.join(sorted(set(bad)))} wall. "
                            f"Vastu asks for south or south-west.",
                remedy=_REMEDY["bed_head_south"],
                item_name=", ".join(b.get("name", "bed") for b in beds),
                item_category="bed",
            ))

    # --- study desk ----------------------------------------------------
    desks = by_category.get("desk", [])
    rule = RULES["desk_east_or_north"]
    if desks:
        walls = [d.get("wall") for d in desks]
        if all(w in _SATISFYING["desk_east_or_north"] for w in walls):
            near = ""
            if window_direction in ("east", "northeast", "north", "southeast"):
                near = " The window is on that side, so it gets daylight."
            results.append(VastuResult(
                rule=rule, passed=True,
                explanation=f"The study desk is against the {', '.join(sorted(set(walls)))} "
                            f"wall.{near}",
                item_name=", ".join(d.get("name", "desk") for d in desks),
                item_category="desk",
            ))
        else:
            results.append(VastuResult(
                rule=rule, passed=False,
                explanation=f"The study desk is on the "
                            f"{', '.join(sorted(set(walls)))} wall; it should face east or north.",
                remedy=_REMEDY["desk_east_or_north"],
                item_name=", ".join(d.get("name", "desk") for d in desks),
                item_category="desk",
            ))

    # --- wardrobe ------------------------------------------------------
    rule = RULES["wardrobe_southwest"]
    w = wall_of("wardrobe")
    if w:
        ok = w in _SATISFYING["wardrobe_southwest"]
        results.append(VastuResult(
            rule=rule, passed=ok,
            explanation=(
                f"The wardrobe is on the {w} wall."
                if ok else
                f"The wardrobe is on the {w} wall; the south-west is preferred for heavy storage."
            ),
            remedy="" if ok else _REMEDY["wardrobe_southwest"],
            item_name="Wardrobe", item_category="wardrobe",
        ))

    # --- mirror opposite a bed ----------------------------------------
    rule = RULES["mirror_not_opposite_bed"]
    mirrors = by_category.get("decor", [])
    mirrors = [m for m in mirrors if "mirror" in str(m.get("name", "")).lower()]
    if mirrors and beds:
        # Opposite means the two sit on the facing walls.
        opposing = {"north": "south", "south": "north", "east": "west", "west": "east"}
        conflicts = [
            (b.get("name", "bed"), m.get("name", "mirror"))
            for b in beds for m in mirrors
            if opposing.get(str(b.get("wall"))) == str(m.get("wall"))
        ]
        if conflicts:
            bed_name, mirror_name = conflicts[0]
            results.append(VastuResult(
                rule=rule, passed=False,
                explanation=f"The {mirror_name.lower()} is on the wall directly opposite the "
                            f"{bed_name.lower()}, so it faces the sleeper head on.",
                remedy=_REMEDY["mirror_not_opposite_bed"],
                item_name=mirror_name, item_category="decor",
            ))
        else:
            results.append(VastuResult(
                rule=rule, passed=True,
                explanation="No mirror is placed on the wall directly opposite a bed.",
                item_name=", ".join(m.get("name", "mirror") for m in mirrors),
                item_category="decor",
            ))

    # --- TV unit -------------------------------------------------------
    rule = RULES["tv_west_or_north"]
    t = wall_of("tv_unit")
    if t:
        ok = t in _SATISFYING["tv_west_or_north"]
        results.append(VastuResult(
            rule=rule, passed=ok,
            explanation=(
                f"The TV unit is on the {t} wall." if ok else
                f"The TV unit is on the {t} wall; west or north is preferred."
            ),
            remedy="" if ok else _REMEDY["tv_west_or_north"],
            item_name="TV unit", item_category="tv_unit",
        ))

    # --- kitchen-specific ---------------------------------------------
    if room_type == "kitchen":
        rule = RULES["cook_south_east"]
        # Without a per-item "role" field, judge the room's own orientation.
        if window_direction in _SATISFYING["cook_south_east"] or door_direction in _SATISFYING["cook_south_east"]:
            results.append(VastuResult(
                rule=rule, passed=True,
                explanation="The kitchen orientation supports a south-east cooking position.",
                item_category="kitchen",
            ))
        else:
            results.append(VastuResult(
                rule=rule, passed=False,
                explanation="The kitchen does not currently support a south-east cooking "
                            "position; the platform would face another direction.",
                remedy=_REMEDY["cook_south_east"],
                item_category="kitchen",
            ))

    # --- entrance clear ------------------------------------------------
    rule = RULES["entrance_clear"]
    if door_direction:
        blocking = [
            i for i in (layout_items or [])
            if isinstance(i, dict)
            and str(i.get("wall", "")).lower() == str(door_direction).lower()
            and float(i.get("height", 0) or 0) > 100
        ]
        if blocking:
            results.append(VastuResult(
                rule=rule, passed=False,
                explanation=f"{', '.join(i.get('name', 'an item') for i in blocking)} "
                            f"stands on the same wall as the door and may obstruct the entrance.",
                remedy=_REMEDY["entrance_clear"],
                item_category="entrance",
            ))
        else:
            results.append(VastuResult(
                rule=rule, passed=True,
                explanation=f"Nothing tall stands against the {door_direction} wall where "
                            f"the door opens.",
                item_category="entrance",
            ))

    # --- light from the north-east -------------------------------------
    if window_direction:
        rule = RULES["light_from_north_east"]
        good = window_direction in ("north", "northeast")
        results.append(VastuResult(
            rule=rule, passed=good,
            explanation=(
                f"The window faces {window_direction}, which is the preferred direction for "
                f"daylight." if good else
                f"The window faces {window_direction}. A north-east window gives the most "
                f"even daylight, so an east window is workable but warmer in the afternoon."
            ),
            remedy="" if good else _REMEDY["light_from_north_east"],
            item_category="window",
        ))

    # --- score ---------------------------------------------------------
    # Severity-weighted: a violated strong rule costs more than a soft one.
    if results:
        earned = 0.0
        possible = 0.0
        for r in results:
            weight = 1.0 if r.severity == "error" else 0.5
            possible += weight
            if r.passed:
                earned += weight
        score = int(round((earned / possible) * 100)) if possible else 100
    else:
        score = 100

    notes: List[str] = []
    if missing:
        notes.append(
            "Add the door and window directions for a complete assessment; "
            "without them several rules are skipped."
        )
    if compliance == "soft":
        notes.append("Vastu was requested as a soft preference, so this is advisory.")

    report = VastuReport(
        score=score,
        grade=_grade(score),
        results=results,
        notes=notes,
        missing_inputs=missing,
    )
    metrics.gauge("vastu.score", float(score))
    metrics.increment("vastu.evaluated")
    logger.info("vastu_evaluated score=%d passed=%d/%d", score, report.passed_count, len(results))
    return report
