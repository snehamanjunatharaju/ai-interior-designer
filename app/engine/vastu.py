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
    "seating_faces_north_or_east": VastuRule(
        code="seating_faces_north_or_east",
        title="Seating faces north or east",
        principle="Main living-room seating faces north or east.",
        severity="advice",
        rooms=("living_room",),
    ),
    "northeast_corner_light": VastuRule(
        code="northeast_corner_light",
        title="Keep the north-east corner light",
        principle="Avoid placing heavy furniture in the north-east corner.",
        severity="advice",
        rooms=("living_room",),
    ),
    "heavy_seating_south_or_west": VastuRule(
        code="heavy_seating_south_or_west",
        title="Heavy seating in the south or west",
        principle="Place heavy living-room seating along the south or west wall.",
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
    "cook_south_east": ("southeast",),
    "sink_north_east": ("northeast", "north", "east"),
    "tv_west_or_north": ("west", "northwest", "north"),
    "seating_faces_north_or_east": ("north", "east"),
    "heavy_seating_south_or_west": ("south", "west"),
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
    "seating_faces_north_or_east": "Rotate the main seating to face north or east.",
    "northeast_corner_light": "Move heavy furniture out of the north-east corner.",
    "heavy_seating_south_or_west": "Move heavy seating to the south or west wall.",
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


def _known_walls(items: List[Dict[str, Any]]) -> List[str]:
    """Walls actually recorded on placed items, never None."""
    return sorted({str(i.get("wall")) for i in items if i.get("wall")})


def evaluate_vastu(
    layout_items: List[Dict[str, Any]],
    room_type: str = "",
    door_direction: Optional[str] = None,
    window_direction: Optional[str] = None,
    compliance: str = "soft",
    room_width_cm: Optional[float] = None,
    room_depth_cm: Optional[float] = None,
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
    beds = [b for b in by_category.get("bed", []) if b.get("wall")]
    rule = RULES["bed_head_south"]
    if beds:
        walls = [str(b.get("wall")) for b in beds]
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
    desks = [d for d in by_category.get("desk", []) if d.get("wall")]
    rule = RULES["desk_east_or_north"]
    if desks:
        walls = [str(d.get("wall")) for d in desks]
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
    mirrors = [
        m for m in mirrors
        if "mirror" in str(m.get("name", "")).lower() and m.get("wall")
    ]
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

    if room_type == "living_room":
        seating = [
            item for item in by_category.get("seating", [])
            if item.get("wall") or item.get("facing")
        ]
        if seating:
            facing_by_wall = {
                "north": "south", "south": "north",
                "east": "west", "west": "east",
            }
            facings = [
                str(item.get("facing") or facing_by_wall.get(str(item.get("wall")), ""))
                for item in seating
            ]
            ok = all(direction in _SATISFYING["seating_faces_north_or_east"]
                     for direction in facings)
            results.append(VastuResult(
                rule=RULES["seating_faces_north_or_east"],
                passed=ok,
                explanation=(
                    f"Main seating faces {', '.join(sorted(set(facings)))}."
                    if ok else
                    f"Main seating faces {', '.join(sorted(set(facings)))}; north or east is preferred."
                ),
                remedy="" if ok else _REMEDY["seating_faces_north_or_east"],
                item_name=", ".join(item.get("name", "seating") for item in seating),
                item_category="seating",
            ))

        heavy_seating = [
            item for item in by_category.get("seating", [])
            if item.get("wall")
        ]
        if heavy_seating:
            walls = [str(item.get("wall")) for item in heavy_seating]
            ok = all(wall in _SATISFYING["heavy_seating_south_or_west"]
                     for wall in walls)
            results.append(VastuResult(
                rule=RULES["heavy_seating_south_or_west"],
                passed=ok,
                explanation=(
                    f"Heavy seating is on the {', '.join(sorted(set(walls)))} wall."
                    if ok else
                    f"Heavy seating is on the {', '.join(sorted(set(walls)))} wall; south or west is preferred."
                ),
                remedy="" if ok else _REMEDY["heavy_seating_south_or_west"],
                item_name=", ".join(item.get("name", "seating") for item in heavy_seating),
                item_category="seating",
            ))

        if room_width_cm and room_depth_cm:
            heavy = [
                item for item in layout_items or []
                if isinstance(item, dict)
                and str(item.get("category", "")).lower()
                    in {"seating", "storage", "wardrobe"}
                and item.get("x") is not None
                and item.get("y") is not None
            ]
            in_corner = [
                item for item in heavy
                if float(item["x"]) + float(item.get("width", 0)) / 2 >= room_width_cm / 2
                and float(item["y"]) + float(item.get("depth", 0)) / 2 <= room_depth_cm / 2
            ]
            results.append(VastuResult(
                rule=RULES["northeast_corner_light"],
                passed=not in_corner,
                explanation=(
                    "The north-east corner has no heavy furniture."
                    if not in_corner else
                    f"{', '.join(item.get('name', 'Heavy item') for item in in_corner)} "
                    "occupies the north-east corner."
                ),
                remedy="" if not in_corner else _REMEDY["northeast_corner_light"],
                item_name=", ".join(item.get("name", "") for item in in_corner),
                item_category="living_room",
            ))

    # --- kitchen-specific ---------------------------------------------
    if room_type == "kitchen":
        rule = RULES["cook_south_east"]
        # Judge the cooking platform's own wall. The old version judged the
        # room's door/window bearings instead, so a kitchen with a north-east
        # window "passed" while the layout put the hob on the south wall - the
        # exact contradiction the client saw ("respects Vastu" above a failed
        # south-east rule).
        cooking = [
            i for i in (layout_items or [])
            if isinstance(i, dict) and (
                i.get("role") == "cooking"
                or i.get("is_cooking") is True
                or "hob" in str(i.get("name", "")).lower()
                or "stove" in str(i.get("name", "")).lower()
            )
        ]
        wall = cooking[0].get("wall") if cooking else None
        facing = cooking[0].get("facing") if cooking else None
        if wall is not None:
            ok = wall == "southeast"
            results.append(VastuResult(
                rule=rule, passed=ok,
                explanation=(
                    "The cooking platform is on the south-east wall"
                    + (f", facing {facing}." if facing else ".")
                    if ok else
                    f"The cooking platform is on the {wall} wall; Vastu asks for "
                    f"the south-east."
                ),
                remedy="" if ok else _REMEDY["cook_south_east"],
                item_name=cooking[0].get("name", "cooking platform"),
                item_category="kitchen",
            ))
        else:
            # No placed platform: the rule is unassessable from geometry. The
            # DesignPipeline attaches the declared cooking position and
            # `analyse_design` replaces this result from it.
            results.append(VastuResult(
                rule=rule, passed=False,
                explanation=(
                    "No cooking platform is placed in the plan, so compliance "
                    "with the south-east rule cannot be shown."
                ),
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

    # Notes are generated from the rule results, never written by the model.
    # The old pipeline mixed the model's prose into this section, so
    # "Limited decorative flexibility" appeared under "Vastu Notes" and a
    # "respects Vastu" sentence sat next to a failed rule.
    notes: List[str] = [
        ("" if r.passed else "Needs attention: ") + r.explanation
        for r in results
    ]
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


def analyse_design(
    design: Dict[str, Any],
    brief: Any,
) -> VastuReport:
    """Assess a design dict against the client's own form inputs.

    Everything the rules need - room type, door, window, compliance level and
    the positions the layout engine produced - is read from the structured
    brief and the design itself. Agents do not pass these in prose.

    Items have no explicit wall when the layout has not been computed, so the
    room's own openings are used for the kitchen rule. Under Strict a kitchen
    must still name a south-east cooking position; that is the condition the
    validator blocks on.
    """
    room_type = str(getattr(brief, "room_type", "") or "").lower()
    compliance = str(getattr(brief, "vastu_compliance", "soft") or "soft").lower()

    items: List[Dict[str, Any]] = []
    layout = design.get("layout") or {}
    for placed in layout.get("items") or []:
        if isinstance(placed, dict):
            items.append({
                "name": placed.get("name", ""),
                "category": placed.get("category", ""),
                "wall": placed.get("wall"),
                "height": placed.get("height", 0),
                "x": placed.get("x"),
                "y": placed.get("y"),
                "width": placed.get("width", 0),
                "depth": placed.get("depth", 0),
                "facing": placed.get("facing"),
            })
    if not items:
        # No plan yet. Placement rules cannot be judged without geometry, so
        # the room's own openings are the only real signal; inventing a wall
        # for each item would produce a verdict about a layout that does not
        # exist. Non-directional observations (entrance clear, daylight) still
        # run from the door and window bearings.
        items = []

    report = evaluate_vastu(
        items,
        room_type=room_type,
        door_direction=getattr(brief, "door_direction", None),
        window_direction=getattr(brief, "window_direction", None),
        compliance=compliance,
        room_width_cm=(layout.get("room") or {}).get("width_cm"),
        room_depth_cm=(layout.get("room") or {}).get("depth_cm"),
    )

    # A kitchen under Strict is *required* to show a south-east cooking
    # position. When the plan carries no wall for the counter, the rule could
    # not be checked at all - record that as an unpassed rule with an explicit
    # reason, rather than silently omitting it and reporting a high score.
    if room_type == "kitchen":
        status = design.get("cooking_platform") or {}
        wall = status.get("wall")
        facing = status.get("facing")
        existing = next(
            (r for r in report.results if r.rule.code == "cook_south_east"), None
        )
        if status:
            ok = wall == "southeast"
            if compliance == "strict":
                ok = ok and facing == "east"
            if existing is not None:
                existing.passed = ok
                existing.explanation = (
                    f"The cooking platform is placed on the south-east wall"
                    + (f", facing {facing}." if facing else ".")
                    if ok else
                    f"The cooking platform is placed on the {wall or 'unspecified'} "
                    f"wall"
                    + (f" facing {facing}" if facing else "")
                    + "; Vastu asks for the south-east."
                )
                existing.remedy = "" if ok else _REMEDY["cook_south_east"]
                existing.item_name = status.get("name", "cooking platform")
            else:
                report.results.append(VastuResult(
                    rule=RULES["cook_south_east"],
                    passed=ok,
                    explanation=(
                        f"The cooking platform is placed on the {wall} wall"
                        + (f", facing {facing}." if facing else ".")
                    ),
                    remedy="" if ok else _REMEDY["cook_south_east"],
                    item_name=status.get("name", "cooking platform"),
                    item_category="kitchen",
                ))

    # Kitchen placement can replace the initially unassessable cooking rule;
    # recompute the score from that final result rather than leaving the
    # pre-placement score beside a passing rule.
    if report.results:
        earned = 0.0
        possible = 0.0
        for result in report.results:
            weight = 1.0 if result.rule.severity == "error" else 0.5
            possible += weight
            if result.passed:
                earned += weight
        report.score = int(round((earned / possible) * 100)) if possible else 100
    else:
        report.score = 100
    report.grade = _grade(report.score)

    # Notes must reflect the final rule set.
    report.notes = [
        ("" if r.passed else "Needs attention: ") + r.explanation
        for r in report.results
    ]
    if report.missing_inputs:
        report.notes.append(
            "Add the door and window directions for a complete assessment; "
            "without them several rules are skipped."
        )
    if compliance == "soft":
        report.notes.append("Vastu was requested as a soft preference, so this is advisory.")
    report.notes = report.notes[:6]
    metrics.gauge("vastu.score", float(report.score))
    return report
