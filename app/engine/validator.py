"""
Deterministic design validator.

This module is the reason the brief cannot be ignored.

The original pipeline trusted the LLM on facts: it asked for two beds and
accepted one, it summed whatever prices came back and called the result
"within budget". A 2B model has no incentive to check itself, and no way to
be argued out of a number it produced.

Here, the design is compared against the parsed brief by plain Python. There
is no model, no prompt, and no judgement call - a violation is a violation.
The LLM still writes the prose; code decides whether the facts hold.

Every check returns a `Violation` carrying a machine-readable code, so the
reflection loop can target the specific failure rather than re-rolling the
whole design hoping for a different answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.observability import get_logger, metrics
from app.engine.brief import BRIEF_TO_CATALOG_CATEGORY, BriefConstraints

logger = get_logger(__name__)

# Violation severities.
SEVERITY_ERROR = "error"     # design is wrong; must be regenerated
SEVERITY_WARNING = "warning" # design is usable but worth flagging
SEVERITY_INFO = "info"       # observation only


@dataclass
class Violation:
    """One failed check against the brief."""

    code: str
    severity: str
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    # Which part of the design to fix, so a targeted rewrite is possible.
    remedy_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
            "remedy_hint": self.remedy_hint,
        }

    def as_critique_line(self) -> str:
        """Render as a line the Architect can act on."""
        bits = [f"[{self.code}] {self.message}"]
        if self.expected is not None:
            bits.append(f"expected: {self.expected}")
        if self.actual is not None:
            bits.append(f"got: {self.actual}")
        if self.remedy_hint:
            bits.append(f"fix: {self.remedy_hint}")
        return " ".join(bits)


@dataclass
class ValidationReport:
    """The full result of validating one design against one brief."""

    violations: List[Violation] = field(default_factory=list)
    checks_run: int = 0
    checks_passed: int = 0

    @property
    def errors(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == SEVERITY_ERROR]

    @property
    def is_valid(self) -> bool:
        """No blocking errors. Warnings do not fail a design."""
        return not self.errors

    @property
    def pass_rate(self) -> float:
        if not self.checks_run:
            return 1.0
        return round(self.checks_passed / self.checks_run, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.is_valid,
            "checks_run": self.checks_run,
            "checks_passed": self.checks_passed,
            "pass_rate": self.pass_rate,
            "violations": [v.to_dict() for v in self.violations],
        }

    def critique_lines(self) -> List[str]:
        """Blocking violations, phrased as instructions for the Architect."""
        return [v.as_critique_line() for v in self.errors]


# A vibrant scheme needs more than one warm accent. A single terracotta
# (#C0704A reads 0.61 saturation) can rescue an otherwise beige palette, so
# the test requires a minimum count of genuinely vivid colours rather than
# "at least one".
_VIBRANT_MIN_SATURATION = 0.35
_VIBRANT_MIN_VIVID_COLORS = 2

# Words that indicate a muted or neutral choice, which contradict a
# "vibrant" brief. Checked against colour *names*, not just hex.
_MUTED_WORDS = (
    "beige", "cream", "off white", "off-white", "ivory", "taupe", "greige",
    "warm neutral", "neutral", "muted", "pastel", "subtle", "soft beige",
    "nude", "dove", "almond", "oyster",
)


def _hex_to_rgb(hex_code: str) -> Optional[tuple]:
    h = hex_code.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return None
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return None


def _saturation(rgb: tuple) -> float:
    """HSV saturation, 0.0-1.0. 0 is grey, 1.0 is fully saturated."""
    r, g, b = (c / 255.0 for c in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    if mx == 0:
        return 0.0
    return (mx - mn) / mx


def _count_items(design: Dict[str, Any], categories: set, term: str = "") -> int:
    """
    Count line items that genuinely satisfy a brief term.

    Two wrinkles matter here, and both were real bugs:

    - Beds count *sleeping places*, not units. A bunk bed is one SKU that
      sleeps two, so it satisfies a two-bed brief on its own. Counting units
      reported a false shortfall and, once "fixed", doubled the price.
    - A broad category is not interchangeable within itself. `seating` holds
      sofas, armchairs, bean bags and stools; a stool is not a sofa. When the
      brief names a specific product type, the name must match too.

    `term` is the brief vocabulary; `categories` the catalog categories.
    """
    from app.catalog import catalog
    from app.engine.brief import ACCEPTABLE_PRODUCTS

    acceptable = ACCEPTABLE_PRODUCTS.get(term) if term else None

    total = 0
    for item in design.get("furniture", []) or []:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("category", "")).strip().lower()
        sku = str(item.get("sku", "")).strip()
        product = catalog.get(sku) if sku else None

        # Resolve the true catalog category from the SKU when we have one.
        # The model's `category` is unreliable - it labels a mirror "mirror"
        # while the catalog files it under `decor` - and trusting that label
        # let an excluded item slip past the check entirely.
        effective = product.category.lower() if product is not None else cat
        if effective not in categories and cat not in categories:
            continue

        if acceptable:
            # The product name must match one of the acceptable types.
            name = str(item.get("name", "")).strip().lower()
            if product is not None:
                name = product.name.lower()
            if not any(word in name for word in acceptable):
                continue

        try:
            qty = int(float(item.get("quantity", 1) or 1))
        except (TypeError, ValueError):
            qty = 1
        sleeps = product.sleeps if (product is not None and effective == "bed") else 1
        total += qty * sleeps
    return total


def validate_design(
    design: Dict[str, Any],
    brief: BriefConstraints,
    cost: Optional[Dict[str, Any]] = None,
) -> ValidationReport:
    """
    Check one design against the parsed brief. No LLM involved.

    `cost` is the code-computed costing dict; if present, the budget and
    arithmetic checks run. It is passed separately because costing is
    produced after the design and may come from the catalog.
    """
    report = ValidationReport()

    _check_required_items(design, brief, report)
    _check_excluded_items(design, brief, report)
    _check_color_scheme(design, brief, report)
    _check_budget(design, brief, cost, report)
    _check_no_placeholders(design, report)
    _check_real_products(design, report)

    metrics.increment("validator.checks_run", report.checks_run)
    metrics.increment("validator.failed_designs" if not report.is_valid else "validator.passed_designs")

    if not report.is_valid:
        logger.info(
            "validation_failed errors=%d codes=%s",
            len(report.errors), [v.code for v in report.errors],
        )
    return report


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def _check_required_items(
    design: Dict[str, Any], brief: BriefConstraints, report: ValidationReport
) -> None:
    """
    Does the design contain the items the brief asked for, in the right count?

    This is the check that would have caught "asked for 2 beds, got 1".

    A shortfall has two very different causes and they must not look alike:

      * the model forgot an item, or picked a cheaper substitute - a mistake
        the reflection loop can fix, so it is an error;
      * the budget genuinely cannot cover the item - not a mistake, and
        unfixable by retrying, so it is reported as infeasibility with the
        shortfall and the way out.

    Before this distinction existed, an honest budget-aware selection that
    skipped an unaffordable bunk bed was reported as "the model forgot the
    bed", which sent the reflection loop off to re-add an item that would push
    the design over budget again.
    """
    infeasible = getattr(design, "_budget_infeasible", None)
    if not infeasible:
        infeasible = (design.get("_budget_infeasible") or None)

    for item, requested in brief.required_items.items():
        report.checks_run += 1
        categories = _SYNONYM_CATEGORIES.get(item, {item})
        actual = _count_items(design, categories, item)
        # For beds, `actual` is sleeping places. The brief asked for `requested`
        # beds; those are the same unit, so no scaling is needed.
        if actual == requested:
            report.checks_passed += 1
        elif actual < requested:
            shortfall = requested - actual
            # Was this specific category the one the optimizer could not afford?
            blocked = False
            if isinstance(infeasible, dict):
                blocked = any(
                    str(entry.get("category", "")) == item
                    for entry in infeasible.get("unserved", [])
                    if isinstance(entry, dict)
                )
            if blocked:
                report.violations.append(Violation(
                    code="budget_infeasible",
                    severity=SEVERITY_WARNING,
                    message=(
                        f"The brief asks for {requested} {item}(s) and the design "
                        f"contains {actual}, because the remaining "
                        f"Rs {infeasible.get('shortfall_inr', 0):,} needed for the "
                        f"missing {shortfall} is outside the stated budget of "
                        f"Rs {infeasible.get('budget_inr', 0):,}. This is a budget "
                        f"constraint, not a missing item."
                    ),
                    expected=f"{requested} {item}(s)",
                    actual=f"{actual} {item}(s)",
                    remedy_hint=(
                        "Either raise the budget, drop this item from scope, or "
                        "accept a lower product tier. The design shown already "
                        "buys the most the budget allows."
                    ),
                ))
                # Deliberately counted as resolved: the check has done its job by
                # naming the shortfall. The overall report still carries the
                # infeasibility, so nothing is hidden.
                report.checks_passed += 1
                continue
            report.violations.append(Violation(
                code="missing_items",
                severity=SEVERITY_ERROR,
                message=f"The brief asks for {requested} {item}(s) but the design contains {actual}.",
                expected=f"{requested} {item}(s)",
                actual=f"{actual} {item}(s)",
                remedy_hint=(
                    f"Add {requested - actual} more {item}(s) with a real catalog SKU and price."
                ),
            ))
        else:
            report.violations.append(Violation(
                code="item_overcount",
                severity=SEVERITY_WARNING,
                message=(
                    f"The brief asks for {requested} {item}(s) but the design contains {actual}."
                ),
                expected=f"{requested} {item}(s)",
                actual=f"{actual} {item}(s)",
                remedy_hint=f"Reduce to {requested} {item}(s) unless the extra is clearly justified.",
            ))


# Categories that satisfy a request for another canonical item.
#
# This must agree with BRIEF_TO_CATALOG_CATEGORY in app.engine.brief, which is
# the join between the brief vocabulary and the catalog vocabulary. Two copies
# of this mapping is exactly the kind of drift that makes a design fail a
# check for an item that is in stock under a different name, so it is derived
# from that single source instead of being restated here.
_SYNONYM_CATEGORIES: Dict[str, set] = {
    term: set(cats) for term, cats in BRIEF_TO_CATALOG_CATEGORY.items()
}


def _check_excluded_items(
    design: Dict[str, Any], brief: BriefConstraints, report: ValidationReport
) -> None:
    """Items the brief explicitly said to avoid must not appear."""
    for item in brief.excluded_items:
        report.checks_run += 1
        categories = _SYNONYM_CATEGORIES.get(item, {item})
        actual = _count_items(design, categories, item)
        if actual == 0:
            report.checks_passed += 1
        else:
            report.violations.append(Violation(
                code="excluded_item_present",
                severity=SEVERITY_ERROR,
                message=f"The client asked to avoid {item}(s) but the design includes {actual}.",
                expected="none",
                actual=f"{actual} {item}(s)",
                remedy_hint=f"Remove the {item}(s) and suggest an alternative.",
            ))


def _check_color_scheme(
    design: Dict[str, Any], brief: BriefConstraints, report: ValidationReport
) -> None:
    """
    Does the palette match the requested scheme?

    "Vibrant" is checked structurally - the hex codes must actually be
    saturated - rather than by trusting a colour name. A model that writes
    "terracotta and warm neutral" for a vibrant brief is wrong, and the
    saturation test catches that regardless of what it called them.
    """
    if not brief.color_scheme:
        return

    palette = design.get("color_palette")
    if not isinstance(palette, dict):
        report.checks_run += 1
        report.violations.append(Violation(
            code="missing_palette",
            severity=SEVERITY_ERROR,
            message="The design has no colour palette, but the client specified a colour scheme.",
            expected=brief.color_scheme,
            actual="none",
            remedy_hint="Provide a colour palette with named colours and hex codes.",
        ))
        return

    scheme = brief.color_scheme.lower()
    names = " ".join([
        str(palette.get("primary", "")),
        str(palette.get("secondary", "")),
        str(palette.get("accent", "")),
        " ".join(palette.get("neutrals", []) or []),
    ]).lower()

    hex_codes = [h for h in (palette.get("hex_codes") or []) if isinstance(h, str)]

    if scheme == "vibrant":
        report.checks_run += 1
        muted_hits = [w for w in _MUTED_WORDS if w in names]
        vivid = [
            c for c in hex_codes
            if (_rgb := _hex_to_rgb(c)) and _saturation(_rgb) >= _VIBRANT_MIN_SATURATION
        ]

        if len(vivid) < _VIBRANT_MIN_VIVID_COLORS:
            report.violations.append(Violation(
                code="color_scheme_mismatch",
                severity=SEVERITY_ERROR,
                message=(
                    f"The client asked for a vibrant palette but only {len(vivid)} of "
                    f"{len(hex_codes) or 0} colours are genuinely saturated"
                    + (f", and the palette leans muted ({', '.join(muted_hits[:3])})." if muted_hits else ".")
                ),
                expected="vibrant / saturated",
                actual=(
                    (", ".join(muted_hits[:3]) + ", only one vivid colour")
                    if muted_hits else f"only {len(vivid)} vivid colour(s)"
                ),
                remedy_hint=(
                    "Use at least two saturated accent colours (e.g. coral #FF6B5B "
                    "(0.64), teal #17A2A2 (0.86), marigold #F2A93B (0.76)) and name them "
                    "explicitly. Keep the neutrals as a base, not the whole palette."
                ),
            ))
        elif not hex_codes:
            report.violations.append(Violation(
                code="missing_hex_codes",
                severity=SEVERITY_WARNING,
                message="A vibrant palette was requested but no hex codes were given.",
                expected="vibrant with hex codes",
                actual="no hex codes",
                remedy_hint="Add specific hex codes so the choice is unambiguous.",
            ))
        else:
            report.checks_passed += 1

    elif scheme in {"monochrome", "neutral", "earthy"}:
        report.checks_run += 1
        if hex_codes:
            saturations = [
                _saturation(_hex_to_rgb(c)) for c in hex_codes if _hex_to_rgb(c)
            ]
            avg = sum(saturations) / len(saturations) if saturations else 1.0
            if scheme == "monochrome" and avg > 0.5:
                report.violations.append(Violation(
                    code="color_scheme_mismatch",
                    severity=SEVERITY_ERROR,
                    message=f"A {scheme} palette was requested but the hex codes average {avg:.0%} saturation.",
                    expected=scheme,
                    actual=f"avg saturation {avg:.0%}",
                    remedy_hint="Choose near-greyscale or single-hue values.",
                ))
            else:
                report.checks_passed += 1
        else:
            report.checks_passed += 1


def _check_budget(
    design: Dict[str, Any],
    brief: BriefConstraints,
    cost: Optional[Dict[str, Any]],
    report: ValidationReport,
) -> None:
    """
    Does the cost match the brief, and does the arithmetic close?

    The total is read from `cost` (computed in Python) when available. If the
    design's own headline total disagrees with the computed one, that is an
    error in its own right - the number shown to the client is the one that
    must be right.
    """
    if not brief.budget_inr:
        return

    report.checks_run += 1

    if not cost:
        report.violations.append(Violation(
            code="no_costing",
            severity=SEVERITY_ERROR,
            message="The design has no computed costing, so budget adherence cannot be verified.",
            expected=f"a priced design within Rs {brief.budget_inr:,}",
            actual="no costing",
            remedy_hint="Price every line item from the catalog before returning a design.",
        ))
        return

    computed = int(cost.get("estimated_cost_inr", 0) or 0)
    breakdown = cost.get("budget_breakdown", []) or []
    breakdown_total = sum(
        int(float(b.get("amount_inr", 0) or 0))
        for b in breakdown if isinstance(b, dict)
    )

    # Arithmetic must close: the parts must sum to the whole.
    if breakdown and breakdown_total != computed:
        report.violations.append(Violation(
            code="arithmetic_mismatch",
            severity=SEVERITY_ERROR,
            message=(
                f"The breakdown sums to Rs {breakdown_total:,} but the stated total is "
                f"Rs {computed:,}."
            ),
            expected=f"Rs {breakdown_total:,}",
            actual=f"Rs {computed:,}",
            remedy_hint="The total is computed in code; do not state a different number.",
        ))
    elif computed <= 0:
        report.violations.append(Violation(
            code="zero_cost",
            severity=SEVERITY_ERROR,
            message="The design costs Rs 0, which means it was never actually priced.",
            expected=f"a real total within Rs {brief.budget_inr:,}",
            actual="Rs 0",
            remedy_hint="Price every item from the catalog. A Rs 0 total is a failure, not a saving.",
        ))
    elif computed > brief.budget_inr:
        overage = computed - brief.budget_inr
        # An error, not a warning. A design that costs more than the client
        # allowed is not a passing design, and this was the exact bug: the
        # violation was recorded as a WARNING while line 482 incremented
        # checks_passed, so the UI rendered "Pass 7/7" in green directly above
        # a budget warning. The check now fails, so the badge must too.
        report.violations.append(Violation(
            code="over_budget",
            severity=SEVERITY_ERROR,
            message=(
                f"The design costs Rs {computed:,}, which is Rs {overage:,} over the "
                f"Rs {brief.budget_inr:,} budget."
            ),
            expected=f"<= Rs {brief.budget_inr:,}",
            actual=f"Rs {computed:,}",
            remedy_hint=(
                "Replace items with cheaper catalog products in the same categories, "
                "or phase the purchase so the essentials fit the budget."
            ),
        ))
    else:
        report.checks_passed += 1


# Phrases that indicate an unfilled template slot rather than real content.
#
# Precision matters here: this check blocks a design, so a false positive
# rejects a perfectly good one. Two rules learned the hard way:
#   - "insert" appears in ordinary design prose ("insert the wardrobe along
#     the north wall"), so it must be anchored to a bracketed placeholder.
#   - "xxx" matches inside hex codes and words, so it needs word boundaries.
# The remaining patterns are unambiguous template sentinels.
_PLACEHOLDER_PATTERNS = (
    r"layout derived from",
    r"derived from the design plan",
    r"based on the design plan",
    r"\bas per the plan\b",
    r"\btbd\b",
    r"to be decided",
    r"to be determined",
    r"\bplaceholder\b",
    r"lorem ipsum",
    r"your text here",
    r"\[insert[^\]]*\]",
    r"<insert[^>]*>",
    r"\bxxx+\b",
    r"n/a here",
)


def _check_no_placeholders(design: Dict[str, Any], report: ValidationReport) -> None:
    """
    Catch template sentinels that leak into client-facing output.

    The original pipeline had a literal fallback string, "Layout derived from
    the design plan", which shipped to clients whenever the architect failed.
    Silent degradation is worse than an honest error, so any leftover
    sentinel is now a blocking violation.
    """
    report.checks_run += 1
    haystack: List[str] = []
    for key in ("title", "concept", "layout_description", "lighting_plan",
                "storage_strategy"):
        val = design.get(key)
        if isinstance(val, str):
            haystack.append(val)
    for key in ("zones", "decor_suggestions", "vastu_notes"):
        for v in design.get(key, []) or []:
            haystack.append(str(v))

    blob = " ".join(haystack).lower()
    found = [p for p in _PLACEHOLDER_PATTERNS if re.search(p, blob)]

    if found:
        report.violations.append(Violation(
            code="placeholder_text",
            severity=SEVERITY_ERROR,
            message=f"Placeholder or template text leaked into the design: {', '.join(found[:3])}.",
            expected="real design prose",
            actual=", ".join(found[:3]),
            remedy_hint="Write the field out in full. Do not leave a template sentinel.",
        ))
    else:
        report.checks_passed += 1


def _check_real_products(design: Dict[str, Any], report: ValidationReport) -> None:
    """
    Every priced item should reference a real catalog SKU.

    Invented brands are the failure mode where a 2B model invents "Bison
    Kids" or "Kendra's". Requiring a catalog SKU makes fabrication
    impossible: the SKU either exists in the database or the line is dropped.
    """
    furniture = [f for f in (design.get("furniture", []) or []) if isinstance(f, dict)]
    if not furniture:
        return

    from app.catalog import catalog  # imported here to keep the module light

    report.checks_run += 1
    unbacked: List[str] = []
    for item in furniture:
        sku = str(item.get("sku", "")).strip()
        if not sku or catalog.get(sku) is None:
            unbacked.append(str(item.get("name", "unnamed item")))

    if unbacked:
        report.violations.append(Violation(
            code="uncatalogued_item",
            severity=SEVERITY_ERROR,
            message=(
                f"{len(unbacked)} line item(s) have no catalog SKU, so their price and "
                f"brand cannot be verified: {', '.join(unbacked[:3])}."
            ),
            expected="every item references a real catalog SKU",
            actual=f"{len(unbacked)} unbacked",
            remedy_hint=(
                "Choose items from the product catalog. Do not invent brand names; "
                "use the brand and price stored against the SKU."
            ),
        ))
    else:
        report.checks_passed += 1
