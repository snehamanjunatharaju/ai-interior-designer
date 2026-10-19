"""
Pydantic schemas: request/response models and core domain types.

These are the contracts used by the API, the agent pipeline, and the
evaluation harness.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class DesignStyle(str, Enum):
    MODERN = "modern"
    MINIMALIST = "minimalist"
    BOHEMIAN = "bohemian"
    INDUSTRIAL = "industrial"
    SCANDINAVIAN = "scandinavian"
    JAPANESE = "japanese"
    MIDCENTURY = "midcentury"
    LUXURY = "luxury"
    COZY = "cozy"
    TRADITIONAL_INDIAN = "traditional_indian"


class RoomType(str, Enum):
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    HOME_OFFICE = "home_office"
    DINING = "dining"
    BATHROOM = "bathroom"
    BALCONY = "balcony"
    KIDS_ROOM = "kids_room"
    WARDROBE = "wardrobe"
    ENTRANCE = "entrance"


class BudgetTier(str, Enum):
    LOW = "low"          # < ₹1L
    MEDIUM = "medium"    # ₹1L - ₹5L
    HIGH = "high"        # ₹5L - ₹15L
    LUXURY = "luxury"    # > ₹15L


class VastuCompliance(str, Enum):
    IGNORE = "ignore"
    SOFT = "soft"
    STRICT = "strict"


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------


class DesignRequest(BaseModel):
    """The primary design-brief input from a client."""

    room_type: RoomType = RoomType.LIVING_ROOM
    client_preferences: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Free-text brief describing the client's needs. Required.",
    )
    style: DesignStyle = DesignStyle.MODERN
    color_scheme: str = Field(default="neutral")
    budget: BudgetTier = BudgetTier.MEDIUM
    room_dimensions: Optional[str] = Field(
        default=None, description='e.g. "12ft x 10ft, ceiling 9ft"'
    )
    # Explicit Length / Width in feet.
    #
    # Added because a free-text dimension field is parsed, and parsing guesses.
    # "about 12 by 10" is ambiguous; "12ft x 10ft" and "240cm x 200cm" are not.
    # A floor plan drawn from a guessed room size is a floor plan of the wrong
    # room, and nothing downstream can tell the client that - the plan looks
    # perfectly valid. So the structured fields are the source of truth when
    # present, and the free-text field remains only as a fallback for callers
    # that have not migrated.
    #
    # Bounds are deliberately generous but finite: 4ft is smaller than any real
    # room and 100ft is larger than any Indian apartment, so a typo is caught
    # at the boundary instead of producing an unplaceable layout.
    length_ft: Optional[float] = Field(
        default=None, ge=4, le=100,
        description="Room length in feet. Preferred over free text.",
    )
    width_ft: Optional[float] = Field(
        default=None, ge=4, le=100,
        description="Room width in feet. Preferred over free text.",
    )
    ceiling_ft: Optional[float] = Field(
        default=None, ge=6, le=20,
        description="Ceiling height in feet. Optional; defaults to 9ft.",
    )
    # Optional compass inputs. Free-text directions inside the brief are
    # still parsed, but an explicit field is authoritative and avoids the
    # ambiguity of "the window faces the morning light".
    door_direction: Optional[str] = Field(
        default=None, description="Compass direction the main door opens toward"
    )
    window_direction: Optional[str] = Field(
        default=None, description="Compass direction the main window faces"
    )
    num_options: int = Field(default=3, ge=1, le=8)
    include_details: bool = True
    furniture: List[str] = Field(default_factory=list)
    specific_requirements: List[str] = Field(default_factory=list)
    occupants: List[str] = Field(default_factory=list)
    vastu_compliance: VastuCompliance = VastuCompliance.SOFT
    budget_amount_inr: Optional[int] = Field(
        default=None, ge=1000, description="Explicit budget cap in INR."
    )
    lifestyle_notes: str = Field(default="", max_length=1000)

    @field_validator("furniture", "specific_requirements", "occupants")
    @classmethod
    def _dedupe(cls, v: List[str]) -> List[str]:
        # Preserve order while removing duplicates/blanks.
        seen, out = set(), []
        for item in v:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(item.strip())
        return out

    @model_validator(mode="after")
    def _both_dimensions_or_neither(self) -> "DesignRequest":
        """
        Length without width is not a room.

        Accepting one and defaulting the other invents a dimension, which is
        the exact failure the structured fields exist to prevent. Rejecting it
        here is better than drawing a plan for a room that is half imaginary.
        """
        if (self.length_ft is None) != (self.width_ft is None):
            raise ValueError(
                "Provide both length_ft and width_ft, or neither. A single "
                "dimension is not enough to draw a floor plan."
            )
        return self

    def dimension_text(self) -> str:
        """
        The structured dimensions as parseable text, or the free text.

        Lets every existing consumer of `room_dimensions` pick up the new
        fields without each one being rewritten.
        """
        if self.length_ft and self.width_ft:
            base = f"{self.length_ft:g}ft x {self.width_ft:g}ft"
            if self.ceiling_ft:
                base += f", ceiling {self.ceiling_ft:g}ft"
            return base
        return self.room_dimensions or ""

    def brief_object(self) -> Dict[str, Any]:
        """
        The whole submitted form as one structured object.

        This is what every agent is given, and what the validator reads. It
        exists because the agents used to each re-derive the brief from prose:
        the architect saw a text rendering, the validator saw a separately
        parsed constraint set, and the form's own door/window/colour/vastu
        selections reached neither. One object means one answer to "what did
        the client ask for?".
        """
        raw_dims = (self.room_dimensions or "").strip()
        dims = None
        if self.length_ft and self.width_ft:
            dims = {
                "length_ft": self.length_ft,
                "width_ft": self.width_ft,
                "ceiling_ft": self.ceiling_ft,
            }
        elif raw_dims:
            # Free text is kept verbatim and parsed in code, once, so the
            # numeric view below and the parser cannot disagree.
            from app.engine.brief import parse_dimensions

            parsed = parse_dimensions(raw_dims)
            if parsed.is_known():
                dims = {
                    "length_ft": parsed.length_ft,
                    "width_ft": parsed.width_ft,
                    "ceiling_ft": parsed.ceiling_ft,
                }
        return {
            "room_type": self.room_type.value,
            "style": self.style.value,
            "dimensions": dims,
            "dimensions_text": self.dimension_text(),
            "door_direction": self.door_direction,
            "window_direction": self.window_direction,
            "color_scheme": self.color_scheme,
            "vastu_level": self.vastu_compliance.value,
            "budget_tier": self.budget.value,
            "budget_inr": self.budget_amount_inr,
            "occupants": list(self.occupants),
            "furniture": list(self.furniture),
            "special_requirements": list(self.specific_requirements),
            "lifestyle_notes": self.lifestyle_notes,
            "client_brief": self.client_preferences,
        }


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)


# --------------------------------------------------------------------------
# Response / domain models
# --------------------------------------------------------------------------


class FurnitureItem(BaseModel):
    """
    One line item.

    `sku` links back to the catalog row, which is where the price and brand
    actually come from. `estimated_price_inr` is populated by code after
    pricing; the LLM never sets it.
    """

    name: str
    category: str = ""
    brand_suggestions: List[str] = Field(default_factory=list)
    estimated_price_inr: int = 0
    quantity: int = 1
    sku: Optional[str] = None
    unit_price_inr: int = 0
    line_total_inr: int = 0
    url: Optional[str] = None
    width_cm: int = 0
    depth_cm: int = 0
    height_cm: int = 0


class ColorPalette(BaseModel):
    primary: str = ""
    secondary: str = ""
    accent: str = ""
    neutrals: List[str] = Field(default_factory=list)
    hex_codes: List[str] = Field(default_factory=list)


class BudgetBreakdown(BaseModel):
    category: str
    amount_inr: int = 0
    percentage: float = 0.0
    notes: str = ""


class DesignOption(BaseModel):
    """A single complete design proposal generated by the pipeline."""

    title: str = "Design Option"
    concept: str = ""
    layout_description: str = ""
    zones: List[str] = Field(default_factory=list)
    furniture: List[FurnitureItem] = Field(default_factory=list)
    color_palette: Optional[ColorPalette] = None
    lighting_plan: str = ""
    storage_strategy: str = ""
    decor_suggestions: List[str] = Field(default_factory=list)
    estimated_cost_inr: int = 0
    remaining_budget_inr: Optional[int] = None
    budget_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    budget_breakdown: List[BudgetBreakdown] = Field(default_factory=list)
    # Vastu notes are generated from the rule-check results, never from model
    # prose, so a note cannot claim compliance the code check contradicts.
    vastu_notes: List[str] = Field(default_factory=list)
    # The rule results themselves, so the UI and PDF can show what passed and
    # what failed rather than only the sentences summarising them.
    vastu_results: List[Dict[str, Any]] = Field(default_factory=list)
    vastu_score: Optional[int] = None
    # Honest trade-offs, kept apart from the Vastu notes. "Limited decorative
    # flexibility" is a trade-off, not a Vastu observation, and mixing the two
    # made the Vastu section read as nonsense.
    trade_offs: List[str] = Field(default_factory=list)
    brands: List[str] = Field(default_factory=list)
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    # The same rows the option total is computed from, with quantity, unit
    # price and line total. Present as a field so no client has to recompute
    # a price from anywhere else.
    shopping_list: List[Dict[str, Any]] = Field(default_factory=list)
    # Deterministic brief check for this option specifically. The response-level
    # `validation` describes the design that won the reflection loop.
    validation: Optional[Dict[str, Any]] = None
    # Dimensions and zone sizes, so the layout text and the plan agree.
    layout: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    critic_score: Optional[float] = None
    revision_round: int = 0


class RetrievedContext(BaseModel):
    content: str
    source: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    text: str
    provider: str
    model: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_inr: float = 0.0
    cached: bool = False


class DesignResponse(BaseModel):
    request_id: str
    options: List[DesignOption]
    context_sources: List[RetrievedContext] = Field(default_factory=list)
    total_cost_inr: float = 0.0
    latency_ms: float = 0.0
    cost_inr: float = 0.0
    model_used: str = ""
    cached: bool = False
    evaluation: Optional[Dict[str, Any]] = None

    # --- added by the deterministic engine ---
    # Deterministic brief check: violations list, pass rate, check counts.
    validation: Optional[Dict[str, Any]] = None
    # Code-computed costing, with per-line catalog provenance.
    cost_breakdown: Optional[Dict[str, Any]] = None
    # Essentials/upgrade split when the budget cannot cover the wish list.
    budget_plan: Optional[Dict[str, Any]] = None
    # What the parser understood the client to be asking for.
    brief_constraints: Optional[Dict[str, Any]] = None

    # True when the pipeline exhausted its rewrite attempts and is returning
    # the best attempt anyway, still failing its brief checks. The UI must
    # show this rather than presenting an unchecked design as final.
    partial: bool = False

    # Everything the design had to assume rather than measure, plus whether the
    # Vastu claim is verifiable. A Strict-Vastu brief with no door or window
    # bearing sets `vastu_withheld`, and the UI must not show a compliance
    # score it cannot justify.
    assumptions: Optional[Dict[str, Any]] = None

    # Options dropped as near-duplicates, with the reason. Present so a client
    # who asked for three and received two is told why rather than left to
    # assume something failed.
    rejected_options: List[str] = Field(default_factory=list)

    # Per-option self-consistency findings: colour chips vs swatch labels,
    # requested items vs shopping list, prose vs what was bought. A design can
    # satisfy the validator and still disagree with itself, and the client is
    # the one who notices.
    consistency: List[Dict[str, Any]] = Field(default_factory=list)

    # Each revision the reflection loop produced, in order, with its score and
    # whether it was blocking-clean. The best one is returned; this is the
    # evidence for that choice.
    revisions: List[Dict[str, Any]] = Field(default_factory=list)


class EvaluationScore(BaseModel):
    coherence: float = 0.0
    feasibility: float = 0.0
    budget_adherence: float = 0.0
    aesthetics: float = 0.0
    overall: float = 0.0
    rationale: str = ""


class HealthStatus(BaseModel):
    status: str
    version: str
    environment: str
    providers: Dict[str, Any] = Field(default_factory=dict)
    uptime_seconds: float = 0.0


class AgentStep(BaseModel):
    """One step of the multi-agent pipeline, streamed to the client."""

    agent: str
    status: str
    message: str = ""
    duration_ms: float = 0.0
    data: Dict[str, Any] = Field(default_factory=dict)
