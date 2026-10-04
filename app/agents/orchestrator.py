"""
Multi-agent design pipeline.

Four specialised agents collaborate per request:

    Planner  -> decomposes the brief, extracts hard constraints
    Architect-> produces the spatial + aesthetic design
    Budget   -> prices it against the tier, validates the arithmetic
    Critic   -> scores quality and drives a reflection loop

The Critic's verdict feeds back into the Architect for up to
MAX_REFLECTION_ROUNDS revisions, which is the "reflection" pattern used
in agentic systems: generate, evaluate, improve.

Every agent is independently testable and emits an AgentStep for streaming.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.observability import get_logger, metrics, tracer
from app.core.schemas import (
    AgentStep,
    BudgetBreakdown,
    ColorPalette,
    DesignOption,
    DesignRequest,
    DesignResponse,
    FurnitureItem,
    RetrievedContext,
)
from app.engine.brief import BriefConstraints, parse_brief
from app.engine.pricing import build_phased_plan, price_design
from app.engine.contracts import (
    validate_proposal as validate_selection,
    selections_to_furniture,
)
from app.engine.validator import validate_design
from app.services.llm_provider import extract_json, registry
from app.services.retrieval import retrieval_service

logger = get_logger(__name__)

# How many consecutive revisions may fail to improve before the reflection
# loop gives up. One flat round is normal - the model may be addressing a
# different violation - but two in a row means it is not converging.
STALL_LIMIT = 2

DESIGNER_SYSTEM_PROMPT = """You are a senior interior designer with 15 years of \
experience in the Indian residential market. You have deep knowledge of Vastu, Indian \
climates, space constraints of Indian apartments, local brands, and INR pricing. You \
give specific, actionable, practical advice - never generic platitudes. You always \
respond with valid JSON and no other text."""


def _decode_entities(value: str) -> str:
    """
    Turn HTML entities the model leaks into its prose back into characters.

    gemma2:2b frequently writes `&quot;` when quoting a client requirement
    instead of a real quote mark. Left alone it reaches the user verbatim in
    the pros and the layout description.
    """
    if "&" not in value:
        return value
    try:
        return html.unescape(value)
    except Exception:  # pragma: no cover - unescape is effectively total
        return value


def _render_mapping(value: Dict[str, Any]) -> str:
    """Render a model-produced object as readable prose, not raw JSON."""
    lines: List[str] = []
    for key, val in value.items():
        if val is None or val == "" or val == []:
            continue
        rendered = _text(val)
        if not rendered:
            continue
        label = str(key).replace("_", " ").strip()
        lines.append(f"{label}: {rendered}")
    return ". ".join(lines)


def _text(value: Any, limit: int = 0) -> str:
    """
    Coerce an arbitrary LLM value to a safely-truncatable string.

    Models frequently emit `null` for a field they chose to skip, and
    `None[:800]` raises TypeError. Every prompt interpolation goes through
    here so a null field degrades to an empty string instead of crashing
    the pipeline.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        out = _decode_entities(value)
    elif isinstance(value, (int, float, bool)):
        out = str(value)
    elif isinstance(value, dict):
        # Models sometimes answer a prose field with an object. Dumping it as
        # JSON puts raw `{"name": ...}` in front of the user, so render the
        # useful fields as readable lines instead.
        out = _render_mapping(value)
    elif isinstance(value, (list, tuple)):
        out = "; ".join(_text(v) for v in value if v is not None)
    else:
        try:
            out = json.dumps(value, default=str)
        except (TypeError, ValueError):
            out = str(value)
    return out[:limit] if limit else out


def _brief_to_text(request: DesignRequest) -> str:
    """Render a design brief as a compact structured block for prompting."""
    budget_amount = request.budget_amount_inr or {
        "low": 150_000,
        "medium": 800_000,
        "high": 3_000_000,
        "luxury": 10_000_000,
    }.get(request.budget.value, 800_000)

    return f"""ROOM TYPE: {request.room_type.value.replace('_', ' ').title()}
STYLE: {request.style.value.replace('_', ' ')}
COLOUR SCHEME: {request.color_scheme}
BUDGET TIER: {request.budget.value.upper()} (approximately Rs {budget_amount:,} total, inclusive of furniture and decor)
DIMENSIONS: {request.room_dimensions or 'not specified - assume a standard Indian apartment room'}
CLIENT BRIEF: {request.client_preferences}
FURNITURE REQUESTED: {', '.join(request.furniture) if request.furniture else 'choose appropriate furniture'}
SPECIAL REQUIREMENTS: {', '.join(request.specific_requirements) if request.specific_requirements else 'none'}
OCCUPANTS: {', '.join(request.occupants) if request.occupants else 'not specified'}
VASTU: {request.vastu_compliance.value}
LIFESTYLE NOTES: {request.lifestyle_notes or 'none'}"""


def _catalog_block(brief: BriefConstraints) -> str:
    """
    Render the subset of the catalog relevant to this brief.

    The whole catalog is ~50 rows, which is small enough to include in full
    and far more reliable than describing it abstractly: the model can copy
    a real SKU and the validator can then check it. Relevance-filtering by
    room keeps the prompt tight on larger catalogs.
    """
    from app.catalog import catalog

    rows = catalog.all()
    if brief.room_type:
        from app.engine.brief import ROOM_CATEGORIES

        preferred = set(ROOM_CATEGORIES.get(brief.room_type, ()))
        if preferred:
            on_topic = [i for i in rows if i.category in preferred]
            # Always keep a few cross-category picks (decor, lighting) so the
            # design is not artificially limited to furniture.
            others = [i for i in rows if i.category not in preferred]
            rows = on_topic + others[:12]

    lines = [
        f"{'SKU':<22} {'NAME':<44} {'CAT':<10} {'Rs':>7} {'W x D x H (cm)':<18} TIER"
    ]
    for item in rows:
        dims = f"{item.width_cm} x {item.depth_cm} x {item.height_cm}"
        lines.append(
            f"{item.sku:<22} {item.name[:44]:<44} {item.category:<10} "
            f"{item.price_inr:>7,} {dims:<18} {item.tier}"
        )
    return "\n".join(lines)


def _fallback_palette(scheme: Optional[str]) -> Dict[str, Any]:
    """
    A palette for the fallback path, chosen to satisfy the brief's scheme.

    Must pass the same colour check as an LLM-written palette, so a fallback
    for a "vibrant" brief actually is vibrant. Values are chosen for HSV
    saturation well above the 0.35 threshold the validator enforces.
    """
    presets: Dict[str, Dict[str, Any]] = {
        "vibrant": {
            "primary": "coral", "secondary": "teal", "accent": "marigold",
            "neutrals": ["white", "pale grey"],
            "hex_codes": ["#FF6B5B", "#17A2A2", "#F2A93B"],
        },
        "cool": {
            "primary": "teal", "secondary": "navy", "accent": "sky blue",
            "neutrals": ["white", "light grey"],
            "hex_codes": ["#17A2A2", "#1E3A5F", "#4A9FD8"],
        },
        "earthy": {
            "primary": "terracotta", "secondary": "olive", "accent": "rust",
            "neutrals": ["sand", "off white"],
            "hex_codes": ["#C0704A", "#7A8B4F", "#A63D2F"],
        },
        "pastel": {
            "primary": "pale pink", "secondary": "powder blue", "accent": "mint",
            "neutrals": ["white"],
            "hex_codes": ["#F7C6D0", "#BBD7EE", "#BFEAD9"],
        },
        "monochrome": {
            "primary": "charcoal", "secondary": "grey", "accent": "silver",
            "neutrals": ["white", "black"],
            "hex_codes": ["#2E2E2E", "#7A7A7A", "#C9C9C9"],
        },
    }
    return presets.get((scheme or "neutral").lower(), {
        "primary": "warm white", "secondary": "greige", "accent": "sage",
        "neutrals": ["white", "soft grey"],
        "hex_codes": ["#F2EFE9", "#CFC7BA", "#9CAF88"],
    })


# ==========================================================================
# Agents
# ==========================================================================


class BaseAgent:
    """Common agent scaffolding: timing, metrics, and step emission."""

    name = "base"
    system_prompt = DESIGNER_SYSTEM_PROMPT

    async def _call_llm(
        self, prompt: str, json_mode: bool = True, temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        result = await registry.complete_json(
            prompt, system=self.system_prompt, temperature=temperature
        )
        return result

    def _step(
        self, status: str, message: str, duration_ms: float = 0.0, **data: Any
    ) -> AgentStep:
        return AgentStep(
            agent=self.name,
            status=status,
            message=message,
            duration_ms=round(duration_ms, 2),
            data=data,
        )


class PlannerAgent(BaseAgent):
    """
    Decomposes the brief into a concrete design specification.

    Run first so downstream agents share one agreed interpretation instead of
    each re-reading the raw brief.
    """

    name = "planner"
    system_prompt = DESIGNER_SYSTEM_PROMPT + """

Your job is PLANNING. Convert the client brief into a precise design plan.
Identify the room's constraints, the zones to create, the design intent, and the
non-negotiable requirements. Be specific to the actual room, not generic.

Respond ONLY with valid JSON:
{
  "design_intent": "one sentence capturing the design direction",
  "constraints": ["hard constraint", ...],
  "zones": ["named zone with purpose", ...],
  "key_principles": ["which design principles apply here", ...],
  "focal_point": "the single visual anchor of the room",
  "risks": ["potential design or budget risk to watch", ...]
}"""

    async def run(self, request: DesignRequest, contexts: List[RetrievedContext]) -> Tuple[Dict[str, Any], AgentStep]:
        start = time.perf_counter()
        context_block = retrieval_service.format_context(contexts)

        prompt = f"""Create a design plan for this room.

{_brief_to_text(request)}

RELEVANT DESIGN KNOWLEDGE (use these facts, cite them where relevant):
{context_block or 'No external knowledge retrieved - rely on your expertise.'}

Produce a focused, room-specific design plan as JSON."""

        with tracer.start("agent.planner"):
            try:
                result = await self._call_llm(prompt)
                plan = result["json"] if isinstance(result["json"], dict) else {}
            except Exception as exc:
                logger.warning("planner_failed error=%s", exc)
                plan = {}
                plan = self._fallback_plan(request)

        plan.setdefault("design_intent", f"{request.style.value} {request.room_type.value}")
        plan.setdefault("constraints", [])
        plan.setdefault("zones", [])

        duration = (time.perf_counter() - start) * 1000
        metrics.observe("agent.planner.ms", duration)
        return plan, self._step(
            "ok",
            f"Planned {len(plan.get('zones', []))} zones for a "
            f"{request.room_type.value.replace('_', ' ')}",
            duration,
            plan=plan,
        )

    def _fallback_plan(self, request: DesignRequest) -> Dict[str, Any]:
        return {
            "design_intent": f"{request.style.value} {request.room_type.value}",
            "constraints": [],
            "zones": ["main functional zone", "circulation", "storage", "accent"],
            "key_principles": ["balance", "proportion", "emphasis"],
            "focal_point": "primary furniture grouping",
            "risks": [],
        }


class ArchitectAgent(BaseAgent):
    """Generates the full design option: layout, palette, lighting, decor."""

    name = "architect"
    system_prompt = DESIGNER_SYSTEM_PROMPT + """

Your job is DESIGN. Produce a complete, specific, buildable design.
Never give generic advice - name actual products with their catalog SKU,
colours with hex codes, and specific materials.

CRITICAL - PRICING IS NOT YOUR JOB. You do NOT invent prices, and you do NOT
invent brand names. Every furniture item MUST carry a `sku` chosen from the
CATALOG below. The price and brand are read from the catalog by code. If you
write your own price it will be discarded. If you invent a brand, the design
is rejected.

Respond ONLY with valid JSON:
{
  "title": "short evocative design name",
  "concept": "2-3 sentence description of the concept",
  "layout_description": "how the room is actually laid out and zoned, with measurements",
  "zones": ["zone: description", ...],
  "furniture": [
    {"name": "exact catalog product name", "category": "...", "sku": "CATALOG-SKU",
     "quantity": 1}
  ],
  "color_palette": {"primary": "", "secondary": "", "accent": "", "neutrals": ["..."], "hex_codes": ["#RRGGBB", ...]},
  "lighting_plan": "layered lighting description with temperatures and wattage",
  "storage_strategy": "storage plan with quantities",
  "decor_suggestions": ["...", ...],
  "vastu_notes": ["...", ...],
  "brands": ["brand name", ...],
  "pros": ["...", ...],
  "cons": ["honest trade-offs", ...]
}

HARD REQUIREMENTS - violating any of these makes the design invalid:
- Every furniture entry has a `sku` that appears verbatim in the catalog.
- The number of items matches what the client asked for, including quantities.
- If the client asked for a specific colour scheme, the palette matches it."""

    async def run(
        self,
        request: DesignRequest,
        plan: Dict[str, Any],
        contexts: List[RetrievedContext],
        revision: int = 0,
        critique: Optional[Dict[str, Any]] = None,
        brief: Optional[BriefConstraints] = None,
    ) -> Tuple[Dict[str, Any], AgentStep]:
        start = time.perf_counter()
        context_block = retrieval_service.format_context(contexts)
        brief = brief or parse_brief(
            client_preferences=request.client_preferences,
            room_type=request.room_type.value,
            color_scheme=request.color_scheme,
            budget_inr=request.budget_amount_inr,
            room_dimensions=request.room_dimensions or "",
        )

        revision_block = ""
        if revision > 0 and critique:
            issues = critique.get("issues", [])
            validation = critique.get("validation") or {}
            violated = [
                v for v in validation.get("violations", [])
                if v.get("severity") == "error"
            ]
            violation_lines = "\n".join(
                f"- {v.get('message')} FIX: {v.get('remedy_hint', '')}"
                for v in violated
            )
            revision_block = f"""

THIS IS REVISION {revision}. Your previous attempt was checked by automated code
and REJECTED. These are measured facts, not opinions - you MUST fix all of them:
{violation_lines or '- the previous attempt was rejected'}

{f'- Additionally, a critic noted: ' + chr(10).join('- ' + i for i in issues) if issues else ''}

Your design is invalid until every one of these is resolved. Double-check the
item counts and the colour scheme against the MANDATORY REQUIREMENTS before
responding."""

        # Spell the hard requirements out, so the model is told exactly what
        # the validator will check rather than inferring it from the brief.
        requirement_lines = []
        for item, qty in brief.required_items.items():
            requirement_lines.append(f"- The client asked for {qty} x {item} - the design MUST contain {qty}.")
        if brief.color_scheme:
            requirement_lines.append(f"- Colour scheme must be '{brief.color_scheme}'.")
        if brief.budget_inr:
            requirement_lines.append(
                f"- Total cost must stay at or under Rs {brief.budget_inr:,}."
            )
        requirements_block = "\n".join(requirement_lines) or "- No specific item counts were given."

        prompt = f"""Design this room.

{_brief_to_text(request)}

DESIGN PLAN (agreed, follow it):
{json.dumps(plan, indent=2)}

RELEVANT DESIGN KNOWLEDGE:
{context_block or 'None retrieved - rely on expertise.'}

PRODUCT CATALOG - choose from this. Use the `sku` column exactly:
{_catalog_block(brief)}

MANDATORY REQUIREMENTS (checked by code after you respond):
{requirements_block}
{revision_block}

Produce the complete design as JSON. Give every furniture item a real `sku`
from the catalog above, copied verbatim. Do not write prices anywhere in your
answer - the `price` field is not part of the schema and code applies prices
from the catalog. An item whose SKU is not in the list above is discarded."""

        with tracer.start("agent.architect", revision=revision):
            try:
                result = await self._call_llm(prompt, temperature=0.8)
                raw = result["json"] if isinstance(result["json"], dict) else {}
                # Validate the selection against the catalog before anything
                # downstream sees it. An invented SKU is dropped here rather
                # than becoming an unpriceable, unbought line item.
                proposal, problems = validate_selection(raw)
                if proposal is not None:
                    design = raw
                    design["furniture"] = selections_to_furniture(proposal)
                    if proposal.color_palette is not None:
                        design["color_palette"] = proposal.color_palette.model_dump()
                    # Prose strings are also trimmed to their validated lengths.
                    design["title"] = proposal.title
                    design["concept"] = proposal.concept
                    design["layout_description"] = proposal.layout_description
                else:
                    for problem in problems[:5]:
                        logger.warning("architect_selection_rejected %s", problem)
                    design = raw
            except Exception as exc:
                logger.warning("architect_failed error=%s", exc)
                design = self._fallback_design(request, plan, brief)

            if not design.get("title"):
                design = self._fallback_design(request, plan, brief)
            design["revision_round"] = revision

        duration = (time.perf_counter() - start) * 1000
        metrics.observe("agent.architect.ms", duration)
        return design, self._step(
            "ok",
            f"Designed '{design.get('title', 'untitled')}'"
            + (f" (revision {revision})" if revision else ""),
            duration,
            title=design.get("title", ""),
        )

    def _fallback_design(
        self, request: DesignRequest, plan: Dict[str, Any],
        brief: Optional[BriefConstraints] = None,
    ) -> Dict[str, Any]:
        """
        Build a usable design entirely from the catalog, with no LLM involved.

        This is the recovery path when the model fails or returns unusable
        JSON. The previous version returned a stub with an empty `furniture`
        list, which meant the whole request produced a Rs 0 design that the
        validator then rejected - the user got nothing for a 90-second wait.

        A catalog-driven fallback is strictly better: it satisfies the brief's
        item counts, prices correctly, and the prose simply stays generic. The
        honesty cost is handled by the `partial` flag on the response.
        """
        from app.engine.brief import find_products

        brief = brief or parse_brief(
            client_preferences=request.client_preferences,
            furniture=request.furniture,
            specific_requirements=request.specific_requirements,
            occupants=request.occupants,
            room_type=request.room_type.value,
            color_scheme=request.color_scheme,
            budget_inr=request.budget_amount_inr,
            vastu_compliance=request.vastu_compliance.value,
            room_dimensions=request.room_dimensions or "",
        )

        # The fallback used to take `find_products(category)[0]` per required
        # category and ignore the budget completely, so a Rs 10,000 brief
        # produced a Rs 18,200 room that the validator then (correctly)
        # rejected. The design the user saw by default was always invalid.
        # Route it through the optimizer, which respects the budget and falls
        # back to the essentials when the brief is unaffordable.
        from app.engine.budget_optimizer import optimize_to_budget

        required = dict(brief.required_items)
        optional = {k: 1 for k in (brief.optional_items or ())}
        if not required:
            from app.engine.brief import ROOM_CATEGORIES
            required = {c: 1 for c in list(ROOM_CATEGORIES.get(brief.room_type, ()))[:3]}

        optimised = optimize_to_budget(
            required=required,
            budget_inr=brief.budget_inr,
            optional=optional,
            style_tier=request.budget.value,
        )
        furniture: List[Dict[str, Any]] = [
            {
                "name": sel["name"],
                "category": sel["category"],
                "sku": sel["sku"],
                "quantity": sel["quantity"],
                "brand_suggestions": [sel["brand"]],
            }
            for sel in optimised["selections"]
        ]
        # Record *why* anything is absent, so the validator reports a budget
        # constraint rather than accusing the model of forgetting an item.
        _budget_infeasible = None
        budget_note = None
        if not optimised["feasible"]:
            _budget_infeasible = {
                "unserved": optimised["unserved"],
                "shortfall_inr": optimised["shortfall_inr"],
                "budget_inr": optimised["budget_inr"],
                "message": optimised["infeasible"],
                "options": optimised["options"],
            }
            budget_note = optimised["infeasible"]

        if not furniture and brief.budget_inr is None:
            # No budget was stated and nothing was requested, so propose a
            # coherent starter set rather than shipping an empty room.
            #
            # Gated on "no budget given" deliberately: when a budget *was*
            # stated and nothing fits, adding unbudgeted items here silently
            # broke the budget the optimizer had just honoured - a Rs 2,000
            # brief came back with a Rs 7,700 room.
            from app.engine.brief import ROOM_CATEGORIES
            for category in ROOM_CATEGORIES.get(brief.room_type, ())[:3]:
                options = find_products(category)
                if options:
                    product = options[0]
                    furniture.append({
                        "name": product.name, "category": product.category,
                        "sku": product.sku, "quantity": 1,
                    })
        elif not furniture and brief.budget_inr is not None:
            # A budget was given and nothing is affordable. Saying so beats
            # inventing furniture the client cannot buy.
            if not budget_note:
                budget_note = (
                    f"Nothing in this brief is affordable within "
                    f"Rs {brief.budget_inr:,}. The cheapest items available start "
                    f"above that figure."
                )
                _budget_infeasible = {
                    "unserved": optimised["unserved"],
                    "shortfall_inr": optimised["shortfall_inr"],
                    "budget_inr": optimised["budget_inr"],
                    "message": budget_note,
                    "options": optimised["options"],
                }

        room = request.room_type.value.replace("_", " ").title()
        listed = ", ".join(f["name"] for f in furniture[:4]) or "the essential pieces"

        fallback = {
            "title": f"{request.style.value.title()} {room}",
            "concept": (
                f"A practical, well-organised {room.lower()} built around "
                f"{listed}."
            ),
            # Never a template sentinel: the placeholder check would block it.
            "layout_description": (
                f"Group the {listed} along the walls, keeping the centre of the "
                f"room clear for circulation. Storage is wall-mounted where "
                f"possible to free floor space."
            ),
            "zones": plan.get("zones", []) or ["sleeping", "study", "storage"],
            "furniture": furniture,
            "color_palette": _fallback_palette(brief.color_scheme),
            "lighting_plan": (
                "Layered lighting: a warm 3000K ceiling light for general "
                "illumination, a 4000K task light at the desk, and dimmable "
                "accent lighting to finish the room."
            ),
            "storage_strategy": (
                "Wall-mounted and under-bed storage, sized to the available "
                "wall length rather than bought as freestanding units."
            ),
            "decor_suggestions": [
                "One large rug to anchor the seating or sleeping area",
                "Curtains with a blackout lining if the window faces west",
            ],
            "vastu_notes": [],
            "brands": sorted({f.get("brand", "") for f in furniture} - {""}),
            "pros": [
                "Every item is a real catalog product with a verifiable price",
                "Fits a constrained budget without inventing a number",
            ],
            "cons": (
                [budget_note] if _budget_infeasible else
                [
                    "Prose is generic because this design was assembled by the "
                    "fallback path rather than written by the design agent",
                ]
            ),
            # Carried through so the validator can distinguish "the model forgot
            # an item" from "the budget could not buy this item".
            "_budget_infeasible": _budget_infeasible,
        }
        return fallback


class BudgetAgent(BaseAgent):
    """
    Prices the design and validates that the total respects the budget tier.

    Also self-corrects: if the LLM's own arithmetic disagrees with the summed
    line items, it recomputes. Small models are frequently sloppy here, and a
    wrong total is the fastest way to lose a client's trust.
    """

    name = "budget"
    system_prompt = DESIGNER_SYSTEM_PROMPT + """

Your job is COSTING. Price a design realistically for the Indian market.
Be honest - if a design exceeds its budget tier, say so and propose reductions.
Use realistic INR prices for Indian furniture retail.

Respond ONLY with valid JSON:
{
  "estimated_cost_inr": 0,
  "budget_breakdown": [{"category": "...", "amount_inr": 0, "percentage": 0, "notes": "..."}],
  "within_budget": true,
  "cost_drivers": ["the main things driving cost", ...],
  "cost_savings": ["how to cut cost without hurting the design", ...]
}"""

    async def run(
        self,
        request: DesignRequest,
        design: Dict[str, Any],
        contexts: List[RetrievedContext],
    ) -> Tuple[Dict[str, Any], AgentStep]:
        start = time.perf_counter()
        budget_amount = request.budget_amount_inr or {
            "low": 150_000, "medium": 800_000,
            "high": 3_000_000, "luxury": 10_000_000,
        }.get(request.budget.value, 800_000)

        furniture_summary = [
            {
                "item": f.get("name", "item"),
                "price": f.get("estimated_price_inr", 0),
                "qty": f.get("quantity", 1),
            }
            for f in design.get("furniture", [])
            if isinstance(f, dict)
        ]

        prompt = f"""Price this design.

{_brief_to_text(request)}

DESIGN TO PRICE:
Title: {_text(design.get('title'))}
Layout: {_text(design.get('layout_description'), 800)}
Furniture line items: {json.dumps(furniture_summary)}
Lighting: {_text(design.get('lighting_plan'), 300)}
Storage: {_text(design.get('storage_strategy'), 300)}

STRICT BUDGET CEILING: Rs {budget_amount:,}

Price every category. The sum of breakdown amounts must equal estimated_cost_inr.
Flag clearly if this design cannot fit under the ceiling."""

        with tracer.start("agent.budget"):
            try:
                result = await self._call_llm(prompt, temperature=0.3)
                costing = result["json"] if isinstance(result["json"], dict) else {}
            except Exception as exc:
                logger.warning("budget_failed error=%s", exc)
                costing = {}

        costing = self._reconcile(costing, design, request, budget_amount)

        duration = (time.perf_counter() - start) * 1000
        metrics.observe("agent.budget.ms", duration)
        metrics.gauge("budget.estimate_inr", float(costing.get("estimated_cost_inr", 0)))
        metrics.increment(
            "budget.within_tier" if costing.get("within_budget") else "budget.over_tier"
        )

        status = "ok" if costing.get("within_budget") else "warning"
        if not costing.get("costing_complete", True):
            message = "Could not price this design - no line items returned"
        else:
            message = (
                f"Estimated Rs {costing.get('estimated_cost_inr', 0):,} "
                f"({'within' if costing.get('within_budget') else 'OVER'} budget)"
            )
        return costing, self._step(
            status,
            message,
            duration,
            estimated_cost_inr=costing.get("estimated_cost_inr", 0),
        )

    def _reconcile(
        self,
        costing: Dict[str, Any],
        design: Dict[str, Any],
        request: DesignRequest,
        budget_amount: int,
    ) -> Dict[str, Any]:
        """Fix arithmetic and enforce the ceiling. Deterministic, not LLM-based."""
        breakdown: List[Dict[str, Any]] = []
        for item in costing.get("budget_breakdown", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                amount = int(float(item.get("amount_inr", 0)))
            except (TypeError, ValueError):
                amount = 0
            breakdown.append({
                "category": str(item.get("category", "misc")),
                "amount_inr": max(0, amount),
                "percentage": float(item.get("percentage", 0) or 0),
                "notes": str(item.get("notes", "")),
            })

        # Independently recompute from the furniture line items.
        furniture_total = sum(
            int(float(f.get("estimated_price_inr", 0) or 0))
            * int(f.get("quantity", 1) or 1)
            for f in design.get("furniture", [])
            if isinstance(f, dict)
        )

        if not breakdown:
            breakdown = [{
                "category": "Furniture and soft furnishings",
                "amount_inr": furniture_total,
                "percentage": 100.0,
                "notes": "Derived from quoted line items",
            }]

        total = sum(b["amount_inr"] for b in breakdown)

        # A design with no priceable line items is *unpriced*, not free.
        # Treating a zero total as "within budget" would report success on
        # a design we know nothing about, so it is flagged explicitly and
        # handed to the Critic as a defect.
        if total <= 0:
            costing["budget_breakdown"] = []
            costing["estimated_cost_inr"] = 0
            costing["within_budget"] = False
            costing["costing_complete"] = False
            costing.setdefault("cost_drivers", [])
            costing["cost_savings"] = [
                "Costing failed: the model returned no priceable line items. "
                "Treat this estimate as unavailable and re-run."
            ]
            metrics.increment("budget.unpriced")
            logger.warning("budget_unpriced design has no priceable line items")
            return costing

        # Trust the sum of the parts over a hallucinated headline total.
        if abs(total - float(costing.get("estimated_cost_inr", 0) or 0)) > 0.2 * max(total, 1):
            logger.info("budget_reconciled llm_total=%s computed_total=%s",
                        costing.get("estimated_cost_inr"), total)
            metrics.increment("budget.arithmetic_corrected")

        # Recompute percentages so they always add to ~100.
        for b in breakdown:
            b["percentage"] = round((b["amount_inr"] / total) * 100, 1) if total else 0.0

        costing["budget_breakdown"] = breakdown
        costing["estimated_cost_inr"] = total
        costing["within_budget"] = total <= budget_amount
        costing["costing_complete"] = True
        costing.setdefault("cost_drivers", [])
        costing.setdefault("cost_savings", [])

        if not costing["within_budget"]:
            overage = total - budget_amount
            costing["cost_savings"].insert(0, (
                f"Design exceeds the budget by Rs {overage:,}. Reduce by: "
                + ", ".join(
                    breakdown_sorted_suggestion(breakdown)
                )
            ))
            metrics.increment("budget.overruns_detected")

        return costing


def breakdown_sorted_suggestion(breakdown: List[Dict[str, Any]]) -> List[str]:
    """Suggest cutting from the largest discretionary categories first."""
    discretionary = ("decor", "lighting", "paint", "soft", "furniture", "accessor")
    ranked = sorted(
        breakdown,
        key=lambda b: (not any(d in b["category"].lower() for d in discretionary), -b["amount_inr"]),
    )
    return [b["category"] for b in ranked[:3]]


class CriticAgent(BaseAgent):
    """
    Evaluates the design and produces actionable critique.

    This is the scoring function of the reflection loop: it decides whether
    the design is good enough or needs another Architect pass.
    """

    name = "critic"
    system_prompt = DESIGNER_SYSTEM_PROMPT + """

Your job is CRITIQUE. You are a demanding design critic. Score honestly - do not
be generous. Penalise: generic advice, missing measurements, unrealistic prices
for the Indian market, ignored constraints, and designs that ignore the budget.
Every score is 0-10.

If the costing below reports "costing_complete": false, the design has no
priceable line items at all. Score budget_adherence 0 and list that as an issue.

Respond ONLY with valid JSON:
{
  "scores": {
    "coherence": 0,
    "feasibility": 0,
    "budget_adherence": 0,
    "aesthetics": 0,
    "specificity": 0
  },
  "overall": 0,
  "issues": ["specific, actionable problem", ...],
  "strengths": ["...", ...],
  "needs_revision": false
}"""

    async def run(
        self,
        request: DesignRequest,
        design: Dict[str, Any],
        costing: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], AgentStep]:
        start = time.perf_counter()

        prompt = f"""Critique this design.

{_brief_to_text(request)}

DESIGN PLAN:
{json.dumps(plan, indent=2)[:1200]}

PROPOSED DESIGN:
{json.dumps(design, indent=2)[:3000]}

COSTING:
{json.dumps(costing, indent=2)[:1200]}

Score it honestly on coherence, feasibility, budget adherence, aesthetics, and
specificity. List the concrete problems that must be fixed. Set needs_revision to
true if overall is below 7."""

        with tracer.start("agent.critic"):
            try:
                result = await self._call_llm(prompt, temperature=0.2)
                critique = result["json"] if isinstance(result["json"], dict) else {}
            except Exception as exc:
                logger.warning("critic_failed error=%s", exc)
                critique = {}

        critique = self._sanitize(critique)
        duration = (time.perf_counter() - start) * 1000
        metrics.observe("agent.critic.ms", duration)
        metrics.observe("critic.overall_score", critique.get("overall", 0))
        metrics.increment(
            "critic.revision_requested" if critique.get("needs_revision") else "critic.accepted"
        )

        return critique, self._step(
            "ok" if not critique.get("needs_revision") else "warning",
            f"Critic scored {critique.get('overall', 0):.1f}/10"
            + (f" - {len(critique.get('issues', []))} issues" if critique.get("issues") else ""),
            duration,
            overall=critique.get("overall", 0),
            needs_revision=critique.get("needs_revision", False),
        )

    def _sanitize(self, critique: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp scores to [0,10] and repair the shape defensively."""
        raw_scores = critique.get("scores") or {}
        scores: Dict[str, float] = {}
        for key in ("coherence", "feasibility", "budget_adherence", "aesthetics", "specificity"):
            try:
                scores[key] = max(0.0, min(10.0, float(raw_scores.get(key, 0) or 0)))
            except (TypeError, ValueError):
                scores[key] = 0.0

        if not any(scores.values()):
            scores = {"coherence": 5, "feasibility": 5, "budget_adherence": 5,
                      "aesthetics": 5, "specificity": 5}

        try:
            overall = float(critique.get("overall", 0) or 0)
        except (TypeError, ValueError):
            overall = 0.0
        if not overall:
            overall = sum(scores.values()) / len(scores)
        overall = max(0.0, min(10.0, overall))

        issues = [str(i) for i in (critique.get("issues") or []) if str(i).strip()]

        return {
            "scores": scores,
            "overall": overall,
            "issues": issues,
            "strengths": [str(s) for s in (critique.get("strengths") or [])][:5],
            # Auto-trigger revision on a low score even if the model forgot to.
            "needs_revision": bool(critique.get("needs_revision")) or overall < 7.0,
        }


# ==========================================================================
# Orchestrator
# ==========================================================================


class DesignPipeline:
    """
    Coordinates the agents and exposes both buffered and streaming APIs.
    """

    def __init__(self) -> None:
        self.planner = PlannerAgent()
        self.architect = ArchitectAgent()
        self.budget = BudgetAgent()
        self.critic = CriticAgent()
        # Tracks which model actually served the most recent request so the
        # response reports reality rather than the configured default.
        self._last_model = settings.DEFAULT_MODEL

    @staticmethod
    def catalog_has(sku: str) -> bool:
        """True when `sku` names a product that exists in the catalog."""
        from app.catalog import catalog

        sku = (sku or "").strip()
        return bool(sku) and catalog.get(sku) is not None

    @staticmethod
    def _repair_invented_sku(item: Dict[str, Any]) -> None:
        """
        Replace a SKU the model made up with a real catalog product.

        The architect is asked to pick from a catalog list, and gemma2:2b
        sometimes invents a plausible-looking code instead ("CHR-01"). An
        invented code cannot be priced, cannot be measured, and cannot be
        bought, so the line would otherwise reach the client as an
        unverifiable product. Matching on the item's name finds the real SKU,
        and the category narrows the search.
        """
        from app.catalog import catalog
        from app.engine.brief import BRIEF_TO_CATALOG_CATEGORY

        name = str(item.get("name", "")).strip()
        if not name:
            return

        category = str(item.get("category", "")).strip().lower()
        mapped = BRIEF_TO_CATALOG_CATEGORY.get(category) or category
        # The mapping's value is a set of category names.
        categories = (
            {str(c).lower() for c in mapped}
            if isinstance(mapped, (set, list, tuple))
            else {str(mapped).lower()}
        )
        pool = [p for p in catalog.all() if p.category.lower() in categories]
        if not pool:
            pool = catalog.all()

        # When the model echoes the SKU back as the name ("name": "CHR-01"), there are
        # no words to match on. The category still identifies the product, so
        # fall back to the catalog's own name for that category. A name that is
        # just a code (letters, digits and separators) is treated the same way,
        # because that is not a product name either.
        if name.upper() == str(item.get("sku", "")).strip().upper():
            product = min(pool, key=lambda p: p.price_inr)
            logger.info(
                "invented_sku_replaced invented=%s resolved=%s (sku echoed as name)",
                item.get("sku"), product.sku,
            )
            item["sku"] = product.sku
            item["name"] = product.name
            item["category"] = product.category
            item.setdefault("brand_suggestions", [product.brand])
            item["width_cm"] = product.width_cm
            item["depth_cm"] = product.depth_cm
            item["height_cm"] = product.height_cm
            return

        def score(product: Any) -> int:
            pname = str(product.name).lower()
            if pname == name.lower():
                return 0
            # Prefer a name that contains the model's, or is contained by it.
            if name.lower() in pname or pname in name.lower():
                return 1
            words = {w for w in name.lower().split() if len(w) > 2}
            return 2 + len(words - {w for w in pname.split() if len(w) > 2})

        best = min(pool, key=score)
        if score(best) <= 2:
            logger.info(
                "invented_sku_replaced invented=%s resolved=%s name=%s",
                item.get("sku"), best.sku, best.name,
            )
            item["sku"] = best.sku
            item["name"] = best.name
            item["category"] = best.category
            item.setdefault("brand_suggestions", [best.brand])
            item["width_cm"] = best.width_cm
            item["depth_cm"] = best.depth_cm
            item["height_cm"] = best.height_cm

    def _normalize_design(
        self, design: Dict[str, Any], constraints: BriefConstraints
    ) -> Dict[str, Any]:
        """
        Repair design details that the model gets wrong in predictable ways.

        This runs *before* validation, so the reflection loop is not asked to
        fix a defect code can fix outright. Two repairs, both deterministic:

        1. Quantities. A brief asking for two beds plus a model that emits one
           bed SKU with `quantity: 1` is the single most common failure. The
           brief is authoritative, so the quantity is corrected here.
        2. Identifiers. Models echo a SKU back as the item name, leaving
           "DSK-KID-01" where "Kids Study Table 90cm" belongs. The catalog
           supplies the real name, brand, and category.

        Only the listed fields are touched; all other content the model wrote
        is left exactly as-is.
        """
        budget = constraints.budget_inr
        # Repair invented SKUs before costing, so the repaired lines are picked
        # up by this pricing pass rather than needing a second one. Only an
        # item that *claims* a SKU and gets it wrong is repaired here; an item
        # with no SKU at all is left for the validator to report, because
        # guessing a product for it would invent a line the model never asked
        # for.
        for item in design.get("furniture", []) or []:
            if not isinstance(item, dict):
                continue
            claimed = str(item.get("sku", "")).strip()
            if claimed and not self.catalog_has(claimed):
                self._repair_invented_sku(item)

        costed = price_design(design, budget_inr=budget)
        priced_by_sku = {li["sku"]: li for li in costed["line_items"]}

        for item in design.get("furniture", []) or []:
            if not isinstance(item, dict):
                continue
            line = priced_by_sku.get(item.get("sku"))
            if not line:
                continue

            name = str(item.get("name", "")).strip()
            # Models often append the SKU to the name, e.g. "Kids Bunk Bed
            # (SKU: BED-KID-BUNK-01)". That is an artefact of the prompt, not
            # a product name, and the catalog already knows the real one.
            name = re.sub(r"\s*\(?\s*sku\s*[:=]?\s*[\w-]+\s*\)?\s*$", "", name, flags=re.I)
            if line["sku"].upper() in name.upper() or not name:
                item["name"] = line["name"]
            item["category"] = item.get("category") or line["category"]
            if not item.get("brand_suggestions"):
                item["brand_suggestions"] = [line["brand"]]

            requested = constraints.required_items.get(line["category"])
            if requested:
                # A unit that sleeps N people satisfies the requirement for
                # ceil(requested / N) units, so a bunk bed is not duplicated.
                sleeps = max(1, line.get("sleeps_per_unit", 1))
                needed = -(-requested // sleeps)
                try:
                    current = int(float(item.get("quantity", 1) or 1))
                except (TypeError, ValueError):
                    current = 1
                if current < needed:
                    item["quantity"] = needed
                    item["_quantity_repaired"] = True

        # 3. Insert required items the model left out entirely.
        #
        # The reflection loop asks the model to fix a missing item, and a 2B
        # model frequently does not - it regenerates prose and quietly omits
        # the same thing again, burning the round budget for nothing. A
        # required bed that the client explicitly asked for is not a creative
        # decision, so code makes it: pick the cheapest catalog product in
        # that category and add it. This is the last step, after quantities
        # are settled, so a two-bed brief yields two beds by construction.
        self._insert_missing_required(design, constraints)

        return design

    @staticmethod
    def _insert_missing_required(
        design: Dict[str, Any], constraints: BriefConstraints
    ) -> None:
        """Add any brief-required item the design does not already contain."""
        from app.catalog import catalog

        furniture = design.setdefault("furniture", [])
        if not isinstance(furniture, list):
            return

        from app.engine.brief import products_satisfying
        from app.engine.validator import _count_items
        from app.engine.validator import _SYNONYM_CATEGORIES

        for category, requested in constraints.required_items.items():
            if requested <= 0:
                continue

            # Count what the design already provides, using the same rule the
            # validator will use to judge the result. Sharing the function is
            # deliberate: two implementations of "does this count?" is how
            # insertion and validation drift apart.
            served = _count_items(
                design, _SYNONYM_CATEGORIES.get(category, {category}), category
            )

            shortfall = requested - served
            if shortfall <= 0:
                continue

            # Resolve through the brief->catalog mapping, so "sofa" finds a
            # sofa rather than any old piece of seating.
            options = products_satisfying(category)
            if not options:
                # Nothing in the catalog for this term; the validator
                # will report it rather than this inventing a product.
                continue
            product = options[0]

            # A unit that sleeps N covers N, so a shortfall of one may be
            # satisfied by a single bunk.
            units = -(-shortfall // max(1, product.sleeps))
            furniture.append({
                "name": product.name,
                # The catalog's category, not the brief's word. Labeling a
                # stool as a sofa category is how a wrong item slips through
                # the checks that follow.
                "category": product.category,
                "sku": product.sku,
                "quantity": units,
                "brand_suggestions": [product.brand],
                "_inserted_by_validator": True,
            })
            logger.info(
                "required_item_inserted category=%s requested=%s served=%d sku=%s",
                category, requested, served, product.sku,
            )
            metrics.increment("design.inserted_required_item")

    @staticmethod
    def _parse_constraints(request: DesignRequest) -> BriefConstraints:
        """
        Turn the API request into a checkable constraint set.

        This is the bridge between the old "ask the model nicely" approach and
        the deterministic checks. Everything the validator needs is derived
        here, once, in code.
        """
        return parse_brief(
            client_preferences=request.client_preferences,
            furniture=request.furniture,
            specific_requirements=request.specific_requirements,
            occupants=request.occupants,
            room_type=request.room_type.value,
            style=request.style.value,
            color_scheme=request.color_scheme,
            budget_inr=request.budget_amount_inr,
            vastu_compliance=request.vastu_compliance.value,
            room_dimensions=request.room_dimensions or "",
        )

    async def generate(
        self, request: DesignRequest, request_id: str = ""
    ) -> DesignResponse:
        """
        Run the full pipeline and return a complete response.

        The pipeline body is factored into `_execute` so both this method and
        `stream` share one implementation - no duplicated orchestration logic.
        """
        request_id = request_id or uuid.uuid4().hex

        async for event in self.stream(request, request_id=request_id):
            if event.get("type") == "complete":
                return DesignResponse(**event["data"])

        # Unreachable in practice: stream always terminates with `complete`.
        raise RuntimeError("pipeline ended without producing a response")

    async def stream(
        self, request: DesignRequest, request_id: str = ""
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute the pipeline, yielding events as they happen.

        Event types: `step` (agent progress) and `complete` (final payload).
        Streaming lets the UI show real pipeline progress rather than an
        opaque spinner, which is a materially better UX on 30-60s requests.
        """
        request_id = request_id or uuid.uuid4().hex
        start = time.perf_counter()

        retrieval_service.initialize()
        contexts = await retrieval_service.retrieve_for_request(request)
        yield {
            "type": "step",
            "data": AgentStep(
                agent="retriever",
                status="ok",
                message=f"Retrieved {len(contexts)} knowledge passages",
                data={"sources": [c.source for c in contexts]},
            ).model_dump(),
        }

        with tracer.start("pipeline.run", request_id=request_id):
            constraints = self._parse_constraints(request)
            plan, step = await self.planner.run(request, contexts)
            yield {"type": "step", "data": step.model_dump()}

            # ---- reflection loop ----
            #
            # The loop is driven by *code* verdicts, not by the LLM critic's
            # opinion. Previously a design with one bed for a two-bed brief
            # could score well enough to be accepted. Now a deterministic
            # validation failure forces a rewrite regardless of what the
            # critic said, which is what makes the guarantee hold.
            best: Optional[Dict[str, Any]] = None
            best_critique: Dict[str, Any] = {}
            best_score = -1.0
            last_report = None

            for revision in range(settings.MAX_REFLECTION_ROUNDS + 1):
                # Tracks convergence across rounds so a flat result can end the
                # loop early instead of spending the full budget on no change.
                previous_score = -1.0
                stalled_rounds = 0
                if revision > 0:
                    yield {
                        "type": "step",
                        "data": AgentStep(
                            agent="pipeline",
                            status="info",
                            message=f"Reflecting: revision {revision} based on validation failures",
                        ).model_dump(),
                    }

                design, step = await self.architect.run(
                    request, plan, contexts, revision=revision,
                    critique=best_critique if revision > 0 else None,
                    brief=constraints,
                )
                yield {"type": "step", "data": step.model_dump()}

                # Fix what code can fix before anything judges the design.
                design = self._normalize_design(design, constraints)

                # Price in Python, not in the prompt. The model never sees a
                # number and never gets to state one.
                costing = price_design(design, budget_inr=constraints.budget_inr)

                report = validate_design(design, constraints, costing)
                yield {
                    "type": "step",
                    "data": AgentStep(
                        agent="validator",
                        status="ok" if report.is_valid else "error",
                        message=(
                            f"All {report.checks_run} brief checks passed"
                            if report.is_valid else
                            f"{len(report.errors)} of {report.checks_run} checks failed: "
                            + ", ".join(sorted({v.code for v in report.errors}))
                        ),
                        duration_ms=0.0,
                        data={"violations": [v.to_dict() for v in report.violations]},
                    ).model_dump(),
                }

                critique, step = await self.critic.run(request, design, costing, plan)
                # Fold the code verdict into the critique so the UI shows one
                # coherent evaluation, and so the loop keeps the attempt that
                # actually satisfies the brief rather than the one the model
                # rated prettiest.
                critique["validation"] = report.to_dict()
                yield {"type": "step", "data": step.model_dump()}

                # Rank: a valid design always beats an invalid one. Among
                # valid designs, prefer the higher critic score.
                validity_bonus = 100.0 if report.is_valid else 0.0
                score = validity_bonus + report.pass_rate * 10.0 + float(
                    critique.get("overall", 0)
                )

                if score > best_score:
                    best_score = score
                    best_critique = critique
                    best = design
                    # Track the verdict for the design actually kept, not
                    # whichever attempt happened to run last. Otherwise the
                    # UI can show a good option next to a failing report,
                    # which reads as a bug in the tool rather than a
                    # discarded draft.
                    last_report = report

                needs_revision = not report.is_valid or bool(
                    critique.get("needs_revision")
                )
                if not needs_revision or revision >= settings.MAX_REFLECTION_ROUNDS:
                    break

                # Early stop on a flat score. The loop exists to improve the
                # design; when a revision scores no better than the attempt it
                # replaced, the model is not converging and another round burns
                # 60-90s of local inference to return the same thing. This was
                # the observed behaviour: two full rounds, no change, and the
                # user waited the whole time for a worse-or-equal result.
                if revision >= 1 and score <= previous_score:
                    stalled_rounds += 1
                    if stalled_rounds >= STALL_LIMIT:
                        yield {
                            "type": "step",
                            "data": AgentStep(
                                agent="pipeline",
                                status="info",
                                message=(
                                    "Stopping early: the last revision did not improve "
                                    "on the previous one."
                                ),
                                data={
                                    "stalled": True,
                                    "last_score": round(score, 2),
                                    "previous_score": round(previous_score, 2),
                                },
                            ).model_dump(),
                        }
                        break
                else:
                    stalled_rounds = 0
                previous_score = score

            if best is None:
                raise RuntimeError("pipeline produced no design")

            # Final costing and phase split for the winning design. Normalise
            # first so the totals published here match the option the user is
            # actually shown - a response total that disagrees with the line
            # items is the original bug in a new place.
            best = self._normalize_design(best, constraints)
            final_cost = price_design(best, budget_inr=constraints.budget_inr)

            options = await self._parallel_options(
                request, plan, contexts, constraints, seed_design=best
            )

            # Report the cheapest option's cost, and make the published total
            # reconcile with the options on screen.
            option_total = min(
                (o.estimated_cost_inr for o in options), default=0
            )
            reported_total = option_total or final_cost["estimated_cost_inr"]

            # The phased plan must be built from the option the user is
            # actually shown. Building it from the reflection-loop design
            # produced a plan whose total contradicted the rendered option -
            # the same class of bug as a total that does not match its lines.
            presented = options[0] if options else None
            if presented is not None and constraints.budget_inr:
                presented_cost = price_design(
                    {"furniture": [
                        {"sku": f.sku, "category": f.category,
                         "quantity": f.quantity, "name": f.name}
                        for f in presented.furniture if f.sku
                    ]},
                    budget_inr=constraints.budget_inr,
                )
                budget_plan = (
                    build_phased_plan(presented_cost["line_items"], constraints.budget_inr)
                    if not presented_cost["within_budget"]
                    else None
                )
            else:
                budget_plan = (
                    build_phased_plan(final_cost["line_items"], constraints.budget_inr)
                    if constraints.budget_inr and not final_cost["within_budget"]
                    else None
                )

            response = DesignResponse(
                request_id=request_id,
                options=options,
                context_sources=contexts,
                total_cost_inr=reported_total,
                latency_ms=(time.perf_counter() - start) * 1000,
                # Token/cost totals are accumulated in the metrics registry,
                # which is the accurate source of truth for spend.
                cost_inr=0.0,
                model_used=self._last_model,
                cached=False,
                evaluation=best_critique,
                validation=last_report.to_dict() if last_report else None,
                cost_breakdown=final_cost,
                budget_plan=budget_plan,
                brief_constraints=constraints.to_dict(),
                partial=bool(last_report and not last_report.is_valid),
            )

        metrics.observe("pipeline.latency_ms", response.latency_ms)
        metrics.increment("pipeline.completed")
        yield {"type": "complete", "data": response.model_dump()}

    async def _parallel_options(
        self,
        request: DesignRequest,
        plan: Dict[str, Any],
        contexts: List[RetrievedContext],
        brief: Optional[BriefConstraints] = None,
        seed_design: Optional[Dict[str, Any]] = None,
    ) -> List[DesignOption]:
        """
        Generate the requested number of options concurrently.

        Concurrency is bounded so we do not overwhelm the LLM provider or
        trip a rate limit; each option is independent.
        """
        if request.num_options <= 1:
            if seed_design is not None:
                return [self._to_option(seed_design, request, brief)]
            design, _ = await self.architect.run(
                request, plan, contexts, brief=brief
            )
            if brief:
                design = self._normalize_design(design, brief)
            return [self._to_option(design, request, brief)]

        # The design that already passed validation leads the list. Generating
        # options from scratch here would show the user designs that were never
        # checked - which is how a failing design reaches the screen despite
        # the loop having produced a valid one.
        options: List[DesignOption] = []
        if seed_design is not None:
            options.append(self._to_option(seed_design, request, brief))

        semaphore = asyncio.Semaphore(min(settings.MAX_CONCURRENT_LLM_CALLS, 4))

        async def build(index: int) -> DesignOption:
            async with semaphore:
                varied = request.model_copy()
                varied.client_preferences = (
                    f"{request.client_preferences} "
                    f"(explore a distinctly different approach from the others, "
                    f"variation {index + 1})"
                )
                design, _ = await self.architect.run(
                    varied, plan, contexts, brief=brief
                )
                if brief:
                    design = self._normalize_design(design, brief)
                return self._to_option(design, request, brief)

        results = await asyncio.gather(
            *(build(i) for i in range(max(0, request.num_options - len(options)))),
            return_exceptions=True,
        )
        options.extend(r for r in results if isinstance(r, DesignOption))
        return options[: request.num_options]

    def _to_option(
        self,
        design: Dict[str, Any],
        request: DesignRequest,
        constraints: Optional[BriefConstraints] = None,
    ) -> DesignOption:
        """Coerce an arbitrary LLM dict into a validated DesignOption."""
        constraints = constraints or self._parse_constraints(request)
        # Re-price in code so each option shows catalog prices, not the
        # model's guesses. This is the same call the validator used.
        budget = request.budget_amount_inr or {
            "low": 150_000, "medium": 800_000,
            "high": 3_000_000, "luxury": 10_000_000,
        }.get(request.budget.value, 800_000)

        # Quantities and identifiers are repaired BEFORE pricing, so the
        # figures below are computed from the corrected quantities. An earlier
        # version repaired quantities after pricing, which let a bunk bed
        # reach the client showing quantity 2 at a unit price for one - a real
        # arithmetic error, not a display one.
        design = self._normalize_design(design, constraints)

        costed = price_design(design, budget_inr=budget)
        priced_by_sku = {li["sku"]: li for li in costed["line_items"]}
        for item in design.get("furniture", []) or []:
            if not isinstance(item, dict):
                continue
            line = priced_by_sku.get(item.get("sku"))
            if line:
                item["_priced"] = line
                # The catalog is the authority on the product name and brand.
                # The model frequently leaves `name` empty, echoes the SKU back,
                # or paraphrases the product - and a line item showing "DSK-KID-01"
                # instead of "Kids Study Table" is not shippable.
                name = str(item.get("name", "")).strip()
                name = re.sub(
                    r"\s*\(?\s*sku\s*[:=]?\s*[\w-]+\s*\)?\s*$", "", name, flags=re.I
                )
                if not name or line["sku"].upper() in name.upper():
                    item["name"] = line["name"]
                item["category"] = item.get("category") or line["category"]
                if not item.get("brand_suggestions"):
                    item["brand_suggestions"] = [line["brand"]]

        def safe_list(val: Any) -> List[str]:
            if not isinstance(val, list):
                return []
            out: List[str] = []
            for v in val:
                if isinstance(v, dict):
                    # The model sometimes answers with objects where the prompt
                    # asked for strings. str(dict) yields a Python repr with
                    # single quotes, which reaches the client verbatim, so
                    # flatten it into readable prose instead.
                    label = v.get("name") or v.get("zone") or v.get("purpose")
                    if label:
                        detail = v.get("purpose") or v.get("layout") or v.get("description")
                        out.append(f"{label}: {detail}" if detail and detail != label else str(label))
                        continue
                    out.append(", ".join(f"{k}: {val2}" for k, val2 in v.items()))
                    continue
                if str(v).strip():
                    out.append(str(v))
            return out

        furniture = []
        for item in design.get("furniture", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                qty = int(float(item.get("quantity", 1) or 1))
            except (TypeError, ValueError):
                qty = 1
            # Price comes from the code-computed costing where available, so
            # the option shows the same number the validation used.
            line = (item.get("_priced") or {})
            furniture.append(FurnitureItem(
                name=str(item.get("name", "Item")),
                category=str(item.get("category", "")),
                brand_suggestions=safe_list(item.get("brand_suggestions"))[:4],
                estimated_price_inr=int(line.get("unit_price_inr", 0) or 0),
                quantity=max(1, qty),
                sku=item.get("sku") or line.get("sku"),
                unit_price_inr=int(line.get("unit_price_inr", 0) or 0),
                line_total_inr=int(line.get("line_total_inr", 0) or 0),
                url=item.get("url") or line.get("url"),
                width_cm=int(line.get("dimensions_cm", {}).get("width", 0) or 0),
                depth_cm=int(line.get("dimensions_cm", {}).get("depth", 0) or 0),
                height_cm=int(line.get("dimensions_cm", {}).get("height", 0) or 0),
            ))

        palette = None
        raw_palette = design.get("color_palette")
        if isinstance(raw_palette, dict):
            palette = ColorPalette(
                primary=str(raw_palette.get("primary", "")),
                secondary=str(raw_palette.get("secondary", "")),
                accent=str(raw_palette.get("accent", "")),
                neutrals=safe_list(raw_palette.get("neutrals"))[:6],
                hex_codes=safe_list(raw_palette.get("hex_codes"))[:6],
            )

        try:
            est = int(float(design.get("estimated_cost_inr", 0) or 0))
        except (TypeError, ValueError):
            est = 0
        # The code-computed total always wins over anything the model said.
        if costed["line_items"]:
            est = costed["estimated_cost_inr"]

        return DesignOption(
            title=str(design.get("title", "Design Option"))[:200],
            concept=_text(design.get("concept"), 4000),
            layout_description=_text(design.get("layout_description"), 4000),
            zones=safe_list(design.get("zones"))[:10],
            furniture=furniture,
            color_palette=palette,
            lighting_plan=_text(design.get("lighting_plan"), 2000),
            storage_strategy=_text(design.get("storage_strategy"), 2000),
            decor_suggestions=safe_list(design.get("decor_suggestions"))[:12],
            estimated_cost_inr=max(0, est),
            budget_breakdown=[
                BudgetBreakdown(
                    category=str(b.get("category", "misc"))[:100],
                    amount_inr=int(float(b.get("amount_inr", 0) or 0)),
                    percentage=float(b.get("percentage", 0) or 0),
                    notes=_text(b.get("notes"), 300),
                )
                for b in (costed["budget_breakdown"] or [])
                if isinstance(b, dict)
            ][:10],
            vastu_notes=safe_list(design.get("vastu_notes"))[:8],
            brands=safe_list(design.get("brands"))[:10],
            pros=safe_list(design.get("pros"))[:6],
            cons=safe_list(design.get("cons"))[:6],
            confidence_score=float(design.get("confidence_score", 0) or 0),
            revision_round=int(design.get("revision_round", 0) or 0),
        )


design_pipeline = DesignPipeline()
