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
from app.engine.assumptions import assess_assumptions
from app.engine.diversity import describe_rejection, select_distinct
from app.engine.pricing import build_phased_plan, price_design, total_inr
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


def _brief_to_text(
    request: DesignRequest, brief: Optional[BriefConstraints] = None
) -> str:
    """Render the structured brief as a compact block for prompting.

    Every field is read from the one brief object every agent shares. The
    door and window directions were previously dropped from this rendering
    entirely (the model was told the dimensions but never the bearings), and
    the colour scheme was passed as the raw select value, so "Warm Earthy"
    reached the model as the string "Warm Earthy" with no palette and no
    constraint behind it.
    """
    budget_amount = request.budget_amount_inr or {
        "low": 150_000,
        "medium": 800_000,
        "high": 3_000_000,
        "luxury": 10_000_000,
    }.get(request.budget.value, 800_000)

    brief = brief or parse_brief(
        client_preferences=request.client_preferences,
        furniture=request.furniture,
        specific_requirements=request.specific_requirements,
        occupants=request.occupants,
        room_type=request.room_type.value,
        style=request.style.value,
        color_scheme=request.color_scheme,
        budget_inr=request.budget_amount_inr,
        vastu_compliance=request.vastu_compliance.value,
        room_dimensions=request.dimension_text(),
        door_direction=request.door_direction,
        window_direction=request.window_direction,
        lifestyle_notes=request.lifestyle_notes,
    )

    dimensions = brief.dimensions
    if dimensions.is_known():
        dims = (
            f"{dimensions.width_cm} cm x {dimensions.length_cm} cm "
            f"({dimensions.width_ft:g} ft x {dimensions.length_ft:g} ft), "
            f"area {dimensions.area_sqft:g} sq ft"
        )
    else:
        dims = "not specified - assume a standard Indian apartment room"

    palette = brief.palette or {}
    palette_line = ""
    if palette:
        palette_line = (
            f"PALETTE (fixed by the selected scheme - use exactly these colours): "
            f"{palette.get('label')} - primary {palette.get('primary')}, "
            f"secondary {palette.get('secondary')}, accent {palette.get('accent')}; "
            f"hex {', '.join(palette.get('hex_codes', []))}"
        )

    vastu_line = f"VASTU: {brief.vastu_compliance}"
    if str(brief.vastu_compliance).lower() == "strict":
        vastu_line += " - every strong rule is mandatory and is checked by code"

    kitchen_line = ""
    if brief.room_type == "kitchen" and str(brief.vastu_compliance).lower() == "strict":
        kitchen_line = (
            "KITCHEN VASTU (mandatory): place the cooking platform on the "
            "SOUTH-EAST wall, facing EAST."
        )

    structured_brief = json.dumps(
        request.brief_object(), ensure_ascii=False, sort_keys=True
    )
    return f"""AUTHORITATIVE STRUCTURED FORM BRIEF (JSON):
{structured_brief}

ROOM TYPE: {brief.room_type.replace('_', ' ').title()}
STYLE: {brief.style.replace('_', ' ')}
COLOUR SCHEME: {brief.form_color_scheme or brief.color_scheme or 'not specified'}
{palette_line}
BUDGET TIER: {request.budget.value.upper()} (approximately Rs {budget_amount:,} total, inclusive of furniture and decor)
DIMENSIONS: {dims}
DOOR DIRECTION: {brief.door_direction or 'not supplied'}
WINDOW DIRECTION: {brief.window_direction or 'not supplied'}
CLIENT BRIEF: {request.client_preferences}
FURNITURE REQUESTED: {', '.join(request.furniture) if request.furniture else 'choose appropriate furniture'}
SPECIAL REQUIREMENTS: {', '.join(brief.special_requirements) if brief.special_requirements else 'none'}
OCCUPANTS: {', '.join(brief.occupants) if brief.occupants else 'not specified'}
{vastu_line}
{kitchen_line}
LIFESTYLE NOTES: {brief.lifestyle_notes or 'none'}"""


_VASTU_COMPLIANCE_CLAIM = re.compile(
    r"\b(?:fully\s+|strictly\s+)?vastu[\s-]*(?:compliant|compliance|friendly)\b"
    r"|\b(?:respects?\s+vastu|as\s+per\s+vastu|according\s+to\s+vastu|"
    r"(?:to\s+)?comply\s+with\s+vastu(?:\s+principles?)?|"
    r"complies?\s+with\s+vastu(?:\s+principles?)?|"
    r"align(?:s|ed|ing)?\s+with\s+vastu(?:\s+principles?)?|"
    r"in\s+accordance\s+with\s+vastu(?:\s+principles?)?)\b",
    re.I,
)


def _remove_vastu_compliance_claim(text: str) -> str:
    """Keep compliance claims in generated prose out of the client output."""
    if not _VASTU_COMPLIANCE_CLAIM.search(text):
        return text
    cleaned = _VASTU_COMPLIANCE_CLAIM.sub("", text)
    cleaned = re.sub(r"(?i)\bprinciples\b(?=\s*[,.;]|$)", "", cleaned)
    cleaned = re.sub(r"\s+([,.;])", r"\1", cleaned)
    cleaned = re.sub(r",\s*(?:,|$)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+(that|which)\s*$", "", cleaned, flags=re.I)
    return cleaned.strip(" ,.;")


def _catalog_block(brief: BriefConstraints) -> str:
    """
    Render the subset of the catalog relevant to this brief.

    The whole catalog is ~50 rows, which is small enough to include in full
    and far more reliable than describing it abstractly: the model can copy
    a real SKU and the validator can then check it. Relevance-filtering by
    room keeps the prompt tight on larger catalogs.
    """
    from app.catalog import catalog
    from app.engine.occupants import product_is_allowed

    # Only products that belong in this room. The old filter showed the whole
    # catalog and then kept "12 others" as padding, which is how a kitchen
    # brief got a kids' study table, a bean bag and a sofa to choose from.
    rows = catalog.for_room(brief.room_type) if brief.room_type else catalog.all()

    # Do not *show* the model products the brief forbids. Filtering only on the
    # way out meant an adults-only room still listed a Kids Study Table and a
    # Kids Cot in the prompt, and a 2B model will cheerfully pick from a list it
    # was given. Prevention is the fix; the validator check is the safety net.
    audience = getattr(brief, "audience", "any")
    if audience != "any":
        allowed = [i for i in rows if product_is_allowed(i, audience)]
        if allowed:
            rows = allowed

    # One row per product. A duplicate SKU in the prompt is what produced a
    # shopping list with "Kids Chair - Ergonomic" twice.
    seen: set = set()
    unique = []
    for item in rows:
        if item.sku in seen:
            continue
        seen.add(item.sku)
        unique.append(item)
    rows = unique

    lines = [
        f"{'SKU':<22} {'NAME':<44} {'CAT':<10} {'BRAND':<16} {'Rs':>7} {'W x D x H (cm)':<18} TIER"
    ]
    for item in rows:
        dims = f"{item.width_cm} x {item.depth_cm} x {item.height_cm}"
        lines.append(
            f"{item.sku:<22} {item.name[:44]:<44} {item.category:<10} "
            f"{item.brand[:16]:<16} {item.price_inr:>7,} {dims:<18} {item.tier}"
        )
    return "\n".join(lines)


def _fallback_palette(scheme: Optional[str]) -> Dict[str, Any]:
    """The palette for a scheme, from the one scheme table.

    There used to be two tables - one here for the fallback and one in the
    validator - and they disagreed for "Warm Earthy". There is now one, in
    `app.engine.palette`, used by the fallback, the install step and the check.
    """
    from app.engine.palette import palette_for

    return palette_for(scheme)


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

    async def run(
        self,
        request: DesignRequest,
        contexts: List[RetrievedContext],
        brief: Optional[BriefConstraints] = None,
    ) -> Tuple[Dict[str, Any], AgentStep]:
        start = time.perf_counter()
        context_block = retrieval_service.format_context(contexts)

        prompt = f"""Create a design plan for this room.

{_brief_to_text(request, brief)}

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
  "trade_offs": ["honest limitation of this design", ...],
  "brands": ["brand name", ...],
  "pros": ["...", ...],
  "cons": ["honest trade-offs", ...]
}

HARD REQUIREMENTS - violating any of these makes the design invalid:
- Every furniture entry has a `sku` that appears verbatim in the catalog.
- The number of items matches what the client asked for, including quantities.
- The palette is FIXED by the selected colour scheme. Use exactly the colours
  and hex codes given under PALETTE. Never use a colour the scheme forbids.
- Use only products from the catalog list, and only brands printed in it.
- Do not write Vastu notes. Vastu notes are generated from the rule-check
  results by code; any compliance claim you write is discarded and checked.
- Every special requirement the client ticked must be addressed explicitly in
  the layout, storage or lighting text and by at least one catalog item."""

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
            room_dimensions=request.dimension_text(),
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

{_brief_to_text(request, brief)}

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
                    # De-duplicate before the design sees its own list: the
                    # model sometimes names the same SKU twice, and downstream
                    # that became a shopping list with a repeated row.
                    merged: Dict[str, int] = {}
                    for sel in proposal.furniture:
                        merged[sel.sku] = max(merged.get(sel.sku, 0), sel.quantity)
                    design["furniture"] = [
                        {**item, "quantity": merged[item["sku"]]}
                        for item in selections_to_furniture(proposal)
                    ]
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

            # The palette is a specification of the selected scheme, so it is
            # installed in code. The model's palette is only used to fill gaps
            # for a scheme the palette table does not know.
            from app.engine.palette import palette_for

            if brief.palette:
                design["color_palette"] = {
                    k: v for k, v in brief.palette.items()
                    if k in ("primary", "secondary", "accent", "neutrals",
                             "hex_codes", "label", "scheme")
                }
            elif not isinstance(design.get("color_palette"), dict):
                design["color_palette"] = palette_for(brief.color_scheme)

            # Vastu notes are not the model's to write: they are generated from
            # the rule-check results, so a "respects Vastu" sentence cannot
            # sit next to a failed south-east rule.
            design.pop("vastu_notes", None)
            # Trade-offs are kept separately from Vastu from here on.
            trade_offs = design.get("trade_offs")
            if not isinstance(trade_offs, list):
                trade_offs = list(design.get("cons") or [])
            design["trade_offs"] = [str(t) for t in trade_offs if str(t).strip()][:8]

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
        variation_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Build a room-specific catalog design without relying on a provider.
        """
        from app.catalog import catalog
        from app.engine.brief import canonical_item_for

        brief = brief or parse_brief(
            client_preferences=request.client_preferences,
            furniture=request.furniture,
            specific_requirements=request.specific_requirements,
            occupants=request.occupants,
            room_type=request.room_type.value,
            color_scheme=request.color_scheme,
            budget_inr=request.budget_amount_inr,
            vastu_compliance=request.vastu_compliance.value,
            room_dimensions=request.dimension_text(),
            door_direction=request.door_direction,
            window_direction=request.window_direction,
            lifestyle_notes=request.lifestyle_notes,
        )

        from app.engine.budget_optimizer import optimize_to_budget

        required = dict(brief.required_items)
        optional = {k: 1 for k in (brief.optional_items or ())}
        selected_products = []
        selected_terms = set()
        for selected in brief.selected_furniture:
            phrase = selected.strip().lower()
            term = canonical_item_for(phrase)
            selected_terms.add(term)
            matches = [
                product for product in catalog.for_room(brief.room_type)
                if phrase in product.name.lower()
            ]
            if matches:
                selected_products.append(
                    min(matches, key=lambda product: (product.price_inr, product.name))
                )
        for term in selected_terms:
            required.pop(term, None)
            optional.pop(term, None)

        requirements = {
            str(value).strip().lower().replace("-", "_").replace(" ", "_")
            for value in brief.special_requirements
        }
        if "heavy_storage" in requirements:
            requirements.add("storage_heavy")
        if "storage_heavy" in requirements and not any(
            product.category in {"storage", "wardrobe"} for product in selected_products
        ):
            selected_products.append(catalog.get("STO-CHV-01"))
        if "entertainment" in requirements and not any(
            product.category == "tv_unit" for product in selected_products
        ):
            selected_products.append(catalog.get("TVU-WAL-01"))
        selected_products = [p for p in selected_products if p is not None]

        # A selected chip is a commitment, not an optional optimizer pick.
        # Reserve its catalog cost first, then fit any other explicitly
        # requested items into the remaining budget.
        selected_cost = sum(p.price_inr for p in selected_products)
        optimizer_budget = (
            max(0, brief.budget_inr - selected_cost)
            if brief.budget_inr is not None else None
        )
        if not required:
            if not selected_products:
                from app.engine.brief import ROOM_CATEGORIES

                defaults = (
                    ("sofa", "tv_unit") if brief.room_type == "living_room"
                    else tuple(ROOM_CATEGORIES.get(brief.room_type, ()))[:2]
                )
                required = {category: 1 for category in defaults}

        optimised = optimize_to_budget(
            required=required,
            budget_inr=optimizer_budget,
            optional=optional,
            style_tier=request.budget.value,
            audience=brief.audience,
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
        furniture.extend({
            "name": product.name,
            "category": product.category,
            "sku": product.sku,
            "quantity": 1,
            "brand_suggestions": [product.brand],
        } for product in selected_products)

        # Catalog alternatives keep every selected and required item, then
        # vary the unselected accessory mix enough to be visibly distinct.
        if variation_index > 0:
            from app.engine.brief import BRIEF_TO_CATALOG_CATEGORY
            variation_skus = set()
            if (
                brief.room_type == "living_room"
                and not any("sofa" in value.lower() for value in brief.selected_furniture)
                and any(
                    item.get("category") == "seating"
                    and any(term in item.get("name", "").lower()
                            for term in ("sofa", "sectional"))
                    for item in furniture
                )
            ):
                sofa_sku = "SOF-3S-01" if variation_index == 1 else "SOF-SEC-01"
                sofa = catalog.get(sofa_sku)
                if sofa is not None:
                    furniture = [
                        item for item in furniture
                        if not (
                            item.get("category") == "seating"
                            and any(term in item.get("name", "").lower()
                                    for term in ("sofa", "sectional"))
                        )
                    ]
                    furniture.append({
                        "name": sofa.name, "category": sofa.category,
                        "sku": sofa.sku, "quantity": 1,
                        "brand_suggestions": [sofa.brand],
                    })

            if (
                brief.room_type == "living_room"
                and variation_index >= 2
                and not any("coffee table" in value.lower()
                            for value in brief.selected_furniture)
            ):
                current_table_skus = {
                    item["sku"] for item in furniture if item.get("category") == "table"
                }
                coffee_table = catalog.get("TBL-COF-02")
                if coffee_table is not None and coffee_table.sku not in current_table_skus:
                    furniture = [
                        item for item in furniture
                        if not (item.get("category") == "table"
                                and "coffee table" in item.get("name", "").lower())
                    ]
                    furniture.append({
                        "name": coffee_table.name, "category": coffee_table.category,
                        "sku": coffee_table.sku, "quantity": 1,
                        "brand_suggestions": [coffee_table.brand],
                    })

            if (
                brief.room_type == "living_room"
                and not any(
                    canonical_item_for(value) in {"bookshelf", "storage"}
                    or "shoe rack" in value.lower()
                    for value in brief.selected_furniture
                )
            ):
                storage = catalog.get("STO-CHV-01")
                if storage is not None:
                    furniture = [
                        item for item in furniture
                        if item.get("category") not in {"storage", "wardrobe"}
                    ]
                    furniture.append({
                        "name": storage.name, "category": storage.category,
                        "sku": storage.sku, "quantity": 1,
                        "brand_suggestions": [storage.brand],
                    })
                    variation_skus.add(storage.sku)

            mandatory_categories = {
                catalog_category
                for category in required
                for catalog_category in BRIEF_TO_CATALOG_CATEGORY.get(
                    category, (category,)
                )
            }
            selected_skus = {product.sku for product in selected_products}
            selected_categories = {product.category for product in selected_products}
            accessory_skus = ("LMP-FLO-01", "DEC-PLT-01", "LMP-CEL-01")
            accessory_categories = {
                product.category for sku in accessory_skus
                if (product := catalog.get(sku)) is not None
                and catalog.allowed_in_room(product, brief.room_type)
            }
            mandatory_categories -= selected_categories | accessory_categories
            furniture = [
                item for item in furniture
                if item["sku"] in selected_skus
                or item["sku"] in variation_skus
                or item["category"] in mandatory_categories
            ]
            current_skus = {item["sku"] for item in furniture}
            current_total = sum(
                catalog.get(item["sku"]).price_inr * int(item.get("quantity", 1))
                for item in furniture if catalog.get(item["sku"]) is not None
            )
            for sku in accessory_skus[: min(variation_index + 1, len(accessory_skus))]:
                accessory = catalog.get(sku)
                if (
                    accessory is None
                    or accessory.sku in current_skus
                    or not catalog.allowed_in_room(accessory, brief.room_type)
                    or (
                        brief.budget_inr is not None
                        and current_total + accessory.price_inr > brief.budget_inr
                    )
                ):
                    continue
                furniture.append({
                    "name": accessory.name,
                    "category": accessory.category,
                    "sku": accessory.sku,
                    "quantity": 1,
                    "brand_suggestions": [accessory.brand],
                })
                current_skus.add(accessory.sku)
                current_total += accessory.price_inr

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

        if not furniture and brief.budget_inr is not None:
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
        names = [item["name"] for item in furniture]
        listed = ", ".join(names[:5]) or "the selected items"
        storage_names = [
            item["name"] for item in furniture
            if item["category"] in {"storage", "wardrobe"}
        ]
        lighting_names = [
            item["name"] for item in furniture if item["category"] == "lighting"
        ]
        decor_names = [
            item["name"] for item in furniture
            if item["category"] in {"decor", "soft"}
        ]
        palette = _fallback_palette(brief.color_scheme)
        palette_variants = {
            "warm": [
                ["#B85F3B", "#718146", "#963D2C", "#D9CAB8"],
                ["#C66A42", "#817F3E", "#A54E2F", "#E1CFB6"],
            ],
            "earthy": [
                ["#B85F3B", "#718146", "#963D2C", "#D9CAB8"],
                ["#C66A42", "#817F3E", "#A54E2F", "#E1CFB6"],
            ],
            "neutral": [
                ["#EAE5DB", "#C8BDAE", "#95A27F", "#F6F2EB"],
                ["#F0E9DE", "#BFB5A8", "#8D9B7C", "#FAF7F0"],
            ],
            "cool": [
                ["#158C90", "#29486A", "#5AA6D8", "#E4EAF0"],
                ["#238E9A", "#263D62", "#68A9CF", "#EFF2F5"],
            ],
            "pastel": [
                ["#F2B8C4", "#AFCBE6", "#B4DFCE", "#FFF8F6"],
                ["#F4C2CC", "#B3D0E8", "#B9E1D4", "#FBF7F4"],
            ],
        }
        if variation_index and palette.get("scheme") in palette_variants:
            palette["hex_codes"] = palette_variants[palette["scheme"]][
                min(variation_index - 1, 1)
            ]

        requirement_notes: List[str] = []
        if "kids_safe" in requirements:
            requirement_notes.append(
                "Kids Safe: specify rounded edges, anchor tall furniture to the wall, "
                "and avoid glass surfaces."
            )
        if "storage_heavy" in requirements:
            requirement_notes.append(
                "Heavy Storage: use the listed storage furniture and anchor tall units."
            )
        if "low_maintenance" in requirements:
            requirement_notes.append(
                "Low Maintenance: choose wipe-clean, durable surfaces for daily use."
            )
        if "natural_light" in requirements and brief.window_direction:
            requirement_notes.append(
                f"Natural Light: keep the {brief.window_direction}-facing window clear "
                "to use available daylight."
            )
        if "entertainment" in requirements:
            requirement_notes.append(
                "Entertainment: centre the viewing area on the listed TV unit."
            )

        if brief.room_type == "living_room":
            zones = [
                "Living area: " + ", ".join(
                    item["name"] for item in furniture
                    if item["category"] in {"seating", "tv_unit", "table"}
                ) or "selected furniture"
            ]
        else:
            zones = plan.get("zones", []) or [f"{room.lower()} furniture zone"]

        fallback = {
            "title": f"{request.style.value.title()} {room}"
            + (f" - Accessory {variation_index + 1}" if variation_index else ""),
            "concept": f"A {request.style.value} {room.lower()} arranged around {listed}.",
            "layout_description": (
                f"Place {listed} to preserve a clear circulation route through the "
                f"{room.lower()}. " + " ".join(requirement_notes)
            ),
            "zones": zones,
            "furniture": furniture,
            "color_palette": palette,
            "lighting_plan": (
                f"Use the listed {', '.join(lighting_names)} for room lighting."
                if lighting_names else
                "Retain the room's existing lighting; no new lighting product is listed."
            ),
            "storage_strategy": (
                f"Storage is provided by {', '.join(storage_names)}."
                if storage_names else
                "No additional storage product is included in this option."
            ),
            "decor_suggestions": [
                f"Use the listed {name} as the room's accessory." for name in decor_names
            ],
            "brands": sorted({f.get("brand", "") for f in furniture} - {""}),
            "pros": [
                "Every listed product has a catalog SKU and price",
                f"The selected room brief is applied to the {room.lower()} layout",
            ],
            "cons": [budget_note] if _budget_infeasible else [],
            "trade_offs": [budget_note] if _budget_infeasible else [],
            # Carried through so the validator can distinguish "the model forgot
            # an item" from "the budget could not buy this item".
            "_budget_infeasible": _budget_infeasible,
        }
        return fallback


class BudgetAgent(BaseAgent):
    """
    Costing is no longer delegated to the model.

    This agent used to ask the LLM to price a design and then "reconcile" the
    answer. A model that emits zeros sums to zero, and the client saw a Rs 0
    shopping list next to a Rs 24,900 headline. `price_design` now computes
    every figure from the catalog, and the pipeline calls it directly. The
    class is kept as a thin, model-free wrapper so existing call sites and
    tests continue to work and so the step still appears in the pipeline UI.
    """

    name = "budget"
    system_prompt = DESIGNER_SYSTEM_PROMPT

    def cost(self, design: Dict[str, Any], budget_inr: Optional[int]) -> Dict[str, Any]:
        """Price a design in code. The only costing path in the system."""
        return price_design(design, budget_inr=budget_inr)

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

        with tracer.start("agent.budget"):
            costing = self.cost(design, budget_amount)

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

    @staticmethod
    def _override_code_scores(
        critique: Dict[str, Any],
        costing: Dict[str, Any],
        validation: Optional[Dict[str, Any]],
        shopping_list: List[str],
        vastu_notes: List[Any],
    ) -> Dict[str, Any]:
        """
        Replace the model's own budget and brief-compliance scores with values
        computed in Python.

        The model was asked to rate "budget adherence" while looking at a JSON
        costing blob, and it did so unreliably - a design Rs 21,100 over budget
        could score 8/10. The arithmetic is not a matter of taste, so it is not
        delegated: `within_budget` and the validator's pass rate are facts, and
        they are now what the score reflects.

        Also drops objections the critic raised about things that plainly exist,
        since "no brands" or "no Vastu explanation" is a hallucinated defect and
        sending the reflection loop to fix it wastes a round.
        """
        scores = critique.setdefault("scores", {})

        # --- Budget: straight from the summed line items --------------------
        total = float(costing.get("estimated_cost_inr", 0) or 0)
        budget = costing.get("budget_inr")
        if not costing.get("costing_complete", True):
            scores["budget_adherence"] = 0.0
        elif budget and total > float(budget):
            over_ratio = (total - float(budget)) / max(float(budget), 1)
            scores["budget_adherence"] = round(max(0.0, 5.0 - over_ratio * 5), 2)
        elif budget:
            headroom = 1.0 - (total / float(budget))
            scores["budget_adherence"] = round(min(10.0, 7.0 + headroom * 3), 2)
        else:
            scores["budget_adherence"] = 7.0

        # --- Brief compliance: the validator's pass rate --------------------
        if validation:
            checks_run = float(validation.get("checks_run", 0) or 0)
            checks_passed = float(validation.get("checks_passed", 0) or 0)
            pass_rate = (checks_passed / checks_run) if checks_run else 0.0
            scores["brief_compliance"] = round(pass_rate * 10.0, 2)
            if not validation.get("valid", True):
                scores["feasibility"] = round(
                    min(float(scores.get("feasibility", 0) or 0), pass_rate * 10.0),
                    2,
                )

        # --- Strip objections that are factually wrong ----------------------
        issues = [
            str(i) for i in (critique.get("issues") or []) if str(i).strip()
        ]
        kept: List[str] = []
        for issue in issues:
            low = issue.lower()
            if shopping_list and "no brand" in low and "brand" not in low.replace("no brand", ""):
                continue
            if shopping_list and ("without brand" in low or "no brands specified" in low):
                continue
            if shopping_list and any(
                phrase in low for phrase in (
                    "incomplete shopping list", "shopping list is incomplete",
                    "missing items from the shopping list",
                )
            ):
                # "Issues" reads as "issues addressed" in the UI, so listing
                # an unresolved shopping list there was backwards. Those
                # defects are reported on the validation line instead.
                continue
            if vastu_notes and "no vastu" in low:
                continue
            kept.append(issue)
        critique["issues"] = kept

        # Overall is recomputed so it cannot contradict the parts.
        scored = [
            float(v) for v in scores.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if scored:
            critique["overall"] = round(sum(scored) / len(scored), 2)
        return critique

    async def run(
        self,
        request: DesignRequest,
        design: Dict[str, Any],
        costing: Dict[str, Any],
        plan: Dict[str, Any],
        validation: Optional[Dict[str, Any]] = None,
        brief: Optional[BriefConstraints] = None,
    ) -> Tuple[Dict[str, Any], AgentStep]:
        start = time.perf_counter()

        # The critic is shown the *shopping list* and the Vastu notes as
        # explicit context. Without them it graded designs it had not fully
        # read and invented objections: it claimed "no brands specified" for a
        # design with eight catalogued brands in front of it, and "no Vastu
        # explanation" for one whose notes were in the payload. Naming the
        # fields it must not claim are missing is cheaper than hoping it reads
        # the whole JSON.
        from app.catalog import catalog as _cat

        # The shopping list is rendered from the code-computed costing, so the
        # critic sees the price the client will see.
        shopping_list = []
        _priced = {
            li["sku"]: li for li in (costing.get("line_items") or [])
            if isinstance(li, dict) and li.get("sku")
        }
        for line in _priced.values():
            shopping_list.append(
                f"- {line.get('name')} x{line.get('quantity', 1)} "
                f"({line.get('brand')}, SKU {line.get('sku')}, "
                f"Rs {int(line.get('line_total_inr') or 0):,})"
            )

        # Brands are read from the catalog, not from the design: the model's
        # suggestions may name a retailer that does not sell the product.
        brands = sorted({
            str(line.get("brand")) for line in _priced.values() if line.get("brand")
        })
        vastu_notes = design.get("vastu_notes") or []

        prompt = f"""Critique this design.

{_brief_to_text(request, brief)}

DESIGN PLAN:
{json.dumps(plan, indent=2)[:1200]}

SHOPPING LIST (every item below is a real catalog product with a real price):
{chr(10).join(shopping_list) or '- (none)'}

BRANDS ALREADY SPECIFIED ON THIS DESIGN ({len(brands)}):
{', '.join(brands) or '- (none)'}

VASTU NOTES ALREADY ON THIS DESIGN ({len(vastu_notes)}):
{chr(10).join(str(n) for n in vastu_notes) or '- (none)'}

PROPOSED DESIGN:
{json.dumps(design, indent=2)[:3000]}

COSTING:
{json.dumps(costing, indent=2)[:1200]}

IMPORTANT - do not report a defect that does not exist:
- If the shopping list above is populated, the design DOES specify brands.
  Never claim "no brands specified".
- If the Vastu notes above are non-empty, the design DOES explain Vastu.
  Never claim "no Vastu explanation".
- The costing figures are computed in Python from the catalog and are
  authoritative. Do not dispute the arithmetic.

Score it honestly on coherence, feasibility, aesthetics and specificity.
Budget adherence and brief compliance are scored in code from the costing and
the validator, so do not guess at them - score the rest.

Set needs_revision to true if overall is below 7."""

        with tracer.start("agent.critic"):
            try:
                result = await self._call_llm(prompt, temperature=0.2)
                critique = result["json"] if isinstance(result["json"], dict) else {}
            except Exception as exc:
                logger.warning("critic_failed error=%s", exc)
                critique = {}

        critique = self._sanitize(critique)
        critique = self._override_code_scores(
            critique, costing, validation, shopping_list, vastu_notes
        )
        # This field is the critic's *remaining* problems, not a list of what
        # was fixed. Naming it otherwise is what made "Issues addressed" list
        # an unresolved shopping list.
        critique["remaining_issues"] = list(critique.get("issues") or [])
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


def catalog_category(item: Dict[str, Any]) -> str:
    """The catalog category of a line item, from its SKU when possible."""
    from app.catalog import catalog as _catalog

    product = _catalog.get(str(item.get("sku", "")).strip())
    return (product.category if product else str(item.get("category", ""))).lower()


def is_cooking_product(item: Dict[str, Any]) -> bool:
    """True for the product the cooking platform is mounted on.

    A hob/hob-unit is the obvious one; a base unit is the counter run the hob
    sits on. Anything else in the kitchen (a sink, a chimney) is not.
    """
    from app.catalog import catalog as _catalog

    product = _catalog.get(str(item.get("sku", "")).strip())
    name = (product.name if product else str(item.get("name", ""))).lower()
    if any(word in name for word in ("sink", "chimney", "exhaust", "trolley",
                                     "shelf", "cabinet", "pantry")):
        return False
    return any(word in name for word in ("base unit", "hob", "stove", "counter"))


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
        self._last_model = (
            settings.DEFAULT_MODEL if registry.all() else "Rule-based mode"
        )
        # Near-duplicate options rejected by the last request, surfaced to the
        # client so a short list is explained rather than looking like a bug.
        self._rejected_options: List[str] = []

    @staticmethod
    def _audience_violates(item: Dict[str, Any], audience: str) -> bool:
        """Does this line item belong to a different audience than the brief?"""
        from app.catalog import catalog
        from app.engine.occupants import product_is_allowed

        if audience == "any":
            return False
        product = catalog.get(str(item.get("sku", "")).strip())
        return product is not None and not product_is_allowed(product, audience)

    @staticmethod
    def _repair_audience(item: Dict[str, Any], audience: str) -> None:
        """
        Swap a forbidden product for the closest allowed one in its category.

        Leaves the item untouched when no acceptable substitute exists, so the
        validator can still report it rather than the item being silently
        dropped.
        """
        from app.catalog import catalog
        from app.engine.brief import find_products
        from app.engine.occupants import product_is_allowed

        category = str(item.get("category", "")).strip()
        candidates = [
            p for p in find_products(category)
            if product_is_allowed(p, audience)
        ]
        if not candidates:
            return
        replacement = min(candidates, key=lambda p: (p.price_inr, p.name))
        logger.info(
            "audience_replaced sku=%s -> %s audience=%s",
            item.get("sku"), replacement.sku, audience,
        )
        item["sku"] = replacement.sku
        item["name"] = replacement.name
        item["category"] = replacement.category
        item.setdefault("brand_suggestions", [replacement.brand])
        item["width_cm"] = replacement.width_cm
        item["depth_cm"] = replacement.depth_cm
        item["height_cm"] = replacement.height_cm

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
        # 0. Drop products that belong to another kind of room, and merge a
        # SKU the model listed twice. Both used to survive all the way to the
        # client: a kitchen shopping list carried a kids' study table, a bean
        # bag and a sofa, and a chair appeared twice. The prompt no longer
        # shows these products, but a model can still emit one, and the
        # constraint is a rule, not a hint.
        from app.catalog import catalog

        room = (constraints.room_type or "").lower()
        filtered = []
        for item in design.get("furniture", []) or []:
            if not isinstance(item, dict):
                continue
            product = catalog.get(str(item.get("sku", "")).strip())
            if product and not catalog.allowed_in_room(product, room):
                filtered.append(f"{product.name} ({product.sku})")
        if filtered:
            logger.info("wrong_room_items_removed items=%s", sorted(set(filtered)))
        design["furniture"] = self._clean_furniture(design, constraints)

        # Repair invented SKUs before costing, so the repaired lines are picked
        # up by this pricing pass rather than needing a second one. Only an
        # item that *claims* a SKU and gets it wrong is repaired here; an item
        # with no SKU at all is left for the validator to report, because
        # guessing a product for it would invent a line the model never asked
        # for.
        for item in design.get("furniture", []) or []:
            if not isinstance(item, dict):
                continue
            # Audience is a hard constraint, not a preference. A kids' product
            # in an adults-only room is swapped for the adult equivalent here,
            # rather than being left in place for the validator to complain about
            # and the reflection loop to fail to fix.
            if self._audience_violates(item, constraints.audience):
                self._repair_audience(item, constraints.audience)
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
            item["name"] = line["name"]
            item["category"] = line["category"]
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
        self._ensure_catalog_accessories(design, constraints)
        if constraints.room_type == "living_room":
            all_furniture = [
                item for item in design.get("furniture", [])
                if isinstance(item, dict)
            ]
            primary = [
                item["name"] for category in ("seating", "tv_unit", "table")
                for item in all_furniture if item.get("category") == category
            ]
            if primary:
                design["concept"] = (
                    "A living room centred on " + ", ".join(primary[:3]) + "."
                )
        design["_normalized"] = True

        return design

    @staticmethod
    def _clean_furniture(
        design: Dict[str, Any], constraints: BriefConstraints
    ) -> List[Dict[str, Any]]:
        """Remove out-of-room products and collapse duplicate SKUs.

        Kept public so tests and other callers normalise furniture exactly the
        way the pipeline does.
        """
        from app.catalog import catalog

        raw = design.get("furniture")
        if not isinstance(raw, list):
            return []
        room = (constraints.room_type or "").lower()
        merged: Dict[str, Dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            sku = str(item.get("sku", "")).strip().upper()
            product = catalog.get(sku) if sku else None
            if product is not None:
                if not catalog.allowed_in_room(product, room):
                    logger.info(
                        "wrong_room_item_dropped sku=%s room=%s", sku, room
                    )
                    metrics.increment("design.dropped_wrong_room_item")
                    continue
                # One line per product. The quantity is the largest asked for,
                # not the sum - the duplicate was the same request emitted
                # twice, not two separate requests.
                try:
                    qty = int(float(item.get("quantity", 1) or 1))
                except (TypeError, ValueError):
                    qty = 1
                existing = merged.get(sku)
                if existing is not None:
                    metrics.increment("design.duplicate_item_merged")
                    existing["quantity"] = max(
                        int(existing.get("quantity", 1) or 1), max(1, qty)
                    )
                    continue
                item = {**item, "quantity": max(1, qty)}
                merged[sku] = item
            else:
                # Uncatalogued entries are left in place: the validator reports
                # them, and guessing a product here would invent a line.
                key = f"__raw__{len(merged)}"
                merged[key] = item
        return list(merged.values())

    @staticmethod
    def _insert_missing_required(
        design: Dict[str, Any], constraints: BriefConstraints
    ) -> None:
        """Add any brief-required item the design does not already contain."""
        from app.catalog import catalog

        furniture = design.setdefault("furniture", [])
        if not isinstance(furniture, list):
            return

        from app.engine.brief import (
            canonical_item_for, products_satisfying, seats_per_product,
        )
        from app.engine.validator import _SYNONYM_CATEGORIES, _count_items

        existing_skus = {
            str(item.get("sku", "")).strip().upper()
            for item in furniture if isinstance(item, dict)
        }
        for selected in constraints.selected_furniture:
            selected_name = str(selected).strip().lower()
            if not selected_name or any(
                selected_name in str(item.get("name", "")).lower()
                for item in furniture if isinstance(item, dict)
            ):
                continue
            candidates = [
                product for product in catalog.for_room(constraints.room_type)
                if selected_name in product.name.lower()
            ]
            if not candidates:
                term = canonical_item_for(selected_name)
                candidates = [
                    product for product in products_satisfying(
                        term, audience=constraints.audience
                    )
                    if catalog.allowed_in_room(product, constraints.room_type)
                ]
            if not candidates:
                logger.warning("selected_catalog_item_unavailable item=%s", selected)
                continue
            product = min(candidates, key=lambda item: (item.price_inr, item.name))
            if product.sku not in existing_skus:
                furniture.append({
                    "name": product.name,
                    "category": product.category,
                    "sku": product.sku,
                    "quantity": 1,
                    "brand_suggestions": [product.brand],
                })
                existing_skus.add(product.sku)

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
            options = [
                product for product in products_satisfying(
                    category, audience=constraints.audience
                )
                if catalog.allowed_in_room(product, constraints.room_type)
            ]
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

        if constraints.room_type == "living_room" and constraints.seating_capacity:
            def seated_count() -> int:
                total = 0
                for entry in furniture:
                    if not isinstance(entry, dict):
                        continue
                    product = catalog.get(str(entry.get("sku", "")))
                    if product is None or product.category != "seating":
                        continue
                    try:
                        quantity = max(1, int(float(entry.get("quantity", 1))))
                    except (TypeError, ValueError):
                        quantity = 1
                    total += seats_per_product(product.name) * quantity
                return total

            while seated_count() < constraints.seating_capacity:
                deficit = constraints.seating_capacity - seated_count()
                present = {
                    str(entry.get("sku", "")).upper()
                    for entry in furniture if isinstance(entry, dict)
                }
                candidates = [
                    item for item in catalog.for_room("living_room")
                    if item.category == "seating"
                    and item.sku not in present
                    and any(word in item.name.lower() for word in (
                        "sofa", "sectional", "armchair",
                    ))
                ]
                if not candidates:
                    break
                product = min(
                    candidates,
                    key=lambda item: (
                        item.price_inr / seats_per_product(item.name),
                        abs(seats_per_product(item.name) - deficit),
                    ),
                )
                furniture.append({
                    "name": product.name,
                    "category": product.category,
                    "sku": product.sku,
                    "quantity": 1,
                    "brand_suggestions": [product.brand],
                    "_inserted_by_validator": True,
                })

    @staticmethod
    def _ensure_catalog_accessories(
        design: Dict[str, Any], constraints: BriefConstraints
    ) -> None:
        """Add functional lighting and a light north-east accent where needed."""
        from app.catalog import catalog

        furniture = design.setdefault("furniture", [])
        if not isinstance(furniture, list):
            return
        existing = {
            str(item.get("sku", "")).upper()
            for item in furniture if isinstance(item, dict)
        }
        skus = ["KIT-LGT-01"] if constraints.room_type == "kitchen" else ["LMP-CEL-01"]
        prefer_closed = any(
            phrase in constraints.raw_text.lower()
            for phrase in ("prefer closed storage", "closed storage preferred")
        )
        if prefer_closed:
            selected = {name.lower() for name in constraints.selected_furniture}
            furniture[:] = [
                item for item in furniture
                if not (
                    isinstance(item, dict)
                    and any(word in str(item.get("name", "")).lower()
                            for word in ("open shelf", "floating wall shelf"))
                    and not any(
                        name in str(item.get("name", "")).lower()
                        or str(item.get("name", "")).lower() in name
                        for name in selected
                    )
                )
            ]
            existing = {
                str(item.get("sku", "")).upper()
                for item in furniture if isinstance(item, dict)
            }
            if not any(
                isinstance(item, dict)
                and "closed" in str(item.get("name", "")).lower()
                for item in furniture
            ):
                skus.insert(
                    0, "KIT-OVC-01"
                    if constraints.room_type == "kitchen" else "STO-CAB-01"
                )
            design["storage_strategy"] = (
                "Prefer closed storage using the listed enclosed cabinet."
            )
        if constraints.room_type == "living_room" and not any(
            any(word in str(item.get("name", "")).lower() for word in ("lamp", "plant"))
            for item in furniture if isinstance(item, dict)
        ):
            skus.append("DEC-PLT-01")

        for sku in skus:
            product = catalog.get(sku)
            if (
                product is None or sku.upper() in existing
                or not catalog.allowed_in_room(product, constraints.room_type)
            ):
                continue
            furniture.append({
                "name": product.name,
                "category": product.category,
                "sku": product.sku,
                "quantity": 1,
                "brand_suggestions": [product.brand],
                "_inserted_by_validator": True,
            })
            existing.add(sku)

        if constraints.room_type == "living_room":
            requirements = {
                str(value).strip().lower().replace("-", "_").replace(" ", "_")
                for value in constraints.special_requirements
            }
            if "entertainment" in requirements and not any(
                str(item.get("category", "")).lower() == "tv_unit"
                for item in furniture if isinstance(item, dict)
            ):
                product = catalog.get("TVU-WAL-01")
                if product is not None:
                    furniture.append({
                        "name": product.name, "category": product.category,
                        "sku": product.sku, "quantity": 1,
                        "brand_suggestions": [product.brand],
                        "_inserted_by_validator": True,
                    })
                    existing.add(product.sku.upper())
            if ("heavy_storage" in requirements or "storage_heavy" in requirements) and not any(
                isinstance(item, dict)
                and item.get("category") in {"storage", "wardrobe"}
                for item in furniture
            ):
                product = catalog.get("STO-CAB-01")
                if product is not None:
                    furniture.append({
                        "name": product.name, "category": product.category,
                        "sku": product.sku, "quantity": 1,
                        "brand_suggestions": [product.brand],
                        "_inserted_by_validator": True,
                    })
                    existing.add(product.sku.upper())
            sentences = []
            if "kids_safe" in requirements:
                sentences.append(
                    "Kids Safe: specify rounded-edge protection, anchor tall units "
                    "with anti-tip hardware, and avoid glass."
                )
            if "low_maintenance" in requirements:
                sentences.append(
                    "Low Maintenance: use wipe-clean, durable finishes for daily use."
                )
            if "heavy_storage" in requirements or "storage_heavy" in requirements:
                sentences.append(
                    "Heavy Storage: use the listed closed storage and anchor tall units."
                )
            if "natural_light" in requirements:
                direction = constraints.window_direction
                if direction:
                    sentences.append(
                        f"Natural Light: keep the {direction}-facing window clear."
                    )
                else:
                    sentences.append("Natural Light: keep the window and its daylight path clear.")
            if "entertainment" in requirements:
                sentences.append("Entertainment: orient the viewing area to the listed TV unit.")
            if sentences:
                current = str(design.get("layout_description") or "").strip()
                design["layout_description"] = " ".join([current, *sentences]).strip()

        light_names = [
            str(item.get("name")) for item in furniture
            if isinstance(item, dict) and item.get("category") == "lighting"
        ]
        if light_names:
            design["lighting_plan"] = (
                "Lighting in this option: " + ", ".join(light_names) + "."
            )

        budget = constraints.budget_inr if constraints.budget_is_exact else None
        if budget:
            current_total = sum(
                (catalog.get(str(item.get("sku", ""))).price_inr
                 * max(1, int(item.get("quantity", 1) or 1)))
                for item in furniture if isinstance(item, dict)
                and catalog.get(str(item.get("sku", ""))) is not None
            )
            target = int(budget * 0.70)
            ceiling = int(budget * 0.95)
            sofa_selected = any(
                "sofa" in name.lower() or "sectional" in name.lower()
                for name in constraints.selected_furniture
            )
            if (
                constraints.room_type == "living_room"
                and not sofa_selected
                and "Accessory" not in str(design.get("title", ""))
            ):
                sofa_entry = next(
                    (
                        item for item in furniture
                        if isinstance(item, dict)
                        and item.get("category") == "seating"
                        and any(term in str(item.get("name", "")).lower()
                                for term in ("sofa", "sectional"))
                    ),
                    None,
                )
                if sofa_entry is not None:
                    current_product = catalog.get(str(sofa_entry.get("sku", "")))
                    alternatives = [
                        item for item in catalog.for_room("living_room")
                        if item.category == "seating"
                        and any(term in item.name.lower() for term in ("sofa", "sectional"))
                        and item.price_inr > (current_product.price_inr if current_product else 0)
                        and current_total
                            - (current_product.price_inr if current_product else 0)
                            + item.price_inr <= ceiling
                    ]
                    if alternatives and current_product is not None:
                        upgraded = max(alternatives, key=lambda item: item.price_inr)
                        sofa_entry.update({
                            "name": upgraded.name, "category": upgraded.category,
                            "sku": upgraded.sku,
                            "brand_suggestions": [upgraded.brand],
                        })
                        existing.discard(current_product.sku.upper())
                        existing.add(upgraded.sku.upper())
                        current_total += upgraded.price_inr - current_product.price_inr
            priority = {
                "table": 0, "storage": 1, "wardrobe": 1,
                "soft": 2, "decor": 3, "lighting": 4,
            }
            candidates = sorted(
                catalog.for_room(constraints.room_type),
                key=lambda item: (
                    priority.get(item.category, 9), -item.price_inr, item.name
                ),
            )
            if constraints.room_type == "kitchen":
                candidates = [
                    item for item in candidates
                    if item.category in {"kitchen", "lighting"}
                ]
            if constraints.room_type == "living_room":
                candidates = [
                    item for item in candidates
                    if item.category in {"table", "storage", "soft", "decor"}
                    if (
                        item.category != "table"
                        or any(term in item.name.lower() for term in (
                            "coffee table", "side table",
                        ))
                    )
                    and (
                        item.category != "storage"
                        or any(term in item.name.lower() for term in (
                            "bookshelf", "shoe rack", "shoe cabinet",
                            "closed storage", "cube storage",
                        ))
                    )
                    and (
                        item.category not in {"decor", "soft"}
                        or any(term in item.name.lower() for term in (
                            "planter", "plant", "rug", "curtain", "mirror", "wall art",
                        ))
                    )
                ]
            for product in candidates:
                if current_total >= target:
                    break
                if product.sku.upper() in existing:
                    continue
                if current_total + product.price_inr > ceiling:
                    continue
                furniture.append({
                    "name": product.name, "category": product.category,
                    "sku": product.sku, "quantity": 1,
                    "brand_suggestions": [product.brand],
                    "_inserted_by_validator": True,
                })
                existing.add(product.sku.upper())
                current_total += product.price_inr
            if (
                constraints.room_type == "living_room"
                and current_total < target
                and not sofa_selected
                and any(
                    isinstance(item, dict)
                    and item.get("category") == "seating"
                    for item in furniture
                )
            ):
                chair = catalog.get("ARM-01")
                if (
                    chair is not None
                    and chair.sku.upper() not in existing
                    and current_total + chair.price_inr <= ceiling
                ):
                    furniture.append({
                        "name": chair.name, "category": chair.category,
                        "sku": chair.sku, "quantity": 1,
                        "brand_suggestions": [chair.brand],
                        "_inserted_by_validator": True,
                    })
                    existing.add(chair.sku.upper())
                    current_total += chair.price_inr

        storage_names = [
            str(item.get("name")) for item in furniture
            if isinstance(item, dict)
            and item.get("category") in {"storage", "wardrobe"}
        ]
        if storage_names and not prefer_closed:
            design["storage_strategy"] = (
                "Storage provided by " + ", ".join(storage_names) + "."
            )
        elif prefer_closed:
            closed_names = [
                str(item.get("name")) for item in furniture
                if isinstance(item, dict)
                and (
                    "closed" in str(item.get("name", "")).lower()
                    or "cabinet" in str(item.get("name", "")).lower()
                    or (
                        constraints.room_type == "kitchen"
                        and "overhead" in str(item.get("name", "")).lower()
                    )
                )
            ]
            design["storage_strategy"] = (
                "Closed storage provided by " + ", ".join(closed_names) + "."
                if closed_names else
                f"The catalog has no closed-storage product suited to "
                f"this {constraints.room_type.replace('_', ' ')}."
            )

    @staticmethod
    def _attach_vastu(design: Dict[str, Any], brief: BriefConstraints) -> None:
        """Run the rule check and replace the model's Vastu notes with it.

        The notes the client reads are the rule results, so the section cannot
        contain a trade-off sentence, and "respects Vastu" cannot appear next
        to a failed rule.

        Under Strict, the cooking platform is forced to the south-east, facing
        east. That is a specification, not a creative choice: the model choosing
        the south wall gets its counter moved by code rather than the design
        being blocked on a rule the model cannot reliably satisfy.
        """
        from app.engine.vastu import analyse_design

        if brief.room_type == "kitchen":
            platform = next(
                (
                    item for item in design.get("furniture", [])
                    if isinstance(item, dict)
                    and catalog_category(item) == "kitchen"
                    and is_cooking_product(item)
                ),
                None,
            )
            if platform is None:
                # Nothing for the cooking rule to act on in the list itself;
                # the base unit is the counter the platform belongs to.
                platform = next(
                    (
                        item for item in design.get("furniture", [])
                        if isinstance(item, dict)
                        and str(item.get("sku", "")).upper().startswith("KIT-")
                    ),
                    None,
                )
            wall = str(
                (platform or {}).get("wall")
                or (platform or {}).get("placement_wall")
                or ""
            ).lower() or None
            strict = str(brief.vastu_compliance).lower() == "strict"
            status = {
                "name": (platform or {}).get("name", "Cooking platform"),
                "sku": (platform or {}).get("sku"),
                "wall": wall,
                "facing": (platform or {}).get("facing"),
                "source": "model" if wall else "unspecified",
            }
            # Strict Vastu is a hard requirement: the platform is placed on the
            # south-east wall facing east, and the rule check verifies it.
            if strict:
                status.update({
                    "wall": "southeast",
                    "facing": "east",
                    "source": "forced_by_strict_vastu",
                })
                if platform is not None:
                    platform["wall"] = "southeast"
                    platform["facing"] = "east"
                    platform["placement_wall"] = "southeast"
            design["cooking_platform"] = status

        report = analyse_design(design, brief)
        design["_vastu"] = report.to_dict()
        design["vastu_notes"] = list(report.notes)
        design["vastu_score"] = report.score
        # Compliance is stated only by the rule results. Strip model-authored
        # claims even when they happen to agree with this assessment.
        for key in ("concept", "layout_description", "lighting_plan", "storage_strategy"):
            value = design.get(key)
            if isinstance(value, str):
                design[key] = _remove_vastu_compliance_claim(value)
        for key in ("zones", "decor_suggestions", "pros", "cons", "trade_offs"):
            value = design.get(key)
            if isinstance(value, list):
                design[key] = [
                    _remove_vastu_compliance_claim(item) if isinstance(item, str) else item
                    for item in value
                ]

    @staticmethod
    def _parse_constraints(request: DesignRequest) -> BriefConstraints:
        """
        Turn the API request into a checkable constraint set.

        This is the bridge between the old "ask the model nicely" approach and
        the deterministic checks. Everything the validator needs is derived
        here, once, in code.
        """
        form = request.brief_object()
        return parse_brief(
            client_preferences=form["client_brief"],
            furniture=form["furniture"],
            specific_requirements=form["special_requirements"],
            occupants=form["occupants"],
            room_type=form["room_type"],
            style=form["style"],
            color_scheme=form["color_scheme"],
            budget_inr=form["budget_inr"],
            vastu_compliance=form["vastu_level"],
            room_dimensions=form["dimensions_text"],
            door_direction=form["door_direction"],
            window_direction=form["window_direction"],
            lifestyle_notes=form["lifestyle_notes"],
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
            plan, step = await self.planner.run(request, contexts, brief=constraints)
            yield {"type": "step", "data": step.model_dump()}

            # ---- reflection loop ----
            #
            # The loop is driven by *code* verdicts, not by the LLM critic's
            # opinion. Previously a design with one bed for a two-bed brief
            # could score well enough to be accepted. Now a deterministic
            # validation failure forces a rewrite regardless of what the
            # critic said, which is what makes the guarantee hold.
            #
            # Every attempt is retained and the best one is returned. The
            # observed failure was the opposite: revision 1 scored 7.4, then
            # revision 2 dropped to 5.8 and went over budget, and the last one
            # - the worst - was what the client received.
            attempts: List[Dict[str, Any]] = []
            best: Optional[Dict[str, Any]] = None
            best_critique: Dict[str, Any] = {}
            best_score = -1.0
            last_report = None
            previous_score = -1.0

            for revision in range(settings.MAX_REFLECTION_ROUNDS + 1):
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
                design["layout"] = self._layout_for(
                    design, costing, constraints
                ) or {}

                # The Vastu assessment is attached before validation, because
                # the validator blocks a Strict design whose cooking rule
                # fails and a design that claims compliance it does not have.
                self._attach_vastu(design, constraints)

                report = validate_design(design, constraints, costing)
                if not report.is_valid and not registry.all():
                    repaired = self.architect._fallback_design(
                        request, plan, constraints
                    )
                    repaired = self._normalize_design(repaired, constraints)
                    repaired_cost = price_design(
                        repaired, budget_inr=constraints.budget_inr
                    )
                    repaired["layout"] = self._layout_for(
                        repaired, repaired_cost, constraints
                    ) or {}
                    self._attach_vastu(repaired, constraints)
                    repaired_report = validate_design(
                        repaired, constraints, repaired_cost
                    )
                    if (
                        repaired_report.is_valid
                        or repaired_report.pass_rate > report.pass_rate
                    ):
                        logger.info(
                            "catalog_design_repaired before=%d after=%d",
                            len(report.errors), len(repaired_report.errors),
                        )
                        design, costing, report = (
                            repaired, repaired_cost, repaired_report
                        )
                        yield {
                            "type": "step",
                            "data": AgentStep(
                                agent="pipeline",
                                status="ok" if report.is_valid else "warning",
                                message=(
                                    "Catalog mode applied deterministic repairs and "
                                    "re-validated the design."
                                ),
                                data={
                                    "remaining_issues": [
                                        violation.to_dict()
                                        for violation in report.violations
                                    ]
                                },
                            ).model_dump(),
                        }
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

                critique, step = await self.critic.run(
                    request, design, costing, plan, validation=report.to_dict(),
                    brief=constraints,
                )
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

                attempts.append({
                    "revision": revision,
                    "score": round(score, 2),
                    "critic_overall": round(float(critique.get("overall", 0) or 0), 2),
                    "valid": report.is_valid,
                    "blocking_issues": [v.code for v in report.errors],
                    "total_cost_inr": int(costing.get("estimated_cost_inr") or 0),
                })

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

                # Stop as soon as nothing blocking remains. A clean design is
                # finished; asking for another round on the strength of a
                # soft critic preference is what produced revision 2, which
                # scored worse and went over budget.
                if report.is_valid:
                    break
                if revision >= settings.MAX_REFLECTION_ROUNDS:
                    break

                # A flat score means the model is not converging. Re-running
                # it changes nothing except latency, and the observed case was
                # worse: a clean-enough revision 1 at 7.4 was followed by a
                # revision 2 at 5.8 that went over budget. The best attempt is
                # already kept, so stop and return it.
                if revision >= 1 and score <= previous_score:
                    yield {
                        "type": "step",
                        "data": AgentStep(
                            agent="pipeline",
                            status="info",
                            message=(
                                "Stopping early: the last revision did not improve; "
                                "returning the best attempt so far."
                            ),
                            data={
                                "stalled": True,
                                "score": round(score, 2),
                                "previous_score": round(previous_score, 2),
                            },
                        ).model_dump(),
                    }
                    break
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

        # Self-consistency audit. The client sees several views of the same
        # design - colour chips, swatch labels, the shopping list, the prose -
        # and they were able to disagree. Anything reported here is a defect
        # between two parts of one response, so it is checked in code rather
        # than asked of the model.
        from app.engine.consistency import audit_option

        has_bearings = bool(
            constraints.door_direction and constraints.window_direction
        )
        strict = str(constraints.vastu_compliance).lower() == "strict"
        requested = list(constraints.required_items.keys())
        option_audits: List[Dict[str, Any]] = []
        for option in options:
            audit = audit_option(
                option.model_dump(),
                requested=requested,
                strict_vastu=strict,
                has_bearings=has_bearings,
            )
            option_audits.append(audit)
            if audit["problems"]:
                logger.info(
                    "option_consistency_issues title=%s codes=%s",
                    option.title,
                    [p["code"] for p in audit["problems"]],
                )

        # One total, reconciled. `option.estimated_cost_inr` and its shopping
        # list are both produced by `price_design`, so the total is literally
        # the sum of the rows the client can see.
        reported_total = min(
            (o.estimated_cost_inr for o in options), default=0
        ) or final_cost["estimated_cost_inr"]

        # The phased plan must be built from the option the user is actually
        # shown. Building it from the reflection-loop design produced a plan
        # whose total contradicted the rendered option - the same class of bug
        # as a total that does not match its lines.
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
                # Everything the design had to assume, and whether Vastu can
                # be claimed at all. Attached to the response rather than buried
                # in the validator output so the client sees it up front.
                assumptions=assess_assumptions(
                    best or {},
                    constraints,
                    strict_vastu=(
                        str(constraints.vastu_compliance).lower() == "strict"
                    ),
                ),
                rejected_options=list(getattr(self, "_rejected_options", [])),
                # Per-option self-consistency findings (B5, C3, D3, D4).
                consistency=option_audits,
                # Every attempt the reflection loop produced, so the choice of
                # the returned one is auditable rather than implicit.
                revisions=attempts,
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
        if not registry.all():
            if brief is None:
                brief = self._parse_constraints(request)
            self._last_model = "Rule-based mode"
            self._rejected_options = []
            options = []
            for index in range(request.num_options):
                if index == 0 and seed_design is not None:
                    design = seed_design
                else:
                    design = self.architect._fallback_design(
                        request, plan, brief, variation_index=index
                    )
                    design = self._normalize_design(design, brief)
                options.append(self._to_option(design, request, brief))
            distinct, rejected = select_distinct(
                [option.model_dump() for option in options]
            )
            self._rejected_options = [
                describe_rejection(item) for item in rejected
            ]
            return [
                option for option in options
                if any(option.model_dump() == candidate for candidate in distinct)
            ][: request.num_options]

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

        # Near-duplicates are dropped here rather than served. Asking a 2B model
        # for "a distinctly different approach" reliably produced the same room
        # with the title swapped, so three "options" were one option shown
        # three times. Anything too close to an earlier option is rejected and
        # reported, rather than padding the count with a copy.
        #
        # Dedup works on plain dicts (that is all a fingerprint needs), so each
        # surviving dict is matched back to its DesignOption. Returning the
        # dicts instead is what broke the pipeline downstream: every later
        # step expects a validated option object.
        distinct, rejected = select_distinct(
            [o.model_dump() for o in options]
        )
        if rejected:
            logger.info(
                "options_deduplicated kept=%d rejected=%d reasons=%s",
                len(distinct), len(rejected),
                "; ".join(describe_rejection(r) for r in rejected[:3]),
            )
            # Carry the explanation through so the client is told why fewer
            # options appeared than were requested, instead of wondering.
            self._rejected_options = [
                describe_rejection(r) for r in rejected
            ]
        else:
            self._rejected_options = []

        # Rebuild the surviving DesignOption objects, in their original order,
        # and only rename where a title genuinely collides.
        # Dedup works on plain dicts (that is all a fingerprint needs), so each
        # surviving dict is matched back to its DesignOption.
        final: List[DesignOption] = []
        title_counts: Dict[str, int] = {}
        for option in options:
            dumped = option.model_dump()
            # Match on identity of the dumped payload rather than content, so
            # two options with identical titles cannot select each other.
            if not any(dumped == d for d in distinct):
                continue
            base = str(option.title or "Design Option").strip()
            title_counts[base] = title_counts.get(base, 0) + 1
            if title_counts[base] > 1:
                option.title = f"{base} (variant {title_counts[base]})"
            final.append(option)

        return (final or options)[: request.num_options]

    @staticmethod
    def _ensure_distinct_titles(options: List[Dict[str, Any]]) -> None:
        """
        Make sure no two surviving options share a title.

        Two identically-named cards are indistinguishable in the UI, and the
        client cannot tell which one they are looking at. The suffix is added
        only where there is an actual collision - renumbering every option
        unconditionally is noise.
        """
        seen: Dict[str, int] = {}
        for option in options:
            title = str(option.get("title") or "Design Option").strip()
            if title not in seen:
                seen[title] = 1
                option["title"] = title
                continue
            seen[title] += 1
            option["title"] = f"{title} (variant {seen[title]})"

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
        #
        # A design that came through the pipeline is already normalised.
        # Running the repair again re-inserted required items onto the option
        # (served was counted from an unnormalised copy), which made the
        # option's shopping list grow between validation and display.
        if not design.get("_normalized"):
            design = self._normalize_design(design, constraints)

        costed = price_design(design, budget_inr=budget)
        priced_by_sku = {li["sku"]: li for li in costed["line_items"]}

        # Compute this option's own geometry, Vastu verdict and brief check.
        # The response-level validation describes the design that won the
        # reflection loop; the option card must describe the option on it.
        layout = self._layout_for(design, costed, constraints)
        design["layout"] = layout or {}
        room_name = constraints.room_type.replace("_", " ")
        primary_names = [
            str(item.get("name"))
            for category in ("seating", "tv_unit")
            for item in design.get("furniture", [])
            if isinstance(item, dict) and item.get("category") == category
        ]
        actual_layout_items = (layout or {}).get("items", [])
        positions = [
            f"{item.get('name')} on the {item.get('wall')} wall"
            for item in actual_layout_items
            if item.get("category") in {"seating", "tv_unit"} and item.get("wall")
        ]
        room_info = (layout or {}).get("room") or {}
        description_parts = []
        if room_info.get("dimension_text"):
            description_parts.append(
                f"In the {room_info['dimension_text']} {room_name}"
            )
        else:
            description_parts.append(f"In the {room_name}")
        if primary_names:
            description_parts.append(
                "the main pieces are " + ", ".join(primary_names[:3])
            )
        if positions:
            description_parts.append("placed " + " and ".join(positions[:3]))
        if constraints.door_direction:
            description_parts.append(f"the door is on the {constraints.door_direction} side")
        if constraints.window_direction:
            description_parts.append(f"the window faces {constraints.window_direction}")
        zones = (layout or {}).get("zones") or []
        if zones:
            description_parts.append(
                "approximate zones: " + "; ".join(
                    f"{zone.get('label')}: {zone.get('approx_area_sqft')} sq ft"
                    for zone in zones
                )
            )
        requirements = {
            str(value).strip().lower().replace("-", "_").replace(" ", "_")
            for value in constraints.special_requirements
        }
        if "kids_safe" in requirements:
            description_parts.append(
                "Kids Safe: specify rounded-edge protection, anchor tall units "
                "with anti-tip hardware, and avoid glass"
            )
        if "low_maintenance" in requirements:
            description_parts.append(
                "Low Maintenance: use wipe-clean, durable finishes"
            )
        if {"heavy_storage", "storage_heavy"} & requirements:
            description_parts.append(
                "Heavy Storage: use the listed storage and anchor tall units"
            )
        if "natural_light" in requirements:
            direction = constraints.window_direction or "available"
            description_parts.append(
                f"Natural Light: keep the {direction}-facing window clear"
            )
        if "entertainment" in requirements:
            description_parts.append(
                "Entertainment: orient the viewing area toward the listed TV unit"
            )
        design["layout_description"] = ". ".join(description_parts) + "."
        self._attach_vastu(design, constraints)
        report = validate_design(design, constraints, costed)
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
                item["name"] = line["name"]
                item["category"] = line["category"]
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

        # One row per piece of furniture, one price source. There is exactly
        # one entry in `design["furniture"]` per catalog product (pricing and
        # the repair paths de-duplicate by SKU), and its price, brand and
        # dimensions are read from the code-computed costing. That was the
        # Rs 0 bug: rows came from model-authored fields while the total came
        # from the catalog, so a row could read zero next to a real total.
        # An item the catalog cannot price still appears, at Rs 0, so the
        # validator can report it instead of it vanishing from the list.
        furniture = []
        shopping_list = []
        for item in design.get("furniture", []) or []:
            if not isinstance(item, dict):
                continue
            sku = str(item.get("sku", "")).strip()
            line = priced_by_sku.get(sku) or {}
            try:
                qty = max(1, int(float(item.get("quantity", 1) or 1)))
            except (TypeError, ValueError):
                qty = 1
            unit = int(line.get("unit_price_inr") or 0)
            furniture.append(FurnitureItem(
                name=str(line.get("name") or item.get("name") or "Item"),
                category=str(line.get("category") or item.get("category") or ""),
                # The catalog's brand, never a model suggestion.
                brand_suggestions=[line["brand"]] if line.get("brand") else [],
                estimated_price_inr=unit,
                quantity=qty,
                sku=line.get("sku") or (sku or None),
                unit_price_inr=unit,
                line_total_inr=unit * qty,
                url=line.get("url"),
                width_cm=int((line.get("dimensions_cm") or {}).get("width") or 0),
                depth_cm=int((line.get("dimensions_cm") or {}).get("depth") or 0),
                height_cm=int((line.get("dimensions_cm") or {}).get("height") or 0),
            ))
            shopping_list.append({
                "sku": line.get("sku") or sku,
                "name": str(line.get("name") or item.get("name") or "Item"),
                "brand": line.get("brand", ""),
                "category": line.get("category") or item.get("category", ""),
                "quantity": qty,
                "unit_price_inr": unit,
                "line_total_inr": unit * qty,
            })

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

        # Brands come from the catalog rows that were actually bought.
        brands = sorted({f.brand_suggestions[0] for f in furniture if f.brand_suggestions})
        option_total = total_inr([f.model_dump() for f in furniture])
        remaining_budget = (
            max(0, constraints.budget_inr - option_total)
            if constraints.budget_inr is not None else None
        )
        purchased_skus = {str(row.get("sku") or "").upper() for row in shopping_list}
        budget_suggestions: List[Dict[str, Any]] = []
        if remaining_budget:
            from app.catalog import catalog

            candidates = catalog.for_room(constraints.room_type)
            if constraints.room_type == "living_room":
                candidates = [
                    product for product in candidates
                    if not any(
                        word in product.name.lower()
                        for word in ("desk", "night lamp", "bedside", "toy", "study")
                    )
                ]
                priority = {
                    "seating": 0, "tv_unit": 1, "table": 2,
                    "decor": 3, "lighting": 4, "storage": 5,
                }
                candidates.sort(key=lambda product: (
                    priority.get(product.category, 9), product.price_inr, product.name
                ))
            for product in candidates:
                if product.sku.upper() in purchased_skus:
                    continue
                if product.price_inr <= remaining_budget:
                    budget_suggestions.append({
                        "sku": product.sku,
                        "name": product.name,
                        "price_inr": product.price_inr,
                    })
                if len(budget_suggestions) == 3:
                    break

        trade_offs = safe_list(design.get("trade_offs")) or safe_list(design.get("cons"))
        vastu = design.get("_vastu") or {}
        layout_description = _text(design.get("layout_description"), 4000)
        room = (layout or {}).get("room") or {}
        dimension_text = str(room.get("dimension_text") or "").strip()
        if dimension_text:
            numbers = set(re.findall(r"\d+(?:\.\d+)?", layout_description))
            room_numbers = {
                f"{room.get('width_ft'):g}",
                f"{room.get('depth_ft'):g}",
            }
            if not room_numbers.issubset(numbers) or not re.search(
                r"\b(?:ft|feet)\b", layout_description, re.I
            ):
                area = room.get("area_sqft")
                area_text = f" (about {area:g} sq ft)" if isinstance(area, (int, float)) else ""
                layout_description = (
                    f"Room dimensions: {dimension_text}{area_text}. "
                    f"{layout_description}"
                ).strip()

        return DesignOption(
            title=str(design.get("title", "Design Option"))[:200],
            concept=_text(design.get("concept"), 4000),
            layout_description=layout_description,
            zones=safe_list(design.get("zones"))[:10],
            furniture=furniture,
            color_palette=palette,
            lighting_plan=_text(design.get("lighting_plan"), 2000),
            storage_strategy=_text(design.get("storage_strategy"), 2000),
            decor_suggestions=safe_list(design.get("decor_suggestions"))[:12],
            # The option total is the sum of the rows above it, by the one
            # shared function the edit step and the PDF also use.
            estimated_cost_inr=option_total,
            remaining_budget_inr=remaining_budget,
            budget_suggestions=budget_suggestions,
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
            vastu_notes=safe_list(vastu.get("notes") or design.get("vastu_notes"))[:8],
            vastu_results=list(vastu.get("results") or []),
            vastu_score=vastu.get("score"),
            trade_offs=trade_offs[:8],
            brands=brands[:10],
            pros=safe_list(design.get("pros"))[:6],
            cons=safe_list(design.get("cons"))[:6],
            shopping_list=shopping_list,
            validation=report.to_dict(),
            layout=layout,
            confidence_score=float(design.get("confidence_score", 0) or 0),
            revision_round=int(design.get("revision_round", 0) or 0),
        )

    def _layout_for(
        self,
        design: Dict[str, Any],
        costing: Dict[str, Any],
        constraints: BriefConstraints,
    ) -> Optional[Dict[str, Any]]:
        """Compute the room plan from the same line items the option lists.

        The old floor plan was built independently from the brief, so it could
        contain different furniture from the option above it. It is now built
        from the priced lines, with real dimensions and with the room's own
        door and window bearings.
        """
        from app.engine.layout import place_furniture

        dims = constraints.dimensions
        if not dims.is_known():
            return None
        items = []
        for line in costing.get("line_items") or []:
            dims_cm = line.get("dimensions_cm") or {}
            items.append({
                "sku": line["sku"],
                "name": line["name"],
                "category": line.get("category", ""),
                "width_cm": dims_cm.get("width", 0),
                "depth_cm": dims_cm.get("depth", 0),
                "height_cm": dims_cm.get("height", 0),
                "quantity": line.get("quantity", 1),
                **({
                    "preferred_walls": ["west", "north"],
                } if constraints.room_type == "living_room"
                   and line.get("category") == "tv_unit" else {}),
                **({
                    "preferred_corner": "northeast",
                } if constraints.room_type == "living_room"
                   and any(word in str(line.get("name", "")).lower()
                           for word in ("plant", "lamp")) else {}),
            })
        try:
            plan = place_furniture(
                items,
                room_width_cm=dims.width_cm,
                room_depth_cm=dims.length_cm,
                ceiling_cm=dims.ceiling_cm or 270,
                door_direction=constraints.door_direction,
                window_direction=constraints.window_direction,
            )
        except Exception as exc:  # a plan is optional; never fail the design for it
            logger.warning("layout_failed error=%s", exc)
            return None
        return plan.to_dict()


design_pipeline = DesignPipeline()
