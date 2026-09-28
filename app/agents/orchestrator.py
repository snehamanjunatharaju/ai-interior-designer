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
import json
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
from app.services.llm_provider import extract_json, registry
from app.services.retrieval import retrieval_service

logger = get_logger(__name__)

DESIGNER_SYSTEM_PROMPT = """You are a senior interior designer with 15 years of \
experience in the Indian residential market. You have deep knowledge of Vastu, Indian \
climates, space constraints of Indian apartments, local brands, and INR pricing. You \
give specific, actionable, practical advice - never generic platitudes. You always \
respond with valid JSON and no other text."""


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
Never give generic advice - name actual products, colours with hex codes,
specific materials, and real Indian retail brand names.
Respect the constraints given to you. Always respect the room dimensions if stated.

Respond ONLY with valid JSON:
{
  "title": "short evocative design name",
  "concept": "2-3 sentence description of the concept",
  "layout_description": "how the room is actually laid out and zoned, with measurements",
  "zones": ["zone: description", ...],
  "furniture": [
    {"name": "...", "category": "...", "brand_suggestions": ["..."],
     "estimated_price_inr": 0, "quantity": 1}
  ],
  "color_palette": {"primary": "", "secondary": "", "accent": "", "neutrals": ["..."], "hex_codes": ["#RRGGBB", ...]},
  "lighting_plan": "layered lighting description with temperatures and wattage",
  "storage_strategy": "storage plan with quantities",
  "decor_suggestions": ["...", ...],
  "vastu_notes": ["...", ...],
  "brands": ["brand name", ...],
  "pros": ["...", ...],
  "cons": ["honest trade-offs", ...]
}"""

    async def run(
        self,
        request: DesignRequest,
        plan: Dict[str, Any],
        contexts: List[RetrievedContext],
        revision: int = 0,
        critique: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], AgentStep]:
        start = time.perf_counter()
        context_block = retrieval_service.format_context(contexts)

        revision_block = ""
        if revision > 0 and critique:
            issues = critique.get("issues", [])
            revision_block = f"""

THIS IS REVISION {revision}. A critic reviewed your previous attempt and found these
problems - you MUST fix all of them:
{chr(10).join('- ' + i for i in issues) if issues else '- Improve specificity and feasibility'}

Scores from previous attempt: {json.dumps(critique.get('scores', {}))}"""

        prompt = f"""Design this room.

{_brief_to_text(request)}

DESIGN PLAN (agreed, follow it):
{json.dumps(plan, indent=2)}

RELEVANT DESIGN KNOWLEDGE:
{context_block or 'None retrieved - rely on expertise.'}
{revision_block}

Produce the complete design as JSON. Be specific with real brand names and INR prices."""

        with tracer.start("agent.architect", revision=revision):
            try:
                result = await self._call_llm(prompt, temperature=0.8)
                design = result["json"] if isinstance(result["json"], dict) else {}
            except Exception as exc:
                logger.warning("architect_failed error=%s", exc)
                design = self._fallback_design(request, plan)

        if not design.get("title"):
            design = self._fallback_design(request, plan)
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

    def _fallback_design(self, request: DesignRequest, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": f"{request.style.value.title()} {request.room_type.value.replace('_', ' ').title()}",
            "concept": plan.get("design_intent", "A practical, well-organised room."),
            "layout_description": "Layout derived from the design plan.",
            "zones": plan.get("zones", []),
            "furniture": [],
            "color_palette": {"primary": "warm neutral", "secondary": "soft beige",
                              "accent": "terracotta", "neutrals": [], "hex_codes": []},
            "lighting_plan": "Layered ambient, task, and accent lighting at 2700-3000K.",
            "storage_strategy": "Modular units sized to available wall space.",
            "decor_suggestions": [],
            "vastu_notes": [],
            "brands": [],
            "pros": ["Practical layout", "Good use of available space"],
            "cons": [],
        }


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
Title: {design.get('title', '')}
Layout: {design.get('layout_description', '')[:800]}
Furniture line items: {json.dumps(furniture_summary)}
Lighting: {design.get('lighting_plan', '')[:300]}
Storage: {design.get('storage_strategy', '')[:300]}

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
        return costing, self._step(
            status,
            f"Estimated Rs {costing.get('estimated_cost_inr', 0):,} "
            f"({'within' if costing.get('within_budget') else 'OVER'} budget)",
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
            plan, step = await self.planner.run(request, contexts)
            yield {"type": "step", "data": step.model_dump()}

            # ---- reflection loop ----
            best: Optional[Dict[str, Any]] = None
            best_critique: Dict[str, Any] = {}

            for revision in range(settings.MAX_REFLECTION_ROUNDS + 1):
                if revision > 0:
                    yield {
                        "type": "step",
                        "data": AgentStep(
                            agent="pipeline",
                            status="info",
                            message=f"Reflecting: revision {revision} based on critique",
                        ).model_dump(),
                    }

                design, step = await self.architect.run(
                    request, plan, contexts, revision=revision,
                    critique=best_critique if revision > 0 else None,
                )
                yield {"type": "step", "data": step.model_dump()}

                costing, step = await self.budget.run(request, design, contexts)
                yield {"type": "step", "data": step.model_dump()}

                critique, step = await self.critic.run(request, design, costing, plan)
                yield {"type": "step", "data": step.model_dump()}

                # Keep the best-scoring attempt rather than merely the last.
                if best is None or critique.get("overall", 0) >= best_critique.get("overall", 0):
                    best_critique = critique
                    best = design

                if not critique.get("needs_revision") or revision >= settings.MAX_REFLECTION_ROUNDS:
                    break

            if best is None:
                raise RuntimeError("pipeline produced no design")

            options = await self._parallel_options(request, plan, contexts)

            response = DesignResponse(
                request_id=request_id,
                options=options,
                context_sources=contexts,
                total_cost_inr=sum(o.estimated_cost_inr for o in options) / max(1, len(options)),
                latency_ms=(time.perf_counter() - start) * 1000,
                # Token/cost totals are accumulated in the metrics registry,
                # which is the accurate source of truth for spend.
                cost_inr=0.0,
                model_used=self._last_model,
                cached=False,
                evaluation=best_critique,
            )

        metrics.observe("pipeline.latency_ms", response.latency_ms)
        metrics.increment("pipeline.completed")
        yield {"type": "complete", "data": response.model_dump()}

    async def _parallel_options(
        self, request: DesignRequest, plan: Dict[str, Any], contexts: List[RetrievedContext]
    ) -> List[DesignOption]:
        """
        Generate the requested number of options concurrently.

        Concurrency is bounded so we do not overwhelm the LLM provider or
        trip a rate limit; each option is independent.
        """
        if request.num_options <= 1:
            design, _ = await self.architect.run(request, plan, contexts)
            return [self._to_option(design, request)]

        semaphore = asyncio.Semaphore(min(settings.MAX_CONCURRENT_LLM_CALLS, 4))

        async def build(index: int) -> DesignOption:
            async with semaphore:
                varied = request.model_copy()
                varied.client_preferences = (
                    f"{request.client_preferences} "
                    f"(explore a distinctly different approach from the others, "
                    f"variation {index + 1})"
                )
                design, _ = await self.architect.run(varied, plan, contexts)
                return self._to_option(design, request)

        results = await asyncio.gather(
            *(build(i) for i in range(request.num_options)), return_exceptions=True
        )
        return [r for r in results if isinstance(r, DesignOption)]

    def _to_option(self, design: Dict[str, Any], request: DesignRequest) -> DesignOption:
        """Coerce an arbitrary LLM dict into a validated DesignOption."""
        def safe_list(val: Any) -> List[str]:
            if isinstance(val, list):
                return [str(v) for v in val if str(v).strip()]
            return []

        furniture = []
        for item in design.get("furniture", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                price = int(float(item.get("estimated_price_inr", 0) or 0))
            except (TypeError, ValueError):
                price = 0
            try:
                qty = int(float(item.get("quantity", 1) or 1))
            except (TypeError, ValueError):
                qty = 1
            furniture.append(FurnitureItem(
                name=str(item.get("name", "Item")),
                category=str(item.get("category", "")),
                brand_suggestions=safe_list(item.get("brand_suggestions"))[:4],
                estimated_price_inr=max(0, price),
                quantity=max(1, qty),
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
        if not est:
            est = sum(f.estimated_price_inr * f.quantity for f in furniture)

        return DesignOption(
            title=str(design.get("title", "Design Option"))[:200],
            concept=str(design.get("concept", ""))[:4000],
            layout_description=str(design.get("layout_description", ""))[:4000],
            zones=safe_list(design.get("zones"))[:10],
            furniture=furniture,
            color_palette=palette,
            lighting_plan=str(design.get("lighting_plan", ""))[:2000],
            storage_strategy=str(design.get("storage_strategy", ""))[:2000],
            decor_suggestions=safe_list(design.get("decor_suggestions"))[:12],
            estimated_cost_inr=max(0, est),
            budget_breakdown=[
                BudgetBreakdown(
                    category=str(b.get("category", "misc"))[:100],
                    amount_inr=int(float(b.get("amount_inr", 0) or 0)),
                    percentage=float(b.get("percentage", 0) or 0),
                    notes=str(b.get("notes", ""))[:300],
                )
                for b in (design.get("budget_breakdown") or [])
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
