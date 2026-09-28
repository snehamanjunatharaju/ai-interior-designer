"""
Evaluation harness.

Makes the system measurable rather than merely impressive. Provides:

- LLM-as-a-judge scoring of design quality
- Retrieval quality metrics (precision, MRR, hit rate)
- Latency and cost benchmarking
- A golden dataset with regression tracking

This is what turns a demo into something a team can iterate on safely.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import settings
from app.core.observability import get_logger, metrics
from app.core.schemas import DesignRequest, DesignResponse
from app.services.llm_provider import registry
from app.services.retrieval import retrieval_service

logger = get_logger(__name__)


class EvalCase(BaseModel):
    """
    A single benchmark case with a reference brief and expectations.

    Declared as a Pydantic model (not a dataclass) so that constructing one
    from a dict with a misspelled key raises rather than silently discarding
    the field - a silently-empty expectation list reports a hit rate of 0.0
    and looks like a retrieval regression.
    """

    name: str
    request: DesignRequest
    expected_keywords: List[str] = Field(default_factory=list)
    max_cost_inr: Optional[int] = None
    notes: str = ""

    # Forbid unknown keys so a misspelled field is an error, not a silent
    # omission that would leave a case with no expectations and score 0.0.
    model_config = ConfigDict(extra="forbid")

    @field_validator("expected_keywords")
    @classmethod
    def _require_keywords(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError(
                "expected_keywords must not be empty; a case without "
                "expectations cannot be scored"
            )
        return v


# A small but diverse golden set spanning styles, budgets, and room types.
GOLDEN_DATASET: List[EvalCase] = [
    EvalCase(
        name="small_budget_living_room",
        request=DesignRequest(
            room_type="living_room",
            client_preferences="Small 2BHK living room, needs to feel spacious, family of 3",
            style="minimalist",
            budget="low",
            room_dimensions="12ft x 10ft",
            num_options=1,
        ),
        expected_keywords=["living room", "apartment", "floating shelf", "vitrified"],
        max_cost_inr=200_000,
    ),
    EvalCase(
        name="luxury_master_bedroom",
        request=DesignRequest(
            room_type="bedroom",
            client_preferences="Master bedroom with a premium, hotel-like feel",
            style="luxury",
            budget="luxury",
            room_dimensions="16ft x 14ft",
            num_options=1,
        ),
        expected_keywords=["headboard", "wardrobe", "bed"],
    ),
    EvalCase(
        name="kids_room",
        request=DesignRequest(
            room_type="kids_room",
            client_preferences="Ages 4-8, modular and safe, lots of storage",
            budget="medium",
            num_options=1,
        ),
        expected_keywords=["storage", "rounded", "reading"],
    ),
    EvalCase(
        name="wfh_home_office",
        request=DesignRequest(
            room_type="home_office",
            client_preferences="Full-time WFH, dual monitor, video calls, needs focus",
            style="modern",
            budget="medium",
            num_options=1,
        ),
        expected_keywords=["desk", "chair", "lighting", "cable"],
    ),
    EvalCase(
        name="compact_kitchen",
        request=DesignRequest(
            room_type="kitchen",
            client_preferences="Compact modular kitchen, heavy Indian cooking, oil extraction",
            budget="medium",
            room_dimensions="8ft x 9ft",
            num_options=1,
        ),
        expected_keywords=["modular", "chimney", "granite", "ventilation"],
    ),
]


class LLMJudge:
    """
    LLM-as-a-judge evaluator for design quality.

    Uses a strong, low-temperature judgement pass with a fixed rubric so
    scores are comparable across runs.
    """

    RUBRIC = """Score the interior design on a 0-10 scale across these dimensions:

1. SPECIFICITY (0-10): Does it name actual products, colours, materials,
   measurements, and real brands rather than generic advice?
2. FEASIBILITY (0-10): Could an interior designer actually build this in
   the stated room dimensions and Indian market?
3. BUDGET_REALISM (0-10): Are the INR prices realistic for the Indian
   furniture market and consistent with the stated budget tier?
4. COHERENCE (0-10): Do layout, palette, lighting and storage work
   together as one design, or do they feel assembled from separate lists?
5. CONTEXT_FIT (0-10): Does it respect Vastu constraints, the room type,
   and the client brief?

Be a harsh grader. Generic or unbuildable advice must score low."""

    system_prompt = """You are a strict, experienced interior design critic evaluating \
submissions. You are precise, you penalise generic advice, and you do not inflate scores. \
You always respond with valid JSON."""

    async def evaluate(self, request: DesignRequest, response: DesignResponse) -> Dict[str, Any]:
        option = response.options[0] if response.options else None
        if option is None:
            return {"error": "no options generated"}

        payload = {
            "title": option.title,
            "concept": option.concept[:1500],
            "layout": option.layout_description[:1500],
            "furniture": [
                {"name": f.name, "price": f.estimated_price_inr, "brands": f.brand_suggestions}
                for f in option.furniture[:15]
            ],
            "palette": option.color_palette.model_dump() if option.color_palette else {},
            "lighting": option.lighting_plan[:800],
            "total_cost_inr": option.estimated_cost_inr,
        }

        prompt = f"""CLIENT BRIEF:
Room: {request.room_type.value}, Style: {request.style.value}, Budget: {request.budget.value}
Requirements: {request.client_preferences}

SUBMITTED DESIGN:
{json.dumps(payload, indent=2)}

Evaluate against the rubric. Respond ONLY with JSON:
{{
  "scores": {{"specificity": 0, "feasibility": 0, "budget_realism": 0,
              "coherence": 0, "context_fit": 0}},
  "overall": 0,
  "verdict": "one sentence",
  "top_issue": "the single biggest weakness"
}}"""

        try:
            result = await registry.complete_json(prompt, system=self.system_prompt,
                                                   temperature=0.1)
            judgement = result["json"]
            if not isinstance(judgement, dict):
                raise ValueError("judge returned non-dict")
            return judgement
        except Exception as exc:
            logger.warning("judge_failed error=%s", exc)
            return {"error": str(exc)}


class RetrievalEvaluator:
    """
    Measures retrieval quality with keyword-recall and ranking metrics.

    Since there is no labelled relevance set, the proxy is whether the
    expected domain concepts actually surface in the top-k chunks - a
    practical, reproducible signal.
    """

    async def evaluate_case(self, case: EvalCase, top_k: int = 4) -> Dict[str, Any]:
        retrieval_service.initialize()
        query = retrieval_service.build_query(case.request)
        contexts = await retrieval_service.retrieve(query, top_k=top_k)

        if not contexts:
            return {"case": case.name, "hit_rate": 0.0, "mrr": 0.0, "precision": 0.0,
                    "retrieved": 0}

        blob = " ".join(c.content.lower() for c in contexts)
        keywords = [k.lower() for k in case.expected_keywords]

        hits = {k: (1.0 if k in blob else 0.0) for k in keywords}
        # Reciprocal rank of the first chunk containing any expected keyword.
        first_hit_rank = 0
        for rank, ctx in enumerate(contexts, 1):
            low = ctx.content.lower()
            if any(k in low for k in keywords):
                first_hit_rank = rank
                break
        mrr = (1.0 / first_hit_rank) if first_hit_rank else 0.0

        return {
            "case": case.name,
            "hit_rate": sum(hits.values()) / len(hits) if hits else 0.0,
            "mrr": round(mrr, 4),
            "precision": len([c for c in contexts if c.score > settings.RAG_MIN_SCORE]) / top_k,
            "retrieved": len(contexts),
            "per_keyword": hits,
        }


class EvaluationHarness:
    """Runs the full evaluation suite and aggregates results."""

    def __init__(self) -> None:
        self.judge = LLMJudge()
        self.retrieval_eval = RetrievalEvaluator()
        self.history: List[Dict[str, Any]] = []

    async def evaluate_retrieval(self) -> Dict[str, Any]:
        """Evaluate retrieval quality across the golden dataset."""
        cases = await asyncio.gather(
            *(self.retrieval_eval.evaluate_case(c) for c in GOLDEN_DATASET),
            return_exceptions=True,
        )
        results = [c for c in cases if isinstance(c, dict)]
        # Report failures rather than dropping them: a silently-skipped case
        # inflates the mean and hides a broken case.
        errors = [
            {"case": GOLDEN_DATASET[i].name, "error": str(c)}
            for i, c in enumerate(cases)
            if isinstance(c, Exception)
        ]
        for err in errors:
            logger.warning("retrieval_eval_failed case=%s error=%s", err["case"], err["error"])

        if not results:
            return {"error": "all retrieval evaluations failed", "failures": errors}

        summary = {
            "cases": len(results),
            "failed_cases": len(errors),
            "mean_hit_rate": round(statistics.mean(r["hit_rate"] for r in results), 4),
            "mean_mrr": round(statistics.mean(r["mrr"] for r in results), 4),
            "mean_precision": round(statistics.mean(r["precision"] for r in results), 4),
            "per_case": results,
        }
        metrics.gauge("eval.retrieval_hit_rate", summary["mean_hit_rate"])
        metrics.gauge("eval.retrieval_mrr", summary["mean_mrr"])
        return summary

    async def evaluate_generation(
        self, num_cases: int = 3, run_generation: bool = True
    ) -> Dict[str, Any]:
        """
        End-to-end evaluation: generate designs and judge them.

        With `run_generation=False`, this reports only the metrics available
        from the previous run, which is useful in CI where generation is slow.
        """
        from app.services.design_service import design_service

        cases = GOLDEN_DATASET[:num_cases]
        results = []

        for case in cases:
            start = time.perf_counter()
            entry: Dict[str, Any] = {"case": case.name}

            if run_generation:
                try:
                    response = await design_service.generate(case.request, use_cache=False)
                    entry["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
                    entry["cost_inr"] = round(response.cost_inr, 4)
                    entry["llm_cost_inr"] = round(response.cost_inr, 4)
                    entry["options"] = len(response.options)
                    entry["estimated_design_cost"] = response.total_cost_inr

                    if case.max_cost_inr and response.total_cost_inr > case.max_cost_inr * 1.5:
                        entry["budget_violation"] = True

                    entry["judgement"] = await self.judge.evaluate(case.request, response)
                except Exception as exc:
                    entry["error"] = str(exc)
            results.append(entry)

        scores = [
            r["judgement"]["overall"]
            for r in results
            if isinstance(r.get("judgement"), dict) and "overall" in r["judgement"]
        ]
        summary = {
            "cases": len(results),
            "mean_quality_score": round(statistics.mean(scores), 3) if scores else None,
            "mean_latency_ms": round(
                statistics.mean([r["latency_ms"] for r in results if "latency_ms" in r]), 1
            ) if any("latency_ms" in r for r in results) else None,
            "per_case": results,
        }
        if scores:
            metrics.gauge("eval.quality_score", statistics.mean(scores))
        self.history.append(summary)
        return summary

    async def run_all(self) -> Dict[str, Any]:
        retrieval, generation = await asyncio.gather(
            self.evaluate_retrieval(),
            self.evaluate_generation(num_cases=2, run_generation=True),
        )
        return {
            "timestamp": time.time(),
            "retrieval": retrieval,
            "generation": generation,
        }


evaluation_harness = EvaluationHarness()
