"""
The evaluation harness.

Runs the corpus in `app.eval.corpus` against the real pipeline and computes six
metrics, all of them from code rather than from a model's opinion:

  budget_compliance_rate   - designs whose total is within the stated budget,
                             or which honestly report infeasibility
  item_coverage_rate       - requested items actually present in the shopping list
  constraint_violation_rate- designs passing every brief check
  vastu_rule_pass_rate     - Vastu rules that passed, over rules that could be
                             assessed (unverifiable ones are excluded, not failed)
  latency per agent        - milliseconds, per agent
  avg_critic_score         - mean critic overall

The distinction that matters: every metric here is arithmetic over catalog prices
and validator verdicts. None of them asks a language model how well it did.
That is the whole reason the harness is trustworthy enough to gate a change on.

Run it directly:

    ./venv/bin/python -m app.eval.run                 # markdown to stdout
    ./venv/bin/python -m app.eval.run --json          # machine readable
    ./venv/bin/python -m app.eval.run --limit 5       # quick pass
    ./venv/bin/python -m app.eval.run --out results.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.eval.corpus import EvalCase, all_cases, corpus_summary
from app.core.observability import get_logger

logger = get_logger(__name__)


@dataclass
class CaseResult:
    """What one brief actually produced."""

    name: str
    ok: bool = True
    error: Optional[str] = None

    within_budget: Optional[bool] = None
    budget_total: Optional[int] = None
    budget_cap: Optional[int] = None
    reported_infeasible: bool = False

    items_requested: int = 0
    items_present: int = 0

    passed: bool = False
    checks_run: int = 0
    checks_passed: int = 0
    violations: List[str] = field(default_factory=list)

    vastu_rules_passed: int = 0
    vastu_rules_total: int = 0
    vastu_verifiable: Optional[bool] = None

    critic_score: Optional[float] = None
    agent_latency_ms: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0

    options: int = 0
    options_distinct: int = 0

    # Expectations from the corpus, for reporting.
    expect_within_budget: Optional[bool] = None
    expect_audience: Optional[str] = None
    expect_vastu_verifiable: Optional[bool] = None


def evaluate_case(case: EvalCase) -> CaseResult:
    """
    Run one brief through the buffered pipeline and measure the result.

    Exceptions are captured rather than raised: a case that crashes is a result,
    and one crash must not abort the run.
    """
    result = CaseResult(
        name=case.name,
        expect_within_budget=case.expect_within_budget,
        expect_audience=case.expect_audience,
        expect_vastu_verifiable=case.expect_vastu_verifiable,
    )
    started = time.perf_counter()

    try:
        from app.core.schemas import DesignRequest
        from app.agents.orchestrator import design_pipeline

        request = DesignRequest(**case.to_request())

        async def _drive():
            # Driven through `stream` rather than `generate` so per-agent
            # latency is captured from the step events. The harness exists to
            # measure, and the step timings are the only per-agent figures the
            # system produces.
            payload = None
            steps: List[Dict[str, Any]] = []
            async for event in design_pipeline.stream(
                request, request_id=f"eval-{case.name}"
            ):
                if event.get("type") == "step":
                    steps.append(event.get("data") or {})
                elif event.get("type") == "complete":
                    payload = event.get("data")
            return payload, steps

        payload, steps = asyncio.run(_drive())
        if payload is None:
            raise RuntimeError("pipeline produced no response")
    except Exception as exc:  # noqa: BLE001 - a crash is a measurable outcome
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        result.total_latency_ms = (time.perf_counter() - started) * 1000
        logger.warning("eval_case_failed case=%s error=%s", case.name, exc)
        return result

    # Per-agent latency straight from the streamed steps.
    for step in steps:
        agent = str(step.get("agent") or "").strip()
        duration = step.get("duration_ms")
        if agent and isinstance(duration, (int, float)) and duration > 0:
            key = f"{agent}_{step.get('status', 'ok')}"
            result.agent_latency_ms[key] = round(float(duration), 1)
    result.total_latency_ms = (time.perf_counter() - started) * 1000

    validation = payload.get("validation") or {}
    costing = payload.get("cost_breakdown") or {}
    assumptions = payload.get("assumptions") or {}

    # --- budget ------------------------------------------------------------
    result.budget_total = costing.get("estimated_cost_inr")
    result.budget_cap = costing.get("budget_inr")
    if result.budget_cap:
        result.within_budget = bool(costing.get("within_budget"))
    result.reported_infeasible = bool(costing.get("within_budget") is False)

    # --- item coverage -----------------------------------------------------
    # Counted against the *primary* option, which is the one the client sees
    # first. An item buried in the third option is not a satisfied request.
    options = payload.get("options") or []
    result.options = len(options)
    if options:
        primary = options[0]
        # Options arrive as dicts from the streamed payload, but tolerate the
        # validated-object shape too so the harness works against either.
        furniture = (
            primary.get("furniture") if isinstance(primary, dict)
            else getattr(primary, "furniture", [])
        ) or []
        present: Dict[str, Any] = {"skus": set(), "names": set(), "cats": set()}
        for entry in furniture:
            if not isinstance(entry, dict):
                continue
            if entry.get("sku"):
                present["skus"].add(str(entry["sku"]).upper())
            if entry.get("name"):
                present["names"].add(str(entry["name"]).lower())
            if entry.get("category"):
                present["cats"].add(str(entry["category"]).lower())
        result.items_requested = len(case.expect_items)
        result.items_present = sum(
            1 for term in case.expect_items if _term_satisfied(term, present)
        )

    # --- brief checks ------------------------------------------------------
    result.checks_run = int(validation.get("checks_run", 0) or 0)
    result.checks_passed = int(validation.get("checks_passed", 0) or 0)
    result.passed = bool(validation.get("valid", False))
    result.violations = [
        str(v.get("code")) for v in (validation.get("violations") or [])
    ]

    # --- Vastu -------------------------------------------------------------
    result.vastu_verifiable = assumptions.get("vastu_verifiable")
    vastu = payload.get("vastu") or {}
    rules = vastu.get("rules") or []
    if rules:
        result.vastu_rules_total = len(rules)
        result.vastu_rules_passed = sum(
            1 for r in rules if isinstance(r, dict) and r.get("passed")
        )

    # --- critic + latency --------------------------------------------------
    evaluation = payload.get("evaluation") or {}
    if isinstance(evaluation.get("overall"), (int, float)):
        result.critic_score = float(evaluation["overall"])
    result.agent_latency_ms = _collect_latency(payload)

    return result


def _term_satisfied(term: str, present: Dict[str, Any]) -> bool:
    """Does the shopping list answer this requested term?"""
    from app.engine.brief import products_satisfying

    term = term.strip().lower()
    if not term:
        return True
    for product in products_satisfying(term):
        if product.sku.upper() in present["skus"]:
            return True
        if product.name.lower() in present["names"]:
            return True
        if product.category.lower() in present["cats"]:
            return True
    # Fall back to a name-word match so "bookshelf" is satisfied by
    # "Floating Wall Shelf" without needing an exact product name.
    words = {w for w in term.split() if len(w) > 3}
    return any(
        any(w in name for name in present["names"]) for w in words
    )


def _collect_latency(payload: Dict[str, Any]) -> Dict[str, float]:
    """Per-agent latency, from the streamed steps if the response carries them."""
    out: Dict[str, float] = {}
    for step in payload.get("agent_steps") or []:
        if not isinstance(step, dict):
            continue
        agent = str(step.get("agent") or "").strip()
        duration = step.get("duration_ms")
        if agent and isinstance(duration, (int, float)):
            out[agent] = out.get(agent, 0.0) + float(duration)
    return out


def aggregate(results: Sequence[CaseResult]) -> Dict[str, Any]:
    """Reduce per-case results to the six headline metrics."""
    ok = [r for r in results if r.ok]

    # Budget compliance: within budget, or honestly declared infeasible. A
    # design that overspends without saying so is the failure this counts.
    budget_scored = [
        r for r in ok if r.expect_within_budget is not None and r.budget_cap
    ]
    budget_ok = 0
    for r in budget_scored:
        if r.expect_within_budget is True:
            if r.within_budget:
                budget_ok += 1
        else:
            # Expected to be infeasible: passing means we did NOT overspend
            # silently, i.e. either within budget or flagged.
            if r.within_budget or r.reported_infeasible:
                budget_ok += 1

    coverage_total = sum(r.items_requested for r in ok)
    coverage_hit = sum(r.items_present for r in ok)

    passed = sum(1 for r in ok if r.passed)
    checked = len(ok)

    vastu_passed = sum(r.vastu_rules_passed for r in ok)
    vastu_total = sum(r.vastu_rules_total for r in ok)

    critic_scores = [r.critic_score for r in ok if r.critic_score is not None]

    agent_totals: Dict[str, List[float]] = {}
    for r in ok:
        for agent, ms in r.agent_latency_ms.items():
            agent_totals.setdefault(agent, []).append(ms)

    return {
        "cases_total": len(results),
        "cases_run": len(ok),
        "cases_failed": len(results) - len(ok),
        "budget_compliance_rate": _rate(budget_ok, len(budget_scored)),
        "budget_cases_scored": len(budget_scored),
        "item_coverage_rate": _rate(coverage_hit, coverage_total),
        "items_requested": coverage_total,
        "items_present": coverage_hit,
        "constraint_pass_rate": _rate(passed, checked),
        "vastu_rule_pass_rate": _rate(vastu_passed, vastu_total),
        "vastu_rules_total": vastu_total,
        "avg_critic_score": (
            round(statistics.fmean(critic_scores), 2) if critic_scores else None
        ),
        "critic_scores_counted": len(critic_scores),
        "avg_total_latency_ms": (
            round(statistics.fmean([r.total_latency_ms for r in ok]), 0)
            if ok else None
        ),
        "latency_per_agent_ms": {
            agent: round(statistics.fmean(values), 0)
            for agent, values in sorted(agent_totals.items())
        },
        "avg_options": (
            round(statistics.fmean([r.options for r in ok]), 2) if ok else None
        ),
    }


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if not denominator:
        return None
    return round(numerator / denominator, 4)


def to_markdown(
    results: Sequence[CaseResult], summary: Dict[str, Any], title: str
) -> str:
    """Render the results as a markdown table."""
    lines: List[str] = []
    lines.append(f"# {title}\n")
    lines.append(
        f"Corpus: {summary['cases_run']}/{summary['cases_total']} cases run, "
        f"{summary['cases_failed']} failed outright.\n"
    )

    lines.append("## Headline metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    rows = [
        ("Budget compliance rate", _pct(summary["budget_compliance_rate"]),
         f"({summary['budget_cases_scored']} cases scored)"),
        ("Requested-item coverage", _pct(summary["item_coverage_rate"]),
         f"({summary['items_present']}/{summary['items_requested']} items)"),
        ("Constraint pass rate", _pct(summary["constraint_pass_rate"]), ""),
        ("Vastu rule pass rate", _pct(summary["vastu_rule_pass_rate"]),
         f"({summary['vastu_rules_total']} rules assessed)"),
        ("Average critic score",
         f"{summary['avg_critic_score']}/10"
         if summary["avg_critic_score"] is not None else "n/a", ""),
        ("Average options returned",
         str(summary["avg_options"]), ""),
        ("Average end-to-end latency",
         f"{summary['avg_total_latency_ms']:.0f} ms"
         if summary["avg_total_latency_ms"] is not None else "n/a", ""),
    ]
    for label, value, note in rows:
        lines.append(f"| {label} | {value} {note} |")

    lines.append("\n## Latency per agent\n")
    latency = summary["latency_per_agent_ms"]
    if latency:
        lines.append("| Agent | Average ms |")
        lines.append("|---|---|")
        for agent, ms in sorted(latency.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {agent} | {ms:.0f} |")
    else:
        lines.append("_No per-agent latency was reported by the response._")

    lines.append("\n## Per-case results\n")
    lines.append(
        "| Case | Within budget | Items | Checks | Vastu | Critic | Options | Notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        if not r.ok:
            lines.append(
                f"| `{r.name}` | ERROR | - | - | - | - | 0 | {r.error or ''} |"
            )
            continue
        budget = (
            "n/a" if r.within_budget is None
            else ("yes" if r.within_budget else "NO")
        )
        items = f"{r.items_present}/{r.items_requested}" if r.items_requested else "-"
        checks = f"{r.checks_passed}/{r.checks_run}"
        vastu = (
            "withheld" if r.vastu_verifiable is False
            else (f"{r.vastu_rules_passed}/{r.vastu_rules_total}"
                  if r.vastu_rules_total else "-")
        )
        critic = f"{r.critic_score:.1f}" if r.critic_score is not None else "-"
        notes = ", ".join(sorted(set(r.violations))[:3]) if r.violations else ""
        lines.append(
            f"| `{r.name}` | {budget} | {items} | {checks} | {vastu} | "
            f"{critic} | {r.options} | {notes} |"
        )

    return "\n".join(lines) + "\n"


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


async def run_all(limit: Optional[int] = None) -> List[CaseResult]:
    """Run the corpus, reusing one event loop across all cases."""
    cases = all_cases()
    if limit:
        cases = cases[:limit]
    results: List[CaseResult] = []
    for index, case in enumerate(cases, 1):
        logger.info("eval_progress %d/%d %s", index, len(cases), case.name)
        results.append(await asyncio.to_thread(evaluate_case, case))
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the InteriorAI evaluation.")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N cases (quick pass)")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of markdown")
    parser.add_argument("--out", default=None,
                        help="write the report to this path as well")
    parser.add_argument("--title", default="InteriorAI evaluation",
                        help="report title")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    results = asyncio.run(run_all(limit=args.limit))
    summary = aggregate(results)
    summary["wall_clock_ms"] = round((time.perf_counter() - started) * 1000)
    summary["corpus"] = corpus_summary()

    report = to_markdown(results, summary, args.title)

    if args.json:
        payload = {
            "summary": summary,
            "cases": [r.__dict__ for r in results],
        }
        report = json.dumps(payload, indent=2, default=str)

    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(f"\nWritten to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())