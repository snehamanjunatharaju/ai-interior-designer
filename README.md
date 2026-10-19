# InteriorAI

An agentic interior design platform for the Indian market. Takes a client's
brief, retrieves domain knowledge, runs a multi-agent LLM pipeline, and returns
costed design proposals with a measured floor plan.

**The core principle: language models write, code decides.** The LLM chooses
furniture and writes prose. Python does every calculation, every constraint
check, every placement decision, and every Vastu ruling. A 2B model cannot be
trusted to add up a furniture list or to count beds, and this project is
structured so it never has to.

---

## Quick start

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python -m uvicorn app.main:app --port 8001
```

Open http://localhost:8001 · API docs at `/docs`

```bash
./venv/bin/python -m pytest        # 330 tests, ~4s, fully offline
```

Ollama is optional. Without it the pipeline returns a catalog-driven fallback
design rather than failing.

---

## What the system does

**Input:** room type, style, colour scheme, budget, dimensions, door and
window direction, occupants, Vastu preference.

**Output:** priced design options, an interactive 2D floor plan and rotatable
3D view, a rule-based Vastu score with per-rule explanations, a two-phase
budget plan when the budget is unrealistic, targeted chat refinement, and a
PDF client proposal.

---

## Architecture

```
   Brief ──▶ BriefParser ──▶ BriefConstraints (item counts, money, colour,
   │                            dimensions, directions)  ← pure code
   │
   ├──▶ Hybrid retrieval (BM25 + dense, rank-fused, metadata-filtered)
   │
   └──▶ Multi-agent pipeline
         Planner ──▶ Architect ──▶ [normalise] ──▶ price_design ──▶ validate
                          ▲                                            │
                          └──────── violation list (code) ◀────────────┘
                          └──────── reflection loop, max 3
   │
   ├──▶ LayoutEngine  (real cm, 60cm clearance, door swing, collisions)
   ├──▶ VastuEngine   (rule-based, per-rule score + remedy)
   └──▶ Refine / PDF  (targeted edits, client proposal)
```

### The division of labour

| Concern | Owner | Why |
|---|---|---|
| Which products, and why | LLM | Creative judgement |
| Prose, concepts, trade-offs | LLM | Language |
| Every price and total | `engine/pricing.py` | Arithmetic cannot be delegated |
| Does it match the brief | `engine/validator.py` | A model cannot check itself |
| Where the furniture goes | `engine/layout.py` | Geometry is arithmetic |
| Vastu compliance | `engine/vastu.py` | Falsifiable, reproducible |
| Aesthetics, coherence | LLM critic | Genuinely subjective |

**Retrieval is hybrid.** Dense vectors alone are a poor fit for the default
hashing embedder, which has no semantic component: a brief asking for a "bunk
bed" cannot reliably find the passage that says "bunk bed". BM25 supplies
exact-term matching, and the two rankings are combined by Reciprocal Rank
Fusion — by position, not by score, because a hashing cosine (~0.2) and a
BM25 score (unbounded) are on unrelated scales.

**Metadata is inferred, not hand-tagged.** Room, style, budget tier, and
Vastu facets are derived from each passage's text, so they cannot drift out of
step with the content when a passage is edited. A stale `room` tag silently
removes a passage from the room it actually describes.

**Filters narrow, they never exclude.** A passage with no room tag is general
advice — "maintain 90 cm of circulation" applies everywhere — and demanding a
tag would discard the most broadly useful guidance in the corpus.

---

## Modules

```
app/
  catalog/      SQLite price catalog (name, brand, W/D/H, price, URL, sleep capacity)
  engine/
    brief.py    free text -> checkable constraints
    pricing.py  all arithmetic; phased budget negotiator
    validator.py deterministic brief checks
    layout.py   furniture placement with clearances and door swing
    vastu.py    rule-based Vastu scoring
    refine.py   classify + apply targeted changes
  agents/       planner, architect, critic, orchestrator
  services/     LLM providers, hybrid RAG, BM25, cache, evaluation, proposal (PDF)
  api/          FastAPI routes
static/         layout.js (2D SVG), layout3d.js (three.js), app.js
tests/          330 tests
```

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/generate` | Full pipeline, streamed over SSE |
| `POST /api/v1/layout` | Floor plan only — no LLM call, instant |
| `POST /api/v1/vastu` | Rule-based Vastu score with explanations |
| `POST /api/v1/refine` | Change one part of a design |
| `POST /api/v1/proposal.pdf` | Client proposal as PDF |
| `GET /api/v1/evaluation` | Pass rate, latency, cost dashboard |
| `GET /api/v1/metrics/prometheus` | Live telemetry |

---

## Design decisions

**Prices live in a catalog, not in prompts.** The original design asked the
LLM to price furniture and then "reconciled" the numbers in Python. That is
unsound: a model emitting zeros sums to zero, and no downstream correction
invents a price that was never produced. Now the model selects a SKU and the
database supplies the price. A hallucinated `estimated_price_inr` is discarded.

**Required items are inserted in code.** When the model omits a bed the client
explicitly asked for, code adds it rather than asking the model again. A
requested bed is not a creative decision, and the 2B model reliably failed to
fix it across every revision round.

**Beds count sleeping places, not units.** One bunk sleeps two, so a
two-child brief is satisfied by quantity 1. Counting units both reported a
false shortfall and, once "fixed", doubled the price. The catalog carries a
`sleeps` field for exactly this.

**The brief vocabulary and the catalog vocabulary differ.** A client says
"sofa"; the catalog files it under `seating`. `BRIEF_TO_CATALOG_CATEGORY` is
the single join between them, and both the validator and the resolver read
from it, so the two cannot drift apart.

**A broad category is not internally interchangeable.** `seating` contains
sofas, armchairs, bean bags, and stools. A stool does not satisfy a request
for a sofa, so `ACCEPTABLE_PRODUCTS` narrows by product name.

**Vastu is a rule table, not a prompt.** Each rule states a condition on the
measured layout and returns a pass or fail with a plain explanation and a
remedy. The score is therefore reproducible and assertable — unlike a model's
opinion of a design.

**Placement is greedy first-fit by descending footprint.** Large items claim
prime wall space first, which is what makes the result fit. Each candidate
tries its preferred wall, then both orientations, scanning in 5 cm steps
against clearance, door-swing, and collision rules.

**Refinement classifies before it edits.** Misclassifying "make the desk
bigger" as a full redesign is the exact failure the feature exists to prevent,
so an unrecognised request returns "I could not tell which part to change"
rather than guessing.

**Graceful degradation.** When the LLM fails or returns unusable JSON, the
architect falls back to a catalog-driven design: real SKUs, correct prices,
brief-compliant counts, honest prose that admits it is generic. The original
stub returned empty furniture and the literal string "Layout derived from the
design plan", so a 90-second request ended in a ₹0 design.

---

## Honest limitations

- **Catalog prices are indicative**, rounded to ₹50, compiled from public
  listings — not scraped live. Verify a sample before quoting a client.
- **The knowledge corpus is 79 curated passages**, not the 200+ a production
  system would need. Vastu coverage went from one passage to fifteen after this
  was measured, but gaps remain in styling, and the coverage counts are
  exposed at `GET /api/v1/evaluation` under `retrieval` so the limit is
  visible rather than implied.
- **The default embedder is hashing**, not sentence-transformers, so dense
  scores sit low by design. BM25 is what carries exact-term recall. Add
  `sentence-transformers` and set `EMBEDDING_BACKEND` to improve paraphrase
  matching.
- **Latency is dominated by the LLM** (~60-90s for a full run on gemma2:2b).
  The layout endpoint is code-only and returns in milliseconds.
- **No rendered imagery or photo analysis.** Both need an external vision or
  image API and the corresponding API key; neither is wired up.
- **The 3D view is a viewer, not a photorealistic renderer** — boxes, walls,
  and directional lighting.

---

## Testing

```bash
./venv/bin/python -m pytest              # 330 tests
./venv/bin/python -m pytest tests/test_engine.py -v
```

Fully offline and deterministic: LLM calls are stubbed, the embedder is
hashing, and placement is pure geometry. The tests assert invariants rather
than snapshots — no footprint leaves the room, no two overlap, every pair
respects clearance, every total equals the sum of its lines — so they fail on
real regressions rather than on refactors.

## Cloud deployment

This repository serves its UI and API from FastAPI; it is not a Streamlit
application and cannot be deployed directly to Streamlit Community Cloud.
Render is the recommended next step because it can run the existing app
without rewriting the UI. Connect the GitHub repository to Render and deploy
the included `render.yaml` Blueprint. The app serves its existing UI at `/`
and its API under `/api/v1`.

Ollama is local to its host and is not reachable from Render or Vercel. The
Render Blueprint selects the hosted OpenAI and Anthropic providers. Enter at
least one provider key into Render's secret environment settings when prompted;
never put a live key in this repository or a checked-in `.env` file. Production
defaults to hosted providers; development remains local-first.

Before deploying, configure a hard usage/spending limit and billing alert in
the provider account dashboard. The application can estimate token costs but
cannot create or enforce account-level provider limits.
