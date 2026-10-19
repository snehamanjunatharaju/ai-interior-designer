# InteriorAI

**Turn an Indian home-design brief into catalog-priced room options, a measured
floor plan, rule-checked Vastu notes, and a client-ready PDF.**

Live demo: <https://interiorai-vaa8.onrender.com/>. The public demo is
keyless and uses rule-based catalog mode; it does not claim to be LLM-generated.
The interactive UI is also a better live walkthrough than a prerecorded clip.
**Demo media placeholder:** add a 30-60 second GIF here after capturing a
representative run. Add current UI screenshots alongside it when available.

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
./venv/bin/python -m pytest        # fully offline regression suite
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

## How it works

- **Retrieval (RAG):** hybrid BM25 and dense retrieval feed room- and
  style-filtered design guidance to the agents. General, untagged guidance is
  still eligible; passages tagged for a different selected style are not.
- **Validation:** deterministic checks verify selected products, room
  suitability, special requirements, color scheme, budget, dimensions, and
  directional rules. Blocking failures are shown instead of a green score.
- **Reflection:** the pipeline ranks attempts by validation and critic score,
  keeps the best attempt, and stops once blocking checks pass. In catalog mode
  it applies deterministic repairs and re-validates rather than retrying a
  missing model.
- **Semantic cache:** normalized briefs are compared with a conservative
  similarity threshold to avoid serving a different client's design.
- **Observability:** structured logs, request latency, provider metrics, cache
  counters, and retrieval coverage are exposed through the evaluation and
  Prometheus endpoints.

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
tests/          unit and integration regression suite
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

**Graceful degradation.** When no provider is configured, the app explicitly
reports **Rule-based mode** and builds designs from catalog SKUs, prices,
structured form data, and deterministic layout/rule code. A configured but
unavailable provider is not represented as a successful model call.

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
- **The product catalog is small and curated.** Prices are indicative
  snapshots, not live availability or retailer quotes.
- **Keyless public mode is rule-based.** Prose and option diversity are more
  constrained than with a configured hosted model; heuristic scores are not
  human design evaluations.
- **No photorealistic room rendering, photo analysis, or image redesign.**
  The 2D plan and simple 3D viewer are geometric visualizations, not
  construction drawings or rendered interiors.
- **Rate limits are in-memory per process.** The rolling per-IP minute and
  daily ceilings protect the single-worker public demo but are not a
  distributed abuse-prevention system.

## Evaluation

The following one-run evaluation used the deterministic keyless catalog
pipeline with eight room briefs. All returned options passed blocking
validation (8/8 briefs); the request for three options produced two distinct
options on seven briefs, and the UI reports when a near-duplicate was removed.
Average first-option catalog total was ₹9,606, model cost was ₹0, and mean
end-to-end latency was 18 ms. This is a regression check of rule-based
behavior, not a human evaluation of design quality.

| # | Room | Style / constraints | Valid options (returned / requested) | Catalog total(s) | Latency |
|---|---|---|---|---:|---:|
| 1 | Living room | Modern, Warm Earthy, family / kids-safe | 2/2 (2/2) | ₹17,450; ₹18,150 | 48 ms |
| 2 | Kitchen | Modern, strict Vastu, ventilation | 2/2 (2/3) | ₹17,900; ₹21,950 | 6 ms |
| 3 | Bedroom | Minimalist, storage, budget | 2/2 (2/3) | ₹2,900; ₹6,950 | 23 ms |
| 4 | Kids room | Scandinavian, kids-safe | 2/2 (2/3) | ₹14,500; ₹18,550 | 6 ms |
| 5 | Home office | Modern, workstation | 2/2 (2/3) | ₹11,700; ₹15,750 | 23 ms |
| 6 | Dining | Modern, family | 2/2 (2/3) | ₹7,200; ₹11,250 | 5 ms |
| 7 | Balcony | Bohemian, compact | 2/2 (2/3) | ₹1,600; ₹5,650 | 12 ms |
| 8 | Entrance | Minimalist, shoe storage | 2/2 (2/3) | ₹3,600; ₹7,650 | 24 ms |

---

## Testing

```bash
./venv/bin/python -m pytest              # full suite
./venv/bin/python -m pytest tests/test_engine.py -v
```

Fully offline and deterministic: LLM calls are stubbed, the embedder is
hashing, and placement is pure geometry. The tests assert invariants rather
than snapshots — no footprint leaves the room, no two overlap, every pair
respects clearance, every total equals the sum of its lines — so they fail on
real regressions rather than on refactors.

## License

This project is available under the [MIT License](./LICENSE).

## Cloud deployment

This repository serves its UI and API from FastAPI; it is not a Streamlit
application and cannot be deployed directly to Streamlit Community Cloud.
Render is the recommended next step because it can run the existing app
without rewriting the UI. Connect the GitHub repository to Render and deploy
the included `render.yaml` Blueprint. The app serves its existing UI at `/`
and its API under `/api/v1`.

The included Blueprint uses Render's free web-service plan and needs no
provider key to start. Ollama is local to your computer and is not reachable
from Render. Without a hosted LLM key, the app uses its catalog-driven fallback:
it still returns real catalog items and prices, a computed layout and Vastu
checks, and PDF proposals, but the design prose is generic rather than
LLM-written. To enable AI-generated design prose later, add an OpenAI or
Anthropic key as a Render secret; never put a live key in this repository.

Render's free web services spin down after 15 minutes without traffic and can
take about a minute to wake up. They have limited monthly usage and ephemeral
storage, so this configuration is for a public demo, not a production service.
The application catalog is seeded from source and does not rely on persistent
local storage. If you later add a paid provider key, set a spending limit and
billing alert in that provider's dashboard; the app cannot enforce account
limits.
