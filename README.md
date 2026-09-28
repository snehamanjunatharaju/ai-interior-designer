# InteriorAI

An agentic interior design platform. Takes a client's brief, retrieves
domain knowledge, runs a multi-agent LLM pipeline, self-critiques, and returns
fully-costed design proposals for the Indian market.

Built to demonstrate production AI system design: provider abstraction, RAG,
agentic reflection, semantic caching, resilience patterns, and full
observability — with 97 tests and a reproducible evaluation harness.

---

## Table of contents

- [Start here](#start-here)
- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Design decisions](#design-decisions)
- [Evaluation](#evaluation)
- [Project layout](#project-layout)
- [Deployment](#deployment)
- [Known limitations](#known-limitations)

---

## Start here

If you're reviewing this project, these four places contain almost all of the
engineering signal. Read them in this order:

| # | File | What it demonstrates |
|---|---|---|
| 1 | [`app/services/llm_provider.py`](app/services/llm_provider.py) | **Resilience under failure.** A three-state circuit breaker (closed → open → half-open) with a recovery probe, ordered multi-provider failover, bounded concurrency, token/cost accounting, and defensive JSON recovery for malformed model output. |
| 2 | [`app/agents/orchestrator.py`](app/agents/orchestrator.py) | **Agentic design.** Four specialised agents and a reflection loop that regenerates against critique and keeps the best-scoring attempt. Includes a deterministic guard so the Critic cannot talk its way out of a low score. |
| 3 | [`app/services/evaluation.py`](app/services/evaluation.py) | **Measurability.** LLM-as-judge scoring on a fixed rubric, a golden dataset, and retrieval metrics. This is what makes the system iterable rather than merely impressive. |
| 4 | [Design decisions](#design-decisions) below | **Judgement.** Why the budget arithmetic is computed rather than trusted, why the cache threshold is high, why the default embedder is hashing. The trade-offs, not the features. |

Two supporting pieces worth knowing about:

- [`app/core/observability.py`](app/core/observability.py) — labelled metrics
  registry, Prometheus exposition, and a tracer, with no heavy dependency.
- [`tests/`](tests) — 102 tests, fully offline (LLM calls stubbed, embedder
  deterministic), running in ~4 seconds. They cover circuit-breaker state
  transitions, budget reconciliation, coercion of malformed model output, and
  the cache's contradiction-safety property.

**Try it in 60 seconds:**

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python -m uvicorn app.main:app --port 8001
# open http://localhost:8001  ·  docs at /docs
```

Then:

```bash
curl -X POST http://localhost:8001/api/v1/evaluate/retrieval   # RAG metrics
curl -s http://localhost:8001/api/v1/metrics/prometheus          # live telemetry
```

---

## What it does

The original version of this project was a single prompt sent to a model with
the response rendered as markdown. That is a demo. This version is a system.

**Input:** a structured brief — room type, style, budget tier, dimensions,
occupants, furniture, requirements, Vastu compliance.

**Output:** complete design proposals with layout, itemised furniture with INR
pricing and real brand suggestions, colour palette with hex codes, layered
lighting plan, storage strategy, budget breakdown, Vastu notes, and an
explicit trade-off list (pros *and* cons).

The UI shows the agent pipeline executing live over SSE, so a 60-second request
is visibly a system working rather than a spinner.

---

## Architecture

```
                        ┌──────────────────────────────┐
   POST /generate ──────▶│        API Layer             │
   POST /generate/stream │  auth · rate limit · metrics  │
                        └──────────────┬───────────────┘
                                       │
                        ┌──────────────▼───────────────┐
                        │      DesignService           │
                        │  semantic cache · concurrency │
                        └──────┬───────────────┬───────┘
                               │ miss          │ hit
              ┌────────────────▼──────┐   ┌────▼──────────────┐
              │   RAG Retrieval       │   │  cached response   │
              │  embed → search →     │   └───────────────────┘
              │  lexical rerank       │
              └────────────────┬──────┘
                               │ grounding context
   ┌───────────────────────────▼────────────────────────────────┐
   │                   Multi-Agent Pipeline                     │
   │                                                         │
   │   Planner ──▶ Architect ──▶ Budget ──▶ Critic            │
   │                              ▲              │             │
   │                              └──────────────┘             │
   │                          reflection loop                   │
   │                       (best-of-N revisions)               │
   └───────────────────────────┬─────────────────────────────┘
                               │
              ┌────────────────▼─────────────────┐
              │      LLM Provider Registry       │
              │  circuit breaker · failover       │
              │  Ollama → OpenAI → Anthropic     │
              └──────────────────────────────────┘
```

### The agent pipeline

Four agents, each with a distinct responsibility and its own system prompt:

| Agent | Responsibility | Why it's separate |
|---|---|---|
| **Planner** | Brief → design plan (intent, zones, constraints, risks) | All agents share one agreed interpretation instead of re-parsing the brief |
| **Architect** | The actual design: layout, furniture, palette, lighting | Creative generation isolated from evaluation |
| **Budget** | Prices it, validates arithmetic, enforces the ceiling | Pricing needs low temperature; design needs high |
| **Critic** | Scores quality, identifies specific issues | Evaluation must be independent of generation |

The **reflection loop** is the key pattern: the Critic's verdict feeds back
into the Architect for up to `MAX_REFLECTION_ROUNDS` revisions. The pipeline
keeps the *best-scoring* attempt, not the last one.

A deterministic guard runs alongside it: if the Critic returns an overall score
below 7, revision is triggered automatically even if the model itself says
otherwise. The Critic cannot talk its way out of a bad score.

### Resilience

- **Circuit breaker** per provider — closed → open → half-open, with a probe on
  recovery. A dead provider fails fast instead of adding 60s to every request.
- **Ordered failover** — providers are tried in configured order; a failure on
  the primary is invisible to the user if a secondary is healthy.
- **Bounded concurrency** — semaphores at both the provider and pipeline level
  so a burst of requests cannot exhaust local inference capacity.
- **Degradation over failure** — every agent has a deterministic fallback. If
  the Architect's JSON is malformed, you get a usable design, not a 500.

---

## Quick start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) for local inference (optional — a cloud
  provider also works)

### Run it

```bash
git clone <your-repo>
cd ai-interior-designer

python3 -m venv venv
./venv/bin/pip install -r requirements.txt

cp .env.example .env          # then edit as needed

# Free, local, private inference
ollama serve &
ollama pull gemma2:2b

./venv/bin/python -m uvicorn app.main:app --reload --port 8001
```

Open **http://127.0.0.1:8001** for the UI, or **/docs** for Swagger.

### With Docker

```bash
cp .env.example .env
docker compose up --build
# App: http://localhost:8001  ·  Ollama: http://localhost:11434
```

### Using a cloud provider instead

```bash
export OPENAI_API_KEY=sk-...
export LLM_PROVIDERS=openai
export OPENAI_MODEL=gpt-4o-mini
```

No code change required. Anthropic works the same way.

---

## API reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/generate` | Generate design options (buffered) |
| `POST` | `/api/v1/generate/stream` | Generate with live agent progress (SSE) |
| `POST` | `/api/v1/retrieve` | Semantic search over the knowledge base |
| `GET` | `/api/v1/health` | Health + provider reachability |
| `GET` | `/api/v1/health/ready` | Readiness probe (index loaded) |
| `GET` | `/api/v1/metrics` | Full metrics snapshot (JSON) |
| `GET` | `/api/v1/metrics/prometheus` | Prometheus exposition format |
| `GET` | `/api/v1/traces` | Recent trace spans |
| `GET` | `/api/v1/models` | Available models |
| `GET` | `/api/v1/cache/stats` | Cache hit rate and size |
| `POST` | `/api/v1/cache/clear` | Flush the cache |
| `POST` | `/api/v1/evaluate/retrieval` | RAG quality metrics |
| `POST` | `/api/v1/evaluate/generation` | LLM-as-judge quality scores |

### Generate a design

```bash
curl -X POST http://localhost:8001/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "client_preferences": "Small compact living room in a 2BHK, family of 3",
    "room_type": "living_room",
    "style": "minimalist",
    "budget": "low",
    "room_dimensions": "12ft x 10ft",
    "num_options": 1,
    "specific_requirements": ["reading_nook"],
    "vastu_compliance": "soft"
  }'
```

### Stream progress

```bash
curl -N -X POST http://localhost:8001/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"client_preferences": "modern master bedroom", "room_type": "bedroom"}'
```

```
event: step
data: {"agent":"retriever","status":"ok","message":"Retrieved 4 knowledge passages"}

event: step
data: {"agent":"planner","status":"ok","message":"Planned 4 zones for a living room"}

event: step
data: {"agent":"architect","status":"ok","message":"Designed 'Serene Minimalist Living'"}
...
event: complete
data: {"request_id":"...","options":[...]}
```

### Authentication

Set `API_KEY` in `.env` and pass it on protected endpoints:

```bash
curl -H "X-API-Key: your-secret" http://localhost:8001/api/v1/retrieve \
  -d '{"query":"small budget living room"}'
```

`/health` and `/metrics` stay open so orchestrators can scrape them.

---

## Design decisions

These are the judgement calls, and the reasoning behind them. This section is
the most useful part of the README.

### Budget arithmetic is computed, not trusted

Small models produce line items that don't sum to their stated total. The
Budget agent sums the parts itself and discards the model's headline number
when they disagree by more than 20%. Percentages are recomputed to sum to 100.
This is deterministic — no LLM in the path — because a wrong total is the
fastest way to lose a client's trust.

### The semantic cache threshold is high on purpose

Measured similarity with the default hashing embedder:

| Query pair | Cosine |
|---|---|
| `small living room` ≈ `compact living room` | 0.596 |
| `modern family living room` ≈ `contemporary living area for family` | 0.478 |
| **`tiny living room`** vs **`spacious living room`** | **0.552** |

The antonym pair scores *higher* than a true paraphrase pair. Lowering the
threshold to catch paraphrases would cause the cache to serve a spacious-room
design to someone who asked for a tiny one.

Serving the wrong design is much worse than a cache miss, so the default
threshold is **0.9**, which is effectively exact-match-plus. If you install
`sentence-transformers` (set `EMBEDDING_BACKEND=sentence-transformers`), its
similarity distribution is compressed far higher and better separated — you
can lower the threshold to ~0.8 and get real paraphrase matching. This is
documented rather than hidden because it's a genuine trade-off, not a bug.

### Hashing embeddings as the default

`sentence-transformers` gives better semantic quality but adds a ~90MB model
download and a torch dependency. The hashing embedder (feature hashing +
sublinear TF + synonym expansion + bigrams) needs no model, is deterministic
across processes, and is what makes caching and reproducible evaluation
possible. `get_embedder()` upgrades automatically if the library is present.

### Lexical reranking on top of dense retrieval

Dense retrieval captures meaning but misses exact matches — a brand name, a
measurement. Reranking blends 80% semantic + 20% lexical token overlap. Cheap
and measurably better on the entity-heavy queries this domain produces.

### Prompts require JSON; parsing is defensive

Every agent returns structured JSON validated by Pydantic. But small local
models wrap JSON in prose, fences, or truncate it mid-object, so `extract_json`
applies four recovery strategies (direct parse → fence strip → bare span →
outermost brace span). Agents then coerce through `_to_option`, which handles
`None`, strings where dicts are expected, and non-numeric prices without
crashing.

### Per-agent temperature

The Architect runs at 0.8 (creative divergence), the Budget and Critic at
0.2–0.3 (arithmetic and critique demand consistency). A single global
temperature would be wrong for at least half the pipeline.

### The knowledge base is India-specific

Generic models hallucinate brands and prices for this market. The corpus
covers Vastu, Indian apartment constraints (90cm doors, monsoon drainage, west
facing windows), real brand tiers from Nilkamal to Studio Mudhaliar, and
realistic INR price bands. Grounding in this is why the output is usable
rather than plausible-sounding.

---

## Evaluation

A demo you can't measure is a demo. Two harnesses ship with the project.

**Retrieval quality** — keyword hit rate, MRR, and precision over a golden
dataset of 5 briefs:

```bash
curl -X POST http://localhost:8001/api/v1/evaluate/retrieval
```

**Generation quality** — LLM-as-judge on a 0–10 rubric (specificity,
feasibility, budget realism, coherence, context fit), plus latency and cost:

```bash
curl -X POST 'http://localhost:8001/api/v1/evaluate/generation?num_cases=3'
```

The judge is deliberately harsh and uses a fixed rubric at low temperature, so
scores are comparable across runs and usable as a regression gate in CI.

---

## Project layout

```
app/
├── main.py                    Application factory, middleware, lifespan
├── core/
│   ├── config.py              Env-driven settings (12-factor)
│   ├── schemas.py             Pydantic contracts + domain enums
│   ├── security.py            API-key auth, sliding-window rate limiter
│   └── observability.py       Logging, metrics registry, tracer
├── services/
│   ├── llm_provider.py        Provider ABC, circuit breaker, failover
│   ├── embeddings.py          Hashing + sentence-transformers embedders
│   ├── vector_store.py        NumPy cosine-similarity index
│   ├── knowledge_base.py      India-specific domain corpus
│   ├── retrieval.py           RAG: query building, search, reranking
│   ├── semantic_cache.py      Similarity cache with LRU + TTL
│   ├── design_service.py      Use-case layer (cache + pipeline)
│   └── evaluation.py          LLM judge, golden dataset, metrics
├── agents/
│   └── orchestrator.py        The four agents + reflection loop
└── api/
    └── routes.py              13 versioned endpoints

tests/                        102 tests
static/  templates/           Frontend (no framework, no build step)
docs/sample_output.md         Example generated design content
Dockerfile  docker-compose.yml  .github/workflows/ci.yml
```

### Testing

```bash
./venv/bin/python -m pytest              # 97 passed in ~4s
./venv/bin/python -m pytest -v
```

The suite is fully offline — LLM calls are stubbed, the embedder is
deterministic — so it runs in CI with no model or network. Coverage spans
circuit-breaker state transitions, budget reconciliation, schema coercion of
malformed model output, cache TTL/LRU/contradiction safety, vector-store
persistence round-trips, and every API contract.

---

## Deployment

**Container:** multi-stage build, non-root user, healthcheck, one worker per
container. Scale with replicas, not workers — the vector index and cache are
in-process, so a multi-worker setup would duplicate them.

**For horizontal scale** the interfaces are deliberately swappable: move the
vector store to FAISS/Qdrant, the cache to Redis, and the rate limiter to
Redis — the calling code doesn't change.

**Observability:** point Prometheus at `/api/v1/metrics/prometheus`. Metrics
cover request rate and latency, LLM calls by provider with token and cost
counters, cache hit rate, RAG query scores, per-agent latency, and budget
compliance.

---

## Known limitations

Stated plainly, because a project that hides its edges is less useful than one
that draws them.

1. **Speed on local models.** A 1-option request takes ~60–70s on
   `gemma2:2b` because the pipeline makes 4–8 sequential LLM calls. The
   reflection loop is the main cost. Set `MAX_REFLECTION_ROUNDS=0` to
   disable it, or use a faster/cloud model.

2. **Small models degrade gracefully, not gracefully *well*.** The fallbacks
   keep the system running, but a 2B model scores ~6/10 against the judge. A
   7B+ model or a cloud API is a large quality jump. The architecture is
   model-agnostic precisely so this is a config change.

3. **The vector store is in-memory and per-process.** Correct for a single
   instance; needs Redis/Qdrant to scale horizontally.

4. **The knowledge base is ~22 hand-written chunks.** Enough to demonstrate
   and to meaningfully improve grounding, not a production corpus. The chunk
   format is ready for a larger set.

5. **An API key was previously committed in plaintext.** The original
   `main.py` contained a live bearer token. It has been removed and is now
   read from `IVIS_GATEWAY_API_KEY` in `.env`, which is gitignored. **If that
   key was ever valid, treat it as compromised and rotate it.** The default
   gateway URL was also removed so the repo carries no third-party endpoint
   baked in.

6. **No auth on `/health` and `/metrics` by design** — orchestrators need
   them. Don't expose them publicly without a gateway.

---

## License

MIT
