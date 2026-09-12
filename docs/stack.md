# Technology stack — and why each piece fits *this* system

Not "what's popular". Each choice below is justified by a property VERITAS
actually needs.

---

## Backend

### Python 3.12 + PyTorch 2.x — core

The whole point of this project is that the transformer, the tokenizer, the
retrieval scoring and the temporal logic are **visible and modifiable**. PyTorch
is define-by-run: the forward pass is ordinary Python, so `naive_attention` can
sit next to the fused kernel and be asserted numerically equal in a test. A
graph-compiled framework hides exactly what this project exists to show.

Concretely used: `F.scaled_dot_product_attention` (dispatches to FlashAttention
when available), `torch.autocast` for bf16, fused AdamW, `torch.compile` as an
opt-in 1.3–2× speedup, `torch.inference_mode` for generation.

**Why not JAX?** Better for TPU/large-scale parallelism; worse for step-through
debugging and a smaller ecosystem for the retrieval/serving half of the system.
**Why not TensorFlow?** Static-graph heritage works against the pedagogical
goal, and the research ecosystem has moved.

### NumPy — retrieval scoring

BM25 postings are contiguous `int32`/`float32` arrays; scoring is `np.add.at`
over query-term postings only. This is the right tool: the operation is
vectorised gather-accumulate over arrays that fit in cache, and moving it to the
GPU would cost more in transfer than it saves. Embeddings are `float32`
matrices, so exact search is one BLAS GEMM.

### FastAPI — HTTP layer

* **Async.** A VERITAS answer is dominated by I/O waits (source fetches,
  retrieval, model calls). ASGI serves many concurrent investigations on one
  worker; WSGI (Flask) would block a thread per request.
* **Pydantic models = the contract.** The `Answer` object has a rich nested
  shape (claims, verdicts, citations, timeline, factors). Pydantic validates it
  at the boundary and generates OpenAPI automatically, so the frontend types are
  derived, not hand-maintained.
* **Streaming.** `StreamingResponse`/SSE lets the UI show the trace live —
  *plan → search → verify → answer* — which matters because an agentic loop
  takes seconds, and showing the reasoning path is part of the product.

**Why not Django?** Its value is the ORM + admin + batteries; VERITAS's state is
a temporal graph, not CRUD models. **Why not Flask?** Sync-first, and no typed
schema layer.

### PostgreSQL — system of record

Not a preference — a feature match:

* **`tstzrange` + GiST indexes** are native range types. The bitemporal store's
  core query ("the version whose valid interval contains `t`, latest
  transaction") is `WHERE valid_range @> $1` with an index-backed containment
  scan. The in-memory `bisect` implementation is deliberately written behind a
  narrow interface (`as_of`, `history`, `assert_fact`) so it maps onto this
  directly.
* **`EXCLUDE USING gist`** can enforce "no two versions of the same
  (entity, attribute) with overlapping valid time from the same source" as a
  *constraint*, not application logic.
* **JSONB** for evidence payloads that do not deserve columns.
* **`pgvector`** means embeddings can live next to the facts they came from, so
  retrieval and provenance are one transaction and cannot drift apart.
* **ACID.** An ingest that writes a version, a graph edge and an index entry must
  be atomic, or the graph ends up citing a document the store does not have.

**Why not MongoDB?** No range types, no transactional multi-document guarantees
in the shape we need, and temporal containment queries become scans.

### Redis — cache and queue

* **Answer cache with targeted invalidation.** Keyed by `entity+attribute`, so
  an update to Acme's CEO invalidates exactly those answers. Flushing everything
  on each update wastes the cache; never flushing serves stale answers — the one
  thing this system exists to prevent.
* **Sub-millisecond TTL reads** in front of a retrieval path measured in
  hundreds of ms.
* **Streams/lists as the ingest queue**, decoupling polling from processing so a
  slow source cannot stall the pipeline.
* **Distributed locks** so two workers do not ingest the same source twice.

### Vector index: FAISS or pgvector (the code ships its own)

`veritas/rag/embeddings.py` implements exact search, IVF (spherical k-means) and
int8 quantisation from scratch — because understanding the recall/latency knob
matters more than the last 20% of throughput, and because **below ~10⁵ vectors
exact search is a single GEMM and is both faster and exact**. Reach for a
library only when measurement says so:

* **FAISS** — best raw performance, GPU support, HNSW/IVF-PQ. Choose for
  ≥10⁶ vectors in a read-heavy index.
* **pgvector** — one datastore, transactional consistency with the facts,
  filtered search (`WHERE entity = … ORDER BY embedding <-> $1`) which is exactly
  what temporal+entity filtering needs. Choose for consistency over raw speed.
* **Qdrant** — best payload filtering and a real distributed story. Choose when
  metadata filtering dominates.

### NetworkX → Neo4j

NetworkX for research scale: adjacency is `O(1)` dict lookups, zero
infrastructure, and the whole graph fits in memory. Its limit is a single
process and no persistence beyond a JSON dump. The migration path is Neo4j (or
ArangoDB) when provenance graphs exceed memory or need concurrent writers —
which is why `EvidenceGraph` exposes only `add_*` / `evidence_for` /
`provenance_path` / `independent_sources` rather than leaking NetworkX objects.

### Whisper (STT) + Piper/Coqui (TTS)

* **Whisper** — open weights, runs locally (privacy: evidence questions can be
  confidential), robust to accents and noise, and emits word-level timestamps,
  which allows barge-in/interruption handling.
* **`faster-whisper`** (CTranslate2) for ~4× throughput at the same accuracy in
  production.
* **Piper** for TTS — small, fast, fully local, good enough prosody. Coqui XTTS
  when voice quality matters more than latency.

Speech is an **interface** feature. The novelty is the evidence layer; the voice
path simply renders `Answer.to_speech()`, which converts `[E1]` markers into
spoken attributions and states the support level in words rather than reading
brackets aloud.

---

## Frontend

### Next.js (React) + TypeScript

* **Server components / SSR** — an answer page with its evidence is shareable
  and indexable; provenance you cannot link to is much less useful.
* **Streaming UI** — React Suspense + SSE renders the agent trace as it happens,
  which turns a multi-second wait into visible progress.
* **TypeScript end-to-end** — the `Answer` schema is generated from FastAPI's
  OpenAPI, so a backend field rename is a frontend compile error rather than a
  blank panel in production. With a nested claim/verdict/citation structure this
  is worth real money.

**Why not plain React SPA?** No SSR, worse sharing/SEO for evidence pages.
**Why not Streamlit/Gradio?** Excellent for a demo, wrong for a product: no
control over the claim→evidence interaction, which *is* the interface here.

### Visualisation

* **Cytoscape.js** for the provenance graph — built for graphs, handles
  hundreds of nodes with sensible layouts, supports click-to-expand of a
  `claim → evidence → source` path.
* **visx / D3** for the timeline (valid intervals as bars, change events as
  markers) — the timeline is the product's signature view and deserves a custom
  component, not a generic chart library.
* **Tailwind** for styling — the UI is dense and information-first; utility
  classes keep verdict badges, tier chips and freshness indicators consistent
  without a design system.

### The interface's one hard requirement

Every factual sentence must be **click-traceable** to
`claim → evidence span → document → source → date`. That constraint is why the
backend returns a structure rather than a string, and it is why the frontend is
a real application rather than a chat box.

---

## Infrastructure

| piece | why |
|---|---|
| **Docker Compose** | five services (api, worker, postgres, redis, frontend) with one `up`; reproducible for anyone cloning the repo |
| **GitHub Actions** | run `tests/` and `scripts/run_notebook.py` on every push — notebooks rot silently otherwise, and a notebook that does not execute top-to-bottom is worse than no notebook |
| **Prometheus + Grafana** | the metrics that matter here are domain metrics: evidence coverage, abstention rate, freshness lag, conflicts detected per day, ingest queue depth. Latency alone tells you nothing about whether answers are still well-evidenced |
| **APScheduler / Celery beat** | adaptive source polling; Celery when polling must be distributed |
| **structlog** | JSON logs keyed by `trace_id` so an answer's full decision path is queryable after the fact — an audit system whose own decisions are unauditable is a contradiction |

---

## What is deliberately *not* used

| not used | why |
|---|---|
| **LangChain / LlamaIndex** | they abstract precisely the components that are this project's contribution — retrieval scoring, chunking, verification. Using them would leave nothing to explain |
| **A pretrained HF model as the core** | the point is to build the model. External APIs appear only as evaluation baselines or an optional fallback, never as the implementation the project hides behind |
| **sentence-transformers** | same reason: the embedder is trained here with InfoNCE so the objective and its trade-offs are explicit |
| **A managed vector DB (Pinecone etc.)** | fine in production; here it would hide the recall/latency trade-off the notebooks measure |

The rule throughout: **implement what the project claims to understand, use a
library for what it does not claim.** BLAS, regex engines, HTTP servers and
database engines are dependencies. Attention, BPE, BM25, fusion, temporal
scoring, evidence scoring and verification are not.
