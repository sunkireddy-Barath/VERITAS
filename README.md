# VERITAS

**Verified Evolving Reality & Intelligence Tracking System**

> *Don't just retrieve the past. Reconstruct what is true now.*

VERITAS is a temporal, evidence-driven AI system built from scratch — its own
BPE tokenizer, its own transformer, its own retrieval stack, its own evidence
layer. It treats external information as a **continuously changing world state**
rather than as a static pile of documents, and it verifies every factual claim
against dated evidence before it will say it.

---

## The problem, concretely

Ask an ordinary RAG system *"who runs Acme?"* over a corpus containing a 2024
annual report, a 2025 press release and a 2026 filing. All three mention "chief
executive". All three rank highly. The model averages them and answers
confidently — with a name that was correct two years ago.

The root cause is a data-model problem, not a prompting problem. A vector index
stores `(chunk, embedding)`. It has no way to distinguish:

| | |
|---|---|
| **(a)** a source is **wrong** | it contradicts reality |
| **(b)** a source **was right, in 2023** | it contradicts only the present |

VERITAS stores facts on two time axes so it can tell these apart, and refuses
to assert anything it cannot support.

## What it outputs

Not a paragraph — a structure:

```
Answer · Current status · Evidence (claim by claim) · Historical context
Changes (what, when, first reported by whom) · Conflicts · Confidence
Last verified · Unknown
```

Every important claim carries a verdict (`SUPPORTED` / `REFUTED` /
`CONFLICTED` / `TEMPORAL_MISMATCH` / `INSUFFICIENT`), its citations, and the
number of genuinely independent sources behind it. Confidence is always an
evidence-support **label** with its contributing factors — never a truth
percentage. The system is designed so that `"100% true"` is not an expressible
output.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Fetch REAL data (no mock fixtures anywhere in this project)
python scripts/fetch_real_data.py            # corpus + SEC facts + live feeds

# 2. Verify the code is sound
python tests/test_smoke.py                   # 10 correctness tests, ~30s

# 3. Train from scratch, in order (notebooks 01 -> 08)
jupyter lab notebooks/

# 4. Check it against real filings, then run the product
python scripts/verify_real.py                # real-data behaviour checks
python run_veritas.py                        # http://localhost:8000
```

### The real data

| source | what | why |
|---|---|---|
| **SEC EDGAR XBRL** | 1,761 facts, 20 companies | every fact carries a fiscal period **and** a filing date — genuinely bitemporal, plus 61 real restatements |
| **Project Gutenberg** | 10.1 MB public-domain text | pretraining volume, unambiguously redistributable |
| **SEC / WHO / ECB / NASA / arXiv** | 115 live feed items | the continuous-update path against sources that actually move |

Measured on that SEC data: the median gap between a fiscal period ending and the
filing disclosing it is **402 days**. That gap is why one timestamp is not
enough, and it is not visible in any synthetic dataset.

The notebooks are the intended path — they build everything in order and
explain each algorithm before implementing it:

| notebook | phase | what you build |
|---|---|---|
| `01_tokenizer` | 1 | byte-level BPE, incremental merge training, round-trip + compression tests |
| `02_transformer` | 2 | RoPE · GQA · SwiGLU · RMSNorm, with numerical correctness checks |
| `03_pretraining` | 3 | clean → dedup (MinHash+LSH) → memmap shards → AdamW/cosine/bf16 loop |
| `04_instruction_tuning` | 4 | chat format, assistant-only loss masking, **trained abstention** |
| `05_rag_retrieval` | 5–6 | chunking, BM25, InfoNCE embedder, IVF, hybrid fusion, cross-encoder rerank |
| `06_temporal_evidence` | 7–10 | bitemporal store, change detection, evidence graph, claim verification |
| `07_agentic_veritas` | 11–12 | the full loop + live ingestion demo |
| `08_evaluation` | 15 | the benchmark and the baseline ladder |

No pretrained language model is downloaded. The tokenizer, the transformer, the
embedder and the reranker are all trained here from random initialisation.

----

## Code architecture

Eleven packages, each one layer. Dependencies point **downward only** — the
model never imports the agents, the agents never import the API — so any layer
can be tested, replaced or scaled on its own.

```
                       ┌──────────────────────────────────────┐
  interface            │  frontend/  (React, Vercel)          │
                       │  api/       (FastAPI, SSE, voice)    │
                       └───────────────┬──────────────────────┘
                                       │
  orchestration        ┌───────────────▼──────────────────────┐
                       │  veritas/agents/                     │
                       │   planner · researcher · synthesis   │
                       │   orchestrator (the bounded loop)    │
                       └──────┬────────────────────┬──────────┘
                              │                    │
  reasoning        ┌──────────▼─────────┐ ┌────────▼──────────┐
                   │ veritas/evidence/  │ │ veritas/temporal/ │
                   │  claims · graph    │ │  versioning       │
                   │  verifier          │ │  change_detection │
                   │  contradiction     │ │  temporal_retrieval│
                   │  quality·provenance│ │  (bitemporal core) │
                   └──────────┬─────────┘ └────────┬──────────┘
                              │                    │
  retrieval            ┌──────▼────────────────────▼──────┐
                       │  veritas/rag/                    │
                       │   chunking · bm25 · embeddings   │
                       │   hybrid · reranker              │
                       └──────────────┬───────────────────┘
                                      │
  ingestion            ┌──────────────▼───────────────────┐
                       │  veritas/ingest/                 │
                       │   pipeline · streaming (Kafka)   │
                       │   real_sources (SEC, feeds)      │
                       └──────────────┬───────────────────┘
                                      │
  model                ┌──────────────▼───────────────────┐
                       │  veritas/model/    attention·transformer
                       │  veritas/train/    data·trainer·sft
                       │  veritas/tokenizer/ bpe
                       └──────────────────────────────────┘

  measurement          veritas/eval/   benchmark · baselines · metrics
```

| package | responsibility | key entry point |
|---|---|---|
| `tokenizer/` | byte-level BPE, incremental merge training | `BPETokenizer.train` |
| `model/` | RoPE · GQA · SwiGLU · RMSNorm · KV cache | `VeritasLM` |
| `train/` | dedup → memmap shards → AdamW/cosine/bf16; SFT | `train`, `SFTDataset` |
| `rag/` | chunking, BM25, InfoNCE embedder, IVF, fusion, rerank | `HybridRetriever` |
| `temporal/` | **bitemporal store**, change cascade, temporal scoring | `TemporalStore` |
| `evidence/` | claims, graph, verification, contradictions, provenance | `ClaimVerifier` |
| `agents/` | plan → search → assess → refine → verify → answer | `Veritas.answer` |
| `ingest/` | continuous update, Kafka bus, real-source loaders | `IngestionPipeline` |
| `eval/` | TemporalEvidenceBench, ablation ladder, metrics | `compare` |
| `api/` | FastAPI, SSE streaming, structured `Answer` | `api.main:app` |
| `frontend/` | React, voice in/out, claim→evidence tracing | `index.html` |

Two boundaries carry the design:

* **`temporal/` knows nothing about retrieval.** It is a pure bitemporal store,
  so it maps onto PostgreSQL `tstzrange` + GiST without touching callers.
* **`agents/` receives everything by injection.** Each baseline in the
  evaluation ladder is this same pipeline with stages removed — which is what
  makes the comparison an ablation rather than five different systems.

### Data flow of one answer

```
question
  → QueryPlan{entity, attribute, TemporalQuery{intent, anchor, halflife}}
  → List[Candidate]{dense, sparse, freshness, authority, entity, temporal}
  → List[EvidenceItem]{doc_id, text, source, date, tier, valid_from, valid_to}
  → TemporalAssessment{current, historical, changes, fresh, stale}
  → Sufficiency{score, factors, label, gaps, n_independent}
  → List[ClaimVerdict]{claim, verdict, entailment, supporting, refuting}
  → Answer{answer, claims, citations, timeline, conflicts, coverage, trace}
```

Every arrow is a typed dataclass. That is what makes each stage independently
testable and the whole path auditable after the fact.

---

## Deployment

```
Browser ── Vercel (static React) ──HTTPS──▶ Fly.io / Railway (FastAPI + model)
                                                 │
                                 ┌───────────────┼──────────────┐
                             Postgres          Redis          Kafka
                        (tstzrange+pgvector)  (cache)     (optional bus)
```

```bash
docker compose up -d                     # API + Postgres + Redis
docker compose --profile kafka up -d     # + streaming ingestion
```

The backend **cannot** go on Vercel: it is a stateful process holding a model
and indices in memory with a ~25 s startup, which is the opposite of what
serverless is for. Frontend on Vercel, backend on a container host. Kafka is
optional — `make_bus()` returns an in-process queue when no broker is
configured, so the notebooks and tests run with zero infrastructure.

Full guide, including the Postgres schema rationale and the two Kafka settings
that are correctness requirements: **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

--

## The five things that make it different

**1. Bitemporal knowledge, never overwritten.**
Facts are stored with *valid time* (when true in the world) and *transaction
time* (when we learned it). A filing published in March 2026 stating a change
effective January 2026 is representable without lying on either axis. As-of
lookups are a binary search: `O(log n)`.

**2. Change → evidence → current state.**
Ingestion detects what *changed*, not what is present. A cascade of filters
(HTTP validators → BLAKE2b digest → SimHash → claim-level diff) means a
cosmetic edit costs nothing and a state change is recorded with its effective
date, its source, and its predecessor.

**3. Independence is topological.**
Three outlets rewriting one wire story look like three-source corroboration in
a `(claim, source)` table. In the evidence graph they converge on one
`DERIVED_FROM` ancestor, so the corroboration bonus is correctly withheld.
Inflated independence counts are how systems talk themselves into false facts.

**4. Verification is entailment, not similarity.**
*"Acme did **not** appoint Y"* has ~0.95 cosine similarity to *"Acme appointed
Y"*. Symbolic checks (numeric mismatch after unit normalisation, polarity flip,
temporal mismatch) run first and can veto the neural entailment head.

**5. Abstention is trained, not bolted on.**
`<|unknown|>` is a vocabulary token the model is fine-tuned to emit, and the
abstention gate is measured in both directions — correct refusals *and*
over-refusals. A system that always refuses is not a safe system, it is a
useless one.

---

## Evaluation

`TemporalEvidenceBench` — eight categories, each targeting one failure mode:
`CURRENT`, `HISTORICAL`, `CHANGE`, `CONFLICT`, `MULTI_HOP`, `INSUFFICIENT`,
`OUTDATED_SOURCE`, `CROSS_SOURCE`.

Existing QA benchmarks cannot express these: there is no NQ label for *"Person
A was correct in 2024, wrong now"*, none for *"the sources disagree, say so"*,
none for *"abstain"* — and a system quoting a stale source scores identically to
one quoting the current source as long as the string matches.

The comparison is an **ablation ladder** — LLM-only → basic RAG → hybrid RAG →
temporal RAG → VERITAS — sharing one corpus, one tokenizer and one set of
weights, so the only variable is the pipeline. `B4 → VERITAS` is the row that
carries the claim: if the extra machinery does not win on `CONFLICT`,
`INSUFFICIENT` and `OUTDATED_SOURCE`, it is not earning its complexity.

See [docs/evaluation.md](docs/evaluation.md) for how to read the table, and what
the numbers do **not** support.

---

## Honest limitations

* The language model is ~30M parameters. It is not competitive with frontier
  models at generation and is not meant to be. The architecture is deliberately
  built so that the small model is never the thing being trusted: it drafts,
  the verifier decides.
* Claim extraction is rule-based. It is fast, auditable and cannot hallucinate a
  claim that was not in the text — but it misses constructions the rules do not
  cover. This is a precision/recall trade chosen deliberately; see
  [docs/algorithms.md](docs/algorithms.md).
* Entity resolution is alias-table + token overlap. Production would need a
  proper linker.
* The seed benchmark is small. Expand it before quoting any absolute number.
* **The novelty claim is a combination claim, not a "first".** Temporal RAG,
  bitemporal databases, FEVER-style verification and RAG self-critique all
  exist independently. [docs/novelty.md](docs/novelty.md) states precisely what
  exists, what is combined, and what is actually new.

---

## Repository

```
veritas/
  tokenizer/    byte-level BPE (incremental merge training)
  model/        attention.py · transformer.py   RoPE, GQA, SwiGLU, KV cache
  train/        data.py · trainer.py · sft.py
  rag/          chunking · bm25 · embeddings · hybrid · reranker
  temporal/     versioning (bitemporal) · change_detection · temporal_retrieval
  evidence/     claims · graph · quality · verifier · contradiction · provenance
  agents/       planner · researcher · synthesis · orchestrator
  ingest/       pipeline · streaming (Kafka) · real_sources (SEC, feeds)
  eval/         benchmark · baselines · metrics
api/            FastAPI app + served frontend
frontend/       static React for Vercel
notebooks/      01-08, the build order
docs/           RUNBOOK · architecture · algorithms · stack · INNOVATION
                novelty · evaluation · DEPLOYMENT
docker/         Dockerfile.api · Dockerfile.worker · initdb/01_schema.sql
scripts/        fetch_real_data · verify_real · run_eval · run_worker
tests/          11 correctness tests
```

## Verification

```bash
python tests/test_smoke.py       # 11/11 unit + integration
python scripts/verify_real.py    #  7/7 against real SEC filings
python scripts/run_eval.py       #  ablation ladder
python scripts/run_notebook.py notebooks/*.ipynb   # all 8 execute
```

Current results: **11/11** tests, **7/7** real-data checks, **8/8** notebooks,
pretraining val perplexity **36.29** (uniform 4096), abstention **5/5** correct
refusals with **0/5** false refusals.

## Licence

Code: MIT. Bring your own corpus — use only legally redistributable sources.
