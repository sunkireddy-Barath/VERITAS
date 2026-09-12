# Architecture

## The core loop

```
OBSERVE → UPDATE → REMEMBER → COMPARE → INVESTIGATE → VERIFY → EXPLAIN → CITE → RECHECK
```

VERITAS is two pipelines meeting at a shared knowledge layer: a **write path**
that continuously ingests the world, and a **read path** that investigates a
question against what has been ingested.

---

## Write path — continuous world update

```
source ──poll──▶ ① HTTP validators (ETag / Last-Modified)      zero bytes
                        │ survived
                 ② BLAKE2b digest of volatility-stripped text  O(n)
                        │ survived
                 ③ SimHash, Hamming ≤ 3 ⇒ cosmetic             one popcount
                        │ survived
                 ④ claim-level diff  ◀── the only stage that may declare
                        │                  a STATE CHANGE
        ┌───────────────┼────────────────┬──────────────────┐
        ▼               ▼                ▼                  ▼
   chunk + index   temporal store   evidence graph    cache invalidation
   (BM25 merge,    (append-only,    (SUPPORTS,        (entity+attribute
    vector append) valid×txn time)   SUPERSEDES,       keyed — targeted,
                                     DERIVED_FROM)     not a flush)
```

**Nothing is rebuilt.** One changed source touches only that document's chunks,
the affected `(entity, attribute)` timelines, and the cache entries for those
entities. A full re-index of a 10⁶-chunk corpus takes hours and cannot run per
poll — which is why most "live" RAG systems are nightly-batch systems wearing a
live badge.

Three mechanisms make the incremental path *correct*:

1. **Chunk-level replacement** — a document's old chunks are removed by id
   before the new ones are added, so a shrinking document leaves no orphan
   chunks still answering queries.
2. **Append-only knowledge** — the store closes valid intervals rather than
   overwriting, so re-ingest cannot destroy history and re-ingesting unchanged
   content is idempotent (it lands as `REAFFIRMED` and only raises
   corroboration).
3. **Targeted invalidation** — `AnswerCache` tracks `(entity, attribute)`
   dependencies per cached answer.

Polling intervals adapt to each source's observed change rate.

---

## Knowledge layer

### Bitemporal store

```
Version:  entity · attribute · value
          valid_from  ───────────── valid_to        (when true in the world)
          recorded_at ───────────── superseded_at   (when we knew it)
          source_id · evidence_ids · confidence · change_kind · previous_value
```

`change_kind ∈ {CREATED, CHANGED, CORRECTED, REAFFIRMED, CONFLICT}`.

* **CHANGED** — the world moved: close the prior valid interval, open a new one.
* **CORRECTED** — we were wrong: close transaction time, keep valid time.
* **REAFFIRMED** — another source restates the current value: widen the interval,
  raise confidence, write no duplicate row.
* **CONFLICT** — same valid time, different value: both rows kept, surfaced.

Versions per `(entity, attribute)` are sorted by `valid_from`; as-of lookup is a
binary search, `O(log n)`.

### Evidence graph

```
nodes  Entity · Claim · Document · Source · Event · TimePoint · Attribute · Answer
edges  SUPPORTS · CONTRADICTS · UPDATED_BY · DERIVED_FROM · VALID_DURING
       ABOUT · SAME_ENTITY · SUPERSEDES · CITES
```

The edge that earns the graph is `DERIVED_FROM`. `independent_sources(claim)`
walks it to each document's **derivation root**, so three syndicated reprints of
one wire story collapse to one independent source. In a flat `(claim, source)`
table they would count as three, and that inflated count is precisely how a
system convinces itself of a false fact.

---

## Read path — the agentic investigation

```
QUESTION
   ▼
PLANNER          entity · attribute · temporal intent · sub-queries
   │              · required_evidence · stopping condition
   ▼
SEARCH  ◀───────────────────────────────────┐
   │  hybrid retrieval over the union of     │
   │  sub-queries: dense + BM25, then        │  refine against the NAMED gap
   │  freshness · authority · entity ·       │  ("prior state missing",
   │  temporal validity, then rerank         │   "no evidence valid now")
   ▼                                         │
TEMPORAL ASSESSMENT                          │
   │  current vs historical state,           │
   │  fresh vs stale evidence, changes       │
   ▼                                         │
EVIDENCE SUFFICIENT?  ──────no───────────────┘   bounded: max_iterations
   │yes                                           + no-progress break
   ▼
CLAIM VERIFICATION     symbolic checks (numeric / polarity / temporal)
   │                   can VETO the neural entailment head
   ▼
CONTRADICTION ANALYSIS  TEMPORAL vs FACTUAL vs GRANULARITY vs SCOPE
   ▼
ABSTENTION GATE  ──yes──▶  "I cannot establish this" + exactly what is missing
   │no
   ▼
SYNTHESIS from VERIFIED claims ─▶ re-verify ─▶ fall back if worse
   ▼
ANSWER  answer · current status · evidence · timeline · changes · conflicts
        confidence (label + factors) · last verified · unknown · trace
```

### Why the loop condition is sufficiency, not a step count

The planner emits `required_evidence` up front ("a source dated after the most
recent known change", "evidence of the prior state", "the effective date"). The
`EvidenceAgent` checks those explicitly and returns **named gaps**, which the
planner turns into the next query. Rephrasing the original question would mostly
return the same chunks — the index already answered that phrasing.

### Why synthesis is inverted

Ordinary RAG: generate prose from context, then hope it is faithful (or attach
citations afterwards, which is how citation-shaped hallucination happens — the
text is fixed first and the citations fitted to it).

VERITAS: retrieve → extract claims → **verify each** → compose only survivors.
The answer cannot contain an unverified claim, by construction rather than by
prompting. The LM's job shrinks from "be truthful" to "be fluent about these
specific sentences" — a job a 30M-parameter model can actually do. Generated
prose is re-verified and discarded if it scores worse than the extractive
baseline.

---

## Agents

Small agents with one responsibility each, rather than one prompt that does
everything — because each then has a typed input/output and can be measured
alone (retrieval recall, temporal accuracy and abstention quality are separate
metrics), and a regression can be localised.

| agent | responsibility |
|---|---|
| `QueryPlanner` | intent, entity, attribute, temporal anchor, sub-queries, stopping condition, gap-directed refinement |
| `RetrievalAgent` | runs sub-queries through hybrid retrieval, dedupes by best score, reranks |
| `TemporalAgent` | current vs historical state, change list, fresh vs stale evidence |
| `EvidenceAgent` | evidence score + support label + **named gaps** — the loop's stopping condition |
| `ClaimVerifier` | symbolic + neural entailment → `SUPPORTED`/`REFUTED`/`CONFLICTED`/`TEMPORAL_MISMATCH`/`INSUFFICIENT` |
| `ContradictionDetector` | classifies disagreement; distinguishes "wrong" from "was right" |
| `AbstentionAgent` | the last gate; calibrated in both directions |
| `SynthesisAgent` | extractive by default, guarded generative mode |
| `CitationAgent` | post-hoc audit: which factual sentences ended up uncited |
| `ProvenanceBuilder` | assembles the `Answer` structure |

---

## Data flow of a single answer

```
question
  → QueryPlan{entity, attribute, TemporalQuery{intent, anchor, halflife}}
  → List[Candidate]{dense, sparse, freshness, authority, entity, temporal, explain}
  → List[EvidenceItem]{doc_id, text, source, date, tier, valid_from, valid_to}
  → TemporalAssessment{current, historical, changes, fresh, stale}
  → Sufficiency{score, factors, label, gaps, n_independent}
  → List[ClaimVerdict]{claim, verdict, entailment, supporting, refuting}
  → List[Conflict]{type, severity, explanation, preference_reason}
  → Answer{answer, current_status, claims, citations, timeline, changes,
           conflicts, support_level, evidence_score, factors, last_verified,
           unknown, abstained, coverage, iterations, trace}
```

Every arrow is a typed dataclass — which is what makes each stage independently
testable and the whole path auditable after the fact.

---

## Deployment shape

```
             ┌── Next.js (SSR + streaming trace) ──┐
             │                                     │
        FastAPI  ──── Redis (cache + queue) ────  Worker pool
             │                                     │  adaptive polling,
             │                                     │  ingest, reverification
             └──── PostgreSQL (tstzrange + GiST, pgvector, JSONB)
                              │
                         Neo4j / NetworkX (evidence graph)
```

`Prometheus` scrapes **domain** metrics, not just latency: evidence coverage,
abstention rate, freshness lag, conflicts detected, ingest queue depth. Latency
alone says nothing about whether answers are still well-evidenced.
