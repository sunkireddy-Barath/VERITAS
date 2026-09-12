# Novelty analysis — what exists, what is combined, what is new

**Rule for this document: never fabricate novelty.** The honest claim is a
*combination* claim plus a benchmark contribution, and it is more defensible
than a "first" claim would be.

---

## 1. What already exists (and must be credited)

| area | prior work | what it already does |
|---|---|---|
| **Bitemporal databases** | Snodgrass (1995); SQL:2011 temporal tables; PostgreSQL `tstzrange` | valid time × transaction time, as-of queries, append-only history. **This is not new. VERITAS applies it to an evidence layer.** |
| **Temporal IR / time-aware RAG** | temporal QA (TempQuestions, TimeQA, StreamingQA), recency-aware ranking, freshness features in web search | retrieval that accounts for document time and question time |
| **Claim verification / fact-checking** | FEVER, SciFact, MultiVerS, RARR, FactScore, Attributed QA | decompose text into atomic claims, retrieve evidence, label SUPPORTED/REFUTED/NEI |
| **Attribution & citation** | ALCE, Self-RAG, Attributed QA, "according to" prompting | answers carrying citations; measuring whether citations support their claims |
| **Self-critique / verification loops** | Self-RAG, CRAG, Chain-of-Verification, RARR | the model checks or revises its own output before emitting it |
| **Agentic retrieval** | ReAct, IRCoT, FLARE, Self-Ask, Search-o1 | iterative retrieve→reason→retrieve loops |
| **Hybrid retrieval** | BM25+dense fusion, RRF (Cormack et al. 2009), ColBERT, SPLADE | sparse+dense complementarity |
| **Abstention / selective prediction** | selective QA, conformal abstention, "I don't know" calibration | refusing when confidence is low |
| **Knowledge graphs + provenance** | PROV-O, W3C provenance, GraphRAG, temporal KGs (ICEWS, Wikidata qualifiers) | entity-attribute-time graphs with sourcing |
| **Change detection / crawling** | SimHash (Charikar 2002), MinHash+LSH, adaptive crawl scheduling | near-duplicate detection, change-rate-driven recrawl |

**Everything in that table is prior art.** A project claiming to have invented
any row of it would be wrong, and a reviewer would know immediately.

---

## 2. Comparison table

| system / pattern | temporal ranking | **bitemporal state** | continuous update | claim verification | evidence graph | **historical reconstruction** | temporal-vs-factual conflict | independence detection | calibrated abstention |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla RAG | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Hybrid RAG (BM25+dense) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Time-aware / recency RAG | ✓ | ✗ | partial | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Self-RAG / CRAG | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | partial |
| RARR / CoVe | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| ALCE / Attributed QA | ✗ | ✗ | ✗ | partial | ✗ | ✗ | ✗ | ✗ | ✗ |
| GraphRAG | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Temporal KGs (Wikidata-style) | ✓ | ✓ | curated | ✗ | ✓ | ✓ | ✗ | ✗ | n/a |
| FEVER-style pipelines | ✗ | ✗ | ✗ | ✓ | partial | ✗ | ✗ | ✗ | ✓ |
| Bitemporal DB (Postgres) | ✓ | ✓ | n/a | ✗ | ✗ | ✓ | ✗ | ✗ | n/a |
| **VERITAS** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Read the table honestly: **every column is ticked somewhere else.** The right-hand
row is not "we invented these"; it is "these are integrated, and the integration
is what is being evaluated".

---

## 3. What VERITAS combines

Four things that are individually known but, as far as this analysis found, are
not usually combined in one system with a measurement:

1. **Bitemporal storage used as the retrieval substrate** — not as an audit log
   bolted on beside the index. The retriever's temporal signal reads the same
   valid intervals the verifier uses to decide `TEMPORAL_MISMATCH`.
2. **Change detection driving knowledge updates rather than re-indexing** — the
   claim-level diff, not the text diff, is what decides a state change.
3. **Claim verification that is temporally aware** — the distinction between
   *"the source is wrong"* (`REFUTED`) and *"the source was right earlier"*
   (`TEMPORAL_MISMATCH`) requires valid intervals, so a FEVER-style pipeline
   without them structurally cannot make it.
4. **Evidence scoring with topological independence** — corroboration counted
   over derivation roots, not documents.

---

## 4. What is arguably new

Stated carefully, with the caveat that exhaustive novelty search across
industry products and patents was **not** performed:

**(a) `TEMPORAL_MISMATCH` as a first-class verification verdict.**
Standard verification vocabularies are `{SUPPORTED, REFUTED, NEI}`. Adding a
verdict meaning *"entailed, but not for the asked-about time"* changes system
behaviour: the answer is given with a validity qualifier instead of being
dropped as unsupported or asserted as current. This needs bitemporal evidence to
compute, which is why the verification literature does not have it.

**(b) Evidence sufficiency as the agentic loop's stopping condition, with
named gaps.** ReAct-style loops stop on a step budget or a model judgement.
Here the planner declares `required_evidence` up front, the evidence agent
returns the specific unmet requirement, and refinement targets *that gap*. The
loop is decidable and its termination reason is auditable.

**(c) Independence as a graph property feeding the confidence score.**
Corroboration counted over `DERIVED_FROM` roots rather than documents, so
syndicated copies cannot inflate confidence. Syndication detection exists in
news analytics; wiring it into an answer's confidence term is the part not
commonly seen.

**(d) TemporalEvidenceBench** — a benchmark whose labels include
`OUTDATED_SOURCE` (must answer *with a staleness qualifier*, so a system that
states the stale value as current fails even though the string matches) and
`CROSS_SOURCE` (syndicated reprints must not count as independent
corroboration). These labels do not exist in NQ/HotpotQA/FEVER/TimeQA, and they
are what makes the failure modes above measurable.

---

## 5. What can actually be demonstrated experimentally

Demonstrable with the code in this repository:

| claim | how it is measured |
|---|---|
| ordinary RAG conflates current and historical state | `CURRENT` vs `HISTORICAL` accuracy across the ablation ladder |
| freshness ranking alone does not fix it | B4 (temporal RAG) vs VERITAS on `CHANGE` and `OUTDATED_SOURCE` |
| conflicting sources are silently merged without verification | `CONFLICT` category; conflict-detection precision **and** recall |
| unsupported answers are produced when evidence is absent | `INSUFFICIENT` category; abstention precision/recall |
| syndication inflates apparent corroboration | `CROSS_SOURCE`; `independent_sources` vs document count |
| new information reaches answers without retraining | `freshness_lag()` — ingest wall-clock plus whether the answer changed |
| claim-level grounding is measurable | evidence coverage, citation accuracy — structurally undefined for the baselines |

**Not** demonstrable, and therefore not to be claimed:

* that VERITAS beats a frontier LLM at open-domain QA — the model is ~30M
  parameters;
* that the evidence score is *calibrated* in a statistical sense — it is a
  weighted heuristic and is reported as a label, not a probability;
* that the architecture scales to 10⁸ documents — it has not been tested there;
* any number from the 8-item seed benchmark treated as a population estimate.

---

## 6. How to phrase it

**Defensible:**

> VERITAS integrates bitemporal knowledge storage, continuous change detection,
> temporally-aware claim verification and provenance-based independence scoring
> into a single retrieval-and-verification pipeline, and introduces a benchmark
> that measures current-vs-historical reasoning, contradiction handling, evidence
> coverage and abstention quality — capabilities that existing QA benchmarks
> cannot express.

**Not defensible:**

> The first system to do temporal RAG / verified retrieval / trustworthy AI.

The first version survives scrutiny and is more impressive for it, because it
shows you know the literature. The second invites one citation that ends the
conversation.

---

## 7. How to keep this document honest

Before putting "novel" anywhere public, re-run the search — arXiv (cs.IR, cs.CL),
ACL/EMNLP/SIGIR/CIKM proceedings, ACM DL, IEEE Xplore, Google Scholar, GitHub,
Papers-with-Code, Google Patents, and the docs of commercial RAG platforms.
Query terms that matter here: *temporal RAG · bitemporal knowledge base LLM ·
claim verification temporal · evidence graph provenance RAG · knowledge
versioning retrieval · stale evidence detection · contradiction detection
retrieval augmented · abstention calibration RAG*.

Update the table in §2 with anything found, and move rows from §4 to §1 when the
prior work exists. A novelty claim that shrinks after a literature search is a
sign the process is working.
