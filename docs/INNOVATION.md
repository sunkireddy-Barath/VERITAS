# Innovation, features, and how VERITAS differs

Everything below is demonstrable on **real data** in this repository — real SEC
EDGAR filings, real live institutional feeds. Nothing here is a mock fixture.

---

## 1. The problem nobody else's data model can express

Two sources disagree. There are two completely different reasons:

| | |
|---|---|
| **(a)** the source is **wrong** | it contradicts reality |
| **(b)** the source **was right, in 2016** | it contradicts only the present |

A vector database stores `(chunk, embedding)`. It has no field in which this
distinction could be written down. So it cannot be computed, no matter how good
the retriever or how large the LLM.

**Real example from this repository's data.** Apple's FY2008 net income appears
in SEC filings as **both** `$4.834B` and `$6.119B`. A normal RAG system has
three options, all wrong: pick one silently, average them, or say "sources
disagree" without knowing why. VERITAS knows why — the 2008 filing said 4.834B,
and a later filing restated the same fiscal period as 6.119B after a revenue
recognition change. It reports that as a **restatement with both values, both
dates, and both filings**.

## 2. The core innovation: two independent time axes

```
Version:  entity · attribute · value
          valid_from ──────── valid_to     when it was true IN THE WORLD
          recorded_at ─────── superseded_at   when WE LEARNED it
```

Most "temporal RAG" work has one timestamp: publication date. That is
`recorded_at`. It cannot represent the gap between when something was true and
when it was disclosed.

**Measured on this repository's real SEC data: the median gap between a fiscal
period ending and the filing that discloses it is 402 days. The maximum is
849 days.** With one axis, more than a year of reality is unrepresentable.

This buys four question types, not one:

| question | resolution |
|---|---|
| "What is Apple's revenue?" | valid = now, transaction = latest |
| "What was it in 2016?" | valid = 2016, transaction = latest |
| "What did we *believe* in 2016?" | valid = 2016, transaction = 2016 |
| "When did we learn?" | read `recorded_at` |

The third and fourth are impossible in any store that overwrites — and the
fourth is the audit trail.

## 3. Fifteen features, and which are genuinely differentiating

| # | feature | status | who else does this |
|---|---|---|---|
| 1 | Bitemporal fact store (valid × transaction time) | ✅ real SEC data | temporal DBs, but not wired to RAG |
| 2 | Restatement detection (`CORRECTED` ≠ `CHANGED`) | ✅ 61 real restatements found | **rare** |
| 3 | `TEMPORAL_MISMATCH` verdict | ✅ | **not in standard NLI vocabularies** |
| 4 | Stale-but-known answers with a validity qualifier | ✅ | rare; most systems answer or refuse |
| 5 | Source independence by graph topology | ✅ | syndication detection exists, not wired to confidence |
| 6 | Evidence sufficiency as agent stopping condition | ✅ | ReAct stops on step budget |
| 7 | Gap-directed query refinement | ✅ | most loops just rephrase |
| 8 | Trained abstention (`<\|unknown\|>` token) | ✅ 5/5 vs 0/5 measured | usually post-hoc filtering |
| 9 | Attribute + period scoped evidence routing | ✅ | **not seen elsewhere** |
| 10 | Claim-level provenance to char offsets | ✅ | ALCE, Self-RAG partially |
| 11 | Change → evidence → state ingestion cascade | ✅ live feeds | crawlers do this, RAG doesn't |
| 12 | Per-attribute freshness half-life | ✅ | usually one global decay |
| 13 | Domain-dependent source tiers | ✅ | usually a fixed list |
| 14 | Synthesis from verified claims only | ✅ | inverted vs standard RAG |
| 15 | TemporalEvidenceBench (8 categories) | ✅ | labels don't exist in NQ/FEVER/TimeQA |

Rows **3, 9 and 15** are where I'd put the strongest novelty claim. Rows 1, 5,
6 and 11 exist elsewhere in isolation; the contribution is the integration.

## 4. Two innovations that came directly out of the real data

These were **not** designed up front. Real SEC filings broke the system, and the
fixes are genuine contributions.

### Attribute-scoped evidence routing

SEC filings state revenue and net income for the same period in the same
sentence template:

```
"Apple Inc. reported revenue of 274.51 billion USD for the fiscal period
 2019-09-29 to 2020-09-26, as disclosed in 10-K filed 2020-10-30."
"Apple Inc. reported net income of 57.41 billion USD for the fiscal period
 2019-09-29 to 2020-09-26, as disclosed in 10-K filed 2020-10-30."
```

Lexical overlap ≈ 0.9. The numbers differ. Every claim-verification system built
on similarity + numeric checking marks these as **mutually refuting** — and
returns `CONFLICTED` for every financial question. VERITAS routes evidence to a
claim only when the **attribute matches**, so a revenue document can never
refute a net-income claim.

### Period-scoped verification

The same template across years means FY2019's revenue "refutes" FY2020's. The
verifier must only see evidence whose **validity interval overlaps the claim's
own**. This is the temporal-vs-factual distinction pushed down from the
contradiction detector into evidence routing — where it actually prevents the
error rather than explaining it afterwards.

Neither bug is visible on synthetic benchmark data, where every document is
about one fact. **They only appear on real filings.** That is the argument for
building on real data.

## 5. How it differs, system by system

| system | what it does | what it cannot do |
|---|---|---|
| **ChatGPT / Claude** | fluent, broad knowledge | no provenance; no notion of when a fact stopped being true; cannot cite the filing |
| **Vanilla RAG** | retrieve top-k → generate | conflates 2016 and 2025 evidence; no abstention; citations attached after the fact |
| **Hybrid RAG** | + BM25 + reranking | better retrieval, same temporal blindness |
| **Temporal RAG** | + recency ranking | prefers *recent*, which is wrong for "in 2016" and destroys change questions |
| **GraphRAG** | entity graph over corpus | graph is topical, not temporal; no valid-time |
| **Self-RAG / CRAG** | self-critique | no time model, so cannot distinguish wrong from outdated |
| **FEVER pipelines** | claim verification | `{SUPPORTED, REFUTED, NEI}` only — no vocabulary for "true, but not now" |
| **VERITAS** | reconstructs state at a point in time from dated evidence, verifies each claim within its period, abstains when evidence is absent | not competitive at open-domain fluency (30M params, by design) |

## 6. The measured evidence

**Ablation ladder** (18-item benchmark, identical corpus/tokenizer/weights,
only the pipeline differs):

| system | accuracy | coverage | cite acc | recall@5 |
|---|---|---|---|---|
| B1 LLM-only | 0.167 | 0.00 | 0.00 | 0.056 |
| B2 Basic RAG | 0.167 | 0.00 | 0.00 | 0.306 |
| B3 Hybrid RAG | 0.222 | 0.00 | 0.00 | 0.861 |
| B4 Temporal RAG | 0.500 | 0.00 | 0.00 | 0.944 |
| **VERITAS** | **0.833** | **0.622** | **0.622** | 0.917 |

Per category, VERITAS scores 1.0 on `CURRENT`, `HISTORICAL`, `CONFLICT`,
`INSUFFICIENT` and `MULTI_HOP` — where every baseline scores 0.0 on the last
three. `coverage` and `cite_acc` are structurally 0 for baselines because they
do not verify claims; that is the point of the column, not a scoring trick.

**Real-data behaviour** (real SEC filings, 20 companies, 1,761 bitemporal
facts):

```
"What was Apple Inc revenue in 2016?"
  → "As of 2016-07-01, Apple Inc's revenue was 215.64 billion USD
     (valid 2015-09-27 to 2016-09-24)."          coverage 1.00, 8 citations

"What is Apple Inc revenue?"
  → "The most recently reported revenue is 416.16 billion USD, covering
     2024-09-29 to 2025-09-27 (reported 2025-10-31 by sec.gov).
     No newer figure has been published."        ← stale, and says so

"What is Apple Inc headcount?"
  → "I cannot establish this from the available evidence."   ← abstains

"What was Apple Inc net income in 2008?"
  → reports the real 4.834B / 6.119B restatement, both filings cited
```

**Pretraining** (real 10.1 MB corpus, from-scratch tokenizer and transformer):
val loss **3.5916**, perplexity **36.29**, against a uniform baseline of 4096.

**Abstention calibration**: 5/5 on unanswerable questions, 0/5 false refusals on
answerable ones.

## 7. What I will not claim

* **Not "the first temporal RAG".** Temporal IR, bitemporal databases,
  FEVER-style verification and RAG self-critique all predate this. The claim is
  *integration* plus the three items in §3 marked as strongest.
* **Not a frontier model.** 14M parameters trained for 8 minutes. It is
  deliberately never the source of truth — it drafts, the verifier decides.
* **Not calibrated probabilities.** The evidence score is a weighted heuristic
  reported as a **label**. `"100% true"` is not an expressible output.
* **Not production-hardened at scale.** In-process indices, no auth, no
  multi-tenancy. See the open items in the handover.

## 8. The one-sentence version

> Most RAG systems retrieve documents and hope the model reads them correctly.
> VERITAS maintains a versioned model of *what was true when*, verifies every
> generated claim against evidence from the same period and about the same
> attribute, distinguishes a wrong source from an outdated one, and refuses to
> answer when the evidence is not there — demonstrated on real SEC filings where
> the gap between a fact becoming true and being disclosed averages 402 days.
