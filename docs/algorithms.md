# Algorithms — what was chosen, and why

Every non-trivial algorithm in VERITAS, with the alternatives it beat and the
cost model that decided it. This is the document to read before an interview.

---

## 1. Byte-level BPE (`veritas/tokenizer/bpe.py`)

**Alternatives.** Word-level (unbounded vocab, `<unk>` on every new entity
name — fatal here, since entity names *are* the payload). Character/byte-level
(no `<unk>`, but ~4× longer sequences against `O(L²)` attention — the most
expensive possible choice). WordPiece (likelihood-based merges, needs an LM
scoring pass). Unigram/SentencePiece (better theory — EM over a token lattice —
but iterative EM to train and Viterbi to encode).

**Choice: BPE.** Greedy frequency merges. Training near-linear with the index
below, encoding is a deterministic merge replay, compression within a few
percent of Unigram. Best accuracy-per-unit-compute, which is why GPT-2/3/4 use
it. *Byte*-level makes the vocabulary closed: `decode(encode(x)) == x` exactly,
for any byte string.

**The optimisation.** The textbook loop is `O(merges × corpus)`. Restructured:

1. Pre-tokenize once → `{word: frequency}`. Zipf collapses 10⁸ characters into
   ~10⁵ distinct words; every later step is paid per *distinct word*.
2. Maintain `pair_counts` plus an inverted index `pair → {word ids}`.
3. Each merge rewrites only the affected words and applies local ± deltas,
   pushed into a lazy-deletion max-heap.

Result: `O(Σ|affected words|)` per merge instead of `O(corpus)`.
Encoding: doubly-linked list + heap keyed by merge rank, `O(L log L)` per word,
with a word-level LRU cache — Zipf means the cache hit rate on real text is very
high.

**Trap that bit us.** `re.findall` returns only what the pattern matches;
anything unmatched is silently **dropped**. The stdlib fallback pattern
(used when the `regex` package is absent) had no catch-all, so CJK and emoji
vanished and round-trip failed. `assert_lossless()` now guards it, and it is a
regression test. The lesson generalises: a pre-tokenizer must *partition* its
input.

---

## 2. Attention: RoPE + GQA + fused SDPA (`veritas/model/attention.py`)

$$A=\mathrm{softmax}\!\left(\tfrac{QK^\top}{\sqrt d}+M\right),\quad O=AV$$

**Why `1/√d`.** With i.i.d. unit-variance entries, `q·k` has variance `d`.
Unscaled, logits grow like `√d`, softmax saturates, and its Jacobian
`diag(a) − aaᵀ` → 0: gradients vanish. The scale pins logit variance at ~1 for
any head size.

**Why causal masking.** `p(x₁..x_L) = Π p(x_t|x_<t)`. `M_ij = −∞` for `j > i`
gives each position exactly its past, so all `L` positions train in parallel
from one forward pass.

**RoPE over learned absolute positions.** Rotation is orthogonal, so
`⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩` — the logit depends only on relative distance.
For VERITAS specifically: evidence chunks can be concatenated in any order
without absolute-position artifacts, context extrapolates past the trained
length, and zero position parameters are learned.

**GQA over full MHA.** The KV cache is `2·n_layers·batch·n_kv·d_head·L·2` bytes
and dominates inference memory. 8 query heads sharing 2 KV heads is a **4×
smaller cache** — that is what fits a long evidence context on one consumer GPU.
K/V are far more redundant across heads than Q, so quality loss is small.

**Fused SDPA.** FlashAttention's tiled online softmax never materialises the
`L×L` matrix: memory `O(L²) → O(L)`, 2–4× faster. `naive_attention` is kept as
the numerical reference and is asserted equal in the tests.

---

## 3. Transformer block (`veritas/model/transformer.py`)

| choice | rejected | reason |
|---|---|---|
| pre-norm | post-norm | leaves an unnormalised identity path from loss to embedding; deep stacks train without warmup tricks |
| RMSNorm | LayerNorm | drops mean-subtraction and bias: ~2× fewer reduction passes, fewer params, equal quality |
| SwiGLU, `d_ff = 8/3·d` | GELU, `4·d` | the gate `W₃x` gives a multiplicative, data-dependent interaction a single-matrix FFN cannot express; the `8/3` width keeps parameter count equal |
| tied embedding/head | separate | they are inverse maps over one vocabulary; saves `V·d` params and regularises |
| residual init `×1/√(2L)` | plain `0.02` | each layer adds into the residual stream; without the scale its variance grows linearly with depth and the logits explode |

RMSNorm computes its reduction in fp32 even under bf16 autocast — `mean(x²)` is
the one place low precision actually bites.

---

## 4. Data pipeline (`veritas/train/data.py`)

* **NFKC normalisation.** The same visible string has several byte encodings
  (full-width digits, ligatures, NBSP). Without it the tokenizer learns
  duplicate merges *and* entity matching in the evidence layer silently fails.
* **MinHash + LSH dedup.** `P(min h(A) = min h(B)) = J(A,B)`; averaging over
  `k` permutations estimates Jaccard with s.e. `~1/√k`. LSH banding (`b` bands
  of `r` rows) gives `P(collide) = 1 − (1−J^r)^b`, an S-curve with threshold
  `≈(1/b)^{1/r}`. Turns `O(N²)` all-pairs into `O(N)` bucket lookups.
* **uint16 memmap + random-offset sampling.** 2 bytes/token; the OS pages it in,
  so RAM is `O(batch)`. Random offsets are `O(1)` and unbiased; a shuffled list
  of all windows is `O(N)` memory and destroys locality.
* **Contiguous tail val split.** A random split puts windows overlapping the
  same documents in both sets — train leaks into val and perplexity looks
  better than it is.

---

## 5. Optimisation (`veritas/train/trainer.py`)

* **AdamW.** Classic L2 folds decay into the gradient, so it gets divided by
  `√v` — high-gradient-variance parameters end up barely regularised. AdamW
  applies `p −= lr·wd·p` separately. Decay **matrices only**: biases and
  RMSNorm gains are 1-D and shrinking them damages the residual scale.
* **Warmup → cosine.** At step 0, Adam's `v ≈ 0` so the effective step is
  enormous; one bad batch destroys the init. Cosine then anneals to `lr/10`,
  and the late small steps do the fine fitting.
* **Gradient accumulation.** Gradient noise `∝ 1/√B`; `B` is capped by VRAM.
  Accumulating `k` micro-batches buys the statistics of `k·B` at the memory of
  `B`.
* **bf16.** fp32's exponent range with fewer mantissa bits — ~2×
  throughput/memory and, unlike fp16, it cannot overflow, so no GradScaler.
* **Clip at global norm 1.0.** Language data is heavy-tailed; a single unusual
  batch can produce a 100× gradient.

---

## 6. Retrieval

### BM25 (`veritas/rag/bm25.py`)

$$\text{score}(D,Q)=\sum_{t\in Q}\text{IDF}(t)\cdot\frac{f(t,D)(k_1+1)}{f(t,D)+k_1\!\left(1-b+b\frac{|D|}{\text{avgdl}}\right)}$$

* IDF's `1 + …` smoothing keeps the value positive; plain RSJ IDF goes negative
  for common terms, letting a stopword *subtract* score.
* `f/(f+k₁)` saturation: the 1st "revenue" is evidence, the 20th is not. Raw TF
  lets one keyword-stuffed page dominate.
* `b·|D|/avgdl`: long documents otherwise accumulate matches by size alone.

Implementation: postings as contiguous numpy arrays; scoring touches only query
terms' postings and accumulates with `np.add.at` — `O(Σ|postings|)`, not
`O(N_docs)`. `finalize()` **merges** new postings rather than rebuilding, which
is what lets the continuous ingest pipeline add one document without an
`O(corpus)` re-index. (IDF is still recomputed — `O(vocabulary)` — because `N`
changed; skipping it leaves every term's IDF quietly wrong.)

### Dense (`veritas/rag/embeddings.py`)

Mean pooling over tokens (not last-token: in a causal model only the last
position has seen everything, but it is also a single position dominated by
whatever token ends the chunk). L2 normalisation makes inner product = cosine,
and since `‖a−b‖² = 2 − 2a·b`, MIPS, cosine and Euclidean k-NN give the *same*
ranking — one normalisation makes the index metric-agnostic and keeps scores in
`[−1,1]`, which is what makes fusion tractable.

Training: InfoNCE with in-batch negatives, `τ≈0.05`. A batch of `B` gives `B−1`
free negatives — the reason contrastive retrieval wants large batches.

Index: exact search is one GEMM and is both fastest *and* exact below ~10⁵
vectors. IVF (spherical k-means, k-means++ seeding) for larger corpora; int8
scalar quantisation gives 4× memory for ~1% recall loss because normalised
components are bounded in `[−1,1]`.

### Why keep both

They fail **differently**. Dense fails on exact strings it never saw — a ticker,
a docket number, a surname, a version string, precisely the tokens that identify
entities. BM25 has no vocabulary problem and needs no training. Hybrid covers
both failure modes; it is not decoration.

### Fusion (`veritas/rag/hybrid.py`)

BM25 is unbounded (0–40), cosine is `[−1,1]`; adding them directly lets BM25
dominate silently.

* **RRF** `Σ w/(K+rank)`, `K=60` — ranks only, so immune to scale, calibration
  drift and outliers. No tuning.
* **Weighted min-max sum** — preserves *margins* (the gap between rank 1 and 2),
  which the evidence-sufficiency check downstream actually reads. Used when the
  consumer reads scores, not just order.

Extra signals (freshness, authority, entity match, temporal validity) are applied
to the union of both backends' `candidate_k` results **before** the top-k cut —
applying them after a tight cut would be pointless, the right evidence would
already be gone.

### Reranking (`veritas/rag/reranker.py`)

A bi-encoder embeds query and document independently — that is what makes it
fast (documents embedded offline) and also its ceiling: the document vector is
computed without ever seeing the query, so negation, date disambiguation and
pronoun binding do not survive the compression. A cross-encoder concatenates
them, at `O(candidates)` forward passes — hence two stages. Trained with
`−log σ(s⁺−s⁻)`: optimising the *margin between* good and bad is the right
objective for ranking; pointwise regression wastes capacity calibrating absolute
scores that only get sorted. Hard negatives are mined from top-ranked non-gold
hits — random negatives are trivially separable and produce no gradient.

---

## 7. Chunking (`veritas/rag/chunking.py`)

Chunking decides what a citation can point at. Three constraints:
self-contained (a chunk read alone must still assert what it asserts), small
enough to be precise (embedding a 2000-token page averages away the one
sentence that matters), offset-preserving (so the UI can highlight the span).

Recursive structural splitting — headings → paragraphs → sentences — with a
token budget and ~15% overlap, plus a heading breadcrumb prepended to what gets
embedded. Structural boundaries correlate with semantic ones for free, far
cheaper than embedding-based semantic chunking (one encoder pass per sentence).

Tables are chunked per row with the header re-attached: flat-text chunking loses
the column→value binding entirely, so the retrieved text says "2026 1,240" with
no way to know 1,240 is revenue.

---

## 8. Bitemporal storage (`veritas/temporal/versioning.py`)

Two independent axes (Snodgrass, 1995): **valid time** `[valid_from, valid_to)`
and **transaction time** `[recorded_at, superseded_at)`. They genuinely come
apart — a March 2026 filing stating a January 2026 effective date — and with one
axis you must choose which lie to tell.

Four question types become answerable, including *"who did we think was CEO in
2024, back in 2024?"* — the audit trail, impossible in any store that overwrites.

A **change** closes valid time and opens a new interval; a **correction** closes
transaction time and keeps valid time. Different operations, both preserving
history. Versions are kept sorted by `valid_from`, so as-of lookup is
`bisect` — `O(log n)`.

Four cases for a second assertion about the same valid time:

| second assertion | kind | effect |
|---|---|---|
| any source, **same** value | `REAFFIRMED` | widens the interval, raises confidence; no duplicate row |
| **same** source, **different** value, filed later | `CORRECTED` | a restatement: stamps `superseded_at` on the original, keeps it |
| **different** source, **different** value | `CONFLICT` | a dispute: both rows stay current and are surfaced |
| any source, later `valid_from` | `CHANGED` | the world moved: closes the old valid interval |

The `CORRECTED` row matters on real data. Apple filed FY2008 net income as
4.83B in October 2009 and restated it to 6.12B in January 2010. Treating that
as a `CONFLICT` answers "sources disagree" about a company correcting itself.
As a correction, the present answer is 6.12B with the original named, and
`as_of(valid=2008, known_at=2009-12-01)` still returns 4.83B. Documents backing
only a superseded version are also withheld from claim verification, so the
original filing cannot "refute" its own restatement.

---

## 9. Change detection (`veritas/temporal/change_detection.py`)

A cascade, cheapest filter first, each stage running only on what survives:

1. HTTP validators (ETag/Last-Modified) — zero bytes transferred.
2. BLAKE2b digest of volatility-stripped text — catches identical content.
3. **SimHash** — catches near-identical content. Projects a document to a 64-bit
   signature where similar documents have small Hamming distance, so comparison
   is one `popcount(a^b)`. MinHash estimates Jaccard better, but here we only
   need a threshold test and one XOR beats 128 array comparisons.
4. **Claim-level diff** — the only stage that may declare a state change. Text
   changing is not news; a *claim* changing is.

Polling interval is derived from the observed change rate: multiplicative
back-off on stable sources, speed-up on volatile ones. A fixed schedule either
hammers static filings or misses a status page that flips in minutes.

---

## 10. Temporal retrieval (`veritas/temporal/temporal_retrieval.py`)

Intent classification (`CURRENT` / `HISTORICAL` / `AS_OF` / `RANGE` / `CHANGE` /
`ATEMPORAL`) is **rule-based on purpose**: the cues are a closed high-precision
set, the decision must be auditable ("why did you treat this as a CURRENT
query?"), and it runs in microseconds on every query.

`CHANGE` deliberately switches freshness weighting **off** — a change question
needs the before *and* the after, so preferring recent evidence destroys the
answer. This is the case ordinary "freshness-aware RAG" gets wrong.

Freshness is exponential decay `0.5^(age/t½)`, not a cutoff: a cliff at "one
year" makes ranking discontinuous and drops a 366-day-old document that is the
only evidence there is. Half-life is **per-attribute** (a share price decays in
hours, a founding date never) and can be re-estimated empirically from observed
interval lengths in the store. One global half-life is the usual mistake and it
degrades accuracy on stable facts.

---

## 11. Evidence graph (`veritas/evidence/graph.py`)

The questions are *path* questions, and the decisive one is **are these sources
independent?** Three outlets rewriting one wire story look like three-source
corroboration in a `(claim, source)` table; in a graph they converge on one
`DERIVED_FROM` ancestor. Independence is topological, and inflated independence
counts are how a system talks itself into a false fact.

`MultiDiGraph`: multi-edge because two nodes can be related in more than one way
(a document can `SUPPORT` a claim and later `CONTRADICT` it via a correction);
directed because provenance has a direction. The interface is deliberately
narrow (`add_*`, `evidence_for`, `provenance_path`) so it can be swapped for
Neo4j without touching callers.

---

## 12. Evidence quality (`veritas/evidence/quality.py`)

Seven bounded factors — authority, recency, independence, directness,
specificity, agreement, validity — weighted, with a **hard floor**: zero
independent sources ⇒ score 0 regardless of everything else. A plain weighted
sum lets five weak signals outvote the absence of evidence, which is the failure
mode of "confidence scores" in most RAG demos.

Independence is log-damped: `1→0.4, 2→0.63, 3→0.77, 5→0.93`. Concave because the
*second* independent source is the big jump (it rules out a single-source
error); the fifth adds little.

Source tiers are **per-domain**, and this is a correctness issue, not a nicety.
For a regulatory question the regulator outranks Reuters; for a software outage
the vendor status page outranks the regulator; for a scientific finding the
peer-reviewed paper outranks the university press release about it. There is no
universal ranking.

---

## 13. Claim extraction (`veritas/evidence/claims.py`)

Rules, not a model, for two reasons: it runs on every ingested document *and*
every generated answer (hottest path in the system), and an LM extractor can
**hallucinate a claim that was never in the text**, silently corrupting the
evidence store — the one failure this project exists to prevent.

Ordering matters: person-valued attributes are resolved *before* the numeric
path, because "Marcus Lund was appointed chief executive effective February
2026" contains a number and a numeric-first rule stores the CEO as `2026`.
Status is categorical, so the numeric path must not run for it at all. Role
words are excluded from the name lexicon, or "…her tenure as CEO…" stores the
CEO as the literal string `"CEO"`. All three were real bugs caught by the
benchmark.

---

## 14. Verification (`veritas/evidence/verifier.py`)

Entailment, not similarity: *"Acme did **not** appoint Y"* has ~0.95 cosine
similarity to *"Acme appointed Y"*.

**Channel 1 — symbolic, runs first, can veto:**
numeric mismatch (after unit normalisation, so `1.4B == 1,400M` but `15 ≠ 12` —
the highest-yield check, since fabricated numbers are the most common and most
damaging RAG error); polarity flip; **temporal mismatch**, which produces
`TEMPORAL_MISMATCH` rather than `REFUTED` — the source is not wrong, it is
stale, and conflating those is the error the whole project exists to avoid.

**Channel 2 — neural NLI** for paraphrase, blended `0.5·lexical + 0.5·neural` so
the rule is a floor the model cannot talk past, and the model cannot be vetoed
by lexical mismatch alone.

Lexical entailment is **asymmetric** on purpose: a long passage covering every
content word of a short claim is strong support; the reverse is not. Symmetric
similarity gets this backwards.

---

## 15. Contradiction analysis (`veritas/evidence/contradiction.py`)

Four kinds, only one of which is a real conflict: `TEMPORAL` (successive
states — output a timeline, not a conflict), `FACTUAL` (same valid time,
different value), `GRANULARITY` (rounding), `UNRESOLVED_SCOPE` (segment vs
group, different currency/period).

Comparability is strict, and every relaxation produced false positives in
practice: same attribute, same subject, extracted value present on both sides,
and a containment check so "construction" vs "begun construction" is phrasing,
not disagreement. A detector that fires on those trains the reader to ignore the
conflicts section — worse than having none.

On a genuine `FACTUAL` conflict VERITAS does **not** pick a winner. A tier-1
source against a tier-4 blog is the one case where a tentative preference is
stated, and even then it is labelled a preference with its reason.

---

## 16. The agentic loop (`veritas/agents/orchestrator.py`)

The loop condition is **evidence sufficiency, not a step count**, and refinement
targets the *named gap* ("prior state missing", "no evidence valid at the
current time") — the index already answered the original phrasing, so
rephrasing mostly returns the same chunks.

It is **bounded**: `max_iterations` plus a no-progress break when an iteration
adds no new documents. Unbounded agent loops turn one question into 200
retrievals.

Synthesis is inverted relative to ordinary RAG: compose **from already-verified
claims**, so the answer cannot contain an unverified claim by construction
rather than by prompting. Generative output is re-verified and discarded if it
scores worse than the extractive baseline — the model never gets the last word.
