# RUNBOOK — train VERITAS from scratch, step by step

Read this with the notebooks open. For each phase: **what you run**, **what is
actually happening**, **what you should see**, and **what it means if you see
something else**.

Nothing here downloads a pretrained model. The tokenizer, the transformer, the
embedder and the reranker are all trained from random initialisation on your
machine.

---

## Before you start

```bash
cd C:\VERITAS
pip install -r requirements.txt
python tests/test_smoke.py          # 10 tests, ~30s — all must pass
jupyter lab notebooks/
```

If a test fails, stop and fix it — every later phase builds on these.

**Optional but strongly recommended:** put a few MB of your own domain text in
`data/raw/corpus.txt` (and/or `data/raw/*.txt`). Every notebook falls back to a
generated stand-in corpus so it runs out of the box, but a real corpus is the
difference between a demo and a model. Use legally redistributable text:
public-domain books, open government data, permissively licensed datasets, your
own documents.

**Hardware.** You have a CUDA GPU, so the defaults are sized for it. Notebook 03
took **468 seconds** for 1500 steps here. On CPU, cut `max_steps` to ~300 and
`d_model` to 128.

---

## Phase 1 — `01_tokenizer.ipynb` (~1 min)

### What you run
Run all cells. The knob is `VOCAB_SIZE = 4096`.

### What is happening
Byte-level BPE is being trained. Your corpus is collapsed into a
`{word: frequency}` dictionary (Zipf's law makes this ~1000× smaller than the
corpus), then the most frequent adjacent byte pair is merged, repeatedly, 3830
times. The speed comes from an inverted index `pair → {word ids}`: each merge
rewrites only the words that contain that pair, instead of rescanning the
corpus.

### What you should see
* Merge log: early merges are character pairs (`'th'`, `'in'`), late merges are
  whole domain words (`' chief executive'`, `' filing'`).
* **All five round-trip cases PASS.** This is non-negotiable — a byte-level
  vocabulary makes `decode(encode(x)) == x` a theorem.
* Compression curve: bytes-per-token rising with vocab size, flattening out.
  Typically 3–4 bytes/token on English at vocab 4096.
* `checkpoints/tokenizer.json` written.

### If something is wrong
* **Round-trip FAIL** → the pre-tokenizer is dropping characters. Run
  `assert_lossless(your_text)`; the regex needs a catch-all alternative. (This
  was a real bug in this repo — see `docs/algorithms.md` §1.)
* **Late merges are still character pairs** → your corpus is too small. The
  merges never reach domain vocabulary.
* **`bytes/token < 2`** → corpus too small or too diverse for the vocab size.

### The question to be able to answer
*"Why byte-level BPE and not word-level?"* → Word-level emits `<unk>` for every
unseen entity name, and in an evidence system entity names **are** the payload.

---

## Phase 2 — `02_transformer.ipynb` (~1 min)

### What you run
Run all cells. Nothing trains here; this is six correctness checks on the
architecture.

### What is happening
You are proving the model is built correctly *before* spending GPU hours on it.

### What you should see
| check | expected |
|---|---|
| fused SDPA vs naive attention | max abs diff `< 1e-5` |
| RoPE relativity | same offset ⇒ same logit at any absolute position |
| **init loss ≈ ln(V)** | `8.31` vs `ln(4096) = 8.32` |
| KV cache vs no cache | **bit-identical** greedy output, plus a large speedup |
| GQA memory | 4× smaller KV cache than MHA |
| gradient norms | same order of magnitude across layers |

### If something is wrong
* **Init loss much below `ln(V)`** → label leakage. The model is seeing the
  target. Stop and fix; this is the cheapest bug-catcher in LM training.
* **Init loss much above `ln(V)`** → broken initialisation (usually a missing
  residual-scale or a wrong `std`).
* **KV cache output differs** → position-indexing bug, almost always the RoPE
  offset when decoding (`cache.pos`).
* **Layer-0 gradients ~0** → the residual path is broken; you are effectively in
  post-norm.

### The question to be able to answer
*"Why divide by √d?"* → `q·k` has variance `d`; unscaled, softmax saturates and
its Jacobian `diag(a) − aaᵀ` goes to zero, so gradients vanish.

---

## Phase 3 — `03_pretraining.ipynb` (~8 min on GPU)

### What you run
Run all cells. Knobs: `SEQ_LEN`, `BATCH`, `ACCUM`, and `max_steps` in
`TrainConfig`.

### What is happening
1. Clean (NFKC) → quality filter → MinHash+LSH dedup.
2. Tokenize everything into one flat `uint16` memmap, EOS-separated.
3. Contiguous tail split into train/val (a *random* split leaks train into val).
4. Train with AdamW + warmup→cosine + gradient accumulation + bf16 + clipping.

### What you should see
* Survival rates printed per stage. Real web data: ~60–80 % survive quality
  filtering, ~50–80 % survive dedup.
* The Chinchilla cell tells you whether you are data-limited (**want ≥ 20 tokens
  per parameter**).
* Training log: `loss` falling fast then slowly; `grad_norm` settling around
  0.2–1.0; `tok/s` steady.
* Val loss tracking train loss.
* `checkpoints/best.pt`, `last.pt`, `final.pt`.

### If something is wrong
* **`cannot mmap an empty file`** → the whole corpus was removed by
  `quality_filter` or `dedupe`. Check the printed survival rates.
* **Loss → NaN** → LR too high, or clipping disabled. Halve `lr`.
* **Loss flat at `ln(V)`** → LR too low, or the targets are misaligned.
* **`grad_norm` pinned at 1.0 forever** → LR too high; the clip is doing all the
  work.
* **Val loss rising while train falls** → overfitting. You are data-limited:
  raise `dropout`, cut `max_steps`, or get more data.

### Reality check
Generation at the end will be fluent-ish nonsense. **That is expected** at this
scale, and it is exactly why VERITAS never lets the model assert a fact
unverified. The model is a drafter; the verifier decides.

---

## Phase 4 — `04_instruction_tuning.ipynb` (~2 min)

### What you run
Run all cells. This loads `checkpoints/best.pt` from Phase 3.

### What is happening
Supervised fine-tuning with **assistant-only loss masking**. The chat format
uses real vocabulary tokens (`<|user|>`, `<|assistant|>`), not strings — a
string costs tokens *and* can be forged by the model mid-answer.

Four behaviours are being installed: answer from evidence, emit `<|claim|>`
spans, emit `<|time|>` qualifiers, and emit `<|unknown|>` to abstain. **Abstention
is trained, not bolted on** — otherwise the model always produces a fluent guess
that the verifier has to delete.

### What you should see
* The mask inspection cell: `0` on every prompt token, `1` on assistant tokens.
  Typically ~15–30 % of positions are scored.
* Masked loss falling.
* The final cell: abstention on the **unanswerable** question, no abstention on
  the answerable one.
* `checkpoints/sft.pt`.

### If something is wrong
* **Mask is 1 on prompt tokens** → you are training the model to generate
  questions. Check `build_example`.
* **Abstains on everything** → too many abstention examples, or LR too high.
  Both directions matter: a model that always refuses is useless, not safe.
* **Never abstains** → too few abstention examples; raise their share above ~25 %.
* **Output degenerates vs Phase 3** → SFT LR too high (catastrophic forgetting).
  It is `5e-5` here, ~6× below pretraining, for exactly this reason.

---

## Phase 5 — `05_rag_retrieval.ipynb` (~2 min)

### What is happening
Chunking with preserved character offsets → BM25 inverted index → dense
embedder trained with InfoNCE → IVF → hybrid fusion → cross-encoder reranker
trained with a pairwise ranking loss.

### What you should see
* Chunks carrying `heading_path` and `(start_char, end_char)` — this is what
  makes a citation point at a span rather than a document.
* **The failure-mode cell**: BM25 wins on the exact rare string (`"Marcus Lund"`),
  dense wins on the paraphrase (`"who runs the company these days"`), and
  overlap is *low*. That low overlap **is** the argument for hybrid retrieval,
  measured rather than asserted.
* Dense recall@3 improving after InfoNCE training.
* RRF and weighted fusion producing different orderings, with per-signal
  explanations.

### If something is wrong
* **Recall does not improve after contrastive training** → too few pairs, or LR
  too high. You need ≥ a few hundred `(query, positive)` pairs for a real gain.
* **BM25 and dense return identical lists** → your corpus is too small for the
  failure modes to separate.

---

## Phase 6 — `06_temporal_evidence.ipynb` (~1 min)

This is the conceptual core. No training; run it and read the outputs.

### What you should see
* The bitemporal store answering **all four** question types, including *"what
  did we believe in 2025?"*.
* Change detection: `identical` and `cosmetic banner` → **not changed**;
  `Priya Raman → Marcus Lund` → **changed, state-change=True**.
* Temporal intent classification, and the per-attribute half-life decay curves.
* **The independence cell**: 4 supporting documents, **2 independent sources**.
  This one number is the clearest single demonstration of why the evidence graph
  exists.
* Verification: numeric mismatch caught (`15` vs `12`), unit normalisation not
  flagged (`1.4 billion` == `1,400 million`), polarity flip caught, temporal
  mismatch reported as `TEMPORAL_MISMATCH` — **not** as `REFUTED`.
* Contradiction: successive CEOs classified `TEMPORAL` (severity 0.1), the
  same-period offices disagreement classified `FACTUAL` (severity 1.0).

### The question to be able to answer
*"Why two time axes?"* → A filing published in March 2026 can state a change
effective January 2026. With one axis you must lie on one of them, and you can
no longer say *when you learned* something — which is the audit trail.

---

## Phase 7 — `07_agentic_veritas.ipynb` (~2 min)

### What you should see
* The trace: `PLAN → SEARCH#1 → SUFFICIENCY → VERIFY → CONTRADICTIONS →
  SYNTHESIS → CITATION_AUDIT → ELAPSED`.
* Full markdown answer with all nine sections.
* Same entity asked three ways (current / historical / change) giving three
  different answers from **the same index**.
* The abstention case refusing, and naming what is missing.
* The conflict case reporting *"15 vs 12, unresolved"*.
* **The live-update demo**: ingest one new filing, the answer changes, weights
  untouched, only affected chunks/timelines/cache entries touched.
* Adaptive polling backing off on a stable source.

This is the demo to show. Run the live-update cell twice so you can narrate it.

---

## Phase 8 — `08_evaluation.ipynb` (~3 min)

### What you should see
The ablation ladder — LLM-only → basic RAG → hybrid RAG → temporal RAG →
VERITAS — plus a per-category breakdown.

Look for the categories where the ladder is **flat until VERITAS**:
`CONFLICT`, `INSUFFICIENT`, `OUTDATED_SOURCE`. Those are the capabilities the
architecture adds. `coverage` and `cite_acc` are structurally `0` for B1–B4 —
they do not verify claims, so there is nothing to measure. That is the point of
the column, not a scoring trick.

### What you must NOT claim
* Any number from a run without a trained checkpoint (with a random encoder,
  dense retrieval is noise and the table swings run to run).
* Any absolute accuracy from the 8-item seed set — expand it first
  (`expand_synthetic`, or hand-write more items).
* *"First"* or *"never been done"*. See `docs/novelty.md`.

---

## Command-line equivalents

```bash
python tests/test_smoke.py                       # correctness gate
python scripts/run_eval.py --synthetic 5 --checkpoint checkpoints/sft.pt
python scripts/run_notebook.py notebooks/*.ipynb # execute all notebooks headless
python scripts/build_notebooks.py                # regenerate notebooks 01-04
cd scripts && python build_notebooks2.py         # regenerate notebooks 05-08
```

`run_notebook.py` runs every code cell top-to-bottom in a clean interpreter.
That is the property that actually breaks when notebooks rot, and it is what CI
should run.

---

## Suggested order for real training

1. Put **≥ 10 MB** of domain text in `data/raw/`.
2. Notebook 01 with `VOCAB_SIZE = 8192`.
3. Notebook 02 — confirm all six checks pass (2 minutes that save hours).
4. Notebook 03 with `max_steps` set so that `steps × batch × accum × seq_len ≈
   20 × params`. Set `compile=True` above ~1000 steps.
5. Notebook 04 — expand the instruction set; keep abstention examples above 25 %.
6. Notebook 05 — mine real `(query, positive)` pairs from your corpus.
7. Notebooks 06–07 — ingest your own sources.
8. Notebook 08 — expand the benchmark with hand-written items in your domain
   before reporting anything.

---

## Where to look when something is confusing

Every module's docstring explains the algorithm and the alternatives it beat.
Start with `docs/algorithms.md`, then read the module itself — they are written
to be read in that order.
