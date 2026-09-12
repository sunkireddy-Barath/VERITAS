"""Generate the VERITAS Jupyter notebooks.

Regenerate with:  python scripts/build_notebooks.py
Keeping the sources here (rather than hand-editing .ipynb JSON) means the
notebooks stay diffable in git and cannot drift into an unrunnable state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

OUT = Path(__file__).resolve().parents[1] / "notebooks"

BOOT = """import sys, os, time, math, json, random
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, torch
torch.manual_seed(1337); np.random.seed(1337); random.seed(1337)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE, '| torch', torch.__version__)
if DEVICE == 'cuda':
    print('gpu:', torch.cuda.get_device_name(0),
          f'| {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"""


def nb(cells: List[Tuple[str, str]]) -> dict:
    out = []
    for kind, src in cells:
        lines = src.split("\n")
        body = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": body})
        else:
            out.append({"cell_type": "code", "execution_count": None, "metadata": {},
                        "outputs": [], "source": body})
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def write(name: str, cells: List[Tuple[str, str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print("wrote", name, f"({len(cells)} cells)")


# ===================================================================== 01
N1 = [
("md", """# VERITAS 01 — Byte-level BPE tokenizer from scratch

**Phase 1 of 16.** Everything downstream is measured in tokens, so this is the
first thing to get right.

## Why a tokenizer at all
A neural network consumes vectors of integers. The tokenizer is the map from
text to integers, and it decides three things that no later component can fix:

| choice | consequence |
|---|---|
| vocabulary size | embedding + output-head size (`2·V·d` parameters when tied) |
| compression (bytes/token) | how much text fits in the context window |
| coverage | whether an unseen entity name survives round-trip |

## Why byte-level BPE, specifically
| approach | `<unk>` risk | seq. length | training cost |
|---|---|---|---|
| word-level | **fatal** — every new entity name | shortest | trivial |
| char / byte | none | **~4× longer** → attention is O(L²) | none |
| WordPiece | none | short | needs an LM scoring pass |
| Unigram (SPM) | none | shortest | EM training + Viterbi encoding |
| **byte-level BPE** | **none** | short | greedy merges, near-linear |

For an evidence system the `<unk>` column decides it: entity names *are* the
payload. A tokenizer that mangles "Sollberg" corrupts the citation.

## The algorithm
1. Pre-tokenize with a regex → `{word: frequency}` (Zipf collapses 10⁸ chars to ~10⁵ words).
2. Count adjacent byte pairs, weighted by word frequency.
3. Merge the most frequent pair; **update only the affected words**.
4. Repeat `vocab_size − 256 − |specials|` times.

Naive cost is `O(merges × corpus)`. With the inverted index
`pair → {word ids}` plus a lazy-deletion heap it becomes
`O(Σ|affected words|)` per merge — the optimisation that makes this trainable
in a notebook."""),
("code", BOOT),
("code", """from veritas.tokenizer.bpe import BPETokenizer, SPECIAL_TOKENS, SPLIT_PATTERN
print('special tokens:')
for t in SPECIAL_TOKENS: print('   ', t)
print('\\nThe last four are VERITAS-specific: they make evidence spans, claim spans,')
print('temporal qualifiers and abstention PARSEABLE instead of regex-guessed from prose.')"""),
("md", """## Training corpus

Swap in your own text. Requirements: legally usable, in the target domain, and
large enough that merges are statistically meaningful (≥ a few MB for a real
run). The cell below builds a small domain corpus so the notebook runs in
seconds; the training call is identical for a 1 GB file."""),
("code", """corpus_path = ROOT/'data'/'raw'/'corpus.txt'
if corpus_path.exists():
    texts = [corpus_path.read_text(encoding='utf-8')]
    print(f'loaded {len(texts[0]):,} chars from {corpus_path}')
else:
    from veritas.eval.benchmark import build_seed_benchmark
    bench = build_seed_benchmark()
    seed_texts = [d.text for it in bench.items for d in it.docs] + [it.question for it in bench.items]
    # repetition here only stands in for a real corpus; with real data, don't repeat
    texts = [' '.join(seed_texts) * 40]
    print(f'no data/raw/corpus.txt — using {len(texts[0]):,} chars of seed text')
    print('For a real run: put a few MB of domain text at data/raw/corpus.txt and re-run.')"""),
("code", """VOCAB_SIZE = 4096   # try 2048 / 4096 / 8192 and compare bytes-per-token below

t0 = time.time()
tok = BPETokenizer.train(texts, vocab_size=VOCAB_SIZE, verbose=True)
print(f'\\ntrained in {time.time()-t0:.1f}s | vocab = {tok.vocab_size} | merges = {len(tok.merges)}')"""),
("md", """## Test 1 — lossless round-trip

The property that matters: `decode(encode(x)) == x` for **any** input,
including emoji, CJK and broken bytes. A byte-level vocabulary makes this a
theorem, not a hope — so a failure here means an implementation bug."""),
("code", """cases = [
    'Acme Industries filing: Marcus Lund was appointed chief executive effective February 2026.',
    'Revenue: $1,412,000,000 (≈ €1.31B) — up 12.4% YoY',
    '日本語テスト and emoji 🚀✓ mixed with ASCII',
    '<|evidence|>[E1] source=sec.gov date=2026-02-02<|claim|>CEO is Marcus Lund<|time|>2026-02-01',
    '',
]
for s in cases:
    ids = tok.encode(s)
    ok = tok.decode(ids) == s
    print(f"{'PASS' if ok else 'FAIL'} | {len(ids):3d} tokens | {s[:58]!r}")
assert all(tok.decode(tok.encode(s)) == s for s in cases), 'round-trip must be exact'"""),
("md", """## Test 2 — compression vs vocabulary size

**Bytes per token** is the number to report. More vocabulary means shorter
sequences (cheaper attention, more text per context) but a bigger embedding
matrix. The curve is logarithmic — doubling the vocab buys steadily less."""),
("code", """import matplotlib.pyplot as plt
sample = texts[0][:20000]
sizes, ratios, params = [], [], []
D_MODEL = 384
for v in [512, 1024, 2048, 4096]:
    t = BPETokenizer.train([sample], vocab_size=v, verbose=False)
    sizes.append(t.vocab_size); ratios.append(t.compression_ratio(sample))
    params.append(t.vocab_size * D_MODEL)   # tied embedding + head
    print(f'vocab {t.vocab_size:5d} | {ratios[-1]:.2f} bytes/token | embedding params {params[-1]:,}')

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(sizes, ratios, 'o-'); ax[0].set_xlabel('vocab size'); ax[0].set_ylabel('bytes / token')
ax[0].set_title('compression (higher = shorter sequences)'); ax[0].grid(alpha=.3)
ax[1].plot(sizes, params, 'o-', color='crimson'); ax[1].set_xlabel('vocab size')
ax[1].set_ylabel('embedding params'); ax[1].set_title('cost'); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()"""),
("md", """## Test 3 — encoding speed

Encoding replays merges with a linked list + heap (`O(L log L)` per word) and
an LRU cache at the word level. Because word frequency is Zipfian, the cache
hit rate on real text is very high, so throughput is far above the
cache-cold number."""),
("code", """big = texts[0][:200000]
t0=time.time(); ids = tok.encode(big); cold = time.time()-t0
t0=time.time(); ids = tok.encode(big); warm = time.time()-t0
print(f'cold: {len(big)/cold/1e6:.2f} MB/s | warm (LRU cache): {len(big)/warm/1e6:.2f} MB/s')
print(f'speedup from word-level caching: {cold/warm:.1f}x')"""),
("md", """## Inspect what was learned

Merges are ranked by frequency. Early merges are common character pairs; later
merges are whole words and domain terms. If you do not see domain vocabulary
("chief executive", "filing") in the later merges, your corpus is too small or
off-domain."""),
("code", """by_rank = sorted(tok.merges.items(), key=lambda kv: kv[1])
print('first 15 merges (most frequent pairs):')
for (a,b), r in by_rank[:15]:
    print(f'  {r:4d}: {tok.vocab[a]!r} + {tok.vocab[b]!r} -> {tok.vocab[256+r]!r}')
print('\\nlast 10 merges (rarest / longest):')
for (a,b), r in by_rank[-10:]:
    print(f'  {r:4d}: -> {tok.vocab[256+r]!r}')"""),
("code", """out = ROOT/'checkpoints'/'tokenizer.json'
tok.save(out)
print('saved ->', out)
reloaded = BPETokenizer.load(out)
assert reloaded.encode('Acme 2026') == tok.encode('Acme 2026')
print('reload verified — notebook 02 will use this file')"""),
("md", """## Design trade-offs to be able to defend

1. **Vocab size.** Small vocab → longer sequences → attention cost grows
   quadratically. Large vocab → most of the parameter budget sits in the
   embedding, and rare tokens get too few gradient updates to be learned well.
   4096–8192 is the sweet spot at this model scale.
2. **`min_frequency`.** Raising it drops hapax words and shrinks the merge
   search, at the cost of splitting rare entity names into more pieces.
3. **Pre-tokenization regex.** It forbids merges *across* its boundaries. The
   GPT-4-style pattern keeps a leading space attached to a word (` the`), which
   halves the token count of ordinary prose versus splitting the space off.
4. **Why not `tokenizers`/`sentencepiece`?** They are faster (Rust), and in
   production you would use them. Implementing it makes the merge ordering,
   the cost model, and the `<unk>` argument above yours to explain.

Next: **02 — the Transformer.**"""),
]

# ===================================================================== 02
N2 = [
("md", """# VERITAS 02 — Transformer from scratch

**Phase 2 of 16.** Decoder-only, pre-norm, RoPE + GQA + SwiGLU + RMSNorm.

## Attention, derived

For one head with `Q, K, V ∈ ℝ^{L×d}`:

$$ A = \\mathrm{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d}} + M\\right), \\qquad O = AV $$

**Why `1/√d`?** If `q,k` have i.i.d. unit-variance entries, `q·k = Σᵢ qᵢkᵢ` has
variance `d`. Unscaled, logits grow like `√d`, softmax saturates, and its
Jacobian `diag(a) − aaᵀ` → 0: the gradient vanishes. Dividing by `√d` pins the
logit variance at ~1 for any head size.

**Why the causal mask?** An LM factorises `p(x₁..x_L) = Π p(x_t | x_<t)`.
Setting `M_ij = −∞` for `j > i` gives every position exactly its past, so all
`L` positions train in parallel from one forward pass.

**Why multi-head?** One softmax = one convex combination of value vectors = one
lookup. `h` heads of width `d/h` cost identical FLOPs and give `h` independent
lookups.

## Four architecture choices, and the reason for each

| choice | alternative | why this one |
|---|---|---|
| **RoPE** | learned absolute pos. | rotations are orthogonal ⇒ `⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩`: logits depend only on *relative* distance. Evidence chunks concatenate in any order; context extrapolates; zero position parameters. |
| **GQA** (`n_kv < n_heads`) | full MHA | KV cache = `2·L·n_kv·d_head·n_layers·2 B`. 8 heads → 2 KV heads is a **4× smaller cache**, which is what fits a long evidence context on one GPU. |
| **RMSNorm** | LayerNorm | drops mean-subtraction and bias: ~2× fewer reduction passes, fewer params, equal quality. |
| **SwiGLU** | GELU FFN | a gate `W₃x` multiplies the activation — a data-dependent interaction a single-matrix FFN cannot express. Width `8/3·d` keeps params equal to a `4d` GELU FFN. |

Plus **weight tying** (embedding = output head: inverse maps over the same
vocabulary, saves `V·d` params) and **residual-scaled init** (`1/√(2·n_layers)`
on `wo`/`w2`, or the residual stream's variance grows linearly with depth)."""),
("code", BOOT),
("code", """from veritas.model.attention import build_rope_cache, apply_rope, naive_attention, KVCache
from veritas.model.transformer import ModelConfig, VeritasLM, RMSNorm, SwiGLU
import torch.nn.functional as F"""),
("md", """## Check 1 — the fused kernel computes the textbook equation

`F.scaled_dot_product_attention` dispatches to FlashAttention: tiled online
softmax, so the `L×L` matrix is **never materialised** (memory `O(L²) → O(L)`,
2–4× faster). It must agree with the explicit loop to within float error."""),
("code", """q,k,v = [torch.randn(2, 4, 32, 16) for _ in range(3)]
ref  = naive_attention(q, k, v, causal=True)
fast = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print('max abs diff:', (ref-fast).abs().max().item())
assert torch.allclose(ref, fast, atol=1e-5)
print('PASS — the fused kernel is the same function, computed without the L×L matrix')"""),
("md", """## Check 2 — RoPE really is relative

Rotate a query at position `m` and a key at position `n`. The attention logit
must depend only on `n − m`. This is the property that lets the model handle
evidence pasted in at arbitrary offsets."""),
("code", """cos, sin = build_rope_cache(64, 16)
qv, kv_ = torch.randn(1,1,1,16), torch.randn(1,1,1,16)
def logit(m, n):
    qq = apply_rope(qv, cos[m:m+1], sin[m:m+1])
    kk = apply_rope(kv_, cos[n:n+1], sin[n:n+1])
    return (qq*kk).sum().item()
print(f'(m=2,n=5)  d=3 -> {logit(2,5):+.6f}')
print(f'(m=10,n=13) d=3 -> {logit(10,13):+.6f}')
print(f'(m=30,n=33) d=3 -> {logit(30,33):+.6f}')
print(f'\\n(m=2,n=9)  d=7 -> {logit(2,9):+.6f}   <- different distance, different logit')
assert abs(logit(2,5)-logit(30,33)) < 1e-4
print('\\nPASS — same offset gives the same logit anywhere in the sequence')"""),
("md", """## Check 3 — initial loss must be ≈ ln(V)

At initialisation the model knows nothing, so it should predict uniform over
the vocabulary: loss `= −ln(1/V) = ln V`. A materially lower value at step 0
means label leakage; a much higher value means a broken initialisation. This is
the cheapest bug-catcher in all of LM training — run it before every training
job."""),
("code", """tok_path = ROOT/'checkpoints'/'tokenizer.json'
from veritas.tokenizer.bpe import BPETokenizer
if tok_path.exists():
    tok = BPETokenizer.load(tok_path); V = tok.vocab_size
else:
    V = 4096; tok = None
    print('run notebook 01 first for a real tokenizer; using V=4096 for the shape checks')

cfg = ModelConfig(vocab_size=V, d_model=384, n_layers=8, n_heads=8, n_kv_heads=2, max_seq_len=512)
model = VeritasLM(cfg)
print(f'params: {model.num_params():,} total | {model.num_params(True):,} non-embedding')
print(f'd_ff (SwiGLU, 8/3·d rounded to 64): {cfg.d_ff}')

x = torch.randint(0, V, (4, 128)); y = torch.randint(0, V, (4, 128))
_, loss = model(x, y)
print(f'\\ninit loss {loss.item():.4f} vs ln(V) = {math.log(V):.4f}')
assert abs(loss.item() - math.log(V)) < 0.25
print('PASS')"""),
("md", """## Check 4 — the KV cache does not change the output

Without a cache, generating `T` tokens re-encodes the prefix every step:
`O(T³)` total. With it, each step is `O(T)` → `O(T²)` overall. It is pure
memoisation, so **greedy output must be bit-identical** with and without it.
Any mismatch is a position-indexing bug (usually the RoPE offset)."""),
("code", """model.eval()
prompt = torch.randint(0, V, (2, 8))
with torch.inference_mode():
    t0=time.time(); a = model.generate(prompt, max_new_tokens=48, temperature=0.0, use_cache=True);  t_cache=time.time()-t0
    t0=time.time(); b = model.generate(prompt, max_new_tokens=48, temperature=0.0, use_cache=False); t_nocache=time.time()-t0
print('identical output:', torch.equal(a,b))
print(f'with cache {t_cache:.3f}s | without {t_nocache:.3f}s | speedup {t_nocache/t_cache:.1f}x')
assert torch.equal(a,b)
print('PASS')"""),
("md", """## Check 5 — GQA memory saving, measured

The KV cache dominates inference memory at long context. This is the number to
quote when asked why GQA is in the architecture."""),
("code", """def kv_bytes(cfg, seq, batch=1, dtype_bytes=2):
    return 2*cfg.n_layers*batch*cfg.n_kv_heads*(cfg.d_model//cfg.n_heads)*seq*dtype_bytes
mha = ModelConfig(vocab_size=V, d_model=384, n_layers=8, n_heads=8, n_kv_heads=8, max_seq_len=512)
for seq in (512, 2048, 8192):
    a, b = kv_bytes(mha, seq), kv_bytes(cfg, seq)
    print(f'seq {seq:5d}: MHA {a/1e6:7.1f} MB | GQA(2 kv) {b/1e6:6.1f} MB | {a/b:.0f}x smaller')"""),
("md", """## Check 6 — gradients reach layer 0

Pre-norm leaves an unnormalised identity path from the loss to the embedding.
Print the gradient norm per layer: it should be the same order of magnitude
everywhere. A norm collapsing toward zero at layer 0 means the residual path is
broken (the classic post-norm deep-stack failure)."""),
("code", """model.train(); model.zero_grad()
_, loss = model(x, y); loss.backward()
norms = [ (n, p.grad.norm().item()) for n,p in model.named_parameters()
          if p.grad is not None and 'attn.wq.weight' in n ]
for n, g in norms: print(f'{n:42s} grad-norm {g:.5f}')
vals = [g for _, g in norms]
print(f'\\nratio deepest/shallowest = {max(vals)/max(min(vals),1e-12):.2f}  (want O(1), not 10^k)')"""),
("md", """## Parameter budget

Where the parameters actually go, so you can size the model for your GPU."""),
("code", """def budget(cfg):
    d, V_, L = cfg.d_model, cfg.vocab_size, cfg.n_layers
    hd = d//cfg.n_heads
    emb  = V_*d
    attn = L*(d*cfg.n_heads*hd + 2*d*cfg.n_kv_heads*hd + cfg.n_heads*hd*d)
    ffn  = L*3*d*cfg.d_ff
    norm = L*2*d + d
    return {'embedding (tied w/ head)': emb, 'attention': attn, 'ffn (SwiGLU)': ffn, 'norms': norm}
bud = budget(cfg); tot = sum(bud.values())
for k_, v_ in bud.items(): print(f'{k_:26s} {v_:>11,}  ({100*v_/tot:4.1f}%)')
print(f'{"TOTAL":26s} {tot:>11,}')
print(f'\\nfp32 weights {tot*4/1e6:.0f} MB | AdamW states (m,v) {tot*8/1e6:.0f} MB'
      f' | ≈{tot*16/1e6:.0f} MB before activations')"""),
("md", """Next: **03 — pretraining.**"""),
]

# ===================================================================== 03
N3 = [
("md", """# VERITAS 03 — Pretraining

**Phase 3 of 16.** Data pipeline → training loop → perplexity.

## The data pipeline, and why each stage exists

```
raw text → clean → quality filter → dedup → tokenize → uint16 memmap → batches
```

* **clean** — NFKC normalise. The same visible string has several byte
  encodings (full-width digits, ligatures, NBSP). Without normalisation the
  tokenizer learns separate merges per variant *and* entity matching in the
  evidence layer silently fails.
* **quality filter** — C4/Gopher heuristics. Each rejects a known failure:
  too short (no structure), symbol-heavy (nav bars, base64), degenerate words
  (OCR noise), repetitive (spam).
* **dedup (MinHash + LSH)** — web corpora are 20–50 % duplicated. Duplicates
  waste compute and cause memorisation. MinHash estimates Jaccard in `O(k)`
  per doc; LSH banding turns `O(N²)` all-pairs into `O(N)` bucket lookups.
* **uint16 memmap** — vocab ≤ 65535 ⇒ 2 bytes/token. The OS pages it in on
  demand: RAM is `O(batch)`, start-up is instant, workers share it.
* **random-offset sampling** — `O(1)` per sample and unbiased. A shuffled list
  of every window is `O(N)` memory and destroys locality.

## The optimisation recipe

| component | why |
|---|---|
| **AdamW** | classic L2 gets divided by `√v`, so high-variance params end up barely regularised. AdamW applies `p −= lr·wd·p` separately. Decay **matrices only** — shrinking RMSNorm gains and biases just breaks the residual scale. |
| **warmup → cosine** | at step 0 Adam's `v ≈ 0`, so the effective step is enormous and one bad batch destroys the init. Cosine then anneals to `lr/10`; the late small steps do the fine fitting. |
| **grad accumulation** | gradient noise `∝ 1/√B`, but `B` is capped by VRAM. Accumulating `k` micro-batches gives the statistics of `k·B` at the memory of `B`. |
| **bf16 autocast** | fp32's exponent range with fewer mantissa bits: ~2× throughput/memory, and unlike fp16 it cannot overflow — no GradScaler needed. |
| **clip at norm 1.0** | language data is heavy-tailed; one odd batch can produce a 100× gradient and undo hours of training. |"""),
("code", BOOT),
("code", """from veritas.tokenizer.bpe import BPETokenizer
from veritas.train.data import (clean_text, quality_filter, dedupe, build_shard,
                                split_shard, TokenDataset, MinHashDeduper)
from veritas.train.trainer import TrainConfig, train, cosine_lr, estimate_loss
from veritas.model.transformer import ModelConfig, VeritasLM

tok = BPETokenizer.load(ROOT/'checkpoints'/'tokenizer.json')
print('vocab:', tok.vocab_size)"""),
("md", """## Step 1 — build the document list

Point `docs` at your own corpus. One string per document. Legally usable
sources only: public-domain text, permissively-licensed datasets, your own
documents, open government data."""),
("code", """raw_dir = ROOT/'data'/'raw'
docs = []
for p in sorted(raw_dir.glob('*.txt')):
    docs.extend([d for d in p.read_text(encoding='utf-8').split('\\n\\n\\n') if d.strip()])
if not docs:
    # Stand-in corpus: varied enough to survive dedup and to give the model
    # some real structure to learn. Every document is distinct.
    from veritas.eval.benchmark import build_seed_benchmark, expand_synthetic
    bench = expand_synthetic(build_seed_benchmark(), 60)
    base = [d.text for it in bench.items for d in it.docs]
    rng = random.Random(0)
    firms = ['Cobalt Works','Vireo Health','Northwind Rail','Stellar Foods','Kestrel Bank',
             'Orion Systems','Helios Energy','Meridian Port','Nova Logistics','Acme Industries']
    people = ['Alicia Moreau','Tomas Berg','Hana Suzuki','Emeka Obi','Lena Petrov','Raj Malhotra']
    attrs = ['revenue','headcount','valuation','operating margin','order backlog']
    docs = list(base)
    for i in range(3000):
        f, p_, a = rng.choice(firms), rng.choice(people), rng.choice(attrs)
        y = rng.randint(2019, 2027); v = rng.randint(100, 9999)
        docs.append(
            f'{f} annual report {y}. The chief executive is {p_}, appointed in {y-1}. '
            f'The company reported {a} of {v/10:.1f} million euros for the {y} fiscal year, '
            f'compared with {v/11:.1f} million in {y-1}. Operations expanded to {rng.randint(2,40)} '
            f'sites. The board confirmed the figures at its meeting in {rng.choice(["March","June","September"])} {y}.')
    print(f'no data/raw/*.txt — generated {len(docs):,} distinct stand-in documents.')
    print('REPLACE THIS with a real corpus before reporting any perplexity number.')
print(f'{len(docs):,} documents | {sum(len(d) for d in docs):,} chars')"""),
("md", """## Step 2 — clean, filter, dedup

Watch the survival rate at each stage. On real web data expect roughly
60–80 % surviving the quality filter and 50–80 % surviving dedup. If dedup
removes almost nothing, your corpus is probably already curated; if it removes
almost everything, check that you are not feeding the same document repeatedly."""),
("code", """t0 = time.time()
cleaned = [clean_text(d) for d in docs]
filtered = [d for d in cleaned if quality_filter(d, min_chars=80)]
deduped = list(dedupe(filtered))
print(f'raw       {len(docs):>7,}')
print(f'cleaned   {len(cleaned):>7,}')
print(f'filtered  {len(filtered):>7,}  ({100*len(filtered)/max(1,len(cleaned)):.0f}% survive)')
print(f'deduped   {len(deduped):>7,}  ({100*len(deduped)/max(1,len(filtered)):.0f}% survive)')
print(f'{time.time()-t0:.1f}s')"""),
("code", """# How MinHash+LSH decides: near-duplicates collide in at least one band.
d = MinHashDeduper(num_perm=128, bands=16)
a = 'Acme Industries reported revenue of 1.2 billion euros for the 2025 fiscal year.'
print('original      ->', d.add(0, a))
print('exact copy    ->', d.add(1, a))
print('minor edit    ->', d.add(2, a.replace('1.2', '1.2 ')))
print('genuinely new ->', d.add(3, 'Nova Logistics opened twelve offices across India in 2026.'))
print('\\nP(collide) = 1-(1-J^r)^b  with b=16,r=8 -> threshold ≈ (1/16)^(1/8) ≈ 0.71 Jaccard')"""),
("md", """## Step 3 — tokenize into a memmapped shard

Documents are concatenated EOS-separated into one flat `uint16` stream rather
than padded to a fixed length: padding a 512-token window to fit a 40-token
document wastes >90 % of the FLOPs."""),
("code", """shard = ROOT/'data'/'processed'/'train_full.bin'
t0 = time.time()
n_tokens = build_shard(deduped, tok, shard)
print(f'{n_tokens:,} tokens in {time.time()-t0:.1f}s -> {shard} ({shard.stat().st_size/1e6:.1f} MB)')
n_train, n_val = split_shard(shard, shard.with_name('train.bin'), shard.with_name('val.bin'), val_frac=0.02)
print(f'train {n_train:,} | val {n_val:,}  (contiguous tail split — a random split leaks)')"""),
("md", """### Sizing the run (Chinchilla)

Compute-optimal is roughly **20 tokens per parameter**. Below that the model is
data-starved and will overfit; far above it you are spending compute that a
bigger model would use better. Use this to pick `max_steps`."""),
("code", """SEQ_LEN, BATCH, ACCUM = 256, 16, 4
cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=384, n_layers=8, n_heads=8,
                  n_kv_heads=2, max_seq_len=SEQ_LEN, dropout=0.0)
model = VeritasLM(cfg)
P = model.num_params()
tokens_per_step = BATCH*ACCUM*SEQ_LEN
print(f'params {P:,} | Chinchilla-optimal ≈ {20*P:,} training tokens')
print(f'available: {n_train:,} tokens ({n_train/P:.1f} tokens/param)')
print(f'tokens/step {tokens_per_step:,} | one epoch = {n_train//tokens_per_step:,} steps')
if n_train < 20*P:
    print('\\nData-limited: use dropout > 0, fewer steps, or a smaller d_model/n_layers.')"""),
("md", """## Step 4 — train

Watch three things:
1. **loss** falls fast then slowly — a flat line means the LR is too low or the
   data is broken; a spike to NaN means it is too high.
2. **grad_norm** stabilises around 0.2–1.0. Persistently pinned at the clip
   value means the LR is too high.
3. **val loss** tracks train loss. Divergence = overfitting → raise dropout or
   get more data."""),
("code", """train_ds = TokenDataset(shard.with_name('train.bin'), SEQ_LEN)
val_ds   = TokenDataset(shard.with_name('val.bin'),   SEQ_LEN)
print(f'train windows {len(train_ds):,} | val windows {len(val_ds):,}')

tcfg = TrainConfig(
    max_steps=1500, batch_size=BATCH, grad_accum=ACCUM, seq_len=SEQ_LEN,
    lr=3e-4, warmup_steps=100, weight_decay=0.1, grad_clip=1.0,
    eval_every=250, eval_iters=20, log_every=25,
    ckpt_dir=str(ROOT/'checkpoints'), device=DEVICE,
    dtype='bfloat16' if DEVICE=='cuda' else 'float32', compile=False,
)
# Plot the schedule before committing GPU hours to it.
import matplotlib.pyplot as plt
plt.figure(figsize=(7,2.5))
plt.plot([cosine_lr(s, tcfg) for s in range(tcfg.max_steps)])
plt.title('learning rate: linear warmup → cosine decay to lr/10')
plt.xlabel('step'); plt.ylabel('lr'); plt.grid(alpha=.3); plt.show()"""),
("code", """state = train(model, train_ds, val_ds, tcfg)
print('\\ndone. checkpoints:', sorted(p.name for p in (ROOT/'checkpoints').glob('*.pt')))"""),
("md", """## Step 5 — perplexity and the loss curve

**Perplexity = exp(loss)**: the effective number of tokens the model is
choosing between at each step. `ppl = V` means "no better than uniform";
halving perplexity means the model has genuinely halved its uncertainty."""),
("code", """hist = [h for h in state.history if 'loss' in h]
vhist = [h for h in state.history if 'val_loss' in h]
fig, ax = plt.subplots(1,2, figsize=(12,3.5))
ax[0].plot([h['step'] for h in hist], [h['loss'] for h in hist], label='train')
if vhist: ax[0].plot([h['step'] for h in vhist], [h['val_loss'] for h in vhist], 'o-', label='val')
ax[0].axhline(math.log(tok.vocab_size), ls='--', c='grey', label='ln(V) = uniform')
ax[0].set_xlabel('step'); ax[0].set_ylabel('cross-entropy'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].semilogy([h['step'] for h in hist], [h['ppl'] for h in hist])
ax[1].set_xlabel('step'); ax[1].set_ylabel('perplexity (log)'); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
final = estimate_loss(model, val_ds, tcfg, 30)
print(f'final val loss {final:.4f} | perplexity {math.exp(final):.2f} | uniform would be {tok.vocab_size}')"""),
("code", """model.eval()
prompt = 'Acme Industries'
ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
with torch.inference_mode():
    out = model.generate(ids, max_new_tokens=60, temperature=0.8, top_k=40,
                         eos_id=tok.special_tokens['<|eos|>'])
print(tok.decode(out[0].tolist(), skip_special=True))
print('\\nA small model trained on a small corpus produces fluent-ish nonsense.')
print('That is expected and is exactly why VERITAS never lets it assert facts')
print('unverified — see notebook 07.')"""),
("md", """## Trade-offs

* **`torch.compile=True`** gives ~1.3–2×, at a slow first step. Worth it above
  ~1000 steps.
* **Bigger `seq_len`** helps long-range modelling but attention is `O(L²)` —
  cost quadruples when you double it.
* **`beta2 = 0.95`** (not 0.999): faster adaptation to the shifting gradient
  statistics of a short LM run.
* **`dropout = 0`** while tokens ≫ params. Turn it on only when val loss
  diverges from train loss.

Next: **04 — instruction tuning.**"""),
]

# ===================================================================== 04
N4 = [
("md", """# VERITAS 04 — Instruction tuning (SFT)

**Phase 4 of 16.** Teach the pretrained model to follow instructions — and, for
this project specifically, to **ground, cite, qualify in time, and abstain**.

## Chat format uses real vocabulary tokens

```
<|bos|><|system|>…<|user|>…<|assistant|>…<|eos|>
```

Not the string `"### Assistant:"`, because (a) a string costs 4–6 tokens per
turn, and (b) the model can *generate* a convincing fake role header mid-answer.
A dedicated id cannot be confused with content.

## Assistant-only loss masking

Training on the prompt tokens teaches the model to **generate questions** —
wasted capacity, and it raises the probability of the model continuing with a
fabricated user turn. We zero the loss on system/user tokens and score only the
assistant span (plus `<|eos|>`, so the model learns *when to stop*).

## The four behaviours this stage must install

1. answer strictly from the `<|evidence|>` block;
2. emit `<|claim|>` spans so the verifier can align claims to evidence;
3. emit `<|time|>` qualifiers when a fact is time-bounded;
4. emit `<|unknown|>` when the evidence does not support an answer.

(4) is the important one: **abstention must be trained, not bolted on**.
Otherwise the model always produces a fluent guess that the verifier then has
to delete, and the deletion shows up to the user as an empty answer."""),
("code", BOOT),
("code", """from veritas.tokenizer.bpe import BPETokenizer
from veritas.model.transformer import VeritasLM, ModelConfig
from veritas.train.sft import Turn, SFTDataset, build_example, render, evidence_prompt, load_jsonl
tok = BPETokenizer.load(ROOT/'checkpoints'/'tokenizer.json')
ckpt = ROOT/'checkpoints'/'best.pt'
model = VeritasLM.load(ckpt, DEVICE) if ckpt.exists() else VeritasLM(
    ModelConfig(vocab_size=tok.vocab_size, d_model=384, n_layers=8, n_heads=8,
                n_kv_heads=2, max_seq_len=256)).to(DEVICE)
print('loaded pretrained' if ckpt.exists() else 'NO PRETRAINED CHECKPOINT — run notebook 03 first')"""),
("md", """## Build the instruction set

Mix curated examples with *controlled* synthetic ones. Never rely entirely on
generated data: a model trained only on another model's output inherits its
failure modes, including its hallucinations — precisely what this project
exists to prevent.

Each template below teaches one required behaviour."""),
("code", """# Compact renderer. `evidence_prompt` (used at inference) is verbose because it
# helps a weak model; for SFT the token budget is scarce, and a verbose template
# silently pushes long examples over MAX_LEN where they are DROPPED -- which
# preferentially kills the abstention examples (they carry two evidence blocks)
# and produces a model that never abstains. Measure the drop rate, always.
NL = chr(10)

def ev_prompt(question, evidence):
    lines = ['<|evidence|>[E%d] %s %s' % (i, e['source'], e['date']) + NL + e['text']
             for i, e in enumerate(evidence, 1)]
    lines.append('Q: ' + question)
    return NL.join(lines)

def grounded(question, evidence, answer):
    return [Turn('system', 'Answer only from evidence. Cite [E#]. Abstain if insufficient.'),
            Turn('user', ev_prompt(question, evidence)),
            Turn('assistant', answer)]

examples = []

# 1. grounded answer with a citation
examples.append(grounded(
    'Who is the CEO of Acme Industries?',
    [{'text':'Acme Industries filing: Marcus Lund was appointed chief executive effective February 2026.',
      'source':'sec.gov','date':'2026-02-02','tier':1}],
    '<|claim|>Acme Industries\\'s chief executive is Marcus Lund [E1].<|time|>valid from 2026-02-01.'))

# 2. ABSTENTION — evidence about the entity, but not about the attribute
examples.append(grounded(
    "What is Nova Logistics' 2027 revenue guidance?",
    [{'text':'Nova Logistics opened 12 offices in India in 2026.','source':'reuters.com','date':'2026-08-14','tier':2}],
    '<|unknown|>The evidence covers office openings, not 2027 revenue guidance. '
    'I cannot establish this from the available evidence.'))

# 3. CONFLICT — two credible sources disagree; report, do not pick
examples.append(grounded(
    'How many offices did Nova Logistics open in India in 2026?',
    [{'text':'Nova Logistics said it opened 15 new offices in India during 2026.','source':'ir.novalogistics.com','date':'2026-08-01','tier':1},
     {'text':'Nova Logistics opened 12 offices in India in 2026, according to filings.','source':'reuters.com','date':'2026-08-14','tier':2}],
    'Sources disagree: [E1] reports 15 offices, [E2] reports 12. '
    '<|unknown|>The evidence is insufficient to establish the exact number.'))

# 4. STALE EVIDENCE — answer, but qualify the validity window
examples.append(grounded(
    'What is the current status of the Meridian Port expansion?',
    [{'text':'The Meridian Port expansion is in the planning phase, with approval expected in 2024.','source':'gov.example.gov','date':'2023-05-01','tier':1}],
    '<|claim|>The Meridian Port expansion was in the planning phase [E1].'
    '<|time|>valid as of May 2023. No more recent evidence confirms the current status.'))

# 5. HISTORICAL vs CURRENT — the model must honour the tense of the question
examples.append(grounded(
    'Who was the CEO of Acme Industries in 2024?',
    [{'text':'Acme Industries annual report 2024. The chief executive is Dana Whitfield.','source':'ir.acme.com','date':'2024-03-01','tier':1},
     {'text':'Acme Industries filing: Marcus Lund was appointed chief executive effective February 2026.','source':'sec.gov','date':'2026-02-02','tier':1}],
    '<|claim|>In 2024 the chief executive of Acme Industries was Dana Whitfield [E1].'
    '<|time|>valid 2024. The current chief executive is Marcus Lund [E2].'))

print(f'{len(examples)} curated templates')
print('\\n--- rendered example ---')
print(render(examples[1])[:400])"""),
("md", """### Scale it with controlled substitution

Only names, dates and values vary; the *behaviour* being taught stays fixed.
That buys volume without importing a generator model's errors."""),
("code", """firms = ['Cobalt Works','Vireo Health','Northwind Rail','Stellar Foods','Kestrel Bank','Orion Systems']
people = ['Alicia Moreau','Tomas Berg','Hana Suzuki','Emeka Obi','Lena Petrov','Raj Malhotra']
rng = random.Random(0)
synthetic = []
for _ in range(400):
    f, p = rng.choice(firms), rng.choice(people)
    yr = rng.choice([2023,2024,2025,2026])
    kind = rng.random()
    if kind < 0.45:
        synthetic.append(grounded(f'Who is the CEO of {f}?',
            [{'text':f'{f} filing: {p} was appointed chief executive effective January {yr}.',
              'source':'sec.gov','date':f'{yr}-01-15','tier':1}],
            f'<|claim|>{f}\\'s chief executive is {p} [E1].<|time|>valid from January {yr}.'))
    elif kind < 0.75:
        synthetic.append(grounded(f'What is {f}\\'s current headcount?',
            [{'text':f'{f} filing: {p} was appointed chief executive effective January {yr}.',
              'source':'sec.gov','date':f'{yr}-01-15','tier':1}],
            '<|unknown|>The evidence does not state headcount. I cannot establish this.'))
    else:
        a, b = rng.randint(10,40), rng.randint(10,40)
        synthetic.append(grounded(f'How many sites does {f} operate?',
            [{'text':f'{f} said it operates {a} sites.','source':f'ir.{f.split()[0].lower()}.com','date':f'{yr}-06-01','tier':1},
             {'text':f'{f} operates {b} sites, according to filings.','source':'reuters.com','date':f'{yr}-06-10','tier':2}],
            f'Sources disagree: [E1] reports {a}, [E2] reports {b}. '
            f'<|unknown|>The evidence is insufficient to establish the exact number.'))

conversations = examples*20 + synthetic
rng.shuffle(conversations)
print(f'{len(conversations)} conversations '
      f'({sum("<|unknown|>" in t[-1].content for t in conversations)} teach abstention)')"""),
("md", """## Encode with the loss mask

Inspect the mask before training. Green (1) positions contribute to the loss;
everything in the prompt must be 0."""),
("code", """MAX_LEN = model.cfg.max_seq_len    # must match what the model was pretrained with
encoded, kept_abstain, total_abstain = [], 0, 0
for c in conversations:
    is_abstain = '<|unknown|>' in c[-1].content
    total_abstain += is_abstain
    e = build_example(tok, c, MAX_LEN)
    if e:
        encoded.append(e); kept_abstain += is_abstain

fit = len(encoded)/len(conversations)
print(f'{len(encoded)}/{len(conversations)} fit in {MAX_LEN} tokens  ({fit:.0%})')
print(f'abstention examples kept: {kept_abstain}/{total_abstain} '
      f'= {kept_abstain/max(1,len(encoded)):.0%} of the training set')
# Length filtering must not silently rebalance the classes. If abstention
# examples drop below ~20% the model will not learn to abstain at all.
assert fit > 0.6, 'too many examples dropped -- shorten the template or raise MAX_LEN'
assert kept_abstain/max(1,len(encoded)) > 0.2, 'abstention examples were filtered away'

ex = encoded[0]
toks = [tok.decode([i]) for i in ex['ids']]
print('\\nfirst 40 positions (mask | token):')
print(' '.join(f"{m}:{t.strip()[:10] or '·'}" for m,t in list(zip(ex['mask'], toks))[:40]))
frac = sum(e['mask'].count(1) for e in encoded)/sum(len(e['ids']) for e in encoded)
print(f'\\n{frac:.1%} of positions are scored — the rest is prompt, correctly masked out')"""),
("code", """ds = SFTDataset(encoded, tok.special_tokens['<|pad|>'], MAX_LEN)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9,0.95))
# LR is ~6x lower than pretraining: SFT adapts an existing model, and a high LR
# causes catastrophic forgetting of everything pretraining bought.
STEPS, BS = 600, 8
losses = []
model.train()
for step in range(STEPS):
    x, y, m = ds.batch(BS, DEVICE)
    with torch.autocast(device_type=DEVICE.split(':')[0], dtype=torch.bfloat16, enabled=DEVICE=='cuda'):
        _, loss = model(x, y, loss_mask=m)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    losses.append(loss.item())
    if step % 40 == 0: print(f'step {step:4d} | masked loss {loss.item():.4f}')
import matplotlib.pyplot as plt
plt.figure(figsize=(7,2.5)); plt.plot(losses); plt.xlabel('step')
plt.ylabel('assistant-only loss'); plt.grid(alpha=.3); plt.show()"""),
("md", """## Test the trained behaviours

The question is not "is the prose good" — at 30M parameters it will not be.
The question is **does it abstain when it should**. Compare an answerable
prompt with an unanswerable one."""),
("code", """def ask(question, evidence, max_new=70):
    prompt = ('<|bos|><|system|>Answer only from evidence. Cite [E#]. Abstain if '
              'insufficient.<|user|>' + ev_prompt(question, evidence) + '<|assistant|>')
    ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
    model.eval()
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=max_new, temperature=0.2, top_k=30,
                             eos_id=tok.special_tokens['<|eos|>'])
    return tok.decode(out[0, ids.shape[1]:].tolist())

ev = [{'text':'Kestrel Bank filing: Lena Petrov was appointed chief executive effective January 2026.',
       'source':'sec.gov','date':'2026-01-15','tier':1}]
print('ANSWERABLE:\\n ', ask('Who is the CEO of Kestrel Bank?', ev), '\\n')
print('NOT ANSWERABLE FROM THIS EVIDENCE:\\n ', ask("What is Kestrel Bank's current headcount?", ev))"""),
("code", """unk = tok.special_tokens['<|unknown|>']
answerable = sum(unk in tok.encode(ask('Who is the CEO of Kestrel Bank?', ev)) for _ in range(5))
unanswerable = sum(unk in tok.encode(ask("What is Kestrel Bank's headcount?", ev)) for _ in range(5))
print(f'abstained on ANSWERABLE   : {answerable}/5   (want 0 — over-abstention is also a failure)')
print(f'abstained on UNANSWERABLE : {unanswerable}/5 (want 5)')
print('\\nThis pair IS the abstention-quality metric. Both directions matter:')
print('a model that always refuses scores perfectly on one and uselessly on the other.')
model.save(str(ROOT/'checkpoints'/'sft.pt')); print('\\nsaved -> checkpoints/sft.pt')"""),
("md", """Next: **05 — retrieval.**"""),
]

for name, cells in [("01_tokenizer.ipynb", N1), ("02_transformer.ipynb", N2),
                    ("03_pretraining.ipynb", N3), ("04_instruction_tuning.ipynb", N4)]:
    write(name, cells)
