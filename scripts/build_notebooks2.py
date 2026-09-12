"""Generate VERITAS notebooks 05-08. Run: python scripts/build_notebooks2.py"""
from __future__ import annotations

from build_notebooks import BOOT, write  # reuse helpers

# ===================================================================== 05
N5 = [
("md", """# VERITAS 05 — Retrieval from scratch: chunking, BM25, dense, hybrid, rerank

**Phases 5–6 of 16.**

## Chunking decides what a citation can point at

VERITAS cites *claim → chunk → document span*, so a chunk must be
(1) **self-contained** — "He stepped down in March" is worthless alone;
(2) **small enough to be precise** — embedding a 2000-token page averages away
the one sentence that matters; (3) **offset-preserving** — we keep
`(start_char, end_char)` so the UI can highlight the exact span.

Strategy: recursive *structural* splitting (headings → paragraphs → sentences)
with a token budget and ~15 % overlap, plus a heading breadcrumb prepended to
each chunk. Structural boundaries correlate with semantic ones for free —
far cheaper than embedding-based semantic chunking, which costs one encoder
pass per sentence.

## BM25, derived

$$\\text{score}(D,Q)=\\sum_{t\\in Q}\\text{IDF}(t)\\cdot\\frac{f(t,D)(k_1+1)}{f(t,D)+k_1\\left(1-b+b\\frac{|D|}{\\text{avgdl}}\\right)}$$

* **IDF** — a rare term is discriminative. The `1 + …` smoothing keeps it
  positive; plain RSJ IDF goes negative for common terms, letting a stopword
  *subtract* score.
* **TF saturation** `f/(f+k₁)` — concave: the 1st "revenue" is evidence, the
  20th adds nothing. Raw TF lets one keyword-stuffed page dominate.
* **Length norm** `b·|D|/avgdl` — long documents otherwise accumulate matches
  by sheer size.

## Why keep BM25 when we have embeddings?

They fail **differently**. Dense retrieval fails on exact strings it never saw:
a ticker, a docket number, a surname, a version string — exactly the tokens
that identify entities. BM25 has no vocabulary problem and needs no training.
Hybrid exists to cover both failure modes, not to be fancy.

## Fusion: RRF vs weighted sum

BM25 scores are unbounded (0–40); cosine lives in [−1,1]. Adding them directly
lets BM25 silently dominate.
* **RRF** `Σ w/(60+rank)` uses only ranks → immune to scale, no tuning.
* **Weighted min-max sum** preserves *margins* (the gap between rank 1 and 2),
  which the evidence-sufficiency check downstream actually reads.

## Reranking: bi-encoder → cross-encoder

A bi-encoder embeds query and document **independently** — that is what makes
it fast (documents embedded offline) and also its ceiling: the document vector
is computed without ever seeing the query. A cross-encoder concatenates them so
every query token attends to every document token, at `O(candidates)` forward
passes. Hence two stages: retrieve 50 cheaply, rerank 50 expensively, keep 8."""),
("code", BOOT),
("code", """from veritas.tokenizer.bpe import BPETokenizer
from veritas.model.transformer import VeritasLM, ModelConfig
from veritas.rag.chunking import chunk_document, chunk_table
from veritas.rag.bm25 import BM25, tokenize
from veritas.rag.embeddings import Embedder, VectorIndex, info_nce_loss, train_embedder
from veritas.rag.hybrid import HybridRetriever, FusionWeights, rrf_fuse, weighted_fuse
from veritas.rag.reranker import CrossEncoder, train_reranker, pairwise_rank_loss

tok = BPETokenizer.load(ROOT/'checkpoints'/'tokenizer.json')
ck = ROOT/'checkpoints'/'sft.pt'
if not ck.exists(): ck = ROOT/'checkpoints'/'best.pt'
model = VeritasLM.load(ck, DEVICE) if ck.exists() else VeritasLM(
    ModelConfig(vocab_size=tok.vocab_size, d_model=384, n_layers=8, n_heads=8,
                n_kv_heads=2, max_seq_len=256)).to(DEVICE)
model.eval(); print('model from', ck.name if ck.exists() else '(random init — run 03/04 first)')"""),
("md", """## Chunking with offsets preserved"""),
("code", """doc = '''# Acme Industries — Annual Report 2026

## Leadership
Marcus Lund was appointed chief executive effective February 2026, succeeding
Priya Raman. The board confirmed the appointment at its January meeting.

## Financial results
Revenue for the year was 1.42 billion euros, up 12% year on year. Operating
margin improved to 14.1%. The company opened 15 new offices during the period.

## Outlook
Management expects continued expansion in Asia-Pacific markets during 2027.
'''
chunks = chunk_document(doc, 'acme_ar_2026', {'title':'Acme Annual Report 2026','tier':1,
                        'date':'2026-03-01','entity':'Acme Industries'}, target_tokens=60)
for c in chunks:
    print(f'[{c.chunk_id}] heading={c.heading_path!r} chars {c.start_char}-{c.end_char}')
    print(f'   {c.text[:90]}...')
print('\\nWhat gets EMBEDDED (breadcrumb restores the referent):')
print(' ', chunks[1].contextualized[:130])"""),
("code", """# Tables need row-level linearisation or the column->value binding is lost.
rows = [['Year','Revenue','Margin'], ['2024','1.10B','11.2%'], ['2025','1.27B','12.8%'], ['2026','1.42B','14.1%']]
for c in chunk_table(rows, 'acme_fin'):
    print(f'[{c.chunk_id}] {c.text}')
print('\\nFlat-text chunking would yield "2026 1.42B 14.1%" — no way to know which is revenue.')"""),
("md", """## Build the corpus and both indices"""),
("code", """from veritas.eval.benchmark import build_seed_benchmark, expand_synthetic
bench = expand_synthetic(build_seed_benchmark(), 4)
corpus, metadata = {}, {}
for it in bench.items:
    for d in it.docs:
        if d.doc_id in corpus: continue
        corpus[d.doc_id] = d.text
        metadata[d.doc_id] = {'source':d.source,'tier':d.tier,'date':d.date,
                              'entity':d.entity or it.entity,'valid_from':d.valid_from,'valid_to':d.valid_to}
print(f'{len(corpus)} chunks')

bm25 = BM25()
for cid, text in corpus.items(): bm25.add(cid, text)
bm25.finalize()
print(f'BM25: {len(bm25.postings):,} terms | avgdl {bm25.avgdl:.1f}')

embedder = Embedder(model, tok, max_len=192, device=DEVICE)
t0=time.time(); vecs = embedder.encode(list(corpus.values()))
vec = VectorIndex(embedder.dim); vec.add(list(corpus), vecs)
print(f'dense: {vecs.shape} in {time.time()-t0:.1f}s | L2-normalised: {np.linalg.norm(vecs[0]):.4f}')"""),
("md", """### Where each retriever fails

Run both on the same queries. The pattern to look for: BM25 wins on exact rare
strings, dense wins on paraphrase. Neither wins both — which is the argument
for hybrid, stated as evidence rather than assertion."""),
("code", """queries = ['Who is the current CEO of Acme Industries?',
           'Marcus Lund',                      # exact rare string -> BM25 should win
           'who runs the company these days']  # paraphrase, no keywords -> dense should win
for q in queries:
    s = [d for d,_ in bm25.search(q, k=3)]
    d_ = [h.doc_id for h in vec.search(embedder.encode_one(q), k=3)]
    print(f'\\nQ: {q}')
    print('  BM25 :', s)
    print('  dense:', d_)
    print('  overlap:', len(set(s)&set(d_)), 'of 3')"""),
("md", """## Contrastive training of the embedder (InfoNCE)

$$L=-\\log\\frac{\\exp(s(q,d^+)/\\tau)}{\\sum_j \\exp(s(q,d_j)/\\tau)}$$

Every other document in the batch is a negative, so a batch of `B` gives `B−1`
free negatives — the reason contrastive retrieval training wants large batches.
`τ≈0.05` sharpens the softmax: too high and the gradient is flat, too low and
it fixates on the single hardest negative."""),
("code", """pairs = []
for it in bench.items:
    for gid in it.gold_docs:
        if gid in corpus: pairs.append((it.question, corpus[gid]))
print(f'{len(pairs)} (query, positive) pairs')

def recall_at(k=3):
    hit = 0; n = 0
    for it in bench.items:
        if not it.gold_docs: continue
        n += 1
        got = [h.doc_id for h in vec.search(embedder.encode_one(it.question), k=k)]
        hit += len(set(got) & set(it.gold_docs)) > 0
    return hit/max(1,n)

before = recall_at()
hist = train_embedder(model, tok, pairs, steps=150, batch_size=8, lr=5e-5, device=DEVICE, log_every=50)
embedder = Embedder(model, tok, max_len=192, device=DEVICE)
vec = VectorIndex(embedder.dim); vec.add(list(corpus), embedder.encode(list(corpus.values())))
print(f'\\ndense recall@3: {before:.2f} -> {recall_at():.2f}')"""),
("md", """## Approximate search: IVF

k-means into √N cells, probe the `nprobe` nearest. On normalised vectors,
Euclidean k-means **is** spherical k-means (minimising `‖x−c‖²` = maximising
`x·c`), so assignment is one GEMM + argmax. Below ~10⁵ vectors, exact search is
a single GEMM and is both faster *and* exact — measure before reaching for ANN."""),
("code", """import matplotlib.pyplot as plt
q = embedder.encode_one('Who is the current CEO of Acme Industries?')
exact = [h.doc_id for h in vec.search(q, k=5)]
vec.build_ivf()
if vec.centroids is not None:
    for nprobe in (1,2,4,8):
        got = [h.doc_id for h in vec.search(q, k=5, nprobe=nprobe)]
        print(f'nprobe={nprobe}: recall vs exact = {len(set(got)&set(exact))/5:.2f}')
    print('\\nThe recall/speed knob: more probes -> closer to exact, more vectors scanned.')
else:
    print(f'{len(vec.ids)} vectors — too few for IVF; exact search is already optimal here.')"""),
("md", """## Hybrid fusion"""),
("code", """retr = HybridRetriever(vec, bm25, embedder, FusionWeights(dense=1.0, sparse=1.0,
                        freshness=0.0, authority=0.0, entity=0.0, temporal=0.0), metadata=metadata)
q = 'Who is the current CEO of Acme Industries?'
print('RRF (rank-only, scale-immune):')
for c in retr.retrieve(q, k=4, mode='rrf'): print(f'  {c.score:.4f}  {c.doc_id}')
print('\\nWeighted min-max sum (preserves margins, with per-signal explanation):')
for c in retr.retrieve(q, k=4, mode='weighted'):
    parts = ' '.join(f'{k_}={v:.2f}' for k_,v in c.explain.items() if v)
    print(f'  {c.score:.4f}  {c.doc_id:22s} {parts}')"""),
("md", """## Cross-encoder reranking

Trained with a pairwise ranking loss `−log σ(s⁺ − s⁻)`: optimising the *margin
between* a good and a bad passage is the right objective for ranking, and
pointwise regression would waste capacity calibrating absolute scores that only
ever get sorted. **Hard** negatives matter — random negatives are trivially
separable and produce no gradient."""),
("code", """from veritas.rag.reranker import mine_hard_negatives
triples = []
for it in bench.items:
    for gid in it.gold_docs:
        if gid not in corpus: continue
        for neg in mine_hard_negatives(retr, it.question, it.gold_docs, n=2):
            if neg in corpus: triples.append((it.question, corpus[gid], corpus[neg]))
print(f'{len(triples)} (query, positive, hard-negative) triples')
ce = CrossEncoder(model, tok, max_len=256, device=DEVICE)
if triples: train_reranker(ce, triples, steps=120, batch_size=4, lr=5e-5, log_every=40)"""),
("code", """cands = [c.doc_id for c in retr.retrieve(q, k=8)]
print('before rerank:', cands[:5])
print('after  rerank:', [i for i,_ in ce.rerank(q, [corpus[c] for c in cands], cands, top_k=5)])
print('\\nThe cross-encoder sees query and passage TOGETHER, so it can judge')
print('entailment — which is why the same architecture is reused as the NLI')
print('head of the claim verifier in notebook 06.')"""),
("md", """Next: **06 — the temporal + evidence layer** (the actual novelty)."""),
]

# ===================================================================== 06
N6 = [
("md", """# VERITAS 06 — Temporal knowledge + evidence graph + verification

**Phases 7–10.** This is where VERITAS stops being a RAG system.

## The failure being fixed

Two sources say different things. There are two very different reasons:

* **(a) a source is WRONG** — it contradicts reality
* **(b) a source WAS RIGHT, in 2023** — it contradicts only the present

A vector index cannot tell these apart: both chunks mention "CEO", both rank
highly, the model averages them into a confident wrong answer. Distinguishing
them requires storing *when a fact was true*.

## Bitemporal storage (Snodgrass, 1995)

Two independent time axes:

* **valid time** `[valid_from, valid_to)` — when it was true **in the world**
* **transaction time** `[recorded_at, superseded_at)` — when **we knew** it

They genuinely come apart: a filing published March 2026 states a CEO change
effective January 2026. With one axis you must choose which lie to tell. With
both, all four questions are answerable — including *"who did we think was CEO
in 2024, back in 2024?"*, which is the audit trail.

**Nothing is deleted.** A change closes valid time and opens a new interval; a
correction closes transaction time and keeps valid time. Different operations,
both preserving history.

As-of lookups are a **binary search** over versions sorted by `valid_from`:
`O(log n)`, not a scan.

## Evidence graph: why a graph, not a table

The questions are *path* questions. The decisive one:
**are these three sources independent?** Three outlets rewriting one wire story
look like three-source corroboration in a table. In a graph they converge on
one `DERIVED_FROM` ancestor, so the corroboration bonus is correctly withheld.
Independence is a **topological** property.

## Verification is entailment, not similarity

"Acme did **not** appoint Y" has ~0.95 cosine similarity to "Acme appointed Y".
Retrieval scores are useless here. Two channels:

1. **symbolic** — numeric mismatch (after unit normalisation), polarity flip,
   temporal mismatch. Auditable, ~free, no hallucination mode.
2. **neural NLI** — paraphrase, which no rule catches.

Symbolic runs first and can **veto** the model. A claim is SUPPORTED only when
no symbolic check fires *and* entailment clears threshold."""),
("code", BOOT),
("code", """from veritas.temporal.versioning import TemporalStore, ChangeKind, now_utc
from veritas.temporal.change_detection import ChangeDetector, simhash, hamming, content_digest
from veritas.temporal.temporal_retrieval import (parse_temporal_query, TemporalIntent,
        freshness_score, interval_overlap_score, make_signal_fns, HALFLIFE_DAYS)
from veritas.evidence.claims import extract_claims, claims_to_state
from veritas.evidence.graph import EvidenceGraph, EdgeType
from veritas.evidence.quality import SourcePolicy, score_evidence, support_label, explain
from veritas.evidence.verifier import ClaimVerifier, EvidenceItem, Verdict, lexical_entailment
from veritas.evidence.contradiction import ContradictionDetector, ConflictType"""),
("md", """## 1. The bitemporal store — all four question types"""),
("code", """store = TemporalStore()
store.alias('acme', 'Acme Industries')   # entity resolution: without it, two timelines

store.assert_fact('Acme Industries','ceo','Dana Whitfield', valid_from='2024-01-01',
                  recorded_at='2024-03-01', source_id='ir.acme.com', confidence=0.8)
store.assert_fact('Acme Industries','ceo','Priya Raman', valid_from='2025-07-01',
                  recorded_at='2025-06-15', source_id='ir.acme.com', confidence=0.8)
v, ev = store.assert_fact('Acme Industries','ceo','Marcus Lund', valid_from='2026-02-01',
                  recorded_at='2026-02-02', source_id='sec.gov', confidence=0.9)
print('change kind:', ev.kind, '| previous:', ev.old_value)

print('\\nQ1 who is CEO now              ->', store.current('Acme Industries','ceo').value)
print('Q2 who was CEO in 2024         ->', store.as_of('Acme Industries','ceo','2024-06-01').value)
print('Q3 what did we believe in 2025 ->', store.as_of("Acme Industries",'ceo','2025-09-01',
                                                       known_at='2025-09-01').value)
print('Q4 when did we learn of Lund   ->', v.recorded_at.date(),
      f'(effective {v.valid_from.date()} — {(v.recorded_at-v.valid_from).days:+d} days)')
print('\\nOUTDATED (a naive retriever would quote these as fact):',
      [x.value for x in store.outdated('Acme Industries','ceo')])"""),
("code", """print('full timeline:')
for x in store.timeline('Acme Industries'):
    end = 'present' if x.valid_to.year > 9000 else x.valid_to.date()
    print(f'  {x.valid_from.date()} → {end:>10}  {x.attribute}={x.value:16s} '
          f'[{x.change_kind:10s}] src={x.source_id} recorded={x.recorded_at.date()}')

# A restatement by a different source is corroboration, not a new fact:
before = store.current('Acme Industries','ceo').confidence
store.assert_fact('Acme Industries','ceo','Marcus Lund', valid_from='2026-02-01',
                  source_id='reuters.com', recorded_at='2026-02-05')
print(f'\\nREAFFIRMED by a second source: confidence {before:.2f} -> '
      f'{store.current("Acme Industries","ceo").confidence:.2f} (no duplicate row written)')"""),
("md", """## 2. Change detection — a cascade, cheapest filter first

Re-ingesting a source costs a parse + claim extraction + embedding + index
write. Most polls return an identical page, or one whose only difference is a
rotating banner. A system that reprocesses everything cannot run continuously.

**SimHash** projects a document to a 64-bit signature where similar documents
have small Hamming distance — comparison is one `popcount(a^b)` instruction.
Stage 4 is the one that matters: text changing is not news, a **claim**
changing is."""),
("code", """det = ChangeDetector()
base = 'Acme Industries filing. The chief executive is Priya Raman. Revenue was 1.27 billion euros.'
tests = [
    ('first sight',        base),
    ('byte-identical',     base),
    ('cosmetic banner',    base + '\\nLast updated 14:03:22'),
    ('reworded, same fact',base.replace('The chief executive is','Chief executive:')),
    ('REAL state change',  base.replace('Priya Raman','Marcus Lund')),
]
for label, text in tests:
    claims = extract_claims(text, 'd','d','Acme Industries')
    r = det.check('acme_ir', text, claims_to_state(claims))
    print(f'{label:22s} changed={str(r.changed):5s} reason={r.reason:16s} sim={r.similarity:.3f} '
          f'state-change={r.has_state_change}')
print(f'\\npolls={det.state("acme_ir").poll_count} changes={det.state("acme_ir").change_count} '
      f'volatility={det.state("acme_ir").volatility:.2f}')
print(f'adaptive next poll: {det.next_poll_seconds("acme_ir")}s '
      f'(stable sources back off, volatile ones speed up)')"""),
("md", """## 3. Temporal query understanding

The retriever must know the **tense** of the question. Note that CHANGE
deliberately switches *off* freshness weighting — a change question needs the
before *and* the after, so preferring recent evidence would destroy the answer."""),
("code", """for q, attr in [('Who is the current CEO?','ceo'),
                ('Who was the CEO in 2022?','ceo'),
                ('As of March 2025, what was the status?','status'),
                ('How did the CEO change over time?','ceo'),
                ('What happened between 2020 and 2024?','ceo'),
                ('What is the share price?','price')]:
    tq = parse_temporal_query(q, attr)
    print(f'{q:44s} -> {tq.intent:11s} anchor={tq.anchor.date()} '
          f'half-life={tq.halflife_days:>6.1f}d freshness={tq.apply_freshness}')
print('\\nHalf-life is PER-ATTRIBUTE. One global decay is the usual mistake:')
print('it would decay a founding date at the same rate as a share price.')"""),
("code", """import matplotlib.pyplot as plt
ages = np.linspace(0, 1095, 200)
plt.figure(figsize=(7,3))
for attr in ['price','status','revenue','ceo','founded']:
    hl = HALFLIFE_DAYS[attr]
    plt.plot(ages, 0.5**(ages/hl), label=f'{attr} (t½={hl:g}d)')
plt.xlabel('document age (days)'); plt.ylabel('freshness'); plt.legend(fontsize=8)
plt.title('exponential decay, not a cliff — a 366-day cutoff would drop the only evidence there is')
plt.grid(alpha=.3); plt.show()"""),
("md", """## 4. Evidence graph and the independence test

This is the cell that separates VERITAS from a citation-listing RAG."""),
("code", """g = EvidenceGraph()
g.add_source('reuters.com','Reuters',tier=2,domain='corporate')
g.add_source('dailyfeed.example.com','Daily Feed',tier=3)
g.add_source('aggregator.example.net','Aggregator',tier=3)
g.add_source('gov.example.gov','Regulator',tier=1)

g.add_document('wire_orig','reuters.com',date='2026-03-03')
for rid, src in [('reprint_a','dailyfeed.example.com'), ('reprint_b','aggregator.example.net')]:
    g.add_document(rid, src, date='2026-03-04')
    g.link(f'doc:{rid}', 'doc:wire_orig', EdgeType.DERIVED_FROM)   # syndication edge
g.add_document('filing','gov.example.gov',date='2026-03-10')

claim = g.add_claim('c1','Helios Energy has begun construction at Almeria',
                    entity='Helios Energy', attribute='status', value='construction',
                    doc_id='wire_orig', valid_from='2026-03-03')
for d in ['reprint_a','reprint_b','filing']:
    g.supports(f'doc:{d}', claim, weight=0.9)

print(f'documents citing this claim : {len(g.evidence_for(claim))}')
print(f'INDEPENDENT sources         : {g.independent_sources(claim)}')
print('\\nFour documents, two independent roots. A flat (claim, source) table would')
print('report four-source corroboration for what is really a wire story plus one filing —')
print('and that inflated count is how a system talks itself into a false fact.')
print('\\nfirst reported by:', g.first_reported(claim))"""),
("code", """import json as _j
print(_j.dumps(g.provenance_path(claim), indent=1)[:900])
print('\\ngraph:', g.stats())"""),
("md", """## 5. Evidence quality — a support LABEL, never a truth percentage

Seven bounded factors, weighted, with a **hard floor**: zero independent
sources ⇒ score 0, regardless of how well everything else scores. A plain
weighted sum would let five weak signals outvote the absence of evidence."""),
("code", """policy = SourcePolicy()
print('Source tiers are PER-DOMAIN — there is no universal ranking:')
for src in ['sec.gov','gov.example.gov','reuters.com','status.acme.com','arxiv.org','reddit.com']:
    tiers = {d: policy.tier(src, d) for d in ['government','corporate','science','software']}
    print(f'  {src:22s} {tiers}')
print('\\nFor a regulatory question the regulator outranks Reuters; for a software')
print('outage the vendor status page outranks the regulator. Same source, different tier.')"""),
("code", """scenarios = [
 ('one tier-1 primary, fresh',      ['sec.gov'], ['2026-08-01'], 1, 1.0),
 ('two independent tier-1, fresh',  ['sec.gov','gov.example.gov'], ['2026-08-01','2026-08-03'], 2, 1.0),
 ('two sources, disagreeing',       ['sec.gov','reuters.com'], ['2026-08-01','2026-08-02'], 2, 0.5),
 ('one tier-4 blog, 3 years old',   ['reddit.com'], ['2023-01-01'], 1, 1.0),
 ('NO independent source',          ['sec.gov'], ['2026-08-01'], 0, 1.0),
]
for name, srcs, dates, n_ind, agree in scenarios:
    s, f = score_evidence(sources=srcs, dates=dates,
        texts=['Acme reported revenue of 1.42 billion euros on 2026-08-01.'],
        n_independent=n_ind, agreement=agree, validity=1.0, domain='corporate',
        ref_time=__import__('datetime').datetime(2026,9,1,tzinfo=__import__('datetime').timezone.utc))
    lab = support_label(s, n_ind, has_conflict=agree < 0.6)
    print(f'{name:32s} {explain(s, f, lab)}')
print('\\nNote the last row: the hard floor. No corroboration -> INSUFFICIENT, whatever else says.')"""),
("md", """## 6. Claim verification — the symbolic checks that catch real errors"""),
("code", """ver = ClaimVerifier()
def check(claim_text, evidence_text, src='reuters.com', date='2026-08-14', valid_to=None):
    c = extract_claims(claim_text, 'gen','gen','Nova Logistics')[0]
    e = [EvidenceItem('d1', evidence_text, src, date, 2, valid_to=valid_to)]
    v = ver.verify_claim(c, e, anchor_time='2026-09-01')
    print(f'{v.verdict:18s} entail={v.entailment:.2f}  {v.reason}')
    return v

print('NUMERIC MISMATCH — the highest-yield check (fabricated numbers are the most')
print('common and most damaging RAG error, and cosine scores the two as near-identical):')
check('Nova Logistics opened 15 offices in India in 2026.',
      'Nova Logistics opened 12 offices in India in 2026.')
print('\\nUNIT NORMALISATION — 1.4 billion == 1,400 million, NOT a conflict:')
check('Nova Logistics reported revenue of 1.4 billion.',
      'Nova Logistics reported revenue of 1,400 million.')
print('\\nPOLARITY FLIP:')
check('Nova Logistics opened 12 offices in India in 2026.',
      'Nova Logistics did not open offices in India in 2026.')
print('\\nCLEAN SUPPORT:')
check('Nova Logistics opened 12 offices in India in 2026.',
      'Nova Logistics opened 12 new offices across India during 2026, filings show.')
print('\\nTEMPORAL MISMATCH — true earlier, not now. NOT the same as "refuted":')
check('Nova Logistics operates 12 offices in India.',
      'Nova Logistics operated 12 offices in India.', date='2023-01-01', valid_to='2024-01-01')"""),
("md", """## 7. Contradiction analysis — four kinds, only one is a real conflict"""),
("code", """cd = ContradictionDetector()
h = store.history('Acme Industries','ceo')
c = cd.compare_versions(h[0], h[-1], 'corporate')
print(f'[{c.type}] sev={c.severity:.1f}  {c.explanation}')
print('   ^ successive states. Reporting this as a disagreement would fire on')
print('     every entity that ever changed — which is how a conflict detector becomes noise.\\n')

s2 = TemporalStore()
a,_ = s2.assert_fact('Nova Logistics','offices','15', valid_from='2026-01-01',
                     valid_to='2026-12-31', source_id='ir.novalogistics.com')
b,_ = s2.assert_fact('Nova Logistics','offices','12', valid_from='2026-01-01',
                     valid_to='2026-12-31', source_id='reddit.com')
c2 = cd.compare_versions(a, b, 'corporate')
print(f'[{c2.type}] sev={c2.severity:.1f}  {c2.explanation}')
print(f'   preference: {c2.preference_reason or "none — reported unresolved"}')
print('\\n' + cd.summarize([c, c2]))"""),
("md", """Next: **07 — the agentic loop, end to end.**"""),
]

# ===================================================================== 07
N7 = [
("md", """# VERITAS 07 — The agentic loop, end to end

**Phases 11–12.** Plan → search → assess → refine → verify → abstain-or-answer,
plus the continuous update pipeline.

```
QUESTION → PLAN → SEARCH ←──────────────┐
                    ↓                   │ refine against the NAMED gap
              TEMPORAL ASSESSMENT       │
                    ↓                   │
            EVIDENCE SUFFICIENT? ──no───┘   (bounded by max_iterations)
                    │yes
              VERIFY CLAIMS → CONTRADICTIONS → ABSTAIN?
                    │no
              SYNTHESISE → RE-VERIFY → ANSWER + EVIDENCE + TIMELINE + CONFIDENCE
```

Two properties distinguish this from "retrieve once, then answer":

* **The loop condition is evidence sufficiency, not a step count.** It refines
  against the *specific* gap ("prior state missing"), because the index already
  answered the original phrasing — rephrasing returns the same chunks.
* **It is bounded.** `max_iterations` plus a no-progress break. Unbounded agent
  loops are how one question becomes 200 retrievals.

## Synthesis is inverted

Ordinary RAG generates prose from context and hopes it is faithful. VERITAS
composes **from already-verified claims**, so the answer *cannot* contain an
unverified claim — it is impossible by construction, not by prompting. The LM's
job shrinks from "be truthful" to "be fluent about these specific sentences",
which a 30M-parameter model can actually do. Generative output is then
re-verified and discarded if it scores worse than the extractive baseline: the
model never gets the last word."""),
("code", BOOT),
("code", """from veritas.tokenizer.bpe import BPETokenizer
from veritas.model.transformer import VeritasLM, ModelConfig
from veritas.pipeline import VeritasSystemBuilder, load_benchmark_corpus
from veritas.agents.orchestrator import VeritasConfig
from veritas.eval.benchmark import build_seed_benchmark

tok = BPETokenizer.load(ROOT/'checkpoints'/'tokenizer.json')
ck = ROOT/'checkpoints'/'sft.pt'
if not ck.exists(): ck = ROOT/'checkpoints'/'best.pt'
model = VeritasLM.load(ck, DEVICE) if ck.exists() else VeritasLM(
    ModelConfig(vocab_size=tok.vocab_size, d_model=384, n_layers=8, n_heads=8,
                n_kv_heads=2, max_seq_len=256)).to(DEVICE)
model.eval()

bench = build_seed_benchmark()
builder = VeritasSystemBuilder(model, tok, device=DEVICE)
n = load_benchmark_corpus(builder, bench)
veritas = builder.build(config=VeritasConfig(k=8, max_iterations=3, domain='corporate', verbose=True))
print(f'\\ningested {n} documents via the REAL ingest path')
print('system state:', builder.ingest.summary())"""),
("md", """## A CURRENT question — watch the trace"""),
("code", """ans = veritas.answer('Who is the current CEO of Acme Industries?')
print('\\n' + '='*72)
print(ans.to_markdown())"""),
("code", """print('DECISION TRACE (this ships with the answer):')
for t in ans.trace: print('  ', t)"""),
("md", """## The same entity, asked historically

Same corpus, same index — a different **tense**. A system without valid-time
returns the same top-k for both and answers both with the current CEO."""),
("code", """for q in ['Who is the current CEO of Acme Industries?',
          'Who was the CEO of Acme Industries in 2024?',
          'How did the CEO of Acme Industries change over time?']:
    a = veritas.answer(q)
    print(f'\\nQ: {q}\\n   {a.answer[:220]}')
    print(f'   support={a.support_level} | coverage={a.coverage:.2f} | iterations={a.iterations}')"""),
("md", """## Abstention and conflict — the two behaviours ordinary RAG cannot do"""),
("code", """a = veritas.answer("What is Nova Logistics' 2027 revenue guidance?")
print('ABSTENTION:', a.abstained)
print(a.answer)
print('unknown:', a.unknown[:2])

b = veritas.answer('How many offices did Nova Logistics open in India in 2026?')
print('\\n\\nCONFLICT:')
print(b.answer[:300])
print('conflicts:', b.conflicts)
print('support level:', b.support_level)
print('\\nIt reports 15 vs 12 as unresolved. Silently picking either is the failure.')"""),
("md", """## The demonstration: a new source changes the answer, with no retraining

This is spec §35 and the single most convincing thing to show. Ask, inject a
new filing, ask again. The language model weights are **untouched**."""),
("code", """q = 'Who is the current CEO of Acme Industries?'
before = veritas.answer(q)
print('BEFORE:', before.answer[:160])
print('current state:', builder.store.current('Acme Industries','ceo').value)

t0 = time.time()
res = builder.add_document('sec.gov', 'acme_2027_8k',
    'Acme Industries filing: Yuki Tanaka was appointed chief executive effective March 2027, '
    'succeeding Marcus Lund.', '2027-03-02', 'Acme Industries')
ingest_s = time.time()-t0

after = veritas.answer(q)
print(f'\\n--- ingested one document in {ingest_s:.3f}s ---')
print('changed:', res.changed, '| reason:', res.reason)
print('state changes:', res.state_changes)
print('cache keys invalidated:', res.invalidated)
print('\\nAFTER :', after.answer[:160])
print('current state:', builder.store.current('Acme Industries','ceo').value)
print('\\nModel weights changed: NO. Full re-index: NO. Only the affected chunks,')
print('timelines and cache entries were touched.')"""),
("code", """print('The full audit trail, preserved rather than overwritten:')
for v in builder.store.history('Acme Industries','ceo'):
    end = 'present' if v.valid_to.year > 9000 else v.valid_to.date()
    print(f'  {v.valid_from.date()} → {end:>10}  {v.value:16s} [{v.change_kind:10s}] '
          f'src={v.source_id:16s} recorded={v.recorded_at.date()}')
print('\\nAnswering "what was the CEO in 2026?" after the update:',
      builder.store.as_of('Acme Industries','ceo','2026-06-01').value)"""),
("md", """## Continuous polling with adaptive intervals

A source that never changes should not be polled as often as a live status
page. The interval is derived from the *observed* change rate, which is a
better prior than any hand-set number."""),
("code", """from veritas.ingest.pipeline import Source
feed_state = {'n': 0}
def fake_feed():
    feed_state['n'] += 1
    if feed_state['n'] < 3:
        return ('Helios Energy status: construction under way at Almeria.',
                {'doc_id':'helios_status','published':'2026-03-15','entity':'Helios Energy'})
    return ('Helios Energy status: the Almeria plant is now operational.',
            {'doc_id':'helios_status','published':'2026-11-01','entity':'Helios Energy'})

builder.ingest.register(Source('helios_status_page', tier=1, domain='corporate',
                               fetch=fake_feed, entity_hint='Helios Energy', poll_seconds=0))
for i in range(4):
    r = builder.ingest.poll_once('helios_status_page')
    nxt = builder.ingest.sources['helios_status_page'].poll_seconds
    print(f'poll {i+1}: changed={str(r.changed):5s} reason={r.reason:18s} '
          f'state_changes={len(r.state_changes)} next_poll={nxt}s')
print('\\nHelios status timeline:')
for v in builder.store.history('Helios Energy','status'):
    print(f'  {v.valid_from.date()} {v.value:22s} [{v.change_kind}]')"""),
("code", """# Reverification: a fact nobody has restated in a long time is not the same as
# a fact confirmed today, even when it is still the newest thing on record.
stale = builder.ingest.reverify('Meridian Port','status', max_age_days=90)
print('sources to re-poll for Meridian Port status:', stale[:3])"""),
("md", """## Structured output for the API / frontend"""),
("code", """import json as _j
d = ans.to_dict()
print('answer object keys:', list(d))
print('\\nspoken rendering (notebook on voice would feed this to TTS):')
print(' ', ans.to_speech())
print('\\nJSON head:'); print(ans.to_json()[:600])"""),
("md", """Next: **08 — evaluation against the baseline ladder.**"""),
]

# ===================================================================== 08
N8 = [
("md", """# VERITAS 08 — Evaluation

**Phase 15.** The table that decides whether the architecture earns its
complexity.

## The ablation ladder

| system | what it adds |
|---|---|
| **B1 LLM-only** | nothing — measures what the weights memorised |
| **B2 Basic RAG** | dense top-k |
| **B3 Hybrid RAG** | + BM25 + fusion + reranking |
| **B4 Temporal RAG** | + freshness and validity signals (the strongest *published* pattern — the honest bar) |
| **VERITAS** | + bitemporal store, agentic loop, claim verification, contradiction analysis, abstention |

Every row uses the **same corpus, tokenizer and weights**. Only the pipeline
differs. Comparing against a differently-trained model would confound "my
architecture is better" with "my model is bigger".

**B4 → VERITAS is the row that carries the claim.** If VERITAS does not beat
temporal RAG on CONFLICT, INSUFFICIENT and OUTDATED_SOURCE, the extra machinery
is not earning its keep — and the honest thing is to report that.

## Why a purpose-built benchmark

NQ/HotpotQA/TriviaQA assume a static corpus with one right answer. They have no
label for *"Person A was correct in 2024, wrong now"*, none for *"the sources
disagree, say so"*, none for *"abstain"* — and a system quoting a stale source
scores **identically** to one quoting the current source, as long as the string
matches. Those missing labels are the entire subject of this project."""),
("code", BOOT),
("code", """from veritas.tokenizer.bpe import BPETokenizer
from veritas.model.transformer import VeritasLM, ModelConfig
from veritas.pipeline import VeritasSystemBuilder, load_benchmark_corpus
from veritas.eval.benchmark import build_seed_benchmark, expand_synthetic, Category
from veritas.eval.baselines import (LLMOnly, BasicRAG, HybridRAG, TemporalRAG,
                                    VeritasSystem, evaluate, compare, freshness_lag)
from veritas.eval.metrics import format_table

bench = expand_synthetic(build_seed_benchmark(), 5)
print('benchmark:', bench.stats())
bench.save(ROOT/'data'/'bench'/'temporal_evidence_bench.json')"""),
("code", """tok = BPETokenizer.load(ROOT/'checkpoints'/'tokenizer.json')
ck = ROOT/'checkpoints'/'sft.pt'
if not ck.exists(): ck = ROOT/'checkpoints'/'best.pt'
model = VeritasLM.load(ck, DEVICE) if ck.exists() else VeritasLM(
    ModelConfig(vocab_size=tok.vocab_size, d_model=384, n_layers=8, n_heads=8,
                n_kv_heads=2, max_seq_len=256)).to(DEVICE)
model.eval()
if not ck.exists():
    print('WARNING: no trained checkpoint. The embedder is random, so dense retrieval')
    print('is noise and the table below is not reportable. Run notebooks 03-05 first.')

b = VeritasSystemBuilder(model, tok, device=DEVICE)
load_benchmark_corpus(b, bench)
from veritas.agents.orchestrator import VeritasConfig
veritas = b.build(config=VeritasConfig(verbose=False, domain='corporate'))
print('corpus:', len(b.corpus), 'chunks |', b.ingest.summary()['versions'], 'fact versions')"""),
("code", """systems = [LLMOnly(model, tok, DEVICE),
           BasicRAG(b.vec, b.embedder, b.corpus, b.metadata),
           HybridRAG(b.retriever(), b.corpus, b.metadata),
           TemporalRAG(b.retriever(), b.corpus, b.metadata),
           VeritasSystem(veritas)]
print(compare(systems, bench))"""),
("md", """## Reading the table honestly

* **accuracy** is scored per category: abstention items need an abstention,
  conflict items need the conflict surfaced, CHANGE items need *every* state.
* **coverage / cite_acc** are structurally 0 for B1–B4. They do not verify
  claims, so there is nothing to measure — that is the point of the column, not
  a scoring trick.
* **conflict** counts true positives **and** true negatives. A system that
  never surfaces a conflict scores well on the 7 non-conflict items, so read
  this column together with the CONFLICT row of the category table.
* **abstain_f1** balances correct abstention against over-abstention. Trading
  one for the other is the entire design question; a single number would hide
  it, which is why `AbstentionScore` also exposes both rates separately."""),
("code", """m = evaluate(systems[-1], bench)
a = m.abstention
print(f'correct abstentions : {a.correct_abstentions}')
print(f'missed (answered when it should not have) : {a.missed_abstentions}')
print(f'over-abstentions (refused with evidence)  : {a.over_abstentions}')
print(f'precision {a.abstention_precision:.2f} | recall {a.abstention_recall:.2f} | F1 {a.balanced:.2f}')
print('\\nBoth failure directions are reported. A system that always refuses gets')
print('recall 1.0 and is useless; precision is what catches that.')"""),
("md", """## Per-category breakdown

This is the interesting plot. Look for the categories where the ladder is flat
until VERITAS — those are the capabilities the architecture actually adds."""),
("code", """import matplotlib.pyplot as plt
results = [evaluate(s, bench) for s in systems]
cats = list(Category.ALL)
x = np.arange(len(cats)); w = 0.16
plt.figure(figsize=(13,4))
for i, r in enumerate(results):
    plt.bar(x + i*w, [r.by_category.get(c, 0) for c in cats], w, label=r.name)
plt.xticks(x + 2*w, cats, rotation=20, ha='right'); plt.ylabel('accuracy')
plt.title('Accuracy by question category'); plt.legend(fontsize=8); plt.grid(axis='y', alpha=.3)
plt.tight_layout(); plt.show()"""),
("md", """## Freshness lag — the metric a batch-rebuild system cannot produce

Answer, inject a new source, answer again. Report the wall-clock cost of
incorporating new information and whether the answer actually moved."""),
("code", """r = freshness_lag(b.ingest, veritas,
    'Who is the current CEO of Acme Industries?', 'sec.gov',
    'Acme Industries filing: Yuki Tanaka was appointed chief executive effective March 2027.',
    'Acme Industries', '2027-03-02')
for k_, v_ in r.items():
    print(f'{k_:22s}: {str(v_)[:120]}')
print('\\nA nightly-rebuild pipeline cannot report a number here at all.')"""),
("md", """## Retrieval metrics in isolation

Retrieval is the ceiling on everything downstream: evidence not retrieved
cannot be verified. **nDCG** is the headline number — it is the only one of the
four that handles graded relevance *and* position together."""),
("code", """from veritas.eval.metrics import recall_at_k, precision_at_k, mrr, ndcg_at_k
rows = []
for s in systems[1:]:
    R = P = M = N = 0; n = 0
    for it in bench.items:
        if not it.gold_docs: continue
        _, retrieved = s.answer(it)
        docs, seen = [], set()
        for c in retrieved:
            d = c.split('#')[0]
            if d not in seen: seen.add(d); docs.append(d)
        R += recall_at_k(docs, it.gold_docs, 5); P += precision_at_k(docs, it.gold_docs, 5)
        M += mrr(docs, it.gold_docs); N += ndcg_at_k(docs, it.relevance_map, 10); n += 1
    rows.append({'system': s.name, 'recall@5': round(R/n,3), 'precision@5': round(P/n,3),
                 'MRR': round(M/n,3), 'nDCG@10': round(N/n,3)})
print(format_table(rows))"""),
("md", """## What to claim, and what not to

Defensible from this table:

* VERITAS is the only configuration that **abstains** when evidence is absent,
  **surfaces** conflicts instead of silently picking, and **qualifies** stale
  evidence as stale.
* Claim-level coverage and citation accuracy are measurable for VERITAS and
  structurally undefined for the baselines.
* New information reaches answers in **milliseconds, without retraining**.

**Not** defensible:

* "This has never been done." Temporal RAG, bitemporal databases, FEVER-style
  claim verification and RAG self-verification all exist independently. See
  `docs/novelty.md` — the contribution is the *combination* plus the benchmark
  that measures it, and it should be stated that way.
* Any number from a run without a trained checkpoint. With a random encoder,
  dense retrieval is noise and the table swings run to run.
* Any absolute accuracy figure from an 8-item seed set. Expand the benchmark
  (`expand_synthetic`, or hand-write more items) before reporting."""),
("code", """print('Reproduce everything:')
print('  python tests/test_smoke.py          # 9 correctness tests')
print('  python scripts/run_eval.py --synthetic 5 --checkpoint checkpoints/sft.pt')
print('\\nNotebooks 01-08 build the whole system from an empty directory.')"""),
]

for name, cells in [("05_rag_retrieval.ipynb", N5), ("06_temporal_evidence.ipynb", N6),
                    ("07_agentic_veritas.ipynb", N7), ("08_evaluation.ipynb", N8)]:
    write(name, cells)
