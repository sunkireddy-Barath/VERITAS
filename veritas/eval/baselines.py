"""Baselines and the harness that runs them (spec section 26).

The comparison is an ablation ladder, so each row isolates one component:

    B1 LLM-only        no retrieval at all -> measures what the weights know
    B2 Basic RAG       dense top-k -> LM. The standard system.
    B3 Hybrid RAG      + BM25 + fusion + reranking. Better retrieval, same
                       (lack of) reasoning about time or evidence.
    B4 Temporal RAG    + freshness and validity signals. This is the strongest
                       *published* pattern, and the honest bar to beat.
    V  VERITAS         + bitemporal store, agentic loop, claim verification,
                       contradiction analysis, abstention.

Every baseline reuses the same corpus, the same tokenizer and the same model
weights. Only the pipeline differs. Comparing against a differently-trained
model would confound "my architecture is better" with "my model is bigger",
which is the most common way benchmark tables mislead.

B4 -> V is the row that carries the project's claim. If VERITAS does not beat
temporal RAG on CONFLICT, INSUFFICIENT and OUTDATED_SOURCE, the extra machinery
is not earning its complexity, and the honest thing is to report that.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from ..evidence.provenance import Answer
from ..evidence.quality import SupportLevel
from .benchmark import BenchItem, Benchmark, Category
from .metrics import (
    AbstentionScore, RunMetrics, answer_match, citation_accuracy, contradiction_detection,
    format_table, mrr, ndcg_at_k, recall_at_k, temporal_accuracy, token_f1,
    unsupported_claim_rate,
)


class BaselineSystem:
    """Common interface: question -> (Answer, retrieved doc ids)."""

    name = "baseline"

    def answer(self, item: BenchItem) -> tuple:  # (Answer, List[str])
        raise NotImplementedError


class LLMOnly(BaselineSystem):
    """No retrieval. Establishes the floor and exposes parametric guessing."""

    name = "B1 LLM-only"

    def __init__(self, model=None, tokenizer=None, device: str = "cpu") -> None:
        self.model, self.tok, self.device = model, tokenizer, device

    def answer(self, item: BenchItem):
        text = ""
        if self.model is not None and self.tok is not None:
            import torch

            prompt = f"<|bos|><|user|>{item.question}<|assistant|>"
            ids = torch.tensor([self.tok.encode(prompt)], device=self.device)
            out = self.model.generate(ids, max_new_tokens=60, temperature=0.2,
                                      eos_id=self.tok.special_tokens["<|eos|>"])
            text = self.tok.decode(out[0, ids.shape[1]:].tolist(), skip_special=True)
        a = Answer(question=item.question, answer=text.strip())
        a.support_level = SupportLevel.INSUFFICIENT  # it has no evidence, by construction
        return a, []


class BasicRAG(BaselineSystem):
    """Dense retrieval -> top-k -> answer. No fusion, no time, no verification."""

    name = "B2 Basic RAG"

    def __init__(self, vector_index, embedder, corpus, metadata, k: int = 5) -> None:
        self.vec, self.emb, self.corpus, self.md, self.k = vector_index, embedder, corpus, metadata, k

    def answer(self, item: BenchItem):
        hits = self.vec.search(self.emb.encode_one(item.question), k=self.k)
        ids = [h.doc_id for h in hits]
        # the characteristic failure: concatenate top-k and assert it, with no
        # regard for which chunk is current
        text = " ".join(self.corpus.get(i, "") for i in ids[:2])
        a = Answer(question=item.question, answer=text.strip())
        a.support_level = SupportLevel.MODERATE
        return a, ids


class HybridRAG(BaselineSystem):
    """Dense + BM25 + fusion (+ reranker). Better retrieval, same reasoning."""

    name = "B3 Hybrid RAG"

    def __init__(self, retriever, corpus, metadata, k: int = 5, reranker=None) -> None:
        self.r, self.corpus, self.md, self.k, self.rr = retriever, corpus, metadata, k, reranker

    def answer(self, item: BenchItem):
        self.r.signal_fns = {}   # relevance only
        cands = self.r.retrieve(item.question, k=self.k, candidate_k=40)
        ids = [c.doc_id for c in cands]
        if self.rr is not None and ids:
            ranked = self.rr.rerank(item.question, [self.corpus.get(i, "") for i in ids], ids)
            ids = [i for i, _ in ranked]
        text = " ".join(self.corpus.get(i, "") for i in ids[:2])
        a = Answer(question=item.question, answer=text.strip())
        a.support_level = SupportLevel.MODERATE
        return a, ids


class TemporalRAG(BaselineSystem):
    """Adds freshness + validity signals. The strongest published pattern."""

    name = "B4 Temporal RAG"

    def __init__(self, retriever, corpus, metadata, k: int = 5) -> None:
        self.r, self.corpus, self.md, self.k = retriever, corpus, metadata, k

    def answer(self, item: BenchItem):
        from ..temporal.temporal_retrieval import make_signal_fns, parse_temporal_query

        tq = parse_temporal_query(item.question, item.attribute)
        self.r.signal_fns = make_signal_fns(tq, item.entity, self.corpus)
        self.r.metadata = self.md
        cands = self.r.retrieve(item.question, k=self.k, candidate_k=40,
                                query_ctx={"entity": item.entity})
        ids = [c.doc_id for c in cands]
        text = " ".join(self.corpus.get(i, "") for i in ids[:2])
        a = Answer(question=item.question, answer=text.strip())
        a.support_level = SupportLevel.MODERATE
        a.last_verified = str(self.md.get(ids[0], {}).get("date")) if ids else None
        return a, ids


class VeritasSystem(BaselineSystem):
    name = "VERITAS"

    def __init__(self, pipeline) -> None:
        self.p = pipeline

    def answer(self, item: BenchItem):
        a = self.p.answer(item.question)
        ids = [c.doc_id for c in a.citations]
        return a, ids


#: Phrases that qualify a fact as possibly no longer current. Required for
#: OUTDATED_SOURCE credit: stating the stale value *as if current* is the
#: failure the category exists to catch, so the string match alone is not
#: enough -- the hedge has to be there too.
_STALENESS_CUES = (
    "valid until", "last verified", "as of", "no current evidence", "not confirmed",
    "was ", "previously", "earlier period", "may no longer", "at the time",
)


def _is_correct(item: BenchItem, ans) -> bool:
    """Per-category correctness. One rule per category, because "did it answer
    correctly" means a different thing for an abstention item than for a
    conflict item."""
    if item.should_abstain:
        return bool(ans.abstained)
    if item.has_conflict:
        return bool(ans.conflicts) or ans.support_level == SupportLevel.CONFLICTED
    if item.category == Category.OUTDATED_SOURCE:
        low = ans.answer.lower()
        return (answer_match(ans.answer, item.expected_answer)
                and any(c in low for c in _STALENESS_CUES))
    if item.category == Category.CHANGE:
        # every state on the timeline must appear, not just the newest
        return all(answer_match(ans.answer, [g]) for g in item.expected_answer)
    return answer_match(ans.answer, item.expected_answer)


# ------------------------------------------------------------------ harness
def evaluate(system: BaselineSystem, bench: Benchmark, k: int = 5, verbose: bool = False
             ) -> RunMetrics:
    m = RunMetrics(name=system.name, n=len(bench.items))
    acc = f1 = cov = cite = unsup = temp = 0.0
    conf_hits = conf_n = 0
    rec = ndcg = 0.0
    lat = 0.0
    per_cat: Dict[str, List[float]] = {}

    for item in bench.items:
        t0 = time.time()
        ans, retrieved = system.answer(item)
        lat += time.time() - t0

        correct = _is_correct(item, ans)
        acc += float(correct)
        if item.expected_answer:
            f1 += max(token_f1(ans.answer, g) for g in item.expected_answer)
        cov += ans.coverage
        cite += citation_accuracy(ans)
        unsup += unsupported_claim_rate(ans)
        if item.expected_answer:
            temp += temporal_accuracy(ans, item.expected_answer[0], item.expected_outdated)
        cd = contradiction_detection(ans, item.has_conflict)
        if cd is not None:
            conf_n += 1
            conf_hits += int(cd)
        # Retrieval returns CHUNK ids ("doc#3"); gold labels are DOCUMENT ids.
        # Comparing them directly scores every correct retrieval as a miss.
        doc_ids, seen_docs = [], set()
        for cid in retrieved:
            d = cid.split("#")[0]
            if d not in seen_docs:
                seen_docs.add(d)
                doc_ids.append(d)
        rec += recall_at_k(doc_ids, item.gold_docs, k)
        ndcg += ndcg_at_k(doc_ids, item.relevance_map, 10)
        m.abstention.update(item.should_abstain, ans.abstained)
        per_cat.setdefault(item.category, []).append(float(correct))

        if verbose:
            print(f"  [{item.category}] {item.qid}: {'OK' if correct else 'MISS'} "
                  f"| abstained={ans.abstained} | {ans.answer[:70]}")

    n = max(1, len(bench.items))
    m.accuracy, m.f1, m.coverage = acc / n, f1 / n, cov / n
    m.citation_accuracy, m.unsupported_rate, m.temporal_accuracy = cite / n, unsup / n, temp / n
    m.conflict_detection = conf_hits / conf_n if conf_n else 0.0
    m.recall_at_5, m.ndcg_at_10, m.mean_latency_s = rec / n, ndcg / n, lat / n
    m.by_category = {c: sum(v) / len(v) for c, v in per_cat.items()}
    return m


def compare(systems: Sequence[BaselineSystem], bench: Benchmark, k: int = 5) -> str:
    results = [evaluate(s, bench, k) for s in systems]
    table = format_table([r.as_row() for r in results])
    cats = sorted({c for r in results for c in r.by_category})
    rows = [{"system": r.name, **{c: round(r.by_category.get(c, float("nan")), 2) for c in cats}}
            for r in results]
    return f"{table}\n\nAccuracy by category:\n{format_table(rows)}"


def freshness_lag(pipeline, veritas, question: str, source_id: str, new_text: str,
                  entity: str, published: str) -> Dict[str, object]:
    """Measure spec section 35's demo: answer, inject a new source, re-answer.

    Reports the wall-clock cost of incorporating new information and whether the
    answer actually moved. A system that requires a re-index or a retrain cannot
    produce a number here at all -- that is the point of the metric.
    """
    before = veritas.answer(question)
    t0 = time.time()
    res = pipeline.ingest(source_id, new_text, entity_hint=entity, published=published)
    ingest_s = time.time() - t0
    after = veritas.answer(question)
    return {
        "answer_before": before.answer,
        "answer_after": after.answer,
        "changed": before.answer != after.answer,
        "ingest_seconds": round(ingest_s, 4),
        "state_changes": res.state_changes,
        "invalidated_cache_keys": res.invalidated,
        "retrain_required": False,
    }
