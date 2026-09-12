"""Hybrid retrieval: fuse dense, sparse and evidence-quality signals.

The fusion problem
------------------
BM25 scores are unbounded and corpus-dependent (0 .. ~40); cosine scores live in
[-1, 1]. Adding them directly lets BM25 silently dominate. Two principled
fixes, both implemented here:

**1. Reciprocal Rank Fusion (RRF)**   -- default

    RRF(d) = sum_r  w_r / (K + rank_r(d))        K = 60

Uses only *ranks*, so it is immune to score scale, calibration drift and
outliers. It cannot be fooled by one system's inflated magnitudes, needs no
tuning, and is the standard strong baseline (Cormack et al., 2009). The
constant K damps the influence of the very top rank so a single system cannot
unilaterally decide the winner.

**2. Normalised weighted sum**        -- when you need calibrated scores

Min-max normalise each signal into [0, 1] within the candidate set, then take a
weighted sum. This preserves *margins* (the gap between rank 1 and rank 2),
which RRF throws away -- and margins are exactly what the evidence-sufficiency
check later needs. Use this when the downstream consumer reads scores, not just
order.

Beyond relevance: why extra signals belong in retrieval, not just reranking
--------------------------------------------------------------------------
A classic RAG system ranks by "aboutness" alone. For a system that must
reconstruct *what is true now*, three more axes decide whether a chunk is good
*evidence*, and they must act before the top-k cut or the right document never
reaches the reranker:

* **freshness**    -- exponential decay exp(-ln2 * age / halflife). Half-life is
                      per-query, not global: "current CEO" decays in months,
                      "1969 moon landing" does not decay at all.
* **authority**    -- source tier prior (see evidence/quality.py).
* **entity match** -- does the chunk actually mention the entity asked about?
                      Cheap, and it kills the classic failure where a
                      topically-similar chunk about a *different* company wins.

Temporal relevance (is this chunk valid at the asked-about time?) is supplied by
`temporal/temporal_retrieval.py` and enters here as one more signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FusionWeights:
    dense: float = 1.0
    sparse: float = 1.0
    freshness: float = 0.3
    authority: float = 0.3
    entity: float = 0.4
    temporal: float = 0.6

    def as_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Candidate:
    doc_id: str
    dense: float = 0.0
    sparse: float = 0.0
    freshness: float = 0.0
    authority: float = 0.0
    entity: float = 0.0
    temporal: float = 0.0
    score: float = 0.0
    explain: Dict[str, float] = field(default_factory=dict)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def rrf_fuse(
    rankings: Sequence[Sequence[str]], weights: Optional[Sequence[float]] = None, k: int = 60
) -> List[Tuple[str, float]]:
    """Reciprocal rank fusion over any number of ranked id lists."""
    weights = weights or [1.0] * len(rankings)
    acc: Dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            acc[doc_id] = acc.get(doc_id, 0.0) + w / (k + rank + 1)
    return sorted(acc.items(), key=lambda kv: -kv[1])


def weighted_fuse(cands: List[Candidate], w: FusionWeights) -> List[Candidate]:
    """Min-max normalise every signal across the candidate set, then combine."""
    if not cands:
        return []
    fields = ("dense", "sparse", "freshness", "authority", "entity", "temporal")
    norm = {f: _minmax(np.array([getattr(c, f) for c in cands], dtype=np.float32)) for f in fields}
    wd = w.as_dict()
    total_w = sum(abs(v) for v in wd.values()) or 1.0
    for i, c in enumerate(cands):
        parts = {f: wd[f] * float(norm[f][i]) for f in fields}
        c.explain = parts
        c.score = sum(parts.values()) / total_w
    return sorted(cands, key=lambda c: -c.score)


class HybridRetriever:
    """Dense + sparse retrieval with pluggable evidence-quality signals.

    Two-stage by design: each backend returns `candidate_k` (>> final k) ids
    cheaply, the union is scored with the expensive signals, and only then is it
    cut to k. Applying quality signals *after* a tight top-k would be pointless
    -- the good evidence would already have been discarded.
    """

    def __init__(
        self,
        vector_index,
        bm25_index,
        embedder,
        weights: Optional[FusionWeights] = None,
        signal_fns: Optional[Dict[str, Callable[[str, dict], float]]] = None,
        metadata: Optional[Dict[str, dict]] = None,
    ) -> None:
        self.vec = vector_index
        self.bm25 = bm25_index
        self.embedder = embedder
        self.weights = weights or FusionWeights()
        self.signal_fns = signal_fns or {}
        self.metadata = metadata or {}

    def retrieve(
        self,
        query: str,
        k: int = 10,
        candidate_k: int = 50,
        mode: str = "weighted",
        query_ctx: Optional[dict] = None,
    ) -> List[Candidate]:
        ctx = query_ctx or {}
        qv = self.embedder.encode_one(query)
        dense_hits = self.vec.search(qv, k=candidate_k)
        sparse_hits = self.bm25.search(query, k=candidate_k)

        if mode == "rrf":
            fused = rrf_fuse(
                [[h.doc_id for h in dense_hits], [d for d, _ in sparse_hits]],
                [self.weights.dense, self.weights.sparse],
            )
            return [Candidate(doc_id=d, score=s) for d, s in fused[:k]]

        dmap = {h.doc_id: h.score for h in dense_hits}
        smap = dict(sparse_hits)
        cands: List[Candidate] = []
        for doc_id in set(dmap) | set(smap):
            md = self.metadata.get(doc_id, {})
            c = Candidate(
                doc_id=doc_id,
                dense=dmap.get(doc_id, 0.0),
                sparse=smap.get(doc_id, 0.0),
            )
            for name in ("freshness", "authority", "entity", "temporal"):
                fn = self.signal_fns.get(name)
                if fn is not None:
                    setattr(c, name, float(fn(doc_id, {**md, **ctx})))
            cands.append(c)
        return weighted_fuse(cands, self.weights)[:k]
