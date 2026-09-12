"""One-call assembly of a working VERITAS instance.

Everything below is already implemented in the sub-packages; this module just
wires the standard configuration so notebooks and the API do not each re-do it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .agents.orchestrator import Veritas, VeritasConfig
from .evidence.graph import EvidenceGraph
from .evidence.quality import SourcePolicy
from .ingest.pipeline import AnswerCache, IngestionPipeline, Source
from .rag.bm25 import BM25
from .rag.embeddings import Embedder, VectorIndex
from .rag.hybrid import FusionWeights, HybridRetriever
from .temporal.versioning import TemporalStore


class VeritasSystemBuilder:
    """Builds store + graph + indices + retriever + orchestrator."""

    def __init__(self, model, tokenizer, device: str = "cpu",
                 weights: Optional[FusionWeights] = None, domain: str = "corporate") -> None:
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.store = TemporalStore()
        self.graph = EvidenceGraph()
        self.corpus: Dict[str, str] = {}
        self.metadata: Dict[str, dict] = {}
        self.embedder = Embedder(model, tokenizer, device=device)
        self.vec = VectorIndex(self.embedder.dim)
        self.bm25 = BM25()
        self.cache = AnswerCache()
        self.weights = weights or FusionWeights()
        self.domain = domain
        self.ingest = IngestionPipeline(
            self.store, self.graph, self.corpus, self.metadata,
            self.vec, self.bm25, self.embedder, self.cache, SourcePolicy(),
        )

    def add_source(self, source_id: str, tier: int = 3, domain: str = "", **kw) -> None:
        self.ingest.register(Source(source_id, tier=tier, domain=domain or self.domain, **kw))

    def add_document(self, source_id: str, doc_id: str, text: str, published: str,
                     entity: str = "", tier: Optional[int] = None, url: str = "",
                     assert_claims: bool = True):
        """Add a document. Set `assert_claims=False` when the caller supplies
        the facts itself from structured data (see ingest/real_sources.py)."""
        if source_id not in self.ingest.sources:
            self.add_source(source_id, tier or 3)
        return self.ingest.ingest(source_id, text, doc_id, published, entity, url=url,
                                  assert_claims=assert_claims)

    def retriever(self) -> HybridRetriever:
        return HybridRetriever(self.vec, self.bm25, self.embedder, self.weights,
                               metadata=self.metadata)

    def build(self, reranker=None, nli=None, config: Optional[VeritasConfig] = None) -> Veritas:
        if len(self.vec.ids) >= 64:
            self.vec.build_ivf()
        cfg = config or VeritasConfig(domain=self.domain, verbose=False)
        return Veritas(
            retriever=self.retriever(), corpus=self.corpus, metadata=self.metadata,
            store=self.store, graph=self.graph, model=self.model, tokenizer=self.tok,
            reranker=reranker, nli=nli, config=cfg,
        )


def load_benchmark_corpus(builder: "VeritasSystemBuilder", bench) -> int:
    """Ingest every benchmark document through the real ingest pipeline.

    Deliberately the real path, not a shortcut index build: the benchmark then
    also exercises change detection, claim extraction and the temporal store,
    so a bug there shows up as a benchmark regression.
    """
    seen = set()
    for item in bench.items:
        for d in item.docs:
            if d.doc_id in seen:
                continue
            seen.add(d.doc_id)
            builder.add_source(d.source, tier=d.tier)
            builder.add_document(d.source, d.doc_id, d.text, d.date, d.entity or item.entity)
            md = builder.metadata
            for cid in list(md):
                if md[cid].get("doc_id") == d.doc_id:
                    md[cid]["valid_from"] = d.valid_from
                    md[cid]["valid_to"] = d.valid_to
                    md[cid]["tier"] = d.tier
                    md[cid]["source"] = d.source
            if d.derived_from:
                builder.graph.link(f"doc:{d.doc_id}", f"doc:{d.derived_from}", "DERIVED_FROM")
    return len(seen)
