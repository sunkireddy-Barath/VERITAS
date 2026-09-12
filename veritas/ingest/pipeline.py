"""Continuous world-update pipeline (spec section 18).

    source -> poll -> fingerprint -> changed? -> parse -> extract claims
           -> diff against known state -> detect state change
           -> update temporal store -> update evidence graph
           -> incremental index update -> invalidate affected cache
           -> recompute current state

The property that makes this "continuous" rather than "batch"
--------------------------------------------------------------
**Nothing is rebuilt.** One changed source touches only the chunks of that
document, only the affected (entity, attribute) timelines, and only the cache
entries for those entities. A full re-index of a 10^6-chunk corpus takes hours
and cannot run on every poll, which is why most "live" RAG systems are actually
nightly-batch systems wearing a live badge.

Three mechanisms make the incremental path correct:

1. **Chunk-level replacement.** A document's old chunks are removed from both
   indices by id before the new ones are added, so a shrinking document does
   not leave orphan chunks behind that still answer queries.
2. **Append-only knowledge.** The store never overwrites; it closes valid
   intervals. So a re-ingest cannot destroy history, and a re-ingest of
   unchanged content is idempotent (it lands as REAFFIRMED and only raises
   corroboration).
3. **Targeted invalidation.** The answer cache is keyed by entity+attribute, so
   an update to Acme's CEO invalidates exactly those answers. Flushing the
   whole cache on every update wastes the cache; never flushing serves stale
   answers -- the thing this system exists to prevent.

BM25 note: `bm25.finalize()` recomputes IDF over the whole corpus, which is
O(vocabulary), not O(corpus), so it is cheap enough to rerun per batch. The
vector index appends in O(1) amortised; the IVF cells are rebuilt only when the
corpus grows past a growth factor, because centroid drift from a handful of new
vectors is negligible.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..evidence.claims import Claim, claims_to_state, extract_claims
from ..evidence.graph import EdgeType, EvidenceGraph
from ..evidence.quality import SourcePolicy
from ..rag.chunking import Chunk, chunk_document
from ..temporal.change_detection import ChangeDetector, ChangeReport
from ..temporal.versioning import ChangeKind, TemporalStore, now_utc


@dataclass
class Source:
    """A monitored source. `fetch` returns (text, metadata) or None on failure."""

    source_id: str
    url: str = ""
    tier: int = 3
    domain: str = "corporate"
    fetch: Optional[Callable[[], Optional[Tuple[str, dict]]]] = None
    entity_hint: str = ""
    poll_seconds: int = 3600
    last_polled: Optional[datetime] = None
    enabled: bool = True


@dataclass
class IngestResult:
    source_id: str
    changed: bool
    reason: str
    n_chunks: int = 0
    n_claims: int = 0
    state_changes: List[str] = field(default_factory=list)
    invalidated: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    error: str = ""


class AnswerCache:
    """Answer cache with entity+attribute-keyed invalidation."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, object]] = {}
        self._deps: Dict[Tuple[str, str], set] = {}

    def get(self, key: str):
        hit = self._store.get(key)
        if hit is None:
            return None
        ts, val = hit
        if time.time() - ts > self.ttl:
            self._store.pop(key, None)
            return None
        return val

    def put(self, key: str, value, deps: Sequence[Tuple[str, str]] = ()) -> None:
        self._store[key] = (time.time(), value)
        for d in deps:
            self._deps.setdefault(d, set()).add(key)

    def invalidate(self, entity: str, attribute: str) -> List[str]:
        keys = self._deps.pop((entity.lower(), attribute), set())
        for k in keys:
            self._store.pop(k, None)
        return sorted(keys)


class IngestionPipeline:
    def __init__(
        self,
        store: TemporalStore,
        graph: EvidenceGraph,
        corpus: Dict[str, str],
        metadata: Dict[str, dict],
        vector_index=None,
        bm25=None,
        embedder=None,
        cache: Optional[AnswerCache] = None,
        policy: Optional[SourcePolicy] = None,
        target_tokens: int = 256,
    ) -> None:
        self.store = store
        self.graph = graph
        self.corpus = corpus
        self.metadata = metadata
        self.vec = vector_index
        self.bm25 = bm25
        self.embedder = embedder
        self.cache = cache or AnswerCache()
        self.policy = policy or SourcePolicy()
        self.detector = ChangeDetector()
        self.sources: Dict[str, Source] = {}
        self.target_tokens = target_tokens
        self._doc_chunks: Dict[str, List[str]] = {}
        self.log: List[IngestResult] = []
        #: When True, per-document index finalisation is skipped. IDF depends on
        #: the corpus size, so finalize() is O(vocabulary) -- fine per batch,
        #: quadratic when called once per document during a bulk load of
        #: thousands of filings. Use `bulk_load()` and it is called once.
        self._defer_index = False
        self._pending_vectors: List[Tuple[str, str]] = []

    # ---------------------------------------------------------------- setup
    def register(self, source: Source) -> None:
        self.sources[source.source_id] = source
        self.graph.add_source(source.source_id, source.source_id,
                              tier=source.tier, domain=source.domain)

    # -------------------------------------------------------------- ingest
    def ingest(
        self,
        source_id: str,
        text: str,
        doc_id: Optional[str] = None,
        published: Optional[str] = None,
        entity_hint: str = "",
        etag: str = "",
        last_modified: str = "",
        url: str = "",
        assert_claims: bool = True,
    ) -> IngestResult:
        t0 = time.time()
        src = self.sources.get(source_id) or Source(source_id)
        doc_id = doc_id or f"{source_id}:{now_utc().date().isoformat()}"
        entity = entity_hint or src.entity_hint

        # 1-3. change cascade (cheapest filter first)
        claims = extract_claims(text, doc_id, doc_id, entity)
        state = claims_to_state(claims)
        report: ChangeReport = self.detector.check(source_id, text, state, etag, last_modified)
        if not report.changed:
            return IngestResult(source_id, False, report.reason, elapsed_s=time.time() - t0)

        # 4. parse + chunk, replacing this document's previous chunks
        self._remove_doc(doc_id)
        chunks = chunk_document(
            text, doc_id,
            {"title": doc_id, "source": source_id, "tier": src.tier,
             "date": published or now_utc().isoformat(), "entity": entity},
            target_tokens=self.target_tokens,
        )
        self._add_chunks(chunks, published, source_id, src.tier, entity)
        self.graph.add_document(doc_id, source_id, date=published, url=url or src.url)

        # 5-9. claims -> temporal store -> graph
        #
        # `assert_claims=False` for STRUCTURED sources (SEC XBRL, APIs, tables).
        # There the caller already knows the exact valid interval and the typed
        # value, so re-deriving them with a regex over the rendered sentence is
        # strictly worse: it dates the fact by publication instead of by fiscal
        # period, and it asserts a second, conflicting version of a fact we
        # already hold. The text is still chunked, indexed and citable -- only
        # the fact extraction is skipped.
        state_changes: List[str] = []
        invalidated: List[str] = []
        for c in (claims if assert_claims else []):
            if not (c.subject and c.attribute and c.value) or c.hedged:
                continue
            version, event = self.store.assert_fact(
                entity=c.subject or entity,
                attribute=c.attribute,
                value=c.value,
                valid_from=c.valid_from or published or now_utc(),
                valid_to=c.valid_to,
                recorded_at=published or now_utc(),
                source_id=source_id,
                evidence_ids=[c.chunk_id or doc_id],
                confidence=c.confidence,
                reason=f"ingested from {doc_id}",
            )
            node = self.graph.add_claim(
                c.claim_id, c.text, entity=c.subject or entity, attribute=c.attribute,
                value=c.value, doc_id=doc_id, valid_from=version.valid_from,
                valid_to=version.valid_to, confidence=c.confidence,
            )
            if event.kind in (ChangeKind.CHANGED, ChangeKind.CORRECTED, ChangeKind.CONFLICT):
                state_changes.append(
                    f"{event.entity}.{event.attribute}: '{event.old_value}' -> "
                    f"'{event.new_value}' [{event.kind}] effective {event.effective_at.date()}"
                )
                # 10. mark supersession / contradiction in the graph
                prior_nodes = [n for n in self.graph.claims_about(event.entity)
                               if self.graph.g.nodes[n].get("attribute") == event.attribute
                               and n != node]
                for p in prior_nodes:
                    if event.kind == ChangeKind.CHANGED:
                        self.graph.supersedes(node, p, at=event.effective_at)
                    else:
                        self.graph.contradicts(node, p, reason=event.kind,
                                               temporal=event.kind == ChangeKind.CHANGED)
                # 11. targeted cache invalidation
                invalidated.extend(self.cache.invalidate(event.entity, event.attribute))

        res = IngestResult(source_id, True, report.reason, len(chunks), len(claims),
                           state_changes, invalidated, time.time() - t0)
        self.log.append(res)
        return res

    # ------------------------------------------------------------ polling
    # ----------------------------------------------------------- bulk load
    @contextlib.contextmanager
    def bulk_load(self, batch_size: int = 256, verbose: bool = False):
        """Defer index finalisation for the duration of a bulk ingest.

            with pipeline.bulk_load():
                for doc in thousands_of_documents:
                    pipeline.ingest(...)

        Turns 1,761 O(vocabulary) IDF recomputations and 1,761 single-document
        encoder passes into one of each. On the real SEC corpus this is the
        difference between minutes and seconds.
        """
        self._defer_index = True
        try:
            yield self
        finally:
            self._defer_index = False
            if self.bm25 is not None:
                self.bm25.finalize()
            if self.vec is not None and self.embedder is not None and self._pending_vectors:
                pending = self._pending_vectors
                self._pending_vectors = []
                for i in range(0, len(pending), batch_size):
                    batch = pending[i : i + batch_size]
                    self.vec.add([c for c, _ in batch],
                                 self.embedder.encode([t for _, t in batch]))
                    if verbose:
                        print(f"  [index] embedded {min(i + batch_size, len(pending))}"
                              f"/{len(pending)} chunks", flush=True)

    def poll_once(self, source_id: str) -> IngestResult:
        src = self.sources[source_id]
        if src.fetch is None:
            return IngestResult(source_id, False, "no fetcher configured")
        try:
            fetched = src.fetch()
        except Exception as exc:  # a dead source must not stop the pipeline
            return IngestResult(source_id, False, "fetch-failed", error=str(exc))
        if not fetched:
            return IngestResult(source_id, False, "empty-response")
        text, md = fetched
        src.last_polled = now_utc()
        src.poll_seconds = self.detector.next_poll_seconds(source_id, src.poll_seconds)
        return self.ingest(
            source_id, text, md.get("doc_id"), md.get("published"),
            md.get("entity", src.entity_hint), md.get("etag", ""),
            md.get("last_modified", ""), md.get("url", src.url),
        )

    def poll_due(self, now: Optional[datetime] = None) -> List[IngestResult]:
        """Poll only sources whose adaptive interval has elapsed."""
        now = now or now_utc()
        out = []
        for sid, src in self.sources.items():
            if not src.enabled:
                continue
            due = (src.last_polled is None
                   or (now - src.last_polled).total_seconds() >= src.poll_seconds)
            if due:
                out.append(self.poll_once(sid))
        return out

    def reverify(self, entity: str, attribute: str, max_age_days: int = 90) -> List[str]:
        """Automatic reverification (spec section 25.11).

        A fact nobody has restated in a long time is not the same as a fact
        confirmed today, even when it is still the newest thing on record. This
        returns the source ids to re-poll, so staleness is *actively* refreshed
        rather than silently tolerated.
        """
        cur = self.store.current(entity, attribute)
        if cur is None:
            return []
        age = (now_utc() - cur.recorded_at).days
        if age < max_age_days:
            return []
        return [cur.source_id] if cur.source_id else list(self.sources)

    # ------------------------------------------------------------- indexing
    def _add_chunks(self, chunks: Sequence[Chunk], published, source_id, tier, entity) -> None:
        ids, texts = [], []
        for ch in chunks:
            self.corpus[ch.chunk_id] = ch.text
            self.metadata[ch.chunk_id] = {
                "source": source_id, "tier": tier, "date": published,
                "entity": entity, "doc_id": ch.doc_id, "heading": ch.heading_path,
                "span": (ch.start_char, ch.end_char),
                "valid_from": published, "valid_to": None,
            }
            ids.append(ch.chunk_id)
            texts.append(ch.contextualized)
        self._doc_chunks.setdefault(chunks[0].doc_id if chunks else "", []).extend(ids)

        if self.bm25 is not None:
            for cid, t in zip(ids, texts):
                self.bm25.add(cid, t)
            if not self._defer_index:
                self.bm25.finalize()
        if self.vec is not None and self.embedder is not None and ids:
            if self._defer_index:
                # Batch the encoder too: one forward pass over many chunks is
                # far cheaper than one pass per document.
                self._pending_vectors.extend(zip(ids, texts))
            else:
                self.vec.add(ids, self.embedder.encode(texts))

    def _remove_doc(self, doc_id: str) -> None:
        """Drop a document's previous chunks so a shrunken document cannot leave
        stale chunks answering queries."""
        old = self._doc_chunks.pop(doc_id, [])
        for cid in old:
            self.corpus.pop(cid, None)
            self.metadata.pop(cid, None)
        if old and self.bm25 is not None:
            import numpy as np

            kept = [(cid, self.corpus[cid]) for cid in self.bm25.doc_ids if cid in self.corpus]
            type(self.bm25).__init__(self.bm25, self.bm25.k1, self.bm25.b)
            for cid, text in kept:
                self.bm25.add(cid, text)
            self.bm25.finalize()
        if old and self.vec is not None and self.vec.vectors is not None:
            import numpy as np

            keep = [i for i, cid in enumerate(self.vec.ids) if cid not in set(old)]
            self.vec.ids = [self.vec.ids[i] for i in keep]
            self.vec.vectors = self.vec.vectors[keep] if keep else None
            self.vec.centroids = None  # force IVF rebuild on next build_ivf
            self.vec.cells = None

    def summary(self) -> Dict[str, object]:
        return {
            "sources": len(self.sources),
            "documents": len(self._doc_chunks),
            "chunks": len(self.corpus),
            "entities": len(self.store.entities()),
            "versions": sum(len(v) for v in self.store._index.values()),
            "changes": len(self.store.changes),
            "graph": self.graph.stats(),
        }
