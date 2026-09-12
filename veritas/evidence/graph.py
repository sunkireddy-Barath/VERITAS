"""The evidence graph: provenance as a queryable structure.

Why a graph rather than a table of (claim, source) pairs
--------------------------------------------------------
The questions VERITAS must answer are *path* questions, and paths are what
graphs are for:

  "why do you believe this?"        -> ANSWER -SUPPORTS- CLAIM -SUPPORTS- DOC -FROM- SOURCE
  "who reported it first?"          -> min(recorded_at) over incoming SUPPORTS
  "is this superseded?"             -> follow SUPERSEDES forward
  "do these two agree?"             -> is there a CONTRADICTS edge, or a shared
                                       ancestor making them non-independent?
  "are these three sources really
   independent?"                    -> do they share a DERIVED_FROM ancestor?

That last one is the reason a flat table is not enough. Three outlets all
rewriting the same wire story look like three-source corroboration in a table.
In a graph they converge on one ancestor, so the corroboration bonus is
correctly withheld. Independence is a *topological* property.

Node and edge types implement the spec's evidence backbone:

    nodes: Entity, Claim, Document, Source, Event, TimePoint, Attribute, Answer
    edges: SUPPORTS, CONTRADICTS, UPDATED_BY, DERIVED_FROM, VALID_DURING,
           ABOUT, SAME_ENTITY, SUPERSEDES, CITES

Backed by `networkx.MultiDiGraph`: multi-edge because two nodes can be related
in more than one way (a document can both SUPPORT and later CONTRADICT via a
correction), directed because provenance has a direction, and in-memory because
at research scale adjacency lookups are O(1) dict hits. The interface is
deliberately narrow (`add_*`, `evidence_for`, `provenance_path`) so it can be
swapped for Neo4j/ArangoDB without touching callers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx


class NodeType:
    ENTITY = "Entity"
    CLAIM = "Claim"
    DOCUMENT = "Document"
    SOURCE = "Source"
    EVENT = "Event"
    TIME = "TimePoint"
    ATTRIBUTE = "Attribute"
    ANSWER = "Answer"


class EdgeType:
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    UPDATED_BY = "UPDATED_BY"
    DERIVED_FROM = "DERIVED_FROM"
    VALID_DURING = "VALID_DURING"
    ABOUT = "ABOUT"
    SAME_ENTITY = "SAME_ENTITY"
    SUPERSEDES = "SUPERSEDES"
    CITES = "CITES"


class EvidenceGraph:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    # ---------------------------------------------------------------- nodes
    def add_node(self, node_id: str, ntype: str, **attrs) -> str:
        if self.g.has_node(node_id):
            self.g.nodes[node_id].update(attrs)
        else:
            self.g.add_node(node_id, ntype=ntype, **attrs)
        return node_id

    def add_source(self, source_id: str, name: str = "", tier: int = 3, domain: str = "", **kw) -> str:
        return self.add_node(f"src:{source_id}", NodeType.SOURCE, name=name or source_id,
                             tier=tier, domain=domain, **kw)

    def add_document(self, doc_id: str, source_id: str, date=None, url: str = "", **kw) -> str:
        n = self.add_node(f"doc:{doc_id}", NodeType.DOCUMENT,
                          date=_iso(date), url=url, **kw)
        self.g.add_edge(n, self.add_source(source_id), key=EdgeType.DERIVED_FROM,
                        etype=EdgeType.DERIVED_FROM)
        return n

    def add_entity(self, name: str, **kw) -> str:
        return self.add_node(f"ent:{name.strip().lower()}", NodeType.ENTITY, name=name.strip(), **kw)

    def add_claim(self, claim_id: str, text: str, entity: str = "", attribute: str = "",
                  value: str = "", doc_id: str = "", valid_from=None, valid_to=None,
                  confidence: float = 0.5, **kw) -> str:
        n = self.add_node(f"clm:{claim_id}", NodeType.CLAIM, text=text, attribute=attribute,
                          value=value, confidence=confidence,
                          valid_from=_iso(valid_from), valid_to=_iso(valid_to), **kw)
        if entity:
            self.g.add_edge(n, self.add_entity(entity), key=EdgeType.ABOUT, etype=EdgeType.ABOUT)
        if doc_id:
            d = f"doc:{doc_id}"
            if self.g.has_node(d):
                # doc SUPPORTS claim: direction is evidence -> claim, so
                # in-edges of a claim are exactly its evidence
                self.g.add_edge(d, n, key=EdgeType.SUPPORTS, etype=EdgeType.SUPPORTS,
                                weight=confidence)
        if valid_from or valid_to:
            t = self.add_node(f"time:{_iso(valid_from)}~{_iso(valid_to)}", NodeType.TIME,
                              start=_iso(valid_from), end=_iso(valid_to))
            self.g.add_edge(n, t, key=EdgeType.VALID_DURING, etype=EdgeType.VALID_DURING)
        return n

    # ---------------------------------------------------------------- edges
    def link(self, a: str, b: str, etype: str, **attrs) -> None:
        self.g.add_edge(a, b, key=etype, etype=etype, **attrs)

    def supports(self, doc_node: str, claim_node: str, weight: float = 1.0, span=None) -> None:
        self.link(doc_node, claim_node, EdgeType.SUPPORTS, weight=weight, span=span)

    def contradicts(self, claim_a: str, claim_b: str, reason: str = "", temporal: bool = False) -> None:
        """`temporal=True` marks "was true earlier" rather than "is wrong" --
        the distinction that keeps a superseded fact from being called an error."""
        self.link(claim_a, claim_b, EdgeType.CONTRADICTS, reason=reason, temporal=temporal)
        self.link(claim_b, claim_a, EdgeType.CONTRADICTS, reason=reason, temporal=temporal)

    def supersedes(self, newer_claim: str, older_claim: str, at=None) -> None:
        self.link(newer_claim, older_claim, EdgeType.SUPERSEDES, at=_iso(at))
        self.link(older_claim, newer_claim, EdgeType.UPDATED_BY, at=_iso(at))

    def same_entity(self, a: str, b: str, score: float = 1.0) -> None:
        self.link(self.add_entity(a), self.add_entity(b), EdgeType.SAME_ENTITY, score=score)

    # -------------------------------------------------------------- queries
    def evidence_for(self, claim_node: str) -> List[Dict]:
        """Documents with a SUPPORTS edge into this claim, newest first."""
        out = []
        for src, _dst, key, data in self.g.in_edges(claim_node, keys=True, data=True):
            if data.get("etype") == EdgeType.SUPPORTS and self.g.nodes[src].get("ntype") == NodeType.DOCUMENT:
                nd = self.g.nodes[src]
                out.append({"doc": src, "date": nd.get("date"), "url": nd.get("url"),
                            "weight": data.get("weight", 1.0), "span": data.get("span"),
                            "source": self.source_of(src)})
        return sorted(out, key=lambda d: d["date"] or "", reverse=True)

    def source_of(self, doc_node: str) -> Optional[str]:
        for _s, dst, data in self.g.out_edges(doc_node, data=True):
            if data.get("etype") == EdgeType.DERIVED_FROM:
                return dst
        return None

    def contradictions_of(self, claim_node: str) -> List[Tuple[str, Dict]]:
        return [(dst, data) for _s, dst, data in self.g.out_edges(claim_node, data=True)
                if data.get("etype") == EdgeType.CONTRADICTS]

    def claims_about(self, entity: str) -> List[str]:
        e = f"ent:{entity.strip().lower()}"
        if not self.g.has_node(e):
            return []
        return [s for s, _d, data in self.g.in_edges(e, data=True)
                if data.get("etype") == EdgeType.ABOUT]

    def independent_sources(self, claim_node: str) -> int:
        """Distinct source ANCESTORS, not distinct documents.

        Syndicated copies of one wire story collapse to a single root here, so
        three reprints count as one. This number feeds the corroboration term of
        the evidence score, and inflating it is the classic way a system
        convinces itself of a false fact.
        """
        roots: Set[str] = set()
        for ev in self.evidence_for(claim_node):
            doc = ev["doc"]
            root = self._derivation_root(doc)
            roots.add(root)
        return len(roots)

    def _derivation_root(self, node: str, depth: int = 8) -> str:
        """Walk DERIVED_FROM to the original source of a document."""
        cur = node
        for _ in range(depth):
            nxt = None
            for _s, dst, data in self.g.out_edges(cur, data=True):
                if data.get("etype") == EdgeType.DERIVED_FROM:
                    nxt = dst
                    break
            if nxt is None:
                return cur
            cur = nxt
        return cur

    def provenance_path(self, claim_node: str) -> Dict:
        """The full ANSWER -> CLAIM -> EVIDENCE -> SOURCE -> DATE trace."""
        nd = self.g.nodes[claim_node]
        ev = self.evidence_for(claim_node)
        return {
            "claim": nd.get("text"),
            "attribute": nd.get("attribute"),
            "value": nd.get("value"),
            "valid_from": nd.get("valid_from"),
            "valid_to": nd.get("valid_to"),
            "evidence": [
                {
                    "document": e["doc"].removeprefix("doc:"),
                    "source": (e["source"] or "").removeprefix("src:"),
                    "tier": self.g.nodes[e["source"]].get("tier") if e["source"] else None,
                    "date": e["date"],
                    "url": e["url"],
                    "span": e["span"],
                }
                for e in ev
            ],
            "independent_sources": self.independent_sources(claim_node),
            "contradictions": [
                {"claim": self.g.nodes[c].get("text"), "temporal": d.get("temporal"),
                 "reason": d.get("reason")}
                for c, d in self.contradictions_of(claim_node)
            ],
        }

    def first_reported(self, claim_node: str) -> Optional[Dict]:
        ev = self.evidence_for(claim_node)
        dated = [e for e in ev if e["date"]]
        return min(dated, key=lambda e: e["date"]) if dated else None

    # ---------------------------------------------------------------- stats
    def stats(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _n, d in self.g.nodes(data=True):
            counts[d.get("ntype", "?")] = counts.get(d.get("ntype", "?"), 0) + 1
        for _u, _v, d in self.g.edges(data=True):
            counts[d.get("etype", "?")] = counts.get(d.get("etype", "?"), 0) + 1
        counts["nodes"] = self.g.number_of_nodes()
        counts["edges"] = self.g.number_of_edges()
        return counts

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.g, edges="links")
        Path(path).write_text(json.dumps(data, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        eg = cls()
        eg.g = nx.node_link_graph(data, multigraph=True, directed=True, edges="links")
        return eg


def _iso(x) -> Optional[str]:
    if x is None:
        return None
    return x.isoformat() if isinstance(x, datetime) else str(x)
