"""Query Planner agent: decide what to look for before looking.

Why plan at all? A single-shot retriever sends the user's raw words to the
index. That fails on three shapes of question this system must handle:

  * **multi-hop**  -- "who runs the company that acquired X?" needs two
    retrievals, and the second query cannot be written until the first
    returns.
  * **comparative/temporal** -- "how did the status change?" needs two
    retrievals against *different time anchors*, then a diff.
  * **under-specified** -- "what's the latest?" has no entity at all; the
    planner must resolve it from conversation history or ask.

The plan is an explicit, inspectable object: entity, attribute, temporal
intent, sub-queries, the evidence required to consider the question answered,
and the stopping condition. Making it explicit is what allows the loop in
`orchestrator.py` to decide *sufficiency* instead of stopping after a fixed
number of retrievals.

Rules + optional LM. The rule path handles the common shapes deterministically
and costs microseconds; the fine-tuned model refines entity/attribute when the
rules find nothing. The planner never invents an entity that is absent from the
question -- a hallucinated plan produces confidently retrieved evidence about
the wrong subject.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..evidence.claims import ATTRIBUTE_LEXICON
from ..temporal.temporal_retrieval import TemporalIntent, TemporalQuery, parse_temporal_query

_QTYPE = {
    "CURRENT_STATE": ("what is", "who is", "current", "now", "latest", "status of"),
    "HISTORICAL": ("what was", "who was", "in 20", "back in", "used to"),
    "CHANGE": ("how did", "change", "when did", "why did", "timeline", "evolve"),
    "COMPARISON": ("compare", "versus", " vs ", "difference between"),
    "MULTI_HOP": ("that acquired", "whose", "of the company that", "owned by"),
    "EXISTENCE": ("is there", "does ", "has ", "did "),
}

_ENTITY = re.compile(r"\b([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,3})\b")
_STOPHEADS = frozenset("What Who When Where Why How Is Are Was Were Does Did Has Have The A An".split())


@dataclass
class QueryPlan:
    question: str
    question_type: str = "CURRENT_STATE"
    entity: str = ""
    attribute: str = ""
    temporal: Optional[TemporalQuery] = None
    sub_queries: List[str] = field(default_factory=list)
    required_evidence: List[str] = field(default_factory=list)
    domain: str = "government"
    max_iterations: int = 3
    min_independent_sources: int = 2
    needs_both_states: bool = False

    def describe(self) -> str:
        return (f"type={self.question_type} entity='{self.entity}' attr='{self.attribute}' "
                f"time={self.temporal.intent if self.temporal else '?'} "
                f"subqueries={len(self.sub_queries)}")


class QueryPlanner:
    def __init__(self, model=None, tokenizer=None, domain_hint: str = "corporate") -> None:
        self.model = model
        self.tok = tokenizer
        self.domain_hint = domain_hint

    def _entity(self, q: str) -> str:
        """Longest capitalised phrase, after trimming leading question words.

        A sentence-initial capital ("Has Helios Energy started...") otherwise
        glues the question word onto the entity, and dropping the whole
        candidate loses the entity entirely -- which sends the whole pipeline
        after the wrong subject.
        """
        cands = []
        for m in _ENTITY.finditer(q):
            words = m.group(1).split()
            while words and words[0] in _STOPHEADS:
                words = words[1:]
            if words:
                cands.append(" ".join(words).strip())
        return max(cands, key=len) if cands else ""

    def _attribute(self, q: str) -> str:
        low = q.lower()
        best, best_len = "", 0
        for canon, words in ATTRIBUTE_LEXICON.items():
            for w in words:
                if w in low and len(w) > best_len:
                    best, best_len = canon, len(w)
        return best

    def _qtype(self, q: str) -> str:
        low = q.lower()
        for qt, cues in _QTYPE.items():
            if any(c in low for c in cues):
                return qt
        return "CURRENT_STATE"

    def plan(self, question: str, history: Optional[Sequence[str]] = None) -> QueryPlan:
        entity = self._entity(question)
        if not entity and history:
            for prev in reversed(list(history)):  # carry the subject forward
                entity = self._entity(prev)
                if entity:
                    break
        attribute = self._attribute(question)
        qtype = self._qtype(question)
        tq = parse_temporal_query(question, attribute)

        # Sub-queries widen lexical coverage without changing the meaning. The
        # entity+attribute form is the one that actually hits a filing; the raw
        # question is kept because it carries phrasing the index may match.
        subs = [question]
        if entity and attribute:
            subs.append(f"{entity} {attribute}")
            if tq.intent == TemporalIntent.CURRENT:
                subs.append(f"{entity} current {attribute} latest announcement")
            elif tq.intent in (TemporalIntent.HISTORICAL, TemporalIntent.AS_OF):
                subs.append(f"{entity} {attribute} {tq.anchor.year}")
            elif tq.intent == TemporalIntent.CHANGE:
                # a change question needs BOTH endpoints retrieved
                subs.append(f"{entity} {attribute} appointed announced effective")
                subs.append(f"{entity} former previous {attribute}")

        required = [f"a source stating {entity or 'the entity'}'s {attribute or 'attribute'}"]
        if tq.intent == TemporalIntent.CHANGE:
            required.append("evidence of the prior state")
            required.append("evidence of the effective date of the change")
        if tq.intent == TemporalIntent.CURRENT:
            required.append("a source dated after the most recent known change")

        return QueryPlan(
            question=question, question_type=qtype, entity=entity, attribute=attribute,
            temporal=tq, sub_queries=subs, required_evidence=required,
            domain=self.domain_hint,
            max_iterations=4 if qtype in ("MULTI_HOP", "CHANGE") else 3,
            min_independent_sources=2 if qtype != "EXISTENCE" else 1,
            needs_both_states=tq.wants_both_states,
        )

    def refine(self, plan: QueryPlan, gap: str) -> str:
        """Write the next query given what the evidence is missing.

        Refinement targets the gap rather than rephrasing the question: the
        index already answered the original phrasing, so asking it again in
        different words mostly returns the same chunks.
        """
        e, a = plan.entity or "", plan.attribute or ""
        if "prior state" in gap:
            return f"{e} previous former {a} before"
        if "effective date" in gap:
            return f"{e} {a} effective date announcement filing"
        if "current" in gap or "after the most recent" in gap:
            return f"{e} {a} latest update {plan.temporal.anchor.year if plan.temporal else ''}"
        if "independent" in gap:
            return f"{e} {a} confirmed official statement"
        return f"{e} {a} {gap}"
