"""Research agents: retrieval, temporal reasoning, sufficiency, abstention.

Four small agents with one responsibility each, rather than one prompt that
does everything. The reason is testability: each has a typed input and output,
so each can be measured on its own (retrieval recall, temporal accuracy,
abstention quality are separate metrics in `eval/`), and a regression can be
localised. A monolithic agent can only be measured end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..evidence.claims import Claim, extract_claims
from ..evidence.quality import SourcePolicy, SupportLevel, score_evidence, support_label
from ..evidence.verifier import EvidenceItem
from ..temporal.temporal_retrieval import (
    TemporalIntent, TemporalQuery, interval_overlap_score, make_signal_fns,
)
from ..temporal.versioning import FOREVER, Version, now_utc


# --------------------------------------------------------------- retrieval
class RetrievalAgent:
    """Runs the planner's sub-queries through hybrid retrieval and dedupes.

    Sub-queries are unioned by best-score rather than concatenated into one
    long query: a long query dilutes both the embedding (averaging several
    intents into one vector) and BM25 (IDF mass spread across more terms).
    Several focused queries retrieve strictly better than one broad one.
    """

    def __init__(self, retriever, corpus: Dict[str, str], metadata: Dict[str, dict],
                 reranker=None) -> None:
        self.retriever = retriever
        self.corpus = corpus
        self.metadata = metadata
        self.reranker = reranker

    def search(
        self, queries: Sequence[str], plan, k: int = 8, candidate_k: int = 50
    ) -> List[EvidenceItem]:
        best: Dict[str, float] = {}
        signal_fns = make_signal_fns(plan.temporal, plan.entity, self.corpus)
        self.retriever.signal_fns = signal_fns
        self.retriever.metadata = self.metadata

        for q in queries:
            for c in self.retriever.retrieve(q, k=k, candidate_k=candidate_k,
                                             query_ctx={"entity": plan.entity}):
                if c.score > best.get(c.doc_id, -1):
                    best[c.doc_id] = c.score

        ids = sorted(best, key=lambda d: -best[d])[: k * 2]
        if self.reranker is not None and ids:
            texts = [self.corpus.get(i, "") for i in ids]
            ranked = self.reranker.rerank(queries[0], texts, ids, top_k=k)
            ids = [i for i, _ in ranked]

        out: List[EvidenceItem] = []
        for doc_id in ids[:k]:
            md = self.metadata.get(doc_id, {})
            out.append(EvidenceItem(
                doc_id=doc_id, text=self.corpus.get(doc_id, ""),
                source=str(md.get("source", doc_id)), date=md.get("date"),
                tier=int(md.get("tier", 3)),
                valid_from=md.get("valid_from"), valid_to=md.get("valid_to"),
                score=best.get(doc_id, 0.0),
            ))
        return out


# ---------------------------------------------------------------- temporal
@dataclass
class TemporalAssessment:
    current: Optional[Version]
    historical: List[Version]
    changes: List[str]
    stale_evidence: List[EvidenceItem]
    fresh_evidence: List[EvidenceItem]
    anchor: datetime
    note: str = ""
    #: False when `current` is the last KNOWN value but its validity interval
    #: has already closed (a finished fiscal year, say). The answer layer must
    #: then qualify it instead of asserting it in the present tense.
    current_is_valid_now: bool = True


class TemporalAgent:
    """Reconstructs current vs historical state and labels stale evidence."""

    def __init__(self, store) -> None:
        self.store = store

    def assess(self, plan, evidence: Sequence[EvidenceItem]) -> TemporalAssessment:
        tq: TemporalQuery = plan.temporal
        anchor = tq.anchor
        current = hist = None
        changes: List[str] = []

        current_valid_now = True
        if plan.entity and plan.attribute:
            current, current_valid_now = self.store.latest(plan.entity, plan.attribute)
            hist = self.store.history(plan.entity, plan.attribute)
            for v in hist or []:
                if v.change_kind in ("CHANGED", "CORRECTED"):
                    changes.append(
                        f"{plan.attribute}: '{v.previous_value}' -> '{v.value}' "
                        f"effective {v.valid_from.date()}, recorded {v.recorded_at.date()} "
                        f"({v.source_id or 'unknown source'})"
                    )

        fresh, stale = [], []
        for e in evidence:
            s = interval_overlap_score(e.valid_from, e.valid_to, tq)
            (fresh if s >= 0.5 else stale).append(e)

        note = ""
        if tq.intent == TemporalIntent.CURRENT and stale and not fresh:
            note = ("All retrieved evidence describes earlier periods; the current state "
                    "cannot be confirmed from what was retrieved.")
        elif tq.intent == TemporalIntent.CHANGE and len(hist or []) < 2:
            note = "Only one state is on record, so a change cannot be characterised."

        if current is not None and not current_valid_now:
            note = (note + " " if note else "") + (
                f"The most recent recorded value covers a period that has ended; "
                f"it is reported as last-known, not as currently valid.")
        return TemporalAssessment(current, list(hist or []), changes, stale, fresh,
                                  anchor, note, current_valid_now)


# -------------------------------------------------------------- sufficiency
@dataclass
class Sufficiency:
    sufficient: bool
    score: float
    factors: object
    label: str
    gaps: List[str] = field(default_factory=list)
    n_independent: int = 0


class EvidenceAgent:
    """Decides whether the retrieved evidence is enough to answer *yet*.

    This is the loop's stopping condition, and it is the component that makes
    the difference between "retrieved something" and "can answer". It checks
    the planner's `required_evidence` list explicitly, so the gap it reports is
    actionable -- the planner can turn it into the next query.
    """

    def __init__(self, policy: Optional[SourcePolicy] = None) -> None:
        self.policy = policy or SourcePolicy()

    def assess(self, plan, evidence: Sequence[EvidenceItem], assessment: TemporalAssessment,
               agreement: float = 1.0) -> Sufficiency:
        if not evidence:
            return Sufficiency(False, 0.0, None, SupportLevel.INSUFFICIENT,
                               ["no evidence retrieved at all"], 0)

        roots = {e.source or e.doc_id for e in evidence}
        validity = (len(assessment.fresh_evidence) / len(evidence)) if evidence else 0.0
        score, factors = score_evidence(
            sources=[e.source or e.doc_id for e in evidence],
            dates=[e.date for e in evidence],
            texts=[e.text for e in evidence],
            n_independent=len(roots),
            agreement=agreement,
            validity=validity,
            domain=plan.domain,
            halflife_days=plan.temporal.halflife_days if plan.temporal else 240.0,
        )
        label = support_label(score, len(roots), has_conflict=agreement < 0.6)

        gaps: List[str] = []
        if len(roots) < plan.min_independent_sources:
            gaps.append(f"only {len(roots)} independent source(s); need "
                        f"{plan.min_independent_sources}")
        if plan.temporal and plan.temporal.intent == TemporalIntent.CURRENT and not assessment.fresh_evidence:
            gaps.append("no evidence valid at the current time")
        if plan.needs_both_states and len(assessment.historical) < 2:
            gaps.append("evidence of the prior state is missing")
        if plan.temporal and plan.temporal.intent == TemporalIntent.CHANGE and not assessment.changes:
            gaps.append("the effective date of the change is missing")

        sufficient = not gaps and score >= 0.45
        return Sufficiency(sufficient, score, factors, label, gaps, len(roots))


# --------------------------------------------------------------- abstention
class AbstentionAgent:
    """The last gate: refuse rather than guess.

    Abstention is a *feature*, and it must be calibrated. Refusing too often is
    as useless as answering wrongly, so `eval/` scores abstention against
    questions that genuinely have no evidence (should abstain) and questions
    that do (should not). The thresholds below are the tuned knobs.
    """

    def __init__(self, min_score: float = 0.3, min_coverage: float = 0.4,
                 min_independent: int = 1) -> None:
        self.min_score = min_score
        self.min_coverage = min_coverage
        self.min_independent = min_independent

    def should_abstain(self, suff: Sufficiency, coverage: float, n_claims_supported: int,
                       n_claims_stale: int = 0, has_store_value: bool = False
                       ) -> Tuple[bool, str]:
        if suff.n_independent < self.min_independent:
            return True, ("No independent source in the retrieved evidence supports an answer "
                          "to this question.")
        if suff.score < self.min_score:
            return True, (f"The available evidence is too weak to answer "
                          f"(evidence score {suff.score:.2f}). "
                          f"Missing: {'; '.join(suff.gaps) if suff.gaps else 'corroboration'}.")
        if n_claims_supported == 0 and has_store_value:
            # The temporal store holds a sourced, dated value for this exact
            # entity+attribute -- it came from a tier-1 structured filing with
            # full provenance. Free-text claim verification adding no SUPPORTED
            # claim is a limitation of the text extractor, not an absence of
            # evidence. Refusing here would discard a fact we can cite, date and
            # attribute. The synthesis layer still qualifies it if its validity
            # interval has closed.
            return False, ""
        if n_claims_supported == 0 and n_claims_stale > 0:
            # Evidence EXISTS and is entailed -- it simply describes a period
            # that has closed. Refusing here would throw away a real, sourced
            # answer; the honest move is to give it with a staleness qualifier.
            # "Apple's last reported revenue was X for FY2025" is useful and
            # true; "I cannot establish this" is neither.
            return False, ""
        if n_claims_supported == 0:
            return True, ("None of the statements needed to answer could be verified against "
                          "the retrieved evidence.")
        # Coverage is a secondary gate, and only while support is thin.
        #
        # Unsupported claims are *dropped* from the answer by
        # `rewrite_unsupported`, so they never reach the reader. Refusing
        # because of claims that were already discarded is over-abstention:
        # it throws away a well-evidenced answer because the same documents
        # happened to contain unrelated sentences. Once two or more claims are
        # independently supported, the answer stands on those.
        if n_claims_supported < 2 and coverage < self.min_coverage:
            return True, (f"Only {coverage:.0%} of the candidate claims could be supported by "
                          f"evidence, and too few statements were verified to answer safely.")
        return False, ""
