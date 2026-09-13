"""Provenance assembly: the answer object VERITAS actually returns.

An answer here is not a string. It is a structure that carries, for every
important claim, the chain

    ANSWER -> CLAIM -> EVIDENCE SPAN -> DOCUMENT -> SOURCE -> DATE

plus what changed, what is disputed, what is stale, and what could not be
established. The string is a *rendering* of that structure. Building it the
other way round -- generating prose and attaching citations afterwards -- is
how citation-shaped hallucination happens: the text is fixed first and the
citations are fitted to it.

The sections below implement spec section 23 exactly, and section 24's rule is
enforced in one place: `confidence` is always a support *label* plus its
factors, never a truth percentage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from ..temporal.versioning import FOREVER, Version, _dt, now_utc
from .contradiction import Conflict
from .quality import EvidenceFactors, SupportLevel, explain
from .verifier import ClaimVerdict, Verdict


@dataclass
class Citation:
    marker: str          # [E1]
    doc_id: str
    source: str
    date: Optional[str]
    tier: int
    url: str = ""
    span: Optional[Sequence[int]] = None
    quote: str = ""


@dataclass
class TimelineEntry:
    value: str
    valid_from: str
    valid_to: str
    source: str
    recorded_at: str
    change_kind: str
    is_current: bool
    #: Set when a later record replaced this one (a restatement). The row is
    #: kept: it is what we believed until that date.
    superseded_at: str = ""


@dataclass
class Answer:
    question: str
    answer: str = ""
    current_status: str = ""
    claims: List[Dict] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    timeline: List[TimelineEntry] = field(default_factory=list)
    changes: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    support_level: str = SupportLevel.INSUFFICIENT
    evidence_score: float = 0.0
    factors: Dict[str, float] = field(default_factory=dict)
    last_verified: Optional[str] = None
    unknown: List[str] = field(default_factory=list)
    abstained: bool = False
    coverage: float = 0.0
    iterations: int = 0
    trace: List[str] = field(default_factory=list)
    #: One row per entity for a comparison question: entity, value, period,
    #: source and whether that side could be established at all.
    comparison: List[Dict] = field(default_factory=list)

    # ------------------------------------------------------------- rendering
    def to_markdown(self) -> str:
        L: List[str] = []
        L.append(f"## Answer\n\n{self.answer or 'No answer could be established from the evidence.'}")
        if self.current_status:
            L.append(f"\n## Current status\n\n{self.current_status}")
        if self.claims:
            L.append("\n## Evidence\n")
            for c in self.claims:
                cite = " ".join(c.get("citations", []))
                L.append(f"- **{c['verdict']}** — {c['text']} {cite}".rstrip())
                if c.get("note"):
                    L.append(f"    - {c['note']}")
        if self.timeline:
            L.append("\n## Historical context\n")
            for t in self.timeline:
                flag = " ← current" if t.is_current else ""
                L.append(f"- {t.valid_from} → {t.valid_to}: **{t.value}** "
                         f"({t.source}, recorded {t.recorded_at}){flag}")
        if self.changes:
            L.append("\n## Changes\n")
            L.extend(f"- {c}" for c in self.changes)
        L.append("\n## Conflicts\n")
        L.extend(f"- {c}" for c in self.conflicts) if self.conflicts else L.append(
            "- None detected between credible sources.")
        L.append(f"\n## Confidence\n\n{explain(self.evidence_score, EvidenceFactors(**self.factors) if self.factors else EvidenceFactors(), self.support_level)}")
        L.append(f"\n## Last verified\n\n{self.last_verified or 'unknown'}")
        L.append("\n## Unknown\n")
        L.extend(f"- {u}" for u in self.unknown) if self.unknown else L.append(
            "- Nothing material is outstanding for this question.")
        if self.citations:
            L.append("\n## Sources\n")
            for c in self.citations:
                date = c.date or "undated"
                L.append(f"- {c.marker} {c.source} (tier {c.tier}, {date}) {c.url}".rstrip())
        return "\n".join(L)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_speech(self) -> str:
        """A spoken rendering: claims and hedges, no bracket markers.

        Reading "[E1]" aloud is noise, but dropping provenance entirely would
        defeat the point -- so citations become spoken attributions and the
        support level is stated in words.
        """
        parts = [self.answer]
        if self.support_level == SupportLevel.CONFLICTED:
            parts.append("Note that credible sources disagree on this.")
        elif self.support_level == SupportLevel.INSUFFICIENT:
            parts.append("I should say the evidence for this is insufficient.")
        elif self.support_level == SupportLevel.HIGH:
            n = len({c.source for c in self.citations})
            parts.append(f"This is supported by {n} independent source{'s' if n != 1 else ''}.")
        if self.last_verified:
            parts.append(f"Last verified {self.last_verified}.")
        if self.unknown:
            parts.append(f"Still unresolved: {self.unknown[0]}")
        import re as _r
        return _r.sub(r"\[E\d+\]", "", " ".join(p for p in parts if p)).strip()


class ProvenanceBuilder:
    """Turns verdicts + store history + conflicts into an `Answer`."""

    def __init__(self, graph=None, store=None) -> None:
        self.graph = graph
        self.store = store

    def build(
        self,
        question: str,
        verdicts: Sequence[ClaimVerdict],
        metrics: Dict[str, float],
        entity: str = "",
        attribute: str = "",
        conflicts: Optional[Sequence[Conflict]] = None,
        evidence_score: float = 0.0,
        factors: Optional[EvidenceFactors] = None,
        support_level: str = SupportLevel.INSUFFICIENT,
        answer_text: str = "",
        trace: Optional[Sequence[str]] = None,
    ) -> Answer:
        ans = Answer(question=question, answer=answer_text)
        ans.coverage = float(metrics.get("coverage", 0.0))
        ans.evidence_score = evidence_score
        ans.factors = factors.as_dict() if factors else {}
        ans.support_level = support_level
        ans.trace = list(trace or [])

        # --- citations: one marker per distinct document, assigned once ----
        marker_of: Dict[str, str] = {}
        for v in verdicts:
            for ev in list(v.supporting) + list(v.refuting):
                if ev.doc_id not in marker_of:
                    marker = f"[E{len(marker_of) + 1}]"
                    marker_of[ev.doc_id] = marker
                    ans.citations.append(Citation(
                        marker=marker, doc_id=ev.doc_id, source=ev.source or ev.doc_id,
                        date=ev.date, tier=ev.tier, quote=ev.text[:240],
                    ))

        latest: Optional[str] = None
        for v in verdicts:
            cites = [marker_of[e.doc_id] for e in v.supporting if e.doc_id in marker_of]
            note = ""
            if v.verdict == Verdict.CONFLICTED:
                note = (f"disputed by {', '.join(marker_of[e.doc_id] for e in v.refuting if e.doc_id in marker_of)}")
            elif v.verdict == Verdict.TEMPORAL_MISMATCH:
                note = "supported only for an earlier period; not confirmed for the asked-about time"
            elif v.verdict == Verdict.INSUFFICIENT:
                note = v.reason
            ans.claims.append({
                "text": v.claim.text, "verdict": v.verdict, "citations": cites,
                "entailment": round(v.entailment, 3), "independent_sources": v.n_independent,
                "note": note,
            })
            for e in v.supporting:
                if e.date and (latest is None or str(e.date) > latest):
                    latest = str(e.date)
            if v.verdict in (Verdict.INSUFFICIENT, Verdict.REFUTED) and v.claim.checkable:
                ans.unknown.append(f"Could not establish: {v.claim.text}")

        ans.last_verified = latest or (now_utc().date().isoformat() if verdicts else None)

        # --- timeline + changes from the temporal store --------------------
        if self.store is not None and entity and attribute:
            hist = self.store.history(entity, attribute)
            cur = self.store.current(entity, attribute)
            # The timeline shows superseded originals too: a restatement should
            # read as "reported, then restated", not as the first figure vanishing.
            for v in self.store.history(entity, attribute, include_superseded=True):
                ans.timeline.append(TimelineEntry(
                    value=v.value,
                    valid_from=_fmt(v.valid_from), valid_to=_fmt(v.valid_to),
                    source=v.source_id or "unknown",
                    recorded_at=_fmt(v.recorded_at), change_kind=v.change_kind,
                    is_current=bool(cur and v.version_id == cur.version_id),
                    superseded_at=_fmt(v.superseded_at) if v.superseded_at else "",
                ))
            for v in hist:
                if v.change_kind == "CORRECTED":
                    # A restatement is not the world changing; say what it is.
                    ans.changes.append(
                        f"{attribute} for {_fmt(v.valid_from)} to {_fmt(v.valid_to)} was "
                        f"restated from '{v.previous_value}' to '{v.value}' on "
                        f"{_fmt(v.recorded_at)} by {v.source_id or 'unknown source'}; "
                        f"the original figure is kept in the audit trail."
                    )
                elif v.change_kind == "CHANGED":
                    ans.changes.append(
                        f"{attribute} changed from '{v.previous_value}' to '{v.value}' "
                        f"effective {_fmt(v.valid_from)}, first recorded {_fmt(v.recorded_at)} "
                        f"by {v.source_id or 'unknown source'}."
                    )
            if cur:
                ans.current_status = (
                    f"{entity} — {attribute}: **{cur.value}**, valid since {_fmt(cur.valid_from)} "
                    f"(recorded {_fmt(cur.recorded_at)}, source {cur.source_id or 'unknown'})."
                )

        if conflicts:
            ans.conflicts = [c.explanation + (f" Preference: {c.preference_reason}" if c.preference_reason else "")
                             for c in conflicts if c.severity >= 0.4]
        return ans


def _fmt(dt) -> str:
    if dt is None:
        return "unknown"
    d = _dt(dt)
    return "present" if d >= FOREVER else d.date().isoformat()
