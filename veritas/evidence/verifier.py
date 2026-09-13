"""Claim verification: does this evidence actually support this claim?

The question is entailment, not similarity
------------------------------------------
"Acme did not appoint Y as CEO" has ~0.95 cosine similarity to "Acme appointed
Y as CEO". Retrieval scores are useless as verification. Verification must
answer a three-way question:

    ENTAILED       the evidence asserts the claim
    CONTRADICTED   the evidence asserts its negation
    NEUTRAL        the evidence is about it but settles nothing

Verdicts returned to the caller are the spec's four:
SUPPORTED / REFUTED / CONFLICTED / INSUFFICIENT.

A two-channel verifier
----------------------
**Channel 1 -- symbolic checks** run first and can decide alone. They exist
because the three ways a grounded answer goes wrong are all mechanical:

  * numeric mismatch  -- "15 offices" vs "12 offices". Compared after unit
    normalisation with a relative tolerance, so 1.4B == 1,400M but 15 != 12.
    This is the single highest-yield check: fabricated numbers are the most
    common and most damaging RAG error, and a neural model scores the two
    sentences as near-identical.
  * polarity flip     -- negation cues on one side only.
  * temporal mismatch -- right value, wrong interval ("CEO in 2022" cited for
    "CEO now"). This produces TEMPORAL_MISMATCH rather than REFUTED, because
    the source is not wrong -- it is stale. Conflating the two is the error
    this whole project is built to avoid.

**Channel 2 -- the neural NLI head** (the cross-encoder trunk with a 3-way
head) handles paraphrase, which no rule catches.

Symbolic first is deliberate: rules are auditable, near-free, and have no
hallucination mode. The model breaks ties and covers the rest. A claim is
SUPPORTED only when no symbolic check fires *and* the entailment score clears
threshold, so the cheap deterministic check can always veto the model.

Evidence coverage
-----------------
    coverage = supported_claims / checkable_claims

This is the project's headline metric. Opinions and non-checkable statements
are excluded from the denominator, otherwise a fluent answer is punished for
its connective tissue.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .claims import Claim, _parse_number
from .quality import SupportLevel


class Verdict:
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    CONFLICTED = "CONFLICTED"
    INSUFFICIENT = "INSUFFICIENT"
    TEMPORAL_MISMATCH = "TEMPORAL_MISMATCH"   # true, but not for the asked time


_NEG = re.compile(
    r"\b(not|no longer|never|denied|denies|rejected|declined|failed to|"
    r"without|ceased|stopped|cancell?ed|withdrew|refuted)\b", re.I
)
_STOP = frozenset("a an the of and or to in on at for is are was were be by with as that this it its from".split())


@dataclass
class EvidenceItem:
    doc_id: str
    text: str
    source: str = ""
    date: Optional[str] = None
    tier: int = 3
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    score: float = 0.0


@dataclass
class ClaimVerdict:
    claim: Claim
    verdict: str
    entailment: float = 0.0
    supporting: List[EvidenceItem] = field(default_factory=list)
    refuting: List[EvidenceItem] = field(default_factory=list)
    reason: str = ""
    n_independent: int = 0

    @property
    def is_supported(self) -> bool:
        return self.verdict == Verdict.SUPPORTED


def _content_tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


def lexical_entailment(claim: str, evidence: str) -> float:
    """Coverage of the claim's content words by the evidence.

    Asymmetric on purpose: entailment is directional. A long evidence passage
    covering every content word of a short claim is strong support; a short
    passage overlapping a long claim is not. Symmetric similarity (Jaccard,
    cosine) gets this backwards.
    """
    c, e = _content_tokens(claim), _content_tokens(evidence)
    if not c:
        return 0.0
    return len(c & e) / len(c)


def numeric_conflict(claim: Claim, evidence_text: str, rel_tol: float = 0.02) -> Optional[Tuple[float, float]]:
    """Return (claim_value, evidence_value) when the numbers genuinely differ."""
    if claim.numeric is None:
        return None
    ev_val, _unit = _parse_number(evidence_text)
    if ev_val is None:
        return None
    denom = max(abs(claim.numeric), abs(ev_val), 1e-9)
    if abs(claim.numeric - ev_val) / denom > rel_tol:
        return (claim.numeric, ev_val)
    return None


def polarity_conflict(claim_text: str, evidence_text: str) -> bool:
    """Negation on exactly one side, with enough lexical overlap that the two
    are talking about the same thing."""
    cn, en = bool(_NEG.search(claim_text)), bool(_NEG.search(evidence_text))
    if cn == en:
        return False
    return lexical_entailment(claim_text, evidence_text) >= 0.5


def person_conflict(claim: Claim, evidence_text: str) -> bool:
    """Two different people named for the same role.

    Person-valued claims carry no number, so `numeric_conflict` cannot see
    "Tim Cook" against "Steve Jobs". Names are compared by shared tokens, so a
    truncated "Samuel J" still agrees with "Samuel J. Palmisano".
    """
    from .claims import PERSON_ATTRIBUTES, _canonical_attribute, _extract_value

    if claim.attribute not in PERSON_ATTRIBUTES or not claim.value:
        return False
    if _canonical_attribute(evidence_text) != claim.attribute:
        return False
    other = _extract_value(evidence_text, claim.attribute, None, "")
    if not other:
        return False
    a = {t for t in re.findall(r"[a-z]+", claim.value.lower()) if len(t) > 2}
    b = {t for t in re.findall(r"[a-z]+", other.lower()) if len(t) > 2}
    return bool(a) and bool(b) and not (a & b)


def temporal_mismatch(claim: Claim, ev: EvidenceItem, anchor: Optional[str]) -> bool:
    """The evidence's validity interval does not cover the asked-about time."""
    if anchor is None or ev.valid_to is None:
        return False
    return str(ev.valid_to) < str(anchor)


class ClaimVerifier:
    """Symbolic checks + optional neural entailment."""

    def __init__(
        self,
        nli_model=None,
        entail_threshold: float = 0.55,
        support_threshold: float = 0.6,
        min_independent: int = 1,
    ) -> None:
        self.nli = nli_model                 # CrossEncoder-style scorer, optional
        self.entail_threshold = entail_threshold
        self.support_threshold = support_threshold
        self.min_independent = min_independent

    def _entailment(self, claim_text: str, evidence_text: str) -> float:
        lex = lexical_entailment(claim_text, evidence_text)
        if self.nli is None:
            return lex
        try:
            import torch

            with torch.inference_mode():
                raw = float(self.nli([claim_text], [evidence_text])[0])
            neural = 1.0 / (1.0 + np.exp(-raw))
            # average, not max: the rule is a floor the model cannot talk its
            # way past, and the model cannot be vetoed by lexical mismatch alone
            return 0.5 * lex + 0.5 * neural
        except Exception:
            return lex

    def verify_claim(
        self,
        claim: Claim,
        evidence: Sequence[EvidenceItem],
        anchor_time: Optional[str] = None,
        n_independent: Optional[int] = None,
    ) -> ClaimVerdict:
        if not claim.checkable:
            return ClaimVerdict(claim, Verdict.INSUFFICIENT, 0.0, reason="not a checkable claim")
        if not evidence:
            return ClaimVerdict(claim, Verdict.INSUFFICIENT, 0.0, reason="no evidence retrieved")

        supporting: List[EvidenceItem] = []
        refuting: List[EvidenceItem] = []
        stale: List[EvidenceItem] = []
        best = 0.0

        for ev in evidence:
            score = self._entailment(claim.text, ev.text)
            best = max(best, score)

            num = numeric_conflict(claim, ev.text)
            if num is not None and lexical_entailment(claim.text, ev.text) >= 0.4:
                ev.score = score
                refuting.append(ev)
                continue
            if polarity_conflict(claim.text, ev.text) or (
                    person_conflict(claim, ev.text)
                    and lexical_entailment(claim.text, ev.text) >= 0.4):
                ev.score = score
                refuting.append(ev)
                continue
            if score >= self.entail_threshold:
                ev.score = score
                (stale if temporal_mismatch(claim, ev, anchor_time) else supporting).append(ev)

        n_ind = n_independent if n_independent is not None else len({e.source or e.doc_id for e in supporting})

        if supporting and refuting:
            return ClaimVerdict(claim, Verdict.CONFLICTED, best, supporting, refuting,
                                "credible evidence disagrees", n_ind)
        if refuting and not supporting:
            return ClaimVerdict(claim, Verdict.REFUTED, best, [], refuting,
                                "evidence contradicts the claim", 0)
        if not supporting and stale:
            return ClaimVerdict(claim, Verdict.TEMPORAL_MISMATCH, best, stale, [],
                                "evidence was valid earlier but not at the asked-about time",
                                len({e.source or e.doc_id for e in stale}))
        if supporting and best >= self.support_threshold and n_ind >= self.min_independent:
            return ClaimVerdict(claim, Verdict.SUPPORTED, best, supporting, [],
                                "entailed by retrieved evidence", n_ind)
        return ClaimVerdict(claim, Verdict.INSUFFICIENT, best, supporting, refuting,
                            "evidence is related but does not establish the claim", n_ind)

    def verify_answer(
        self,
        claims: Sequence[Claim],
        evidence_lookup,
        anchor_time: Optional[str] = None,
    ) -> Tuple[List[ClaimVerdict], Dict[str, float]]:
        """Verify every claim in a generated answer and compute coverage."""
        verdicts = [
            self.verify_claim(c, evidence_lookup(c), anchor_time)
            for c in claims
        ]
        checkable = [v for v in verdicts if v.claim.checkable]
        n = len(checkable) or 1
        metrics = {
            "n_claims": len(claims),
            "n_checkable": len(checkable),
            "coverage": sum(v.is_supported for v in checkable) / n,
            "unsupported_rate": sum(
                v.verdict in (Verdict.INSUFFICIENT, Verdict.REFUTED) for v in checkable) / n,
            "conflict_rate": sum(v.verdict == Verdict.CONFLICTED for v in checkable) / n,
            "stale_rate": sum(v.verdict == Verdict.TEMPORAL_MISMATCH for v in checkable) / n,
        }
        return verdicts, metrics


def rewrite_unsupported(verdicts: Sequence[ClaimVerdict]) -> List[str]:
    """Turn verdicts into answer text, removing or hedging what failed.

    Section 13 of the spec: an unsupported claim is not silently kept. REFUTED
    claims are dropped entirely; CONFLICTED claims are rewritten to *state the
    disagreement*, which is more useful than either value on its own.
    """
    out: List[str] = []
    for v in verdicts:
        if v.verdict == Verdict.SUPPORTED:
            cites = ", ".join(f"[{e.doc_id}]" for e in v.supporting[:3])
            out.append(f"{v.claim.text} {cites}".strip())
        elif v.verdict == Verdict.CONFLICTED:
            a = v.supporting[0].source if v.supporting else "one source"
            b = v.refuting[0].source if v.refuting else "another source"
            out.append(
                f"Sources disagree on this point: {a} supports \"{v.claim.text}\", "
                f"while {b} reports otherwise. The evidence is insufficient to settle it."
            )
        elif v.verdict == Verdict.TEMPORAL_MISMATCH:
            src = v.supporting[0] if v.supporting else None
            when = src.valid_to if src else "an earlier period"
            out.append(
                f"{v.claim.text} -- but this was last verified as valid until {when}; "
                f"no current evidence confirms it still holds."
            )
        # REFUTED and INSUFFICIENT are dropped from the answer body
    return out
