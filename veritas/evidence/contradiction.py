"""Contradiction analysis with temporal awareness.

The distinction that matters
----------------------------
Two sources say different things. There are four reasons, and treating them
alike is why RAG systems confidently merge incompatible facts:

  1. **TEMPORAL**  -- both were right, at different times. "CEO is X" (2024) vs
     "CEO is Y" (2026). The correct output is a *timeline*, not a conflict.
  2. **FACTUAL**   -- same valid time, different value. A genuine disagreement.
     Report both, name the sources, do not pick.
  3. **GRANULARITY** -- "about 1.4 billion" vs "1,412,000,000". Not a conflict;
     one is a rounding of the other.
  4. **SCOPE**     -- "revenue 1.4B" (group) vs "revenue 0.9B" (one segment);
     or different units, currencies or regions. Not a conflict; the referents
     differ. Flagged as UNRESOLVED_SCOPE rather than guessed at.

Case 1 is the one ordinary systems get wrong, and the fix requires the
bitemporal store: without valid intervals you cannot tell "wrong" from "was
right". This module is therefore the direct consumer of `temporal/versioning`.

Resolution policy
-----------------
When a conflict is FACTUAL, VERITAS does **not** pick a winner by default. It
reports the disagreement with both sources and their tiers. A tier-1 primary
source against a tier-4 blog is the one case where a tentative preference is
stated -- and even then it is stated as a preference with its reason, never as
a resolution.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from ..temporal.versioning import FOREVER, Version, _dt
from .claims import Claim, _parse_number
from .quality import SourcePolicy


class ConflictType:
    TEMPORAL = "TEMPORAL"                    # both true, different times
    FACTUAL = "FACTUAL"                      # same time, different values
    GRANULARITY = "GRANULARITY"              # rounding / precision only
    UNRESOLVED_SCOPE = "UNRESOLVED_SCOPE"    # referents may differ
    NONE = "NONE"


@dataclass
class Conflict:
    type: str
    a: object
    b: object
    explanation: str
    preferred: Optional[object] = None
    preference_reason: str = ""
    severity: float = 0.0     # 0 = benign, 1 = direct unresolved disagreement

    def as_dict(self) -> Dict:
        return {"type": self.type, "explanation": self.explanation,
                "severity": self.severity, "preference_reason": self.preference_reason}


_SCOPE_CUES = re.compile(
    r"\b(segment|division|subsidiary|region|excluding|including|adjusted|"
    r"pro forma|constant currency|per share|q[1-4]|fiscal|calendar)\b", re.I
)
_ROUNDING = re.compile(r"\b(about|approximately|roughly|around|nearly|over|under|~)\b", re.I)


def _values_differ(a: str, b: str, rel_tol: float = 0.02) -> Optional[bool]:
    """None when incomparable; True/False for a numeric or string difference."""
    na, _ = _parse_number(a)
    nb, _ = _parse_number(b)
    if na is not None and nb is not None:
        return abs(na - nb) / max(abs(na), abs(nb), 1e-9) > rel_tol
    sa, sb = _norm(a), _norm(b)
    if not sa or not sb:
        return None
    return sa != sb


def _intervals_overlap(a_from, a_to, b_from, b_to) -> bool:
    af, at = _dt(a_from), _dt(a_to) if a_to else FOREVER
    bf, bt = _dt(b_from), _dt(b_to) if b_to else FOREVER
    return af < bt and bf < at


class ContradictionDetector:
    def __init__(self, policy: Optional[SourcePolicy] = None, rel_tol: float = 0.02) -> None:
        self.policy = policy or SourcePolicy()
        self.rel_tol = rel_tol

    # ------------------------------------------------------- version-level
    def compare_versions(self, a: Version, b: Version, domain: str = "government") -> Conflict:
        """Classify two competing assertions from the temporal store."""
        # Same categorical value, different wording ("construction" vs
        # "begun construction") is a phrasing difference, not a disagreement.
        na, nb = _norm(a.value), _norm(b.value)
        if na and nb and (na in nb or nb in na) and _parse_number(a.value)[0] is None:
            return Conflict(ConflictType.NONE, a, b, "same value, different phrasing",
                            severity=0.0)
        differ = _values_differ(a.value, b.value, self.rel_tol)
        if differ is None or not differ:
            return Conflict(ConflictType.NONE, a, b, "values agree", severity=0.0)

        if not _intervals_overlap(a.valid_from, a.valid_to, b.valid_from, b.valid_to):
            earlier, later = (a, b) if a.valid_from < b.valid_from else (b, a)
            return Conflict(
                ConflictType.TEMPORAL, a, b,
                f"'{earlier.value}' was valid until {_fmt(earlier.valid_to)}; "
                f"'{later.value}' is valid from {_fmt(later.valid_from)}. "
                f"These are successive states, not a disagreement.",
                preferred=later,
                preference_reason="later valid interval covers the present",
                severity=0.1,
            )

        # overlapping valid time: genuine disagreement
        ta = self.policy.tier(a.source_id, domain)
        tb = self.policy.tier(b.source_id, domain)
        preferred, reason = None, ""
        if abs(ta - tb) >= 2:
            preferred = a if ta < tb else b
            reason = (f"tier-{min(ta, tb)} source ({preferred.source_id}) is materially more "
                      f"authoritative than tier-{max(ta, tb)} for the '{domain}' domain; "
                      f"stated as a preference, not a resolution")
        return Conflict(
            ConflictType.FACTUAL, a, b,
            f"Sources disagree for the same period: {a.source_id} reports '{a.value}', "
            f"{b.source_id} reports '{b.value}'.",
            preferred=preferred, preference_reason=reason,
            severity=1.0 if preferred is None else 0.6,
        )

    # --------------------------------------------------------- claim-level
    def compare_claims(self, a: Claim, b: Claim, a_text: str = "", b_text: str = "") -> Conflict:
        # Two claims are comparable only if they assert the SAME attribute of
        # the SAME subject with an extracted value on both sides. Every
        # relaxation of this produced false positives in practice:
        #   - falling back to raw sentence text compares prose, not values
        #     ("Priya Raman" vs a whole sentence about another company);
        #   - allowing different attributes compares "15 offices" with
        #     "completed";
        #   - allowing an empty value compares a name with the literal "CEO".
        # A contradiction detector that fires on those is worse than none: it
        # trains the reader to ignore the conflicts section.
        if not (a.value and b.value):
            return Conflict(ConflictType.NONE, a, b, "no comparable value", severity=0.0)
        if a.attribute != b.attribute:
            return Conflict(ConflictType.NONE, a, b, "different attributes", severity=0.0)
        if a.subject and b.subject and not _same_subject(a.subject, b.subject):
            return Conflict(ConflictType.NONE, a, b, "different subjects", severity=0.0)

        ctx = f"{a_text} {b_text} {a.text} {b.text}"
        # A categorical value that contains the other ("begun construction" vs
        # "construction") is a phrasing difference, not a disagreement.
        # Test the VALUES for numerality, not `claim.numeric` -- that field
        # records any number in the sentence, so a date ("3 March 2026") would
        # wrongly mark a categorical status claim as numeric.
        na, nb = _norm(a.value), _norm(b.value)
        both_categorical = _parse_number(a.value)[0] is None and _parse_number(b.value)[0] is None
        if both_categorical and (na in nb or nb in na):
            return Conflict(ConflictType.NONE, a, b, "same value, different phrasing", severity=0.0)

        differ = _values_differ(a.value, b.value, self.rel_tol)
        if differ is None or not differ:
            return Conflict(ConflictType.NONE, a, b, "values agree", severity=0.0)

        if a.numeric is not None and b.numeric is not None:
            hi, lo = max(abs(a.numeric), abs(b.numeric)), min(abs(a.numeric), abs(b.numeric))
            rel = (hi - lo) / max(hi, 1e-9)
            if rel <= 0.05 and _ROUNDING.search(ctx):
                return Conflict(ConflictType.GRANULARITY, a, b,
                                f"{a.value} and {b.value} differ by {rel:.1%}; one is a rounded "
                                f"restatement of the other.", severity=0.05)
            if _SCOPE_CUES.search(ctx):
                return Conflict(ConflictType.UNRESOLVED_SCOPE, a, b,
                                "the two figures may cover different scopes (segment, period or "
                                "currency); they cannot be compared directly.", severity=0.4)

        if a.valid_from and b.valid_from and not _intervals_overlap(
            a.valid_from, a.valid_to, b.valid_from, b.valid_to
        ):
            return Conflict(ConflictType.TEMPORAL, a, b,
                            "the claims describe different time periods.", severity=0.1)

        return Conflict(ConflictType.FACTUAL, a, b,
                        f"direct disagreement: '{a.value or a.text}' vs '{b.value or b.text}'",
                        severity=1.0)

    # ------------------------------------------------------------ sweeping
    def scan_store(self, store, domain: str = "government") -> List[Conflict]:
        """All unresolved factual conflicts currently in the knowledge store."""
        out: List[Conflict] = []
        for entity in store.entities():
            for (e, attr) in list(store._index.keys()):
                if e != entity:
                    continue
                for a, b in store.conflicts(entity, attr):
                    c = self.compare_versions(a, b, domain)
                    if c.type != ConflictType.NONE and c.severity >= 0.5:
                        out.append(c)
        return out

    def summarize(self, conflicts: Sequence[Conflict]) -> str:
        """The text that goes in the answer's Conflicts section."""
        if not conflicts:
            return "No credible source disagreement detected."
        real = [c for c in conflicts if c.severity >= 0.5]
        if not real:
            return ("Minor differences found (rounding or scope), none of which change the "
                    "substance of the answer.")
        lines = [f"{len(real)} unresolved disagreement(s) between credible sources:"]
        for c in real[:5]:
            lines.append(f"  - {c.explanation}")
            if c.preference_reason:
                lines.append(f"    Preference: {c.preference_reason}")
        return "\n".join(lines)


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().replace(",", "").split())


def _same_subject(a: str, b: str) -> bool:
    """Token overlap on words longer than 2 chars, so "Acme" matches
    "Acme Industries" but not "Orion Systems"."""
    ta = {w for w in _norm(a).replace(".", " ").split() if len(w) > 2}
    tb = {w for w in _norm(b).replace(".", " ").split() if len(w) > 2}
    return bool(ta & tb) if (ta and tb) else _norm(a) == _norm(b)


def _fmt(dt) -> str:
    if dt is None:
        return "unknown"
    d = _dt(dt)
    return "present" if d >= FOREVER else d.date().isoformat()
