"""Evidence quality scoring and source policy.

The honest framing
------------------
This module does NOT compute a probability that a claim is true. No open-world
system can. It computes *how well-evidenced* a claim is, and the output is a
label plus the factors that produced it, so a reader can disagree with the
weighting rather than with an opaque number. Section 24 of the spec is a
design constraint, not a slogan: never emit "100% true".

Seven factors, each a bounded [0, 1] signal:

  authority     source tier prior, DOMAIN-DEPENDENT (see below)
  recency       exponential decay on the attribute's half-life
  independence  distinct source *roots* in the evidence graph, log-damped
  directness    does the doc state the fact, or cite someone who does
  specificity   exact values/dates beat vague prose
  agreement     fraction of retrieved evidence that concurs
  validity      does the evidence's valid interval cover the asked-about time

Aggregation is a weighted geometric-ish mean with a hard floor: a claim with
zero independent sources cannot be rescued by scoring well everywhere else. A
plain weighted sum would let five weak signals outvote the absence of evidence,
which is exactly the failure mode of "confidence scores" in most RAG demos.

Why source tiers are per-domain
-------------------------------
There is no universal source ranking, and pretending otherwise is a real
correctness bug. For a regulatory question, the regulator's filing outranks
Reuters. For a software outage, the vendor's status page outranks the
regulator. For a scientific finding, the peer-reviewed paper outranks the
university's own press release *about* that paper. Policies are therefore
keyed by domain, with a conservative default.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from ..temporal.temporal_retrieval import freshness_score


class SupportLevel:
    HIGH = "HIGH SUPPORT"
    MODERATE = "MODERATE SUPPORT"
    LOW = "LOW SUPPORT"
    CONFLICTED = "CONFLICTED"
    INSUFFICIENT = "INSUFFICIENT EVIDENCE"


#: domain -> {tier: [url/source patterns]}. Tier 1 is primary/authoritative.
DEFAULT_POLICIES: Dict[str, Dict[int, List[str]]] = {
    "government": {
        1: [r"\.gov$", r"\.gov\.", r"europa\.eu", r"\.gov\.uk", r"nic\.in", r"\.gob\."],
        2: [r"\.org$", r"reuters\.com", r"apnews\.com", r"bloomberg\.com"],
        3: [r"\.com$", r"\.net$"],
        4: [r"reddit\.com", r"x\.com", r"twitter\.com", r"medium\.com", r"quora\.com"],
    },
    "corporate": {
        1: [r"sec\.gov", r"investor\.", r"ir\.", r"/investor-relations", r"companieshouse"],
        2: [r"reuters\.com", r"bloomberg\.com", r"ft\.com", r"wsj\.com"],
        3: [r"techcrunch\.com", r"\.com$"],
        4: [r"reddit\.com", r"x\.com", r"blogspot\."],
    },
    "science": {
        1: [r"arxiv\.org", r"doi\.org", r"nature\.com", r"science\.org", r"pubmed", r"\.edu$"],
        2: [r"nih\.gov", r"who\.int", r"\.ac\."],
        3: [r"sciencedaily", r"newscientist"],
        4: [r"reddit\.com", r"substack\."],
    },
    "software": {
        1: [r"status\.", r"statuspage\.io", r"github\.com/[^/]+/[^/]+/releases", r"/docs/"],
        2: [r"github\.com", r"stackoverflow\.com"],
        3: [r"medium\.com", r"dev\.to"],
        4: [r"reddit\.com", r"x\.com"],
    },
}

TIER_PRIOR: Dict[int, float] = {1: 1.0, 2: 0.78, 3: 0.5, 4: 0.2}


@dataclass
class SourcePolicy:
    """Assigns a tier to a source, given the question's domain."""

    policies: Dict[str, Dict[int, List[str]]] = field(default_factory=lambda: dict(DEFAULT_POLICIES))
    default_domain: str = "government"
    overrides: Dict[str, int] = field(default_factory=dict)

    def tier(self, source: str, domain: Optional[str] = None) -> int:
        if source in self.overrides:
            return self.overrides[source]
        table = self.policies.get(domain or self.default_domain, {})
        s = source.lower()
        for t in (1, 2, 3, 4):
            for pat in table.get(t, []):
                if re.search(pat, s):
                    return t
        return 3

    def authority(self, source: str, domain: Optional[str] = None) -> float:
        return TIER_PRIOR.get(self.tier(source, domain), 0.4)


@dataclass
class EvidenceFactors:
    authority: float = 0.0
    recency: float = 0.0
    independence: float = 0.0
    directness: float = 0.0
    specificity: float = 0.0
    agreement: float = 0.0
    validity: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


WEIGHTS: Dict[str, float] = {
    "authority": 0.20, "recency": 0.12, "independence": 0.22, "directness": 0.14,
    "specificity": 0.08, "agreement": 0.16, "validity": 0.08,
}

_SECONDHAND = re.compile(
    r"\b(according to|as reported by|cited by|per a report|sources say|reportedly|"
    r"a spokesperson (?:said|told))\b", re.I
)
_SPECIFIC = re.compile(r"(\d{4}-\d{2}-\d{2}|\b\d[\d,]*\.?\d*\s?(?:%|billion|million)|\bQ[1-4]\s?\d{4})", re.I)


def independence_score(n_independent: int) -> float:
    """log-damped: 1 -> 0.4, 2 -> 0.63, 3 -> 0.77, 5 -> 0.93, 0 -> 0.

    Concave because the *second* independent source is the big jump (it rules
    out a single-source error); the fifth adds little.
    """
    if n_independent <= 0:
        return 0.0
    return min(1.0, math.log1p(n_independent) / math.log1p(6)) * 0.4 + min(1.0, n_independent / 3) * 0.6


def directness_score(text: str) -> float:
    """Second-hand attribution is weaker evidence than a primary statement."""
    return 0.45 if _SECONDHAND.search(text) else 1.0


def specificity_score(text: str) -> float:
    hits = len(_SPECIFIC.findall(text))
    return min(1.0, 0.4 + 0.3 * hits)


def score_evidence(
    *,
    sources: Sequence[str],
    dates: Sequence[Optional[datetime]],
    texts: Sequence[str],
    n_independent: int,
    agreement: float,
    validity: float,
    domain: str = "government",
    halflife_days: float = 240.0,
    ref_time: Optional[datetime] = None,
    policy: Optional[SourcePolicy] = None,
) -> Tuple[float, EvidenceFactors]:
    policy = policy or SourcePolicy()
    if not sources:
        return 0.0, EvidenceFactors()

    f = EvidenceFactors(
        authority=max(policy.authority(s, domain) for s in sources),
        recency=max(freshness_score(d, ref_time, halflife_days) for d in dates) if dates else 0.3,
        independence=independence_score(n_independent),
        directness=max(directness_score(t) for t in texts) if texts else 0.5,
        specificity=max(specificity_score(t) for t in texts) if texts else 0.4,
        agreement=float(agreement),
        validity=float(validity),
    )
    score = sum(WEIGHTS[k] * v for k, v in f.as_dict().items())
    # hard floor: no independent source => no support, whatever else says
    if n_independent <= 0:
        score = 0.0
    return float(min(1.0, max(0.0, score))), f


def support_label(score: float, n_independent: int, has_conflict: bool) -> str:
    """Map a score to a label. Conflict beats score: a high-scoring claim that
    a credible source disputes must never be reported as settled."""
    if has_conflict:
        return SupportLevel.CONFLICTED
    if n_independent == 0 or score < 0.25:
        return SupportLevel.INSUFFICIENT
    if score >= 0.72 and n_independent >= 2:
        return SupportLevel.HIGH
    if score >= 0.45:
        return SupportLevel.MODERATE
    return SupportLevel.LOW


def explain(score: float, f: EvidenceFactors, label: str) -> str:
    """Human-readable justification -- never a truth percentage."""
    top = sorted(f.as_dict().items(), key=lambda kv: -WEIGHTS[kv[0]] * kv[1])[:3]
    drivers = ", ".join(f"{k} {v:.2f}" for k, v in top)
    return f"{label} (evidence score {score:.2f}; strongest factors: {drivers})"
