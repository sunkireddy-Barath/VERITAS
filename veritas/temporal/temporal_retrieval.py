"""Temporal retrieval: make time a first-class ranking signal.

The failure this fixes
----------------------
Ask a standard RAG system "who runs Acme?" and it retrieves by similarity
alone. A 2019 press release about the founder is *more* on-topic than a terse
2026 filing, so it wins, and the model answers with a seven-year-old fact
stated in the present tense. The retriever had no notion that the question has
a tense.

VERITAS resolves the *temporal intent* of the query first, then scores each
candidate on whether it is valid at the asked-about time.

Intents
-------
CURRENT    "who is the CEO now"          -> prefer valid_to = open, recent
HISTORICAL "who was the CEO in 2022"     -> prefer intervals covering 2022
AS_OF      "as of March 2025"            -> same, with an explicit anchor
RANGE      "between 2020 and 2024"       -> any overlap with the range
CHANGE     "how did it change / when"    -> deliberately retrieve BOTH states
ATEMPORAL  "what is photosynthesis"      -> time is irrelevant; do not decay

The CHANGE intent is the one ordinary temporal RAG gets wrong: a change
question must *not* filter to the latest state, because the answer needs the
before and the after. Freshness weighting is therefore switched off and the
retriever is asked for the union of adjacent intervals.

Scoring
-------
    temporal_score = interval_match(chunk, anchor)      in [0, 1]
    freshness      = 0.5 ^ (age_days / halflife_days)

Exponential decay, not a linear cutoff: a cliff at "one year old" makes ranking
discontinuous and drops a 366-day-old document that is the only evidence there
is. Half-life is chosen per query from the attribute's observed churn rate --
a share price decays in hours, a CEO in months, a founding date never. Using
one global half-life is the usual mistake, and it is why "freshness-aware RAG"
often degrades accuracy on stable facts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from .versioning import BEGINNING, FOREVER, _dt, now_utc


class TemporalIntent:
    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"
    AS_OF = "AS_OF"
    RANGE = "RANGE"
    CHANGE = "CHANGE"
    ATEMPORAL = "ATEMPORAL"


#: Attribute half-lives in days. Derived from how fast each kind of fact
#: actually churns; the ingest pipeline can re-estimate these empirically from
#: observed change intervals in the store (see `estimate_halflife`).
HALFLIFE_DAYS: Dict[str, float] = {
    "price": 0.5, "stock_price": 0.5, "status": 3, "outage": 0.25,
    "headcount": 180, "revenue": 120, "ceo": 365, "cto": 365,
    "address": 730, "founded": 1e9, "birth_date": 1e9, "default": 240,
}

_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_MONTH_YEAR = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\b", re.I
)
_RANGE = re.compile(r"\b(?:between|from)\s+(\d{4})\s*(?:and|to|-|until)\s*(\d{4}|now|today|present)\b", re.I)
_RELATIVE = re.compile(r"\b(?:last|past|previous)\s+(\d+)?\s*(day|week|month|year)s?\b", re.I)

_CURRENT_CUES = ("current", "currently", "now", "today", "latest", "as of now",
                 "at present", "right now", "these days", "up to date", "still")
_CHANGE_CUES = ("change", "changed", "changes", "evolve", "evolved", "transition",
                "history", "over time", "when did", "since when", "used to",
                "replaced", "became", "timeline", "how has", "trend")
_PAST_CUES = ("was", "were", "used to", "formerly", "previously", "back in",
              "at the time", "had been", "originally")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


@dataclass
class TemporalQuery:
    intent: str
    anchor: datetime                       # the point in time being asked about
    range_start: Optional[datetime] = None
    range_end: Optional[datetime] = None
    halflife_days: float = HALFLIFE_DAYS["default"]
    explicit_dates: List[datetime] = None
    attribute_hint: str = ""

    @property
    def wants_both_states(self) -> bool:
        return self.intent in (TemporalIntent.CHANGE, TemporalIntent.RANGE)

    @property
    def apply_freshness(self) -> bool:
        return self.intent in (TemporalIntent.CURRENT,)


def parse_temporal_query(query: str, attribute_hint: str = "", ref_time: Optional[datetime] = None) -> TemporalQuery:
    """Rule-based temporal intent classifier.

    Deliberately rules, not a model: the cues are a closed, high-precision set,
    the decision must be auditable ("why did you treat this as a CURRENT
    query?"), and it runs in microseconds on every query. The LM planner can
    override it when the phrasing is genuinely ambiguous.
    """
    ref = ref_time or now_utc()
    q = query.lower()
    dates: List[datetime] = []

    for m in _MONTH_YEAR.finditer(q):
        dates.append(datetime(int(m.group(2)), _MONTHS[m.group(1)[:3].lower()], 1, tzinfo=timezone.utc))
    for m in _YEAR.finditer(q):
        y = int(m.group(1))
        if not any(d.year == y for d in dates):
            dates.append(datetime(y, 7, 1, tzinfo=timezone.utc))  # mid-year midpoint

    halflife = HALFLIFE_DAYS.get(attribute_hint.lower(), HALFLIFE_DAYS["default"])

    rm = _RANGE.search(q)
    if rm:
        start = datetime(int(rm.group(1)), 1, 1, tzinfo=timezone.utc)
        end = ref if rm.group(2).lower() in ("now", "today", "present") else datetime(
            int(rm.group(2)), 12, 31, tzinfo=timezone.utc)
        return TemporalQuery(TemporalIntent.RANGE, end, start, end, halflife, dates, attribute_hint)

    relm = _RELATIVE.search(q)
    if relm:
        n = int(relm.group(1) or 1)
        unit = {"day": 1, "week": 7, "month": 30, "year": 365}[relm.group(2).lower()]
        start = ref - timedelta(days=n * unit)
        return TemporalQuery(TemporalIntent.RANGE, ref, start, ref, halflife, dates, attribute_hint)

    if any(c in q for c in _CHANGE_CUES):
        return TemporalQuery(TemporalIntent.CHANGE, ref, dates[0] if dates else None, ref,
                             halflife, dates, attribute_hint)
    if any(c in q for c in _CURRENT_CUES):
        return TemporalQuery(TemporalIntent.CURRENT, ref, None, None, halflife, dates, attribute_hint)
    if dates:
        intent = TemporalIntent.AS_OF if "as of" in q else TemporalIntent.HISTORICAL
        return TemporalQuery(intent, max(dates), None, None, halflife, dates, attribute_hint)
    if any(c in q for c in _PAST_CUES):
        return TemporalQuery(TemporalIntent.HISTORICAL, ref, None, None, halflife, dates, attribute_hint)
    # Default to CURRENT, not ATEMPORAL: in a changing-world system, an
    # unqualified question almost always means "now".
    return TemporalQuery(TemporalIntent.CURRENT, ref, None, None, halflife, dates, attribute_hint)


# ------------------------------------------------------------------ scoring
def freshness_score(doc_time: Optional[datetime], ref: Optional[datetime] = None,
                    halflife_days: float = 240.0) -> float:
    """0.5 ^ (age / halflife). 1.0 = brand new, 0.5 = one half-life old."""
    if doc_time is None:
        return 0.3  # unknown date: mildly penalised, not excluded
    ref = ref or now_utc()
    age = max(0.0, (ref - _dt(doc_time)).total_seconds() / 86400.0)
    return float(0.5 ** (age / max(halflife_days, 1e-6)))


def interval_overlap_score(
    valid_from: Optional[datetime], valid_to: Optional[datetime], tq: TemporalQuery
) -> float:
    """How well a chunk's validity interval answers the query's time need."""
    vf = _dt(valid_from) if valid_from is not None else BEGINNING
    vt = _dt(valid_to) if valid_to is not None else FOREVER

    if tq.intent == TemporalIntent.ATEMPORAL:
        return 0.5

    if tq.intent in (TemporalIntent.CURRENT,):
        if vt >= FOREVER:                       # still open -> still true
            return 1.0
        if vt >= tq.anchor:
            return 0.9
        # closed in the past: decays with how long ago it closed
        years = (tq.anchor - vt).days / 365.25
        return max(0.0, 0.5 ** years) * 0.4     # keep >0: needed for "outdated" labelling

    if tq.intent in (TemporalIntent.HISTORICAL, TemporalIntent.AS_OF):
        if vf <= tq.anchor < vt:
            return 1.0
        gap_days = min(abs((tq.anchor - vf).days), abs((tq.anchor - vt).days))
        return max(0.0, 0.5 ** (gap_days / 365.25))

    if tq.intent == TemporalIntent.RANGE:
        s = tq.range_start or BEGINNING
        e = tq.range_end or FOREVER
        inter = (min(vt, e) - max(vf, s)).total_seconds()
        if inter <= 0:
            return 0.0
        span = max((e - s).total_seconds(), 1.0)
        return float(min(1.0, inter / span))

    if tq.intent == TemporalIntent.CHANGE:
        # A change question wants the endpoints of intervals, not one state.
        # Anything with a *closed* boundary is more informative than an
        # eternally-open fact, so bounded intervals score highest.
        bounded = (vf > BEGINNING) + (vt < FOREVER)
        return 0.5 + 0.25 * bounded

    return 0.5


def estimate_halflife(store, attribute: str, default: float = 240.0) -> float:
    """Empirical half-life: median observed interval length for an attribute.

    Better than the hand-written table once the store has history, because it
    measures how fast *this corpus* actually changes rather than how fast we
    assumed it would.
    """
    durations = []
    for (_e, attr), versions in store._index.items():
        if attr != attribute:
            continue
        for v in versions:
            if v.valid_to < FOREVER:
                durations.append((v.valid_to - v.valid_from).days)
    if not durations:
        return HALFLIFE_DAYS.get(attribute, default)
    durations.sort()
    return float(max(1.0, durations[len(durations) // 2]))


def make_signal_fns(tq: TemporalQuery, entity: str = "", texts: Optional[Dict[str, str]] = None):
    """Build the signal callables consumed by `rag.hybrid.HybridRetriever`."""
    texts = texts or {}
    ent = entity.lower().strip()

    def freshness(doc_id: str, md: dict) -> float:
        if not tq.apply_freshness:
            return 0.5   # neutral: do not penalise old evidence for a history question
        return freshness_score(md.get("date") or md.get("published"), tq.anchor, tq.halflife_days)

    def temporal(doc_id: str, md: dict) -> float:
        return interval_overlap_score(md.get("valid_from"), md.get("valid_to"), tq)

    def entity_match(doc_id: str, md: dict) -> float:
        if not ent:
            return 0.5
        hay = (str(md.get("entity", "")) + " " + texts.get(doc_id, "")).lower()
        return 1.0 if ent in hay else 0.0

    def authority(doc_id: str, md: dict) -> float:
        tier = int(md.get("tier", 3))
        return {1: 1.0, 2: 0.75, 3: 0.45, 4: 0.15}.get(tier, 0.3)

    return {"freshness": freshness, "temporal": temporal,
            "entity": entity_match, "authority": authority}
