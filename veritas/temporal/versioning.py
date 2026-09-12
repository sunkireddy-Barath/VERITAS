"""Bitemporal knowledge store: the heart of VERITAS.

The idea
--------
An ordinary RAG index stores "chunk text + embedding" and answers "what does my
corpus say about X". That is the wrong data model for a changing world, because
it cannot distinguish these two situations:

    (a) a source is WRONG                -> it contradicts reality
    (b) a source WAS RIGHT, in 2023      -> it contradicts only the present

Conflating them is the single most common failure of production RAG: a 2023
page and a 2026 page both mention "CEO", both rank highly, and the model
averages them into a confident wrong answer.

VERITAS stores facts as versioned assertions on **two independent time axes**
(the bitemporal model from temporal databases, Snodgrass 1995):

* **valid time**  [valid_from, valid_to) -- when the fact was true *in the world*
* **transaction time** [recorded_at, superseded_at) -- when *we knew* it

Two axes, not one, because they genuinely come apart. A filing published in
March 2026 may state a CEO change effective January 2026: valid_from is
January, recorded_at is March. With only one axis you must choose which lie to
tell. With both, VERITAS can answer all four question types:

    "who is CEO now"            -> valid_time = now,  transaction = latest
    "who was CEO in 2024"       -> valid_time = 2024, transaction = latest
    "who did we think was CEO
     in 2024, back in 2024"     -> valid_time = 2024, transaction = 2024
    "when did we learn?"        -> read recorded_at directly

The fourth question is the audit trail: it is what lets the system say *which
source reported the change first*, and it is impossible in any store that
overwrites.

Nothing is ever deleted. A superseded version gets `superseded_at` stamped and
stays in the log. Corrections (we were wrong) and changes (the world moved) are
represented differently: a correction closes transaction time while keeping
valid time, a change closes valid time and opens a new interval.

Complexity
----------
Versions per (entity, attribute) are kept in a list sorted by valid_from, so an
as-of query is a binary search: **O(log n)**, not a scan. Appends are usually
at the end (data arrives roughly in order), which is the O(1) path; out-of-order
backfill costs an O(n) insort. Indices are dicts of lists -- no database
required for the research scale, and the interface is narrow enough to swap for
PostgreSQL range types (`tstzrange` + GiST) in production.
"""
from __future__ import annotations

import bisect
import re
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Open-ended intervals use these sentinels rather than None, so every
# comparison is a plain datetime comparison with no null handling.
BEGINNING = datetime(1, 1, 1, tzinfo=timezone.utc)
FOREVER = datetime(9999, 12, 31, tzinfo=timezone.utc)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


#: Formats accepted from real sources. Natural-language dates ("March 2026",
#: "3 March 2026") appear constantly in filings and press releases, so the
#: parser has to handle them -- a claim whose date cannot be parsed loses its
#: place on the timeline, which is the one thing this store exists to keep.
_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m", "%Y",
    "%B %Y", "%b %Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
    "%B %d %Y", "%b %d %Y",
)


def _dt(x, strict: bool = False) -> datetime:
    """Parse a timestamp. Unparseable input becomes `now` unless `strict`,
    because dropping a whole document over an odd date string loses more
    information than dating it conservatively."""
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    if x is None:
        return FOREVER
    s = str(x).strip().replace("Z", "+00:00")
    if not s:
        return FOREVER
    try:
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    m = re.search(r"\b(1[89]\d{2}|2[01]\d{2})\b", s)  # last resort: a bare year
    if m:
        return datetime(int(m.group(1)), 1, 1, tzinfo=timezone.utc)
    if strict:
        raise ValueError(f"unparseable timestamp: {x!r}")
    return now_utc()


class ChangeKind:
    CREATED = "CREATED"        # first time we ever saw this attribute
    CHANGED = "CHANGED"        # the world changed: new valid interval
    CORRECTED = "CORRECTED"    # we were wrong: same valid interval, new record
    REAFFIRMED = "REAFFIRMED"  # a new source restates the current value
    CONFLICT = "CONFLICT"      # a source disagrees about the SAME valid time


@dataclass
class Version:
    """One assertion: entity.attribute = value, over a valid interval."""

    entity: str
    attribute: str
    value: str
    valid_from: datetime = field(default_factory=lambda: BEGINNING)
    valid_to: datetime = FOREVER
    recorded_at: datetime = field(default_factory=now_utc)
    superseded_at: Optional[datetime] = None
    source_id: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    version_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    change_kind: str = ChangeKind.CREATED
    previous_value: Optional[str] = None
    reason: str = ""

    def valid_at(self, t: datetime) -> bool:
        return self.valid_from <= t < self.valid_to

    def known_at(self, t: datetime) -> bool:
        return self.recorded_at <= t and (self.superseded_at is None or t < self.superseded_at)

    @property
    def is_current_record(self) -> bool:
        return self.superseded_at is None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("valid_from", "valid_to", "recorded_at", "superseded_at"):
            d[k] = d[k].isoformat() if isinstance(d[k], datetime) else d[k]
        return d


@dataclass
class ChangeEvent:
    entity: str
    attribute: str
    kind: str
    old_value: Optional[str]
    new_value: str
    detected_at: datetime
    effective_at: datetime
    source_id: str
    evidence_ids: List[str] = field(default_factory=list)
    note: str = ""


class TemporalStore:
    """Append-only bitemporal store of entity-attribute versions."""

    def __init__(self) -> None:
        # (entity, attribute) -> versions sorted by valid_from
        self._index: Dict[Tuple[str, str], List[Version]] = {}
        self._by_id: Dict[str, Version] = {}
        self.changes: List[ChangeEvent] = []
        self.entity_aliases: Dict[str, str] = {}

    # ------------------------------------------------------------- entities
    #: Corporate suffixes and punctuation carry no identity. "Apple Inc." and
    #: "Apple Inc" and "Apple" must resolve to one timeline, or every question
    #: silently misses the facts we hold.
    _SUFFIXES = frozenset(
        "inc incorporated corp corporation co company ltd limited plc llc lp "
        "holdings group nv sa ag se the".split()
    )

    @classmethod
    def _key(cls, entity: str) -> frozenset:
        toks = re.findall(r"[a-z0-9]+", str(entity).lower())
        core = [t for t in toks if t not in cls._SUFFIXES]
        return frozenset(core or toks)

    def canonical(self, entity: str) -> str:
        """Resolve a surface form to the stored entity name.

        Exact alias first, then token-set matching that ignores punctuation and
        corporate suffixes. Without this, the planner's "Apple Inc" never finds
        the store's "Apple Inc." and every answer about it comes back empty --
        which is exactly the bug real SEC data exposed.
        """
        raw = entity.strip()
        key = raw.lower()
        if key in self.entity_aliases:
            return self.entity_aliases[key]
        known = {e for (e, _a) in self._index}
        if raw in known:
            return raw
        want = self._key(raw)
        if not want:
            return raw
        best, best_score = None, 0.0
        for cand in known:
            have = self._key(cand)
            if not have:
                continue
            score = len(want & have) / len(want | have)
            # require the query's tokens to be fully contained, so "Apple"
            # matches "Apple Inc." but "Apple" never matches "Apple Hospitality"
            if want <= have and score > best_score:
                best, best_score = cand, score
        if best is not None:
            self.entity_aliases[key] = best   # memoise: resolution is hot
            return best
        return raw

    def alias(self, alias: str, canonical: str) -> None:
        self.entity_aliases[alias.strip().lower()] = canonical.strip()

    # --------------------------------------------------------------- writes
    def assert_fact(
        self,
        entity: str,
        attribute: str,
        value: str,
        valid_from=None,
        valid_to=None,
        recorded_at=None,
        source_id: str = "",
        evidence_ids: Optional[List[str]] = None,
        confidence: float = 0.5,
        reason: str = "",
    ) -> Tuple[Version, ChangeEvent]:
        """Insert an assertion and classify it against what we already hold.

        This is step 4-11 of the ingest pipeline in one call: compare with the
        existing state, classify (created/changed/corrected/reaffirmed/
        conflict), close the previous valid interval when the world moved, and
        keep the old row.
        """
        entity = self.canonical(entity)
        vf = _dt(valid_from) if valid_from is not None else now_utc()
        vt = _dt(valid_to) if valid_to is not None else FOREVER
        rec = _dt(recorded_at) if recorded_at is not None else now_utc()
        key = (entity, attribute)
        versions = self._index.setdefault(key, [])

        prior = self._at(versions, vf, transaction_time=None)
        kind, note = ChangeKind.CREATED, ""
        old_value = None

        if prior is not None:
            old_value = prior.value
            if _norm(prior.value) == _norm(value):
                kind = ChangeKind.REAFFIRMED
                # A restatement is not a new fact -- it is independent
                # corroboration. Widen the interval and raise confidence
                # instead of writing a duplicate row.
                prior.evidence_ids.extend(evidence_ids or [])
                if source_id and source_id != prior.source_id:
                    prior.confidence = min(0.99, prior.confidence + 0.1)
                if vt > prior.valid_to:
                    prior.valid_to = vt
                ev = ChangeEvent(entity, attribute, kind, old_value, value, rec, vf,
                                 source_id, evidence_ids or [], "restated by another source")
                self.changes.append(ev)
                return prior, ev
            elif vf > prior.valid_from:
                kind = ChangeKind.CHANGED
                prior.valid_to = min(prior.valid_to, vf)   # close the old interval
                note = "state transition"
            else:
                # same valid time, different value -> genuine disagreement
                kind = ChangeKind.CONFLICT
                note = "same valid time, different value"

        v = Version(
            entity=entity, attribute=attribute, value=value,
            valid_from=vf, valid_to=vt, recorded_at=rec,
            source_id=source_id, evidence_ids=list(evidence_ids or []),
            confidence=confidence, change_kind=kind,
            previous_value=old_value, reason=reason or note,
        )
        # keep sorted by valid_from -> O(log n) as-of lookups
        bisect.insort(versions, v, key=lambda x: x.valid_from)
        self._by_id[v.version_id] = v

        ev = ChangeEvent(entity, attribute, kind, old_value, value, rec, vf,
                         source_id, list(evidence_ids or []), note)
        self.changes.append(ev)
        return v, ev

    def correct(self, version_id: str, new_value: str, source_id: str = "", reason: str = "") -> Version:
        """We were wrong. Close transaction time, keep valid time, keep the row."""
        old = self._by_id[version_id]
        old.superseded_at = now_utc()
        return self.assert_fact(
            old.entity, old.attribute, new_value,
            valid_from=old.valid_from, valid_to=old.valid_to,
            source_id=source_id, confidence=old.confidence,
            reason=reason or f"correction of {version_id}",
        )[0]

    # ---------------------------------------------------------------- reads
    @staticmethod
    def _at(versions: List[Version], t: datetime, transaction_time: Optional[datetime]) -> Optional[Version]:
        """Binary search for the version valid at t (latest wins on overlap)."""
        i = bisect.bisect_right([v.valid_from for v in versions], t)
        for v in reversed(versions[:i]):
            if not v.valid_at(t):
                continue
            if transaction_time is None:
                if v.is_current_record:
                    return v
            elif v.known_at(transaction_time):
                return v
        return None

    def as_of(self, entity: str, attribute: str, t=None, known_at=None) -> Optional[Version]:
        """The value valid at `t`, as we understood it at `known_at`."""
        versions = self._index.get((self.canonical(entity), attribute), [])
        return self._at(versions, _dt(t) if t is not None else now_utc(),
                        _dt(known_at) if known_at is not None else None)

    def current(self, entity: str, attribute: str) -> Optional[Version]:
        return self.as_of(entity, attribute, now_utc())

    def latest(self, entity: str, attribute: str) -> Tuple[Optional[Version], bool]:
        """The most recent version, and whether it is still valid NOW.

        `current()` correctly returns None for a fact whose validity interval
        has closed -- a fiscal-year revenue is not "current revenue" once the
        year ends. But answering "nothing is known" would be wrong and unhelpful:
        the latest *reported* figure is real evidence, it simply needs a
        staleness qualifier. Returning (version, is_current) lets the answer
        layer say "most recently reported, for the period ending X" instead of
        either silence or a false present-tense claim.
        """
        cur = self.current(entity, attribute)
        if cur is not None:
            return cur, True
        versions = self.history(entity, attribute)
        if not versions:
            return None, False
        return max(versions, key=lambda v: (v.valid_to, v.recorded_at)), False

    def history(self, entity: str, attribute: str, include_superseded: bool = False) -> List[Version]:
        vs = self._index.get((self.canonical(entity), attribute), [])
        return list(vs) if include_superseded else [v for v in vs if v.is_current_record]

    def timeline(self, entity: str) -> List[Version]:
        """Every attribute of an entity, ordered -- the narrative of its change."""
        out = [v for (e, _a), vs in self._index.items() if e == self.canonical(entity) for v in vs]
        return sorted(out, key=lambda v: (v.valid_from, v.recorded_at))

    def changes_between(self, entity: str, start, end) -> List[ChangeEvent]:
        s, e = _dt(start), _dt(end)
        ent = self.canonical(entity)
        return [c for c in self.changes if c.entity == ent and s <= c.effective_at < e
                and c.kind in (ChangeKind.CHANGED, ChangeKind.CORRECTED)]

    def outdated(self, entity: str, attribute: str) -> List[Version]:
        """Versions that were once current and no longer are -- the states that
        a naive retriever would happily quote as fact."""
        cur = self.current(entity, attribute)
        return [v for v in self.history(entity, attribute) if cur is None or v.version_id != cur.version_id]

    def conflicts(self, entity: str, attribute: str) -> List[Tuple[Version, Version]]:
        """Pairs asserting different values over overlapping valid intervals."""
        vs = self.history(entity, attribute)
        out = []
        for i, a in enumerate(vs):
            for b in vs[i + 1 :]:
                overlap = a.valid_from < b.valid_to and b.valid_from < a.valid_to
                if overlap and _norm(a.value) != _norm(b.value):
                    out.append((a, b))
        return out

    def entities(self) -> List[str]:
        return sorted({e for e, _ in self._index})

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rows = [v.to_dict() for vs in self._index.values() for v in vs]
        Path(path).write_text(json.dumps({"versions": rows, "aliases": self.entity_aliases}, indent=1),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TemporalStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls()
        store.entity_aliases = payload.get("aliases", {})
        for r in payload["versions"]:
            v = Version(
                entity=r["entity"], attribute=r["attribute"], value=r["value"],
                valid_from=_dt(r["valid_from"]), valid_to=_dt(r["valid_to"]),
                recorded_at=_dt(r["recorded_at"]),
                superseded_at=_dt(r["superseded_at"]) if r.get("superseded_at") else None,
                source_id=r.get("source_id", ""), evidence_ids=r.get("evidence_ids", []),
                confidence=r.get("confidence", 0.5), version_id=r["version_id"],
                change_kind=r.get("change_kind", ChangeKind.CREATED),
                previous_value=r.get("previous_value"), reason=r.get("reason", ""),
            )
            store._index.setdefault((v.entity, v.attribute), []).append(v)
            store._by_id[v.version_id] = v
        for vs in store._index.values():
            vs.sort(key=lambda x: x.valid_from)
        return store


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().replace(",", "").split())
