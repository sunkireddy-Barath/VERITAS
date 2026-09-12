"""Change detection: decide what is actually new before paying to process it.

The economics
-------------
Re-ingesting a monitored source costs a parse, a claim-extraction pass, an
embedding pass and an index write. Most polls return a byte-identical page, or
a page whose only difference is a rotating ad slot or a "last updated" banner.
A system that re-processes everything on every poll cannot run continuously.

So detection is a cascade, cheapest filter first, and each stage only runs on
what survives the previous one:

  1. **HTTP validators** (ETag / Last-Modified) -- zero bytes transferred.
  2. **Exact digest** (BLAKE2b) -- O(n) over bytes, catches identical content.
  3. **SimHash** -- catches *near*-identical content (boilerplate churn).
  4. **Claim-level diff** -- the only stage that decides the world changed.

Stage 3 is the interesting one. SimHash (Charikar, 2002) projects a document
into a 64-bit signature such that similar documents have small Hamming
distance:

    for each feature f with weight w:  v += w * (+1/-1 per bit of hash(f))
    signature bit i = 1 if v_i > 0

Unlike MinHash it yields one machine word, so comparison is
`popcount(a ^ b)` -- a single CPU instruction. A distance <= 3 of 64 bits means
"the same page with cosmetic edits". MinHash estimates Jaccard better, but here
we only need a threshold test, and one XOR beats 128 array comparisons.

Stage 4 is where VERITAS differs from a diff tool. Text changing is not news;
a *claim* changing is. "Revenue was $1.2B" -> "Revenue was $1.4B" is a state
change. The same sentence reworded is not. So the diff runs over extracted
(entity, attribute, value) triples, not over text.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_WORD = re.compile(r"[a-z0-9]+")

# Lines that change on every poll and mean nothing. Stripping them before
# fingerprinting is what stops a clock in the footer from looking like news.
_VOLATILE = re.compile(
    r"(?im)^.*(last updated|generated on|page loaded|copyright \d{4}|session id|"
    r"\d{1,2}:\d{2}:\d{2}|csrf|nonce=)[^\n]*$"
)


def normalize_for_fingerprint(text: str) -> str:
    text = _VOLATILE.sub("", text)
    return " ".join(text.lower().split())


def content_digest(text: str) -> str:
    return hashlib.blake2b(normalize_for_fingerprint(text).encode("utf-8"), digest_size=16).hexdigest()


def simhash(text: str, bits: int = 64, ngram: int = 3) -> int:
    """64-bit SimHash over word n-grams (weighted by frequency)."""
    words = _WORD.findall(normalize_for_fingerprint(text))
    if not words:
        return 0
    feats: Dict[str, int] = {}
    grams = (
        [" ".join(words[i : i + ngram]) for i in range(len(words) - ngram + 1)]
        if len(words) >= ngram else words
    )
    for g in grams:
        feats[g] = feats.get(g, 0) + 1
    vec = [0] * bits
    for feat, w in feats.items():
        h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
        for i in range(bits):
            vec[i] += w if (h >> i) & 1 else -w
    out = 0
    for i in range(bits):
        if vec[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class SourceState:
    """What we remember about a monitored source between polls."""

    source_id: str
    digest: str = ""
    simhash: int = 0
    etag: str = ""
    last_modified: str = ""
    claims: Dict[Tuple[str, str], str] = field(default_factory=dict)  # (entity, attr) -> value
    poll_count: int = 0
    change_count: int = 0

    @property
    def volatility(self) -> float:
        """Observed change rate -- drives adaptive polling. A source that never
        changes should not be polled at the same rate as a live status page."""
        return self.change_count / max(1, self.poll_count)


@dataclass
class ChangeReport:
    changed: bool
    reason: str
    similarity: float = 1.0
    added: List[Tuple[str, str, str]] = field(default_factory=list)     # (entity, attr, new)
    modified: List[Tuple[str, str, str, str]] = field(default_factory=list)  # (.., old, new)
    removed: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def has_state_change(self) -> bool:
        return bool(self.added or self.modified or self.removed)


class ChangeDetector:
    """Cascade detector with per-source memory."""

    def __init__(self, near_dup_bits: int = 3) -> None:
        self.near_dup_bits = near_dup_bits
        self.states: Dict[str, SourceState] = {}

    def state(self, source_id: str) -> SourceState:
        return self.states.setdefault(source_id, SourceState(source_id))

    def check(
        self,
        source_id: str,
        text: str,
        claims: Optional[Dict[Tuple[str, str], str]] = None,
        etag: str = "",
        last_modified: str = "",
    ) -> ChangeReport:
        st = self.state(source_id)
        st.poll_count += 1
        first_seen = st.poll_count == 1

        # stage 1 -- HTTP validators
        if not first_seen and etag and etag == st.etag:
            return ChangeReport(False, "etag-match")
        if not first_seen and last_modified and last_modified == st.last_modified:
            return ChangeReport(False, "last-modified-match")

        # stage 2 -- exact digest
        digest = content_digest(text)
        if not first_seen and digest == st.digest:
            return ChangeReport(False, "identical-content")

        # stage 3 -- near-duplicate
        sh = simhash(text)
        dist = hamming(sh, st.simhash) if st.simhash else 64
        similarity = 1.0 - dist / 64.0
        cosmetic = (not first_seen) and dist <= self.near_dup_bits

        # stage 4 -- claim diff (the only stage that can declare a state change)
        added: List[Tuple[str, str, str]] = []
        modified: List[Tuple[str, str, str, str]] = []
        removed: List[Tuple[str, str, str]] = []
        if claims is not None:
            old = st.claims
            for k, v in claims.items():
                if k not in old:
                    added.append((k[0], k[1], v))
                elif _norm(old[k]) != _norm(v):
                    modified.append((k[0], k[1], old[k], v))
            for k, v in old.items():
                if k not in claims:
                    removed.append((k[0], k[1], v))
            st.claims = dict(claims)

        st.digest, st.simhash = digest, sh
        st.etag, st.last_modified = etag or st.etag, last_modified or st.last_modified

        if cosmetic and not (added or modified or removed):
            return ChangeReport(False, "cosmetic-edit", similarity)

        changed = first_seen or bool(added or modified or removed) or not cosmetic
        if changed:
            st.change_count += 1
        reason = (
            "first-seen" if first_seen
            else "claim-change" if (added or modified or removed)
            else "content-change"
        )
        return ChangeReport(changed, reason, similarity, added, modified, removed)

    def next_poll_seconds(self, source_id: str, base: int = 3600, lo: int = 300, hi: int = 604_800) -> int:
        """Adaptive polling interval.

        Multiplicative back-off on a stable source, aggressive speed-up on a
        volatile one. A fixed schedule either hammers static filings or misses
        a status page that flips in minutes; the observed change rate is a
        better prior than any hand-set number.
        """
        st = self.state(source_id)
        if st.poll_count < 3:
            return base
        v = st.volatility
        interval = base / max(v, 0.01) if v < 0.5 else base * v
        return int(min(hi, max(lo, interval)))


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().replace(",", "").split())
