"""Atomic claim extraction: the unit of verification.

Why claims and not sentences
----------------------------
"Acme, which moved to Berlin in 2023, reported $1.4B revenue and named Y as
CEO in March." One sentence, four independently checkable assertions. Verifying
it as a unit forces a single verdict on four facts, so one wrong number makes
the whole sentence unsupported -- or, worse, three correct facts make it look
supported. Decomposition is what makes *claim-level* provenance possible, and
it is what makes the evidence-coverage metric mean anything.

A good atomic claim is:
  * **standalone** -- no unresolved pronouns or "the company"; the subject is
    substituted back in, because the claim will be matched against evidence in
    isolation,
  * **single-predicate** -- exactly one thing to check,
  * **time-qualified** -- carries its own valid interval when one is stated,
  * **checkable** -- an opinion ("a bold move") is not a claim; marking it
    non-checkable keeps it out of the coverage denominator instead of counting
    as a permanent failure.

Implementation: rules first, model second
-----------------------------------------
The extractor is a deterministic clause splitter plus a typed-value parser.
Rules are used here rather than an LM pass because (a) extraction runs on every
ingested document and every generated answer, so it is the hottest path in the
system; (b) an LM extractor can *hallucinate a claim that was never in the
text*, which silently corrupts the evidence store -- the one failure mode this
project exists to prevent. The fine-tuned model refines borderline cases via
`<|claim|>` spans; the rules stay as the floor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")
_CLAUSE = re.compile(r",\s+(?:and|but|while|whereas)\s+|;\s*|\s+--\s+")
_REL_CLAUSE = re.compile(r",\s+which\s+[^,]+,\s*")
_PRONOUN = re.compile(r"\b(it|its|they|their|he|she|his|her|the company|the firm|this)\b", re.I)

_NUM = re.compile(
    r"(?P<cur>[$€£¥])?\s?(?P<num>\d[\d,]*\.?\d*)\s?(?P<unit>billion|million|bn|m|k|%|percent)?",
    re.I,
)
_DATE = re.compile(
    r"\b(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}|\b(?:19|20)\d{2}\b)", re.I,
)
_SINCE = re.compile(r"\b(?:since|effective|starting|as of|from)\s+(" + _DATE.pattern + r")", re.I)
_UNTIL = re.compile(r"\b(?:until|through|up to|ended?)\s+(" + _DATE.pattern + r")", re.I)

#: Capitalised words that start a sentence but are never a person's name.
#: Never a person's name: sentence-initial function words, and -- importantly --
#: role words. "...her tenure as CEO..." matches the "as <Name>" pattern and
#: would otherwise store the CEO of the company as the literal string "CEO".
_NON_NAMES = frozenset(
    "The A An In On At For And But It He She They This That Its His Her "
    "Chief CEO CTO CFO Executive Officer President Chairman Director Board "
    "Company Group Inc Ltd Corp Limited Holdings".split()
)

_HEDGE = re.compile(
    r"\b(may|might|could|reportedly|allegedly|expected to|plans to|is likely|"
    r"rumou?red|appears|seems|suggests|potentially|possibly)\b", re.I
)
_OPINION = re.compile(
    r"\b(believe|think|feel|arguably|remarkable|impressive|bold|disappointing|"
    r"best|worst|should|ought)\b", re.I
)

#: attribute keyword -> canonical attribute name. Canonicalisation is what lets
#: "chief executive", "CEO" and "chief exec" collapse onto one timeline.
ATTRIBUTE_LEXICON: Dict[str, Tuple[str, ...]] = {
    "ceo": ("ceo", "chief executive"),
    "cto": ("cto", "chief technology officer"),
    "cfo": ("cfo", "chief financial officer"),
    "revenue": ("revenue", "sales", "turnover", "top line"),
    # net_income BEFORE profit: dict order decides which canonical name wins,
    # and the SEC XBRL loader keys this attribute as "net_income". A mismatch
    # here silently splits one timeline into two and the store lookup returns
    # nothing -- which made the system abstain on facts it actually held.
    "net_income": ("net income", "net profit", "net earnings", "netincomeloss"),
    "profit": ("profit", "earnings", "operating income"),
    "headcount": ("headcount", "employees", "staff", "workforce"),
    "status": ("status", "state", "phase", "stage", "construction", "operational",
               "commissioned", "under way", "underway", "completed", "approved",
               "cancelled", "suspended", "delayed", "launched"),
    "valuation": ("valuation", "valued at", "market cap"),
    "headquarters": ("headquarters", "hq", "based in", "headquartered"),
    "price": ("price", "share price", "stock price"),
    "funding": ("funding", "raised", "round", "series"),
}


@dataclass
class Claim:
    text: str
    subject: str = ""
    attribute: str = ""
    value: str = ""
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    doc_id: str = ""
    chunk_id: str = ""
    char_span: Tuple[int, int] = (0, 0)
    checkable: bool = True
    hedged: bool = False
    numeric: Optional[float] = None
    unit: str = ""
    claim_id: str = ""
    confidence: float = 0.5

    @property
    def key(self) -> Tuple[str, str]:
        return (self.subject.lower().strip(), self.attribute)

    def as_triple(self) -> Tuple[str, str, str]:
        return (self.subject, self.attribute, self.value)


def _canonical_attribute(text: str) -> str:
    low = text.lower()
    for canon, words in ATTRIBUTE_LEXICON.items():
        if any(w in low for w in words):
            return canon
    return ""


def _parse_number(text: str) -> Tuple[Optional[float], str]:
    """Normalise magnitudes so '1.4 billion', '$1,400M' and '1400000000'
    compare as equal. Without this every numeric claim reads as a conflict."""
    m = _NUM.search(text)
    if not m:
        return None, ""
    try:
        val = float(m.group("num").replace(",", ""))
    except ValueError:
        return None, ""
    unit = (m.group("unit") or "").lower()
    mult = {"billion": 1e9, "bn": 1e9, "million": 1e6, "m": 1e6, "k": 1e3}.get(unit, 1.0)
    if unit in ("%", "percent"):
        return val, "%"
    return val * mult, m.group("cur") or ""


def split_sentences(text: str) -> List[Tuple[str, int]]:
    out, pos = [], 0
    for part in _SENT.split(text):
        i = text.find(part, pos)
        out.append((part.strip(), i if i >= 0 else pos))
        pos = (i if i >= 0 else pos) + len(part)
    return [(s, p) for s, p in out if s]


def extract_claims(
    text: str,
    doc_id: str = "",
    chunk_id: str = "",
    subject_hint: str = "",
    base_offset: int = 0,
) -> List[Claim]:
    """Decompose text into atomic, offset-anchored claims."""
    claims: List[Claim] = []
    for sent, s_off in split_sentences(text):
        # a relative clause is its own assertion: pull it out first
        parts: List[str] = []
        rel = _REL_CLAUSE.search(sent)
        if rel and subject_hint:
            parts.append(f"{subject_hint} {rel.group(0).strip(', ').replace('which ', '')}")
            sent = _REL_CLAUSE.sub(" ", sent)
        parts.extend(p.strip() for p in _CLAUSE.split(sent) if p.strip())

        for part in parts:
            if len(part.split()) < 3:
                continue
            subject = subject_hint or _leading_subject(part)
            # resolve pronouns to the hint: a claim with a dangling "it" cannot
            # be matched against evidence on its own
            resolved = _PRONOUN.sub(subject, part, count=1) if subject and _PRONOUN.match(part) else part
            attribute = _canonical_attribute(part)
            num, unit = _parse_number(part)
            since = _SINCE.search(part)
            until = _UNTIL.search(part)
            value = _extract_value(part, attribute, num, unit)

            c = Claim(
                text=resolved.strip(),
                subject=subject,
                attribute=attribute,
                value=value,
                valid_from=since.group(1) if since else None,
                valid_to=until.group(1) if until else None,
                doc_id=doc_id,
                chunk_id=chunk_id,
                char_span=(base_offset + s_off, base_offset + s_off + len(part)),
                checkable=not bool(_OPINION.search(part)),
                hedged=bool(_HEDGE.search(part)),
                # A person-valued claim has no numeric value: the year in
                # "named Tim Cook CEO, effective 2011" is a date, and comparing
                # it as a number made every succession look like a conflict.
                numeric=None if attribute in PERSON_ATTRIBUTES else num,
                unit=unit,
                claim_id=f"{chunk_id or doc_id}:c{len(claims)}",
            )
            # a hedged statement is a claim about uncertainty, so it can be
            # supported ("the filing says it *may*"), but it can never be
            # treated as establishing the fact
            c.confidence = 0.3 if c.hedged else 0.6
            claims.append(c)
    return claims


def _leading_subject(sentence: str) -> str:
    """Cheap subject heuristic: the leading capitalised noun phrase."""
    words = sentence.split()
    out = []
    for w in words[:6]:
        if w[:1].isupper() or (out and w.lower() in ("of", "and", "&")):
            out.append(w.strip(",.:;"))
        elif out:
            break
    return " ".join(out)


#: Attributes whose value is a person, not a number. Checked BEFORE the numeric
#: path: "Marcus Lund was appointed chief executive effective February 2026"
#: contains a number, and a numeric-first rule would store the CEO as "2026".
PERSON_ATTRIBUTES = frozenset({"ceo", "cto", "cfo"})

#: Canonical status values, longest first so "construction started" wins over
#: "construction". Ordering matters: a substring match would otherwise collapse
#: distinct lifecycle states onto one another.
STATUS_VALUES = (
    "construction started", "begun construction", "started construction",
    "under construction", "construction", "operational", "commissioned",
    "completed", "approved", "planning phase", "planned", "cancelled",
    "suspended", "delayed", "launched",
)

_NAME = r"([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+){0,2})"
_PERSON_PATTERNS = (
    re.compile(r"\b(?:is|as|named|appointed|becomes?|succeeded by)\s+" + _NAME),
    # name-first phrasing: "<Name> was appointed / has been named / becomes"
    re.compile(_NAME + r"\s+(?:was|is|has been|will be)\s+(?:appointed|named|becoming)"),
    re.compile(_NAME + r"\s+becomes?\b"),
)


def _extract_value(text: str, attribute: str, num: Optional[float], unit: str) -> str:
    if attribute in PERSON_ATTRIBUTES:
        for pat in _PERSON_PATTERNS:
            m = pat.search(text)
            if m:
                name = m.group(1).strip()
                if name.split()[0] not in _NON_NAMES:
                    return name
        return ""
    if attribute == "status":
        # Status is categorical, so the numeric path must not run: the date in
        # "construction began in March 2026" is not the status.
        m = re.search(r"\bstatus\s+(?:is|=|:)\s*([a-z ]+)", text, re.I)
        if m:
            return m.group(1).strip()
        for kw in STATUS_VALUES:
            if kw in text.lower():
                return kw
        return ""
    if num is not None:
        return f"{num:g}{unit}"
    return ""


def claims_to_state(claims: Sequence[Claim]) -> Dict[Tuple[str, str], str]:
    """Collapse claims to a {(entity, attribute): value} snapshot, which is what
    `ChangeDetector` diffs between polls."""
    state: Dict[Tuple[str, str], str] = {}
    for c in claims:
        if c.subject and c.attribute and c.value and not c.hedged:
            state[(c.subject.strip(), c.attribute)] = c.value
    return state


def split_answer_into_claims(answer: str, subject_hint: str = "") -> List[Claim]:
    """Decompose a *generated* answer for self-verification.

    Citation markers are stripped before extraction so "[E2]" never ends up
    inside the claim text being matched against evidence.
    """
    clean = re.sub(r"\[E\d+\]", "", answer)
    return extract_claims(clean, doc_id="__generated__", subject_hint=subject_hint)
