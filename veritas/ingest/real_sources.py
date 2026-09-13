"""Load REAL data into VERITAS. No fixtures, no synthetic facts.

Three loaders, one per kind of real data:

`load_wiki_revisions`  -- the important one
    Wikipedia revision history is genuine dated evidence. When an article said
    "Tim Cook" in a 2015 revision and something else in a 2023 revision, that is
    a real source that really did assert the old value before the new one, with
    a real timestamp and a permanent citable URL (`?oldid=`). Feeding those
    revisions into the bitemporal store *in chronological order* reconstructs a
    real timeline, with real `CHANGED` transitions -- exactly what the store
    models, and something no synthetic dataset can honestly provide.

    Note what this data is and is not. A revision timestamp is *transaction
    time*: when Wikipedia recorded the claim. It is a lower bound on, not the
    same as, the real-world effective date. VERITAS stores it as `recorded_at`
    and only sets `valid_from` when the text states an effective date -- which
    is precisely why the store has two axes.

`load_feeds`
    Live RSS/Atom items from SEC, WHO, ECB, NASA, arXiv, with real publication
    dates. Used for the continuous-update path and freshness measurement.

`load_sec_filings`
    SEC EDGAR full-text search. Tier-1 primary sources for corporate facts.
"""
from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import certifi

    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    _SSL = ssl.create_default_context()

UA = "VERITAS-research/0.1 (temporal evidence RAG; contact: local researcher)"

#: Source tiers for the corporate domain, by host. Used by SourcePolicy and by
#: the evidence-quality model. Wikipedia is deliberately tier 3: it is a
#: tertiary source that is usually right and occasionally, briefly, wrong --
#: which makes it honest training data for a system built to handle exactly
#: that.
REAL_SOURCE_TIERS: Dict[str, int] = {
    "sec.gov": 1, "www.sec.gov": 1,
    "ecb.europa.eu": 1, "www.ecb.europa.eu": 1,
    "who.int": 1, "www.who.int": 1,
    "nasa.gov": 1, "www.nasa.gov": 1,
    "arxiv.org": 2, "export.arxiv.org": 2,
    # Structured, referenced statements with explicit start/end dates: better
    # than an infobox scrape, still not a primary source.
    "wikidata.org": 2,
    "en.wikipedia.org": 3,
}


def _get(url: str, params: Optional[dict] = None, retries: int = 3) -> bytes:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30, context=_SSL) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                time.sleep(int(exc.headers.get("Retry-After", 0) or 0) or 15 * (attempt + 1))
            else:
                time.sleep(2 * (attempt + 1))
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url} ({last})")


def _iso(ts: str) -> str:
    """Normalise the several date formats real feeds actually emit."""
    ts = (ts or "").strip()
    if not ts:
        return datetime.now(timezone.utc).date().isoformat()
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                "%a, %d %b %Y %H:%M:%S", "%d %b %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts, fmt).date().isoformat()
        except ValueError:
            continue
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", ts)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else \
        datetime.now(timezone.utc).date().isoformat()


# --------------------------------------------------------- wiki revisions
#: The only infobox attribute asserted into the store. See load_wiki_revisions.
_WIKI_ASSERTABLE = frozenset({"ceo"})


def _clean_wiki_value(attribute: str, value: str) -> str:
    """A usable infobox value, or "" when only markup survived extraction.

    Older fetches cut values at the first "|" inside a template, leaving
    fragments like "{{US$" or "{{ubl". An unclosed "{{" means everything after
    it was lost, so the value is cut there and kept only if a name (or, for
    revenue, a figure) remains.
    """
    v = str(value or "").split("{{", 1)[0]
    v = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", v)
    v = re.sub(r"[\[\]{}|<>]", " ", v)
    v = re.sub(r"\s+", " ", v).strip(" .,;:")
    if not re.search(r"[A-Za-z]{2}", v):
        return ""
    if attribute == "revenue" and not re.search(r"\d", v):
        return ""
    return v


def load_wiki_revisions(
    builder, path: str | Path, max_entities: int = 0, verbose: bool = True
) -> Dict[str, int]:
    """Replay real revision history into the temporal store, oldest first.

    Chronological order is essential: the store classifies an assertion by
    comparing it with what it already holds, so replaying out of order would
    turn a real succession into a spurious CONFLICT.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/fetch_real_data.py --revisions-only")

    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    records.sort(key=lambda r: r["timestamp"])
    if max_entities:
        keep = list(dict.fromkeys(r["entity"] for r in records))[:max_entities]
        records = [r for r in records if r["entity"] in keep]

    stats = {"records": 0, "versions": 0, "transitions": 0, "entities": 0, "docs": 0,
             "rejected": 0}
    builder.add_source("en.wikipedia.org", tier=3, domain="corporate")

    for r in records:
        entity = r["entity"]
        ts = _iso(r["timestamp"])
        doc_id = f"wiki:{entity}:{r['revid']}"
        facts = {a: _clean_wiki_value(a, v)
                 for a, v in r.get("changed", r.get("facts", {})).items()}
        facts = {a: v for a, v in facts.items() if v}
        if not facts:
            # Only markup survived extraction ("{{US$", "{{ubl"). Storing that
            # as a fact is how "{{US$" became Nvidia's current revenue.
            stats["rejected"] += 1
            continue

        # The revision becomes a real, citable evidence document.
        parts = [f"{entity} (Wikipedia revision {r['revid']}, {ts})."]
        for attr, val in facts.items():
            label = {"ceo": "chief executive", "revenue": "revenue"}.get(attr, attr)
            parts.append(f"The {label} is {val}.")
        if r.get("comment"):
            parts.append(f"Edit summary: {r['comment']}")
        text = " ".join(parts)

        builder.add_document("en.wikipedia.org", doc_id, text, ts, entity,
                             tier=3, url=r.get("url", ""), assert_claims=False)
        stats["docs"] += 1

        # The fact itself, asserted with the revision timestamp as the time we
        # LEARNED it. valid_from is the same here because a Wikipedia revision
        # does not state an effective date -- an SEC filing does, which is why
        # the two loaders differ.
        for attr, val in facts.items():
            # An infobox revenue states no fiscal period, so asserting it "valid
            # since the edit" would outrank the filed figure as the current
            # value. And a tier-3 infobox never rewrites a timeline a structured
            # source already holds: it stays citable text, not a state change.
            if attr not in _WIKI_ASSERTABLE or builder.store.history(entity, attr):
                continue
            _v, event = builder.store.assert_fact(
                entity=entity, attribute=attr, value=val,
                valid_from=ts, recorded_at=ts,
                source_id="en.wikipedia.org",
                evidence_ids=[doc_id], confidence=0.55,
                reason=f"wikipedia revision {r['revid']}",
            )
            stats["versions"] += 1
            if event.kind in ("CHANGED", "CORRECTED"):
                stats["transitions"] += 1
        stats["records"] += 1

    stats["entities"] = len(builder.store.entities())
    if verbose:
        print(f"[real] wiki revisions: {stats['records']} revisions, {stats['docs']} documents, "
              f"{stats['entities']} entities, {stats['transitions']} real state transitions, "
              f"{stats['rejected']} rejected as markup-only")
    return stats


# ------------------------------------------------------------------ feeds
def load_feeds(builder, path: str | Path, verbose: bool = True) -> Dict[str, int]:
    """Ingest live feed items as dated evidence documents."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/fetch_real_data.py --feeds-only")

    stats = {"items": 0, "sources": 0}
    seen_sources = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        host = urllib.parse.urlparse(rec.get("url", "")).netloc or rec["source"]
        tier = REAL_SOURCE_TIERS.get(host, 2)
        if host not in seen_sources:
            builder.add_source(host, tier=tier, domain="corporate")
            seen_sources.add(host)
        text = f"{rec['title']}. {rec.get('summary', '')}".strip()
        if len(text) < 40:
            continue
        builder.add_document(host, f"feed:{rec['source']}:{stats['items']}", text,
                             _iso(rec.get("published", "")), "", tier=tier,
                             url=rec.get("url", ""))
        stats["items"] += 1
    stats["sources"] = len(seen_sources)
    if verbose:
        print(f"[real] feeds: {stats['items']} live items from {stats['sources']} sources")
    return stats


def refresh_feeds(builder, feeds: Sequence[Tuple[str, str]], verbose: bool = True) -> Dict[str, int]:
    """Re-poll live feeds NOW and ingest anything new.

    This is the continuous-update path against real moving sources: the change
    detector suppresses items already seen, so repeated calls are cheap and only
    genuinely new items reach the store.
    """
    stats = {"fetched": 0, "changed": 0}
    for name, url in feeds:
        try:
            xml = _get(url).decode("utf-8", errors="replace")
        except Exception as exc:
            if verbose:
                print(f"[real] {name}: unreachable ({str(exc)[:60]})")
            continue
        host = urllib.parse.urlparse(url).netloc
        tier = REAL_SOURCE_TIERS.get(host, 2)
        builder.add_source(host, tier=tier, domain="corporate")
        for i, item in enumerate(re.findall(r"<(?:item|entry)\b.*?</(?:item|entry)>", xml, re.S)[:25]):
            def tag(t: str) -> str:
                m = re.search(rf"<{t}[^>]*>(.*?)</{t}>", item, re.S)
                if not m:
                    return ""
                v = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", m.group(1), flags=re.S)
                return re.sub(r"<[^>]+>", " ", v).strip()
            title = tag("title")
            if not title:
                continue
            text = f"{title}. {tag('description') or tag('summary')}".strip()
            published = _iso(tag("pubDate") or tag("updated") or tag("published"))
            res = builder.add_document(host, f"live:{name}:{abs(hash(title)) % 10**9}",
                                       text, published, "", tier=tier)
            stats["fetched"] += 1
            if getattr(res, "changed", False):
                stats["changed"] += 1
    if verbose:
        print(f"[real] refreshed {stats['fetched']} items, {stats['changed']} new")
    return stats


# ------------------------------------------------------------ SEC EDGAR
def fetch_sec_filings(query: str, limit: int = 10, forms: str = "8-K,10-K,10-Q") -> List[dict]:
    """SEC EDGAR full-text search -- tier-1 primary corporate sources.

    EDGAR filings are the best real evidence available for this domain: they
    carry an explicit filing date AND frequently state an *effective* date in
    the text, which is the case that makes two time axes necessary.
    """
    try:
        data = json.loads(_get("https://efts.sec.gov/LATEST/search-index", {
            "q": query, "forms": forms, "hits": str(limit)}))
    except Exception:
        try:
            data = json.loads(_get("https://efts.sec.gov/LATEST/search-index?q="
                                   + urllib.parse.quote(query)))
        except Exception:
            return []
    out = []
    for hit in data.get("hits", {}).get("hits", [])[:limit]:
        src = hit.get("_source", {})
        out.append({
            "doc_id": hit.get("_id", ""),
            "form": src.get("file_type", ""),
            "date": src.get("file_date", ""),
            "entity": (src.get("display_names") or [""])[0],
            "text": src.get("file_description", "") or query,
            "url": f"https://www.sec.gov/Archives/edgar/data/{hit.get('_id','').replace(':','/')}",
        })
    return out




# --------------------------------------------------------------- SEC XBRL
def _human(v: float, unit: str) -> str:
    if unit in ("USD", "usd"):
        for div, suf in ((1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
            if abs(v) >= div:
                return f"{v/div:.2f} {suf} USD"
        return f"{v:,.0f} USD"
    return f"{v:,.0f} {unit}"


#: When one filing reports a period under several XBRL tags that map to the same
#: attribute, the lower number wins. Walmart's FY2020 10-K reports Revenues
#: (523.96B, total, including membership income) AND
#: RevenueFromContractWithCustomerExcludingAssessedTax (519.93B, a subset). Both
#: arrive with the same filing date, so without a priority they became a
#: same-source CONFLICT and whichever row loaded last decided the answer.
SEC_TAG_PRIORITY: Dict[str, int] = {
    "Revenues": 0,
    "RevenueFromContractWithCustomerExcludingAssessedTax": 1,
}


def _prefer_primary_tags(rows: List[dict]) -> List[dict]:
    """Keep, per (entity, attribute, period, filing), only the highest-priority tag."""
    best: Dict[tuple, int] = {}
    for r in rows:
        key = (r["entity"], r["attribute"], r["valid_from"], r["valid_to"], r.get("accn", ""))
        rank = SEC_TAG_PRIORITY.get(r.get("tag", ""), 99)
        best[key] = min(best.get(key, 99), rank)
    return [r for r in rows
            if SEC_TAG_PRIORITY.get(r.get("tag", ""), 99) == best[
                (r["entity"], r["attribute"], r["valid_from"], r["valid_to"], r.get("accn", ""))]]


def load_sec_facts(builder, path: str | Path, max_entities: int = 0,
                   verbose: bool = True) -> Dict[str, int]:
    """Load real SEC XBRL facts as genuinely bitemporal assertions.

    This is the cleanest real demonstration of why VERITAS stores two time axes:

        valid_from/valid_to  = the fiscal period the number describes
        recorded_at          = the day the filing disclosed it

    In this dataset those differ by a median of ~400 days. And 10-K restatements
    give real CORRECTED events: the same fiscal period, refiled years later with
    a different value. A store keyed on one timestamp must either lose the
    original or lose the correction; this one keeps both, and can still answer
    "what did we believe in 2009?".

    Records are replayed in FILING order, because that is the order the world
    actually learned them -- replaying by fiscal period would present a 2010
    restatement as if it had been known in 2007.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/fetch_real_data.py --sec-only")

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = _prefer_primary_tags(rows)
    rows.sort(key=lambda r: (r.get("filed") or "", r.get("valid_from") or ""))
    if max_entities:
        keep = list(dict.fromkeys(r["entity"] for r in rows))[:max_entities]
        rows = [r for r in rows if r["entity"] in keep]

    builder.add_source("sec.gov", tier=1, domain="corporate")
    stats = {"facts": 0, "documents": 0, "transitions": 0, "corrections": 0}

    with builder.ingest.bulk_load(verbose=verbose):
      for r in rows:
          entity, attr = r["entity"], r["attribute"]
          value = _human(r["value"], r.get("unit", "USD"))
          filed = r.get("filed") or r["valid_to"]
          doc_id = f"sec:{r['cik']}:{r.get('accn', '')}:{attr}:{r['valid_from']}"

          text = (f"{entity} reported {attr.replace('_', ' ')} of {value} "
                  f"for the fiscal period {r['valid_from']} to {r['valid_to']}, "
                  f"as disclosed in {r.get('form', '10-K')} filed {filed}"
                  + (" (restated)." if r.get("restatement") else "."))

          # assert_claims=False: we hold the typed value and the exact fiscal
          # period already; letting the regex extractor re-derive them from this
          # sentence would date the fact by FILING day and assert a duplicate.
          builder.add_document("sec.gov", doc_id, text, filed, entity, tier=1,
                               url=r.get("url", ""), assert_claims=False)
          # Pin the chunk metadata to the FISCAL period, not the filing date:
          # temporal retrieval must match "revenue in 2016" against the period the
          # number describes, not the day it was published.
          for cid, md in builder.metadata.items():
              if md.get("doc_id") == doc_id:
                  md["valid_from"] = r["valid_from"]
                  md["valid_to"] = r["valid_to"]
          stats["documents"] += 1

          _v, event = builder.store.assert_fact(
              entity=entity, attribute=attr, value=value,
              valid_from=r["valid_from"], valid_to=r["valid_to"],
              recorded_at=filed, source_id="sec.gov",
              evidence_ids=[doc_id], confidence=0.9,
              reason=f"{r.get('form','10-K')} accn {r.get('accn','')}"
                     + (" (restatement)" if r.get("restatement") else ""),
          )
          stats["facts"] += 1
          if event.kind == "CHANGED":
              stats["transitions"] += 1
          if r.get("restatement"):
              stats["corrections"] += 1

    if verbose:
        print(f"[real] SEC: {stats['facts']} bitemporal facts, {stats['documents']} documents, "
              f"{stats['transitions']} transitions, {stats['corrections']} restatements")
    return stats


# ------------------------------------------------------------ Wikidata CEOs
def _fmt_date(iso: str, precision: Optional[str]) -> str:
    """Render a date no more precisely than the source states it."""
    if precision == "year":
        return iso[:4]
    if precision == "month":
        return datetime.fromisoformat(iso).strftime("%B %Y")
    return iso


def load_ceo_facts(builder, path: str | Path, max_entities: int = 0,
                   verbose: bool = True) -> Dict[str, int]:
    """Load dated CEO tenures (Wikidata P169) as one timeline per company.

    Each tenure becomes its own citable document. The store, however, gets the
    timeline cut at every start and end date, because the raw statements
    overlap: a successor who starts weeks before the predecessor leaves, or
    genuine co-CEOs (Netflix, 2020-2023). Asserting tenures one at a time would
    let each start date silently truncate the tenure before it.
    """
    from collections import defaultdict
    from datetime import timedelta

    from ..temporal.versioning import FOREVER, _dt

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/fetch_real_data.py --ceo-only")

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_entity: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_entity[r["entity"]].append(r)
    entities = list(by_entity)[:max_entities] if max_entities else list(by_entity)

    tier = REAL_SOURCE_TIERS["wikidata.org"]
    builder.add_source("wikidata.org", tier=tier, domain="corporate")
    stats = {"tenures": 0, "segments": 0, "transitions": 0, "entities": len(entities)}

    def span(r):
        return _dt(r["valid_from"]), (_dt(r["valid_to"]) if r.get("valid_to") else FOREVER)

    with builder.ingest.bulk_load(verbose=False):
        for entity in entities:
            tenures = sorted(by_entity[entity], key=lambda r: r["valid_from"])
            for r in tenures:
                r["_doc"] = f"wikidata:{r['cik']}:{r['person_qid']}:{r['valid_from']}"
                text = (f"{entity} named {r['value']} chief executive officer (CEO), effective "
                        f"{_fmt_date(r['valid_from'], r.get('start_precision'))}")
                text += (f"; the tenure ended {_fmt_date(r['valid_to'], r.get('end_precision'))}."
                         if r.get("valid_to") else "; no end date is recorded.")
                text += (f" Source: Wikidata statement P169 on {r['item_qid']}, "
                         f"{r.get('references', 0)} reference(s), retrieved {r['retrieved']}.")
                builder.add_document("wikidata.org", r["_doc"], text, r["retrieved"], entity,
                                     tier=tier, url=r.get("url", ""), assert_claims=False)
                for md in builder.metadata.values():
                    if md.get("doc_id") == r["_doc"]:
                        md["valid_from"] = r["valid_from"]
                        md["valid_to"] = r.get("valid_to")
                stats["tenures"] += 1

            cuts = sorted({t for r in tenures for t in span(r)})
            segments: List[dict] = []
            for a, b in zip(cuts, cuts[1:]):
                active = [r for r in tenures if span(r)[0] <= a and span(r)[1] >= b]
                # A successor starting shortly before the predecessor leaves is a
                # handover, not co-leadership: keep the later appointment.
                active = [r for r in active
                          if not any(span(s)[0] > span(r)[0]
                                     and (span(r)[1] - span(s)[0]).days < 90 for s in active)]
                if not active:
                    continue
                names = list(dict.fromkeys(r["value"] for r in active))
                value = " and ".join(names) + (" (co-CEOs)" if len(names) > 1 else "")
                docs = {r["_doc"] for r in active}
                if segments and segments[-1]["value"] == value and segments[-1]["end"] == a:
                    segments[-1]["end"] = b
                    segments[-1]["docs"] |= docs
                else:
                    segments.append({"value": value, "start": a, "end": b, "docs": docs})

            for i, seg in enumerate(segments):
                nxt = segments[i + 1] if i + 1 < len(segments) else None
                # Tenure end dates are inclusive ("until 23 Aug", successor from
                # 24 Aug), so a one-day gap is a handover. Leaving the interval
                # open lets the successor's assertion close it as a CHANGED event.
                contiguous = nxt is not None and nxt["start"] - seg["end"] <= timedelta(days=1)
                end = None if (contiguous or seg["end"] == FOREVER) else seg["end"].date().isoformat()
                _v, event = builder.store.assert_fact(
                    entity=entity, attribute="ceo", value=seg["value"],
                    valid_from=seg["start"].date().isoformat(), valid_to=end,
                    recorded_at=tenures[0]["retrieved"], source_id="wikidata.org",
                    evidence_ids=sorted(seg["docs"]), confidence=0.7,
                    reason="Wikidata P169 tenure",
                )
                stats["segments"] += 1
                if event.kind == "CHANGED":
                    stats["transitions"] += 1

    if verbose:
        print(f"[real] CEOs: {stats['tenures']} Wikidata tenures -> {stats['segments']} "
              f"timeline segments for {stats['entities']} companies, "
              f"{stats['transitions']} successions")
    return stats


def build_real_system(builder, data_dir: str | Path = "data/real",
                      max_entities: int = 0, verbose: bool = True) -> Dict[str, object]:
    """Load every available real source into a VeritasSystemBuilder."""
    data_dir = Path(data_dir)
    report: Dict[str, object] = {}
    sec = data_dir / "sec_facts.jsonl"
    if sec.exists():
        report["sec"] = load_sec_facts(builder, sec, max_entities, verbose)
    # CEOs before Wikipedia revisions: the structured timeline must exist first
    # so a tier-3 infobox cannot rewrite it (see load_wiki_revisions).
    ceo = data_dir / "ceo_facts.jsonl"
    if ceo.exists():
        report["ceo"] = load_ceo_facts(builder, ceo, max_entities, verbose)
    rev = data_dir / "wiki_revisions.jsonl"
    if rev.exists():
        report["revisions"] = load_wiki_revisions(builder, rev, max_entities, verbose)
    feeds = data_dir / "feeds.jsonl"
    if feeds.exists():
        report["feeds"] = load_feeds(builder, feeds, verbose)
    report["summary"] = builder.ingest.summary()
    return report
