"""Fetch REAL data for VERITAS. No mock data, no static fixtures.

Three kinds of real data, because VERITAS needs three different things:

  1. PRETRAINING CORPUS  -> data/raw/corpus.txt
     Wikipedia article text (CC BY-SA) on companies, finance, technology and
     government. Domain-relevant English at volume, legally redistributable.

  2. TEMPORAL EVIDENCE   -> data/real/wiki_revisions.jsonl
     Wikipedia REVISION HISTORY. This is the important one. The same article at
     different points in time is genuine dated evidence of how a fact changed:
     a real CEO succession, with real effective dates, from a real source that
     really did say the old value before it said the new one. Synthetic data
     cannot produce that, and it is exactly what the bitemporal store models.

  3. LIVE FEEDS          -> data/real/feeds.jsonl
     RSS/Atom items with publication dates, for the continuous-update demo.

Politeness: every endpoint gets a descriptive User-Agent and rate limiting.
Wikipedia's API policy asks for <=200 req/s and a UA identifying the client;
SEC EDGAR REQUIRES a UA with contact info and throttles above 10 req/s.

Usage:
    python scripts/fetch_real_data.py --corpus-mb 12
    python scripts/fetch_real_data.py --revisions-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import ssl
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REAL = ROOT / "data" / "real"

UA = "VERITAS-research/0.1 (temporal evidence RAG; contact: local researcher)"

# Windows ships an incomplete root store for Python, so some public endpoints
# fail CERTIFICATE_VERIFY_FAILED. certifi carries Mozilla's CA bundle. We still
# VERIFY -- this fixes trust, it does not disable it.
try:
    import certifi

    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
WIKI_API = "https://en.wikipedia.org/w/api.php"

#: Entities with genuinely time-varying attributes -- the whole point. These are
#: real organisations whose leadership, revenue and status actually changed, so
#: their revision histories contain real state transitions to reconstruct.
ENTITIES: List[str] = [
    "Apple Inc.", "Microsoft", "Alphabet Inc.", "Amazon (company)", "Meta Platforms",
    "Nvidia", "Tesla, Inc.", "Intel", "IBM", "Oracle Corporation",
    "Boeing", "Airbus", "Siemens", "Samsung Electronics", "Toyota",
    "Reliance Industries", "Tata Consultancy Services", "Infosys", "Wipro",
    "HDFC Bank", "State Bank of India", "Adani Group",
    "OpenAI", "Anthropic", "DeepMind", "Hugging Face",
    "Twitter", "Netflix", "Uber", "Airbnb", "Starbucks", "Walmart",
    "Volkswagen Group", "Ford Motor Company", "General Motors",
    "Pfizer", "Moderna", "AstraZeneca", "Novartis",
    "World Health Organization", "International Monetary Fund",
    "European Central Bank", "Federal Reserve", "Reserve Bank of India",
]

#: Broader topical articles, for pretraining language coverage only.
TOPICS: List[str] = [
    "Corporate governance", "Chief executive officer", "Annual report",
    "Financial statement", "Revenue", "Initial public offering",
    "Mergers and acquisitions", "Securities and Exchange Commission",
    "Regulation", "Central bank", "Monetary policy", "Inflation",
    "Supply chain", "Semiconductor industry", "Artificial intelligence",
    "Machine learning", "Large language model", "Information retrieval",
    "Knowledge graph", "Fact-checking", "Provenance", "Temporal database",
    "Renewable energy", "Electric vehicle", "Climate change mitigation",
    "Public company", "Stock market", "Venture capital", "Antitrust",
    "Data center", "Cloud computing", "Cybersecurity", "Privacy law",
]

FEEDS: List[Tuple[str, str]] = [
    ("sec-press", "https://www.sec.gov/news/pressreleases.rss"),
    ("who-news", "https://www.who.int/rss-feeds/news-english.xml"),
    ("ecb-press", "https://www.ecb.europa.eu/rss/press.html"),
    ("nasa-news", "https://www.nasa.gov/rss/dyn/breaking_news.rss"),
    ("arxiv-ai", "http://export.arxiv.org/rss/cs.AI"),
]


def get(url: str, params: Optional[dict] = None, retries: int = 4, pause: float = 0.15) -> bytes:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
                time.sleep(pause)  # be polite; these are free public endpoints
                return r.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if attempt == retries - 1:
                break   # never sleep after the final attempt: pure dead time
            if exc.code == 429:
                # Rate limited. Back off hard and honour Retry-After. Hammering
                # a free public API is both rude and self-defeating.
                wait = int(exc.headers.get("Retry-After", 0) or 0) or 20 * (attempt + 1)
                print(f"  [429] backing off {wait}s", file=sys.stderr)
                time.sleep(wait)
            else:
                time.sleep(2.0 * (attempt + 1))
        except Exception as exc:  # transient network failures are normal at volume
            last = exc
            if attempt == retries - 1:
                break
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed after {retries}: {url} ({last})")


# --------------------------------------------------------------- gutenberg
#: Public-domain works (Project Gutenberg). Chosen for non-fiction, expository
#: English -- closer to filings and reports than novels are. Whole books arrive
#: in ONE request each, so 30 requests give ~15 MB: far gentler on a free public
#: service than thousands of API calls, and unambiguously redistributable.
GUTENBERG_IDS = [
    2130, 3300, 1232, 4280, 5827, 1497, 6762, 2680, 10615, 61,
    1404, 815, 7370, 3207, 14988, 20203, 2456, 33310, 16712, 2009,
    41360, 28054, 9662, 25344, 2600, 1342, 84, 1080, 145, 730,
    174, 2701, 76, 1661, 98, 219, 1400, 120, 46, 205,
]


def gutenberg_text(book_id: int) -> str:
    """One public-domain book, header/footer stripped."""
    for url in (f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
                f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"):
        try:
            raw = get(url, pause=0.5).decode("utf-8", errors="replace")
        except Exception:
            continue
        start = raw.find("*** START OF")
        end = raw.find("*** END OF")
        if start != -1:
            raw = raw[raw.index(chr(10), start) + 1:]

        if end != -1:
            raw = raw[: raw.find("*** END OF")]
        if len(raw) > 20000:
            return raw.strip()
    return ""


def build_gutenberg(target_bytes: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    size = 0
    for bid in GUTENBERG_IDS:
        if size >= target_bytes:
            break
        text = gutenberg_text(bid)
        if text:
            out[f"gutenberg-{bid}"] = text
            size += len(text)
            print(f"  [gutenberg] {bid:6d}  {len(text)/1e6:.2f} MB  (total {size/1e6:.1f} MB)")
    return out


# ------------------------------------------------------------------- corpus
def _one_extract(title: str) -> Tuple[str, str]:
    """Fetch one article's plain text.

    MUST be one title per request: the extracts API caps FULL-text extracts at
    exlimit=1. Passing 10 titles silently returns only the first, which looks
    like a tiny corpus rather than an error -- it cost us a 0.3 MB "12 MB" run.
    """
    try:
        data = json.loads(get(WIKI_API, {
            "action": "query", "prop": "extracts", "explaintext": "1",
            "exlimit": "1", "format": "json", "formatversion": "2",
            "redirects": "1", "titles": title,
        }, retries=1, pause=1.0))   # fail fast: the circuit breaker handles the rest
        pages = data.get("query", {}).get("pages", [])
        if pages and pages[0].get("extract"):
            return pages[0]["title"], pages[0]["extract"]
    except Exception:
        pass
    return title, ""


def wiki_extract(titles: Iterable[str], workers: int = 1) -> Dict[str, str]:
    """Plain-text extracts, fetched concurrently.

    8 workers keeps us well inside Wikipedia's rate policy while turning a
    ~500-request serial crawl from ~6 minutes into well under one.
    """
    from concurrent.futures import ThreadPoolExecutor

    titles = list(dict.fromkeys(titles))
    out: Dict[str, str] = {}
    done = 0
    # Circuit breaker. When a host is rate-limiting, every further request costs
    # a full backoff and returns nothing. Failing fast on a run of consecutive
    # empty results turns an hours-long stall into seconds, and the caller keeps
    # whatever the other sources already produced.
    consecutive_failures = 0
    FAILURE_LIMIT = 5
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for title, text in pool.map(_one_extract, titles):
            done += 1
            if len(text) > 800:
                out[title] = text
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= FAILURE_LIMIT:
                    print(f"  circuit breaker: {FAILURE_LIMIT} consecutive failures, "
                          f"abandoning Wikipedia pass with {len(out)} articles")
                    break
            if done % 25 == 0:
                size = sum(len(t) for t in out.values())
                print(f"  {done}/{len(titles)} requested, {len(out)} kept, "
                      f"{size/1e6:.1f} MB", flush=True)#

    print(f"  {done}/{len(titles)} requested, {len(out)} kept" + " " * 20)
    return out


def wiki_linked_titles(seeds: Iterable[str], per_seed: int = 40) -> List[str]:
    """Expand the seed list via outgoing article links, to reach corpus volume
    without hand-listing hundreds of titles."""
    found: List[str] = []
    for seed in seeds:
        try:
            data = json.loads(get(WIKI_API, {
                "action": "query", "prop": "links", "pllimit": str(per_seed),
                "plnamespace": "0", "format": "json", "formatversion": "2",
                "redirects": "1", "titles": seed,
            }))
        except Exception:
            continue
        for page in data.get("query", {}).get("pages", []):
            for link in page.get("links", []):
                found.append(link["title"])
    seen, uniq = set(), []
    for t in found:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def build_corpus(target_mb: float) -> int:
    """Real text corpus: domain articles + public-domain books for volume.

    Two sources on purpose. Wikipedia supplies the DOMAIN vocabulary the
    tokenizer needs ("chief executive", "filing", "fiscal year"); Gutenberg
    supplies the VOLUME needed for the merges to be statistically meaningful.
    Facts do not have to come from here at all -- in VERITAS the LM is never
    the source of truth, retrieval is -- so a general-English corpus is the
    honest choice for language modelling.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "corpus.txt"
    target = int(target_mb * 1_000_000)

    print(f"[corpus] target {target_mb} MB of real text")
    # Gutenberg first: one request per whole book, no throttling, guaranteed
    # volume. Wikipedia second for domain vocabulary, at 1 req/s so we never
    # trip the rate limiter again.
    print("[corpus] public-domain books (bulk volume) ...")
    articles = build_gutenberg(int(target * 0.8))
    size = sum(len(t) for t in articles.values())
    print(f"[corpus] Gutenberg: {len(articles)} books, {size/1e6:.1f} MB")

    # Wikipedia is a BONUS pass for domain vocabulary. If it is throttled or
    # down, the corpus must still be written -- losing 10 MB of already-fetched
    # books because an optional second source failed would be absurd.
    titles = list(dict.fromkeys(ENTITIES + TOPICS))
    print(f"[corpus] {len(titles)} domain articles from Wikipedia at 1 req/s ...")
    try:
        articles.update(wiki_extract(titles))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"[corpus] Wikipedia pass failed ({str(exc)[:80]}); keeping what we have")
    size = sum(len(t) for t in articles.values())
    print(f"[corpus] total: {len(articles)} documents, {size/1e6:.1f} MB")

    with open(out, "w", encoding="utf-8") as f:
        for title, text in articles.items():
            f.write(title + chr(10)*2 + text + chr(10)*3)
    print(f"[corpus] wrote {out} ({out.stat().st_size/1e6:.1f} MB, {len(articles)} documents)")
    return out.stat().st_size


# ---------------------------------------------------------------- revisions
_CEO_PAT = re.compile(
    r"\|\s*(?:ceo|key_people|leader_name|chief_executive)\s*=\s*([^\n|]{3,160})", re.I
)
_REV_PAT = re.compile(r"\|\s*(?:revenue|net_income|num_employees)\s*=\s*([^\n|]{1,120})", re.I)


def clean_wikitext(value: str) -> str:
    """Strip wiki markup from an infobox value."""
    v = re.sub(r"<ref[^>]*>.*?</ref>", "", value, flags=re.S)
    v = re.sub(r"<ref[^>]*/>", "", v)
    v = re.sub(r"\{\{[^{}]*\}\}", " ", v)
    v = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", v)
    v = re.sub(r"''+", "", v)
    v = re.sub(r"<[^>]+>", " ", v)
    v = re.sub(r"\s+", " ", v)
    return v.strip(" .,;|")


def fetch_revisions(title: str, n: int = 24, years_back: int = 8) -> List[dict]:
    """Sample revisions across time, oldest first.

    Sampling (rather than taking the newest N) is what makes the data temporal:
    consecutive edits differ by minutes, so the newest 24 revisions all describe
    the same state. Spacing them across years is what surfaces real transitions.
    """
    try:
        data = json.loads(get(WIKI_API, {
            "action": "query", "prop": "revisions", "rvprop": "timestamp|ids|comment",
            "rvlimit": "500", "rvdir": "older", "format": "json",
            "formatversion": "2", "redirects": "1", "titles": title,
        }))
    except Exception as exc:
        print(f"  ! {title}: {exc}", file=sys.stderr)
        return []
    pages = data.get("query", {}).get("pages", [])
    if not pages or "revisions" not in pages[0]:
        return []
    revs = pages[0]["revisions"]
    cutoff = datetime.now(timezone.utc).year - years_back
    revs = [r for r in revs if int(r["timestamp"][:4]) >= cutoff]
    if not revs:
        return []
    step = max(1, len(revs) // n)
    sampled = list(reversed(revs[::step][:n]))  # oldest first

    out = []
    for r in sampled:
        try:
            content = json.loads(get(WIKI_API, {
                "action": "query", "prop": "revisions", "rvprop": "content|timestamp",
                "rvslots": "main", "revids": str(r["revid"]), "format": "json",
                "formatversion": "2",
            }))
            page = content["query"]["pages"][0]
            wikitext = page["revisions"][0]["slots"]["main"]["content"]
        except Exception:
            continue

        facts = {}
        m = _CEO_PAT.search(wikitext)
        if m:
            val = clean_wikitext(m.group(1))
            # key_people lists several names+roles; keep the first person only
            val = re.split(r"\(|,|;", val)[0].strip()
            if 3 < len(val) < 60 and not val.lower().startswith(("list", "see")):
                facts["ceo"] = val
        m = _REV_PAT.search(wikitext)
        if m:
            val = clean_wikitext(m.group(1))
            if val:
                facts["revenue"] = val[:60]

        if facts:
            out.append({
                "entity": title, "revid": r["revid"], "timestamp": r["timestamp"],
                "comment": (r.get("comment") or "")[:200], "facts": facts,
                "url": f"https://en.wikipedia.org/w/index.php?oldid={r['revid']}",
                "source": "en.wikipedia.org",
            })
    return out


def build_revisions(entities: List[str], per_entity: int = 20) -> int:
    REAL.mkdir(parents=True, exist_ok=True)
    out = REAL / "wiki_revisions.jsonl"
    total = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, ent in enumerate(entities, 1):
            revs = fetch_revisions(ent, per_entity)
            # Only the revisions where a tracked fact CHANGED are worth storing:
            # that is the state-transition signal, and it is what the temporal
            # store is built to hold.
            kept, last = [], {}
            for r in revs:
                changed = {k: v for k, v in r["facts"].items() if last.get(k) != v}
                if changed:
                    r["changed"] = changed
                    kept.append(r)
                    last.update(r["facts"])
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += len(kept)
            print(f"[revisions] {i}/{len(entities)} {ent:38s} {len(kept):3d} transitions "
                  f"(total {total})")
    print(f"[revisions] wrote {out} ({total} real dated state transitions)")
    return total


# -------------------------------------------------------------------- feeds
def build_feeds() -> int:
    REAL.mkdir(parents=True, exist_ok=True)
    out = REAL / "feeds.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for name, url in FEEDS:
            try:
                xml = get(url).decode("utf-8", errors="replace")
            except Exception as exc:
                print(f"[feeds] {name}: FAILED ({exc})")
                continue
            items = re.findall(r"<(?:item|entry)\b.*?</(?:item|entry)>", xml, re.S)
            for it in items[:40]:
                def tag(t):
                    m = re.search(rf"<{t}[^>]*>(.*?)</{t}>", it, re.S)
                    if not m:
                        return ""
                    v = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", m.group(1), flags=re.S)
                    return re.sub(r"<[^>]+>", " ", v).strip()
                link = tag("link") or (re.search(r'<link[^>]*href="([^"]+)"', it) or [None, ""])[1]
                rec = {"source": name, "title": tag("title"),
                       "summary": tag("description") or tag("summary"),
                       "published": tag("pubDate") or tag("updated") or tag("published"),
                       "url": link}
                if rec["title"]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
            print(f"[feeds] {name}: {len(items)} items")
    print(f"[feeds] wrote {out} ({n} live items)")
    return n




# ------------------------------------------------------------- SEC XBRL
#: SEC EDGAR's XBRL "company concept" API is the best real bitemporal source
#: available for this domain, and it is free and unthrottled for polite use.
#: Every fact carries BOTH axes explicitly:
#:
#:     start..end  -> VALID TIME        (the fiscal period the fact describes)
#:     filed       -> TRANSACTION TIME  (the day the filing told us)
#:
#: The two genuinely diverge: Apple's FY2016 revenue appears in a filing dated
#: 2018-11-05, two years later. And when a company RESTATES a period, the same
#: valid interval gets a second, later-filed value -- a real CORRECTED event,
#: which is the case a single-timestamp store cannot represent at all.
SEC_CONCEPTS = [
    ("us-gaap", "Revenues", "revenue"),
    ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "revenue"),
    ("us-gaap", "NetIncomeLoss", "net_income"),
    ("us-gaap", "Assets", "assets"),
    ("us-gaap", "StockholdersEquity", "equity"),
]

SEC_COMPANIES = [
    ("Apple Inc.", "0000320193"), ("Microsoft Corporation", "0000789019"),
    ("NVIDIA Corporation", "0001045810"), ("Amazon.com Inc.", "0001018724"),
    ("Alphabet Inc.", "0001652044"), ("Meta Platforms Inc.", "0001326801"),
    ("Tesla Inc.", "0001318605"), ("Intel Corporation", "0000050863"),
    ("International Business Machines", "0000051143"), ("Oracle Corporation", "0001341439"),
    ("The Boeing Company", "0000012927"), ("Walmart Inc.", "0000104169"),
    ("Netflix Inc.", "0001065280"), ("Starbucks Corporation", "0000829224"),
    ("Pfizer Inc.", "0000078003"), ("Ford Motor Company", "0000037996"),
    ("General Motors Company", "0001467858"), ("Uber Technologies Inc.", "0001543151"),
    ("Advanced Micro Devices", "0000002488"), ("Qualcomm Incorporated", "0000804328"),
]


def fetch_sec_concept(cik: str, taxonomy: str, tag: str) -> List[dict]:
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json"
    try:
        return json.loads(get(url, pause=0.15))
    except Exception:
        return {}


def build_sec_facts(companies=None, verbose: bool = True) -> int:
    """Write real SEC XBRL facts as dated, bitemporal evidence records."""
    REAL.mkdir(parents=True, exist_ok=True)
    out = REAL / "sec_facts.jsonl"
    companies = companies or SEC_COMPANIES
    total, restatements = 0, 0

    with open(out, "w", encoding="utf-8") as f:
        for name, cik in companies:
            per_company = 0
            for taxonomy, tag, attribute in SEC_CONCEPTS:
                data = fetch_sec_concept(cik, taxonomy, tag)
                if not data:
                    continue
                for unit, facts in data.get("units", {}).items():
                    # Annual figures only: quarterly rows would flood the
                    # timeline with overlapping intervals for the same year.
                    annual = [x for x in facts
                              if x.get("form", "").startswith("10-K")
                              and x.get("start") and x.get("end")
                              and (datetime.fromisoformat(x["end"])
                                   - datetime.fromisoformat(x["start"])).days > 300]
                    seen_periods = {}
                    for x in sorted(annual, key=lambda r: r.get("filed", "")):
                        key = (x["start"], x["end"])
                        is_restatement = key in seen_periods and seen_periods[key] != x["val"]
                        seen_periods[key] = x["val"]
                        rec = {
                            "entity": name, "cik": cik, "attribute": attribute,
                            "value": x["val"], "unit": unit,
                            "valid_from": x["start"], "valid_to": x["end"],
                            "filed": x.get("filed"), "form": x.get("form"),
                            "fy": x.get("fy"), "fp": x.get("fp"),
                            "accn": x.get("accn"), "tag": tag,
                            "restatement": is_restatement,
                            "source": "sec.gov",
                            "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K",
                        }
                        f.write(json.dumps(rec) + chr(10))
                        total += 1
                        per_company += 1
                        restatements += is_restatement
            if verbose:
                print(f"[sec] {name:36s} {per_company:4d} annual facts")
    if verbose:
        print(f"[sec] wrote {out}: {total} real bitemporal facts, "
              f"{restatements} restatements (real CORRECTED events)")
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-mb", type=float, default=12.0)
    ap.add_argument("--entities", type=int, default=24)
    ap.add_argument("--revisions-per-entity", type=int, default=20)
    ap.add_argument("--corpus-only", action="store_true")
    ap.add_argument("--revisions-only", action="store_true")
    ap.add_argument("--feeds-only", action="store_true")
    ap.add_argument("--sec-only", action="store_true")
    args = ap.parse_args()

    do_all = not (args.corpus_only or args.revisions_only or args.feeds_only or args.sec_only)
    if do_all or args.corpus_only:
        build_corpus(args.corpus_mb)
    if do_all or args.revisions_only:
        build_revisions(ENTITIES[: args.entities], args.revisions_per_entity)
    if do_all or args.sec_only:
        build_sec_facts()
    if do_all or args.feeds_only:
        build_feeds()
    print("\nDone. Real data in data/raw/corpus.txt and data/real/*.jsonl")


if __name__ == "__main__":
    main()
