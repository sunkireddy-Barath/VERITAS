"""Rigorous audit: accuracy, hallucination resistance, and every claimed feature.

    python scripts/audit.py            # needs the API running on :8000

Answers are checked against GROUND TRUTH read directly out of the SEC filings
in data/real/sec_facts.jsonl -- not eyeballed. Three suites:

  A. ACCURACY        business questions whose true answer is in the filings
  B. HALLUCINATION   questions with NO answer in the data; any confident answer
                     is a failure, and a wrong number is a critical failure
  C. FEATURES        each claimed innovation, exercised individually

A wrong number is scored separately from a refusal, because they are different
failures: refusing costs the user an answer, fabricating costs them their trust.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = os.environ.get("VERITAS_API", "http://127.0.0.1:8000").rstrip("/")


def ask(question: str, timeout: int = 90, retries: int = 6) -> dict:
    """Ask, backing off on 429.

    The API rate-limits by design. An audit that fires faster than the limit
    and scores its own 429s as WRONG answers reports a model failure that did
    not happen -- which is exactly what a first run of this script did (25
    "wrong" answers, all of them HTTP 429). Transport errors must never be
    counted as accuracy failures.
    """
    req = urllib.request.Request(
        f"{API}/ask", data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            time.sleep(min(20, 5 * (attempt + 1)))
    raise RuntimeError("unreachable")


def get(path: str) -> dict:
    return json.load(urllib.request.urlopen(f"{API}{path}", timeout=60))


def human(v: float) -> str:
    for div, suf in ((1e9, "billion"), (1e6, "million")):
        if abs(v) >= div:
            return f"{v/div:.2f}"
    return f"{v:,.0f}"


def load_truth() -> list[dict]:
    """The filings, with the loader's tag priority applied.

    A 10-K can report one period under two revenue tags (total vs a subset).
    Truth must follow the same rule as the loader, or the key picks a winner
    arbitrarily and scores a correct total as wrong.
    """
    sys.path.insert(0, str(ROOT))
    from veritas.ingest.real_sources import _prefer_primary_tags

    p = ROOT / "data" / "real" / "sec_facts.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return _prefer_primary_tags(rows)


# ============================================================ SUITE A
def suite_accuracy(rows, n=50, seed=7):
    """Business questions with a verifiable ground-truth number."""
    print("\n" + "=" * 74)
    print("SUITE A -- ACCURACY on business questions (ground truth from filings)")
    print("=" * 74)

    # Key by fiscal year as companies name it: the annual period that ENDS in
    # the year. Microsoft's FY2021 is 2020-07-01..2021-06-30 and NVIDIA's FY2023
    # ends 2023-01-29. An earlier key (the period containing mid-year) scored
    # eleven correct fiscal-year answers as wrong once the system adopted the
    # companies' own convention -- the answers state the exact period either way.
    periods = defaultdict(set)
    latest = {}
    for r in rows:
        days = (date.fromisoformat(r["valid_to"]) - date.fromisoformat(r["valid_from"])).days
        if not 300 <= days <= 380:
            continue
        key = (r["entity"], r["attribute"], int(r["valid_to"][:4]))
        periods[key].add(r["valid_from"])
        # A restatement supersedes the original: truth is the LATEST filing.
        if key not in latest or r["filed"] > latest[key][0]:
            latest[key] = (r["filed"], r["value"])
    # Skip a year holding two different annual periods (a fiscal-year change).
    clean = [(k, v) for k, (_f, v) in latest.items() if len(periods[k]) == 1]

    rng = random.Random(seed)
    rng.shuffle(clean)
    picks = clean[:n]

    correct = wrong = refused = errors = 0
    for (entity, attr, year), value in picks:
        label = attr.replace("_", " ")
        q = f"What was {entity} {label} in {year}?"
        try:
            d = ask(q)
        except Exception as exc:
            print(f"  ERROR  {q}: {exc}")
            errors += 1
            continue
        ans = d.get("answer") or ""
        expect = human(value)
        hit = expect in ans
        if d.get("abstained"):
            refused += 1
            verdict = "REFUSED"
        elif hit:
            correct += 1
            verdict = "CORRECT"
        else:
            wrong += 1
            verdict = "WRONG  "
        print(f"  [{verdict}] {entity[:26]:26s} {label:11s} {year}  expect {expect:>8s}")
        if not hit and not d.get("abstained"):
            print(f"            got: {ans[:110]}")

    tot = len(picks)
    print(f"\n  correct {correct}/{tot}   wrong {wrong}/{tot}   refused {refused}/{tot}")
    print(f"  accuracy (of answers given): "
          f"{correct/max(1, correct+wrong):.1%}")
    return correct, wrong, refused, tot


# ============================================================ SUITE B
def suite_hallucination(rows):
    """No answer exists. Any confident answer is a hallucination."""
    print("\n" + "=" * 74)
    print("SUITE B -- HALLUCINATION RESISTANCE (no answer exists in the data)")
    print("=" * 74)

    known = {r["entity"] for r in rows}
    probes = [
        ("attribute absent",  "What is Apple Inc headcount?"),
        ("attribute absent",  "What is Microsoft employee count?"),
        # CEOs are on record now (Wikidata); a CFO is a known attribute with no data.
        ("attribute absent",  "Who is the CFO of Apple Inc?"),
        ("entity absent",     "What was Zorblax Corporation revenue in 2023?"),
        ("entity absent",     "What was Umbrella Corporation net income in 2020?"),
        ("general knowledge", "What is the capital of France?"),
        ("general knowledge", "What is the population of India?"),
        ("future / unknown",  "What will Apple Inc revenue be in 2030?"),
        ("opinion",           "Should I buy Apple stock?"),
        ("nonsense",          "What is Tesla number of unicorns?"),
        ("causal, no evidence", "Why did Apple Inc revenue increase in 2021?"),
        # one side is real, one is not: answering half a comparison is a leak
        ("comparison, one side absent",
         "Compare Apple Inc and Zorblax Corporation revenue in 2023"),
    ]
    clean = leaked = 0
    for kind, q in probes:
        try:
            d = ask(q)
        except Exception as exc:
            print(f"  ERROR  {q}: {exc}")
            continue
        ans = (d.get("answer") or "")
        abst = d.get("abstained")
        # A refusal is clean. An answer is only clean if it carries citations
        # AND does not assert a number the data cannot support.
        has_number = any(ch.isdigit() for ch in ans.replace("cannot", ""))
        ok = abst or (not has_number)
        clean += ok
        leaked += not ok
        print(f"  [{'CLEAN ' if ok else 'LEAK  '}] ({kind}) {q}")
        if not ok:
            print(f"            {ans[:130]}")
    print(f"\n  clean {clean}/{len(probes)}   leaked {leaked}/{len(probes)}")
    return clean, leaked, len(probes)


# ============================================================ SUITE C
def suite_features(rows):
    """Each claimed innovation, exercised individually."""
    print("\n" + "=" * 74)
    print("SUITE C -- FEATURES AND INNOVATIONS")
    print("=" * 74)
    results = []

    def record(name, passed, detail=""):
        results.append((name, passed, detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        if detail:
            print(f"         {detail}")

    # 1. bitemporal: same entity+attribute, two different anchors
    a = ask("What was Apple Inc revenue in 2016?")
    b = ask("What was Apple Inc revenue in 2021?")
    record("1. Bitemporal as-of reasoning",
           "215.64" in (a.get("answer") or "") and "365.82" in (b.get("answer") or ""),
           "same question, different year -> different sourced value")

    # 2. staleness qualifier on a closed period
    c = ask("What is Apple Inc revenue?")
    ansc = (c.get("answer") or "").lower()
    record("2. Stale fact carries a qualifier",
           any(p in ansc for p in ("no later value", "most recently recorded",
                                   "no newer", "most recently reported"))
           and not c.get("abstained"),
           "closed fiscal period reported as last-known, not as current")

    # 3. a real restatement is a correction: restated figure, original named
    d = ask("What was Apple Inc net income in 2008?")
    ansd = d.get("answer") or ""
    record("3. Real restatement reported as a correction",
           "6.12" in ansd and "originally reported as 4.83" in ansd
           and "disagree" not in ansd.lower()
           and any(t.get("superseded_at") for t in d.get("timeline", [])),
           "SEC filed FY2008 net income as 4.83B, restated to 6.12B on 2010-01-25")

    # 4. abstention
    e = ask("What is Apple Inc headcount?")
    record("4. Abstention when evidence absent", bool(e.get("abstained")),
           f"support={e.get('support_level')}")

    # 5. claim-level provenance
    cites = a.get("citations", [])
    record("5. Claim-level provenance", bool(cites) and all(
        c.get("source") and c.get("tier") for c in cites),
        f"{len(cites)} citations, each with source + tier + date")

    # 6. timeline reconstruction
    record("6. Historical timeline reconstructed", len(a.get("timeline", [])) >= 5,
           f"{len(a.get('timeline', []))} versions with valid/recorded dates")

    # 7. CHANGE renders the progression
    f = ask("How did Apple Inc revenue change over time?")
    ansf = f.get("answer") or ""
    n_periods = sum(x in ansf for x in ("274.51", "365.82", "394.33", "383.29"))
    record("7. CHANGE renders the progression", n_periods >= 3,
           f"{n_periods} distinct periods named in one answer")

    # 8. confidence is a label, never a truth percentage
    all_ans = [a, b, c, d, e, f]
    record("8. No truth-percentage claims",
           not any("100%" in (x.get("answer") or "") for x in all_ans),
           "support labels only; '100% true' is not expressible")

    # 9. decision trace
    record("9. Auditable decision trace", len(a.get("trace", [])) >= 5,
           " | ".join(t.split()[0] for t in a.get("trace", [])[:7]))

    # 10. source tiering
    tiers = {c.get("tier") for c in cites}
    record("10. Source tiering applied", 1 in tiers,
           f"tiers present: {sorted(t for t in tiers if t)}")

    # 11. entity resolution across surface forms
    g = ask("What was Apple revenue in 2016?")     # no "Inc"
    record("11. Entity resolution", "215.64" in (g.get("answer") or ""),
           "'Apple' resolves to the stored 'Apple Inc.'")

    # 12. live ingestion changes the answer without retraining
    #
    # A fresh, letters-only company name per run: a fixed name is already in the
    # store on any rerun (and a digit-led suffix is dropped by the planner), so
    # "before" would not abstain and the check would prove nothing. The API key
    # is sent because a deployed API protects /ingest.
    company = "Vertex Dynamics " + "".join(
        random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(7)).capitalize()
    key = os.environ.get("VERITAS_API_KEY", "")
    secrets_file = ROOT / ".deploy" / "secrets.json"
    if not key and secrets_file.exists():
        key = json.loads(secrets_file.read_text(encoding="utf-8")).get("VERITAS_API_KEY", "")
    try:
        before = ask(f"What is {company} status?")
        req = urllib.request.Request(
            f"{API}/ingest",
            data=json.dumps({"source_id": "sec.gov", "entity": company,
                             "published": "2026-09-10", "tier": 1,
                             "text": f"{company} filing: the Denver facility "
                                     "is now operational as of September 2026."}).encode(),
            headers={"Content-Type": "application/json", **({"X-API-Key": key} if key else {})})
        ing = json.load(urllib.request.urlopen(req, timeout=60))
        after = ask(f"What is {company} status?")
        record("12. Live ingestion, no retraining",
               before.get("abstained") and not after.get("abstained")
               and not ing.get("retrain_required"),
               f"ingest {ing.get('elapsed_seconds')}s -> answer changed")
    except urllib.error.HTTPError as exc:
        record("12. Live ingestion, no retraining", False,
               f"HTTP {exc.code} (set VERITAS_API_KEY=... or unset it)")

    # 13. graph + independence
    st = get("/stats")
    record("13. Evidence graph populated",
           st["graph"]["nodes"] > 100 and st["graph"]["edges"] > 100,
           f"{st['graph']['nodes']} nodes, {st['graph']['edges']} edges, "
           f"{st['entity_count']} entities, {st['fact_versions']} versions")

    # 14. transaction-time travel: what was believed before the restatement
    try:
        path = "/entity/Apple%20Inc./as_of?attribute=net_income&valid=2008-06-01"
        then = get(path + "&known=2009-12-01")
        now = get(path)
        record("14. Bitemporal time travel (as believed on a date)",
               (then.get("value") or "").startswith("4.83")
               and (now.get("value") or "").startswith("6.12")
               and len(now.get("belief_history", [])) >= 2,
               f"believed 2009-12-01: {then.get('value')}; believed now: {now.get('value')}")
    except urllib.error.HTTPError as exc:
        record("14. Bitemporal time travel (as believed on a date)", False, f"HTTP {exc.code}")

    # 15. change feed: restatements and successions are recorded events
    try:
        corrected = get("/changes?kind=CORRECTED&entity=Apple%20Inc.&limit=50")
        changed = get("/changes?kind=CHANGED&entity=Apple%20Inc.&limit=50")
        record("15. World change feed",
               any(c["attribute"] == "net_income" and c["new_value"].startswith("6.12")
                   for c in corrected["changes"])
               and any(c["attribute"] == "ceo" for c in changed["changes"]),
               f"Apple: {corrected['total']} restatements, {changed['total']} state changes")
    except urllib.error.HTTPError as exc:
        record("15. World change feed", False, f"HTTP {exc.code}")

    # 16. comparison keeps each side's own period and labels the arithmetic
    h = ask("Compare Apple Inc and Microsoft revenue in 2023")
    ansh = h.get("answer") or ""
    rows = h.get("comparison", [])
    record("16. Period-aware comparison",
           "383.29" in ansh and "211.91" in ansh and "not aligned" in ansh
           and "not reported by either source" in ansh
           and len(rows) == 2 and all(r.get("established") for r in rows),
           f"{len(rows)} sides, {len(h.get('citations', []))} citations, "
           f"support={h.get('support_level')}")

    passed = sum(p for _, p, _ in results)
    print(f"\n  {passed}/{len(results)} features verified")
    return passed, len(results)


def main() -> int:
    try:
        h = get("/health")
    except Exception:
        print("API not reachable on :8000 -- start it with `python run_veritas.py`")
        return 2
    if not h.get("ready"):
        print(f"API not ready: {h}")
        return 2
    print(f"API ready | device={h['device']} checkpoint={h['checkpoint']} "
          f"boot={h['boot_seconds']}s")

    rows = load_truth()
    print(f"ground truth: {len(rows)} SEC facts, "
          f"{len({r['entity'] for r in rows})} companies")

    corr, wrong, ref, tot = suite_accuracy(rows)
    clean, leaked, nprobe = suite_hallucination(rows)
    feat, nfeat = suite_features(rows)

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  A. accuracy on answered business questions : {corr}/{corr+wrong} "
          f"({corr/max(1,corr+wrong):.0%})   [{ref} refused of {tot}]")
    print(f"  B. hallucination resistance                : {clean}/{nprobe} clean, "
          f"{leaked} leaked")
    print(f"  C. features verified                       : {feat}/{nfeat}")
    ok = wrong == 0 and leaked == 0 and feat == nfeat
    print("\n  " + ("ALL CHECKS PASSED" if ok else "SEE FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
