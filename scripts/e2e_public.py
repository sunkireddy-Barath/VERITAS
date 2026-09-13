"""End-to-end check of a RUNNING deployment, through its public URL.

    python scripts/e2e_public.py https://<your-deployment>
    python scripts/e2e_public.py http://127.0.0.1:8000 --rate-limit

Everything a user or client touches, over the real network path (tunnel, CDN,
proxy) rather than in-process: the UI and config.js, every read endpoint, the
streaming endpoint through the proxy, write-endpoint auth, CORS, live ingestion
changing an answer, and refusals. The API key for /ingest is read from
VERITAS_API_KEY or .deploy/secrets.json.

`--rate-limit` also floods /ask until it returns 429. It locks the client out
for a minute, so it runs last and only when asked.
"""
from __future__ import annotations

import json
import os
import random
import string
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "veritas-e2e/1.0"}
results: list[tuple[str, bool, str]] = []


def call(base, path, method="GET", body=None, headers=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    h = dict(UA, **(headers or {}))
    if data is not None:
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(base + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def ask(base, question):
    status, _h, raw = call(base, "/ask", "POST", {"question": question})
    if status != 200:
        raise RuntimeError(f"/ask HTTP {status}: {raw[:120]!r}")
    return json.loads(raw)


def check(name, fn):
    t0 = time.time()
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001 - a crash is a failed check, not a crashed run
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  ({time.time() - t0:.1f}s)")
    if detail:
        print(f"       {str(detail)[:160]}")


def api_key() -> str:
    if os.environ.get("VERITAS_API_KEY"):
        return os.environ["VERITAS_API_KEY"]
    f = ROOT / ".deploy" / "secrets.json"
    return json.loads(f.read_text(encoding="utf-8")).get("VERITAS_API_KEY", "") if f.exists() else ""


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    base = args[0].rstrip("/")
    print(f"target: {base}\n")

    def health():
        s, _h, raw = call(base, "/health")
        d = json.loads(raw)
        return s == 200 and d.get("ready"), f"device={d.get('device')} bus={d.get('bus')}"
    check("health: ready", health)

    def ui():
        s, h, raw = call(base, "/")
        page = raw.decode("utf-8", "replace")
        s2, h2, cfg = call(base, "/config.js")
        return (s == 200 and "VERITAS" in page and 'src="config.js"' in page
                and s2 == 200 and "VERITAS_API" in cfg.decode()
                and "javascript" in h2.get("content-type", h2.get("Content-Type", ""))), \
            f"page {len(raw)} bytes, config.js {s2}"
    check("UI page + config.js served", ui)

    def fiscal():
        d = ask(base, "What was Apple Inc revenue in 2016?")
        return "215.64" in d["answer"] and d["citations"], d["answer"][:110]
    check("historical fiscal-year answer, cited", fiscal)

    def ceo():
        d = ask(base, "Who was the CEO of Microsoft in 2020?")
        return "Satya Nadella" in d["answer"], d["answer"][:110]
    check("point-in-time CEO answer", ceo)

    def restated():
        d = ask(base, "What was Apple Inc net income in 2008?")
        a = d["answer"]
        return "6.12" in a and "originally reported as 4.83" in a and "disagree" not in a.lower(), a[:130]
    check("restatement reported as a correction", restated)

    def compare():
        d = ask(base, "Compare Apple Inc and Microsoft revenue in 2023")
        return ("383.29" in d["answer"] and "not aligned" in d["answer"]
                and len(d.get("comparison", [])) == 2), d["answer"][:130]
    check("period-aware comparison", compare)

    def refusals():
        f = ask(base, "What will Apple Inc revenue be in 2030?")
        u = ask(base, "What was Zorblax Corporation revenue in 2023?")
        return f["abstained"] and u["abstained"], "forecast and unknown entity both refused"
    check("refuses forecasts and unknown entities", refusals)

    def stream():
        s, h, raw = call(base, "/ask/stream", "POST", {"question": "Who is the current CEO of Apple Inc.?"})
        text = raw.decode("utf-8", "replace")
        events = [line[7:] for line in text.splitlines() if line.startswith("event: ")]
        answer = next((json.loads(line[6:]) for line in reversed(text.splitlines())
                       if line.startswith("data: ") and '"answer"' in line), {})
        return (s == 200 and events.count("trace") >= 3 and "answer" in events
                and "Apple" in answer.get("answer", "")), \
            f"{len(events)} SSE events ({events.count('trace')} trace) through the proxy"
    check("SSE streaming through the network path", stream)

    def time_travel():
        q = "/entity/Apple%20Inc./as_of?attribute=net_income&valid=2008-06-01"
        then = json.loads(call(base, q + "&known=2009-12-01")[2])
        now = json.loads(call(base, q)[2])
        return (str(then.get("value")).startswith("4.83") and str(now.get("value")).startswith("6.12")), \
            f"believed 2009-12-01: {then.get('value')}; now: {now.get('value')}"
    check("bitemporal time travel", time_travel)

    def feeds():
        s, _h, raw = call(base, "/changes?kind=CORRECTED&entity=Apple%20Inc.")
        t = json.loads(call(base, "/entity/Apple%20Inc./timeline?attribute=ceo")[2])
        d = json.loads(raw)
        return (s == 200 and d["total"] >= 1 and len(t["attributes"]["ceo"]["versions"]) >= 3), \
            f"{d['total']} Apple restatements; {len(t['attributes']['ceo']['versions'])} CEO versions"
    check("change feed + timeline", feeds)

    def bad_input():
        s1 = call(base, "/entity/Apple%20Inc./as_of?attribute=revenue&valid=someday")[0]
        s2 = call(base, "/ask", "POST", {"question": "x"})[0]
        return s1 == 422 and s2 == 422, f"bad date -> {s1}, too-short question -> {s2}"
    check("rejects malformed input", bad_input)

    def cors():
        _s, h, _raw = call(base, "/ask", "OPTIONS", headers={
            "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
        allowed = {k.lower(): v for k, v in h.items()}.get("access-control-allow-origin")
        return allowed not in ("*", "https://evil.example"), f"foreign origin allow-origin={allowed!r}"
    check("CORS refuses a foreign origin", cors)

    key = api_key()
    # Letters only: a digit-led suffix is dropped by the planner, and a rerun would
    # then match the entity an earlier run created.
    entity = "Norvale Aerospace " + "".join(
        random.choice(string.ascii_lowercase) for _ in range(7)).capitalize()
    doc = {"source_id": "sec.gov", "tier": 1, "entity": entity, "published": "2026-09-12",
           "text": f"{entity} filing: Mira Castellanos was appointed chief executive officer "
                   f"of {entity}, effective September 2026."}

    def auth():
        s1 = call(base, "/ingest", "POST", doc)[0]
        s2 = call(base, "/ingest", "POST", doc, headers={"X-API-Key": "wrong"})[0]
        return s1 == 401 and s2 == 401, f"no key -> {s1}, wrong key -> {s2}"
    check("write endpoint requires the API key", auth if key else (lambda: (False, "no API key configured")))

    def live_ingest():
        q = f"Who is the current CEO of {entity}?"
        before = ask(base, q)
        s, _h, raw = call(base, "/ingest", "POST", doc, headers={"X-API-Key": key})
        after = ask(base, q)
        res = json.loads(raw) if s == 200 else {}
        return (before["abstained"] and s == 200 and "Mira Castellanos" in after["answer"]
                and res.get("retrain_required") is False), \
            f"ingest {s} in {res.get('elapsed_seconds')}s, logged_to_kafka={res.get('logged_to_kafka')}"
    check("live ingestion changes the answer, no retraining", live_ingest if key else
          (lambda: (False, "no API key configured")))

    if "--rate-limit" in sys.argv:
        def limit():
            codes = [call(base, "/ask", "POST", {"question": "What is Apple Inc revenue?"})[0]
                     for _ in range(70)]
            return 429 in codes, f"first 429 after {codes.index(429) if 429 in codes else '-'} requests"
        check("rate limiter returns 429", limit)

    passed = sum(ok for _n, ok, _d in results)
    print(f"\n{passed}/{len(results)} end-to-end checks passed against {base}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
