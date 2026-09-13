"""End-to-end Kafka check: a document published to the log reaches the answers.

    VERITAS_API=http://127.0.0.1:8000 KAFKA=localhost:29092 python scripts/kafka_smoke.py

Needs a broker and an API started with VERITAS_KAFKA_BOOTSTRAP pointing at the
same broker (`make kafka` does both). The script checks the whole path the
stack claims -- produce to veritas.raw -> API consumer thread -> change
detection -> chunk/embed/index -> bitemporal assert -> /ask -- rather than
only that a broker accepts messages, which proves nothing about the answers.

It uses a fictional company, so the answer can only have come from the message.
"""
from __future__ import annotations

import json
import os
import random
import string
import sys
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

API = os.environ.get("VERITAS_API", "http://127.0.0.1:8000").rstrip("/")
KAFKA = os.environ.get("KAFKA", os.environ.get("VERITAS_KAFKA_BOOTSTRAP", "localhost:29092"))


def get(path: str) -> dict:
    return json.load(urllib.request.urlopen(f"{API}{path}", timeout=30))


def ask(question: str, retries: int = 8) -> dict:
    """Ask, backing off on 429. The API rate-limits per client; a limiter doing
    its job must not be reported as a Kafka failure."""
    req = urllib.request.Request(f"{API}/ask", data=json.dumps({"question": question}).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            return json.load(urllib.request.urlopen(req, timeout=90))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            time.sleep(min(20, 5 * (attempt + 1)))
    raise RuntimeError("unreachable")


def main() -> int:
    from veritas.ingest.streaming import KafkaBus, StreamingIngest, make_bus

    health = get("/health")
    print(f"API {API}: ready={health.get('ready')} bus={health.get('bus')}")
    if not health.get("ready"):
        print("FAIL: API not ready")
        return 1
    if (health.get("bus") or {}).get("kind") != "kafka":
        print("FAIL: the API is not consuming Kafka (start it with VERITAS_KAFKA_BOOTSTRAP set)")
        return 1

    bus = make_bus(KAFKA, group_id=f"veritas-smoke-{uuid.uuid4().hex[:6]}", attempts=6)
    if not isinstance(bus, KafkaBus):
        print(f"FAIL: cannot reach Kafka at {KAFKA}")
        return 1

    # A unique name per run, LETTERS ONLY. A hex suffix like "3F2A" starts with a
    # digit, the planner drops it, and the question silently matched the entity
    # a previous run had created -- so the check "passed" without this run's
    # message mattering at all.
    tag = "".join(random.choice(string.ascii_lowercase) for _ in range(7)).capitalize()
    entity = f"Quillon Robotics {tag}"
    question = f"Who is the current CEO of {entity}?"
    before = ask(question)
    print(f"before: abstained={before.get('abstained')}  {before.get('answer', '')[:90]}")
    if not before.get("abstained"):
        print("FAIL: the entity was already known before publishing; this run proves nothing")
        return 1

    StreamingIngest(None, bus).publish_raw(
        source_id="sec.gov", entity=entity, tier=1, published="2026-09-01",
        doc_id=f"smoke:{entity}",
        text=f"{entity} filing: Ada Okafor was appointed chief executive officer "
             f"of {entity}, effective September 2026.")
    bus.flush()
    print(f"published to veritas.raw at {KAFKA}; waiting for the API to consume it")

    deadline = time.time() + 90
    while time.time() < deadline:
        time.sleep(3)
        after = ask(question)
        if "Ada Okafor" in (after.get("answer") or ""):
            print(f"after:  abstained={after.get('abstained')}  {after['answer'][:110]}")
            print(f"bus stats: {get('/stats').get('bus')}")
            print("PASS: a Kafka message changed the answer, with no restart and no retraining")
            bus.close()
            return 0
    print(f"FAIL: answer did not change within 90s: {after.get('answer', '')[:120]}")
    print(f"bus stats: {get('/stats').get('bus')}")
    bus.close()
    return 1


if __name__ == "__main__":
    sys.exit(main())
