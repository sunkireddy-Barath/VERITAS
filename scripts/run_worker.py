"""Ingestion worker entrypoint.

    python -m scripts.run_worker --poll --interval 900     # the source poller
    python -m scripts.run_worker --poll --once             # one polling pass
    python -m scripts.run_worker --topics veritas.raw      # standalone consumer

Where the knowledge base lives decides the topology. The store, indices and
evidence graph are held in memory by the process that answers questions, so
**the API is the consumer**: with `VERITAS_KAFKA_BOOTSTRAP` set it drains
`veritas.raw` in a background thread (see `api/main.py`). A separate consumer
container would build a second, private knowledge base that no question ever
reaches -- which is what the first version of this stack did.

This script therefore has two roles:

* **poller** (`--poll`) fetches the monitored feeds and publishes each item to
  `veritas.raw`. It loads no model. Run exactly one: duplicate pollers double
  every fetch, and politeness to free public APIs is a hard requirement.
* **standalone consumer** (`--topics`) runs the pipeline in its own process,
  for a single-machine setup or for replaying the log into a fresh store. Its
  state is local to it; it does not serve answers.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veritas.ingest.streaming import ALL_TOPICS, StreamingIngest, make_bus


def build_pipeline(device: str):
    from veritas.ingest.real_sources import build_real_system
    from veritas.model.transformer import ModelConfig, VeritasLM
    from veritas.pipeline import VeritasSystemBuilder
    from veritas.tokenizer.bpe import BPETokenizer

    root = Path(__file__).resolve().parents[1]
    tok = BPETokenizer.load(root / "checkpoints" / "tokenizer.json")
    ck = root / "checkpoints" / "sft.pt"
    if not ck.exists():
        ck = root / "checkpoints" / "best.pt"
    model = (VeritasLM.load(str(ck), device) if ck.exists()
             else VeritasLM(ModelConfig(vocab_size=tok.vocab_size, d_model=256,
                                        n_layers=4, n_heads=4, n_kv_heads=2,
                                        max_seq_len=256)).to(device))
    model.eval()
    builder = VeritasSystemBuilder(model, tok, device=device, domain="corporate")
    if (root / "data" / "real").exists():
        build_real_system(builder, root / "data" / "real", verbose=True)
    return builder


def feed_items(xml: str, limit: int = 25):
    """(title, body, published) for each RSS <item> / Atom <entry>."""
    from veritas.ingest.real_sources import _iso

    for item in re.findall(r"<(?:item|entry)\b.*?</(?:item|entry)>", xml, re.S)[:limit]:
        def tag(t: str) -> str:
            m = re.search(rf"<{t}[^>]*>(.*?)</{t}>", item, re.S)
            if not m:
                return ""
            v = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", m.group(1), flags=re.S)
            return re.sub(r"<[^>]+>", " ", v).strip()

        title = tag("title")
        if title:
            yield title, tag("description") or tag("summary"), _iso(tag("pubDate") or tag("updated"))


def stable_doc_id(source: str, title: str) -> str:
    # Not hash(): Python salts it per process, so every restart gave each item
    # a new id and the log could never be deduplicated or replayed.
    return f"live:{source}:{hashlib.blake2b(title.encode('utf-8'), digest_size=8).hexdigest()}"


def poll(bus, stream: StreamingIngest, interval: int, once: bool, stop: threading.Event) -> None:
    from scripts.fetch_real_data import FEEDS
    from veritas.ingest.real_sources import _get

    print(f"[worker] poller started, interval={interval}s, feeds={len(FEEDS)}", flush=True)
    while not stop.is_set():
        for name, url in FEEDS:
            try:
                xml = _get(url).decode("utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001 - one dead feed must not stop the cycle
                print(f"[worker] {name} unreachable: {str(exc)[:70]}", flush=True)
                continue
            n = 0
            for title, body, published in feed_items(xml):
                stream.publish_raw(source_id=name, text=f"{title}. {body}",
                                   doc_id=stable_doc_id(name, title), published=published)
                n += 1
            print(f"[worker] {name}: published {n} items", flush=True)
        if hasattr(bus, "flush"):
            bus.flush()
        if once:
            break
        stop.wait(interval)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default=",".join(ALL_TOPICS))
    ap.add_argument("--poll", action="store_true", help="run as the source poller")
    ap.add_argument("--interval", type=int, default=900)
    ap.add_argument("--once", action="store_true", help="single pass, then exit")
    args = ap.parse_args()

    device = os.environ.get("VERITAS_DEVICE", "cpu")
    bootstrap = os.environ.get("VERITAS_KAFKA_BOOTSTRAP") or None
    # A poller started beside a broker that is still booting should wait for
    # it, not silently fall back to an in-process queue nobody reads.
    bus = make_bus(bootstrap, group_id="veritas-worker", attempts=24 if bootstrap else 1)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Graceful shutdown: finish the in-flight message, commit, then exit.
        signal.signal(sig, lambda *_: stop.set())

    if args.poll:
        if bootstrap and not hasattr(bus, "flush"):
            print("[worker] VERITAS_KAFKA_BOOTSTRAP is set but Kafka is unreachable; "
                  "refusing to poll into an in-process queue", flush=True)
            return 2
        # The poller only publishes, so it gets a pipeline-less stream.
        poll(bus, StreamingIngest(pipeline=None, bus=bus), args.interval, args.once, stop)
    else:
        stream = StreamingIngest(build_pipeline(device).ingest, bus)
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
        print(f"[worker] consuming {topics} (device={device})", flush=True)
        if args.once:
            print(stream.run_once())
        else:
            stream.run_worker(topics, stop)

    bus.close()
    print("[worker] stopped cleanly", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
