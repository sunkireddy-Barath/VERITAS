"""Ingestion worker entrypoint.

    python -m scripts.run_worker --topics veritas.raw,veritas.changed
    python -m scripts.run_worker --poll --interval 900

Two roles, one image:

* **consumer** (`--topics`) drains the bus and runs the pipeline stages. Scale
  the replica count of this one -- embedding is the bottleneck.
* **poller** (`--poll`) fetches monitored sources on an adaptive interval and
  publishes to `veritas.raw`. Exactly one replica: duplicate pollers would
  double every fetch, and politeness to free public APIs is a hard requirement.

With `VERITAS_KAFKA_BOOTSTRAP` unset the bus is in-process, so this script is
also the way to run a single-process pipeline locally.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default=",".join(ALL_TOPICS))
    ap.add_argument("--poll", action="store_true", help="run as the source poller")
    ap.add_argument("--interval", type=int, default=900)
    ap.add_argument("--once", action="store_true", help="single pass, then exit")
    args = ap.parse_args()

    device = os.environ.get("VERITAS_DEVICE", "cpu")
    bootstrap = os.environ.get("VERITAS_KAFKA_BOOTSTRAP") or None
    bus = make_bus(bootstrap, group_id="veritas-worker")

    builder = build_pipeline(device)
    stream = StreamingIngest(builder.ingest, bus)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Graceful shutdown: finish the in-flight message, commit, then exit.
        # Killing mid-handler with auto-commit off would reprocess it, which is
        # safe (the store is idempotent) but wastes an embedding pass.
        signal.signal(sig, lambda *_: stop.set())

    if args.poll:
        from scripts.fetch_real_data import FEEDS
        from veritas.ingest.real_sources import _get, _iso
        import re

        print(f"[worker] poller started, interval={args.interval}s, feeds={len(FEEDS)}")
        while not stop.is_set():
            for name, url in FEEDS:
                try:
                    xml = _get(url).decode("utf-8", errors="replace")
                except Exception as exc:
                    print(f"[worker] {name} unreachable: {str(exc)[:70]}")
                    continue
                items = re.findall(r"<(?:item|entry)\b.*?</(?:item|entry)>", xml, re.S)
                for item in items[:25]:
                    def tag(t: str) -> str:
                        m = re.search(rf"<{t}[^>]*>(.*?)</{t}>", item, re.S)
                        if not m:
                            return ""
                        v = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", m.group(1), flags=re.S)
                        return re.sub(r"<[^>]+>", " ", v).strip()
                    title = tag("title")
                    if not title:
                        continue
                    stream.publish_raw(
                        source_id=name,
                        text=f"{title}. {tag('description') or tag('summary')}",
                        doc_id=f"live:{name}:{abs(hash(title)) % 10**9}",
                        published=_iso(tag("pubDate") or tag("updated")),
                    )
                print(f"[worker] {name}: published {len(items[:25])} items")
            if hasattr(bus, "flush"):
                bus.flush()
            if args.once:
                break
            stop.wait(args.interval)
    else:
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
        print(f"[worker] consuming {topics} (device={device})")
        if args.once:
            print(stream.run_once())
        else:
            stream.run_worker(topics, stop)

    bus.close()
    print("[worker] stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
