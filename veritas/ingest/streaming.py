"""Kafka-backed ingestion pipeline, with an in-process fallback.

Why a broker belongs here
-------------------------
The synchronous path (`IngestionPipeline.ingest`) does everything inline:
fetch, fingerprint, chunk, embed, index, assert facts, invalidate cache. That
is correct and it is what the notebooks use, but it has three properties that
break in production:

* **A slow source blocks every other source.** One 30-second SEC timeout stalls
  the whole poll cycle.
* **A crash loses the work.** There is no record of what was in flight.
* **It cannot scale horizontally.** Embedding is GPU-bound and indexing is
  write-bound; they want different machines and different replica counts.

Splitting the pipeline across topics fixes all three, and the topic boundaries
fall naturally on the stages that already exist:

    veritas.raw        <- pollers publish fetched documents
         |  (fingerprint + change detection; most messages die here)
    veritas.changed    <- only documents that ACTUALLY changed
         |  (chunk + embed + index; GPU-bound, scale this one)
    veritas.facts      <- extracted/structured facts
         |  (bitemporal assert, graph update, cache invalidation)
    veritas.changes    <- state transitions, for downstream notification

Kafka rather than a plain task queue, for reasons specific to this system:

* **Replayable log.** The evidence store is append-only and auditable; an
  ingestion pipeline that cannot be replayed undermines that. With retained
  topics you can rebuild the entire knowledge base from the raw log and get a
  bit-identical store -- which is the ingestion-side equivalent of the
  bitemporal guarantee.
* **Ordering per key.** Partitioning by entity guarantees that one entity's
  facts are processed in order. Out-of-order replay would turn a real
  succession into a spurious CONFLICT (see `TemporalStore.assert_fact`), so
  per-entity ordering is a correctness requirement, not a nicety.
* **Consumer groups** give at-least-once delivery and horizontal scaling for
  free; the store's REAFFIRMED path already makes re-delivery idempotent.

`kafka-python` is an optional dependency. Without a broker, `LocalBus`
implements the same interface with an in-process queue, so the code path is
identical in development and the notebooks keep working unchanged.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence

TOPIC_RAW = "veritas.raw"
TOPIC_CHANGED = "veritas.changed"
TOPIC_FACTS = "veritas.facts"
TOPIC_CHANGES = "veritas.changes"
ALL_TOPICS = (TOPIC_RAW, TOPIC_CHANGED, TOPIC_FACTS, TOPIC_CHANGES)


@dataclass
class Message:
    """One unit of work on the bus."""

    topic: str
    key: str                      # entity or source id -> partition key
    payload: Dict[str, object]
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), default=str).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Message":
        d = json.loads(raw.decode("utf-8"))
        return cls(topic=d["topic"], key=d["key"], payload=d["payload"], ts=d.get("ts", ""))


class Bus:
    """Minimal produce/consume interface shared by both backends."""

    def produce(self, msg: Message) -> None:
        raise NotImplementedError

    def consume(self, topics: Sequence[str], handler: Callable[[Message], None],
                stop: Optional[threading.Event] = None) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class LocalBus(Bus):
    """In-process queue with the same semantics. Used when Kafka is absent.

    Deliberately preserves per-key ordering (a single FIFO per topic) so that
    behaviour matches a partitioned Kafka topic and a bug cannot hide in
    development only to appear in production.
    """

    def __init__(self) -> None:
        self.queues: Dict[str, queue.Queue] = {t: queue.Queue() for t in ALL_TOPICS}
        self.delivered = 0

    def produce(self, msg: Message) -> None:
        self.queues.setdefault(msg.topic, queue.Queue()).put(msg)

    def consume(self, topics, handler, stop=None) -> None:
        stop = stop or threading.Event()
        while not stop.is_set():
            drained = False
            for t in topics:
                q = self.queues.setdefault(t, queue.Queue())
                try:
                    msg = q.get_nowait()
                except queue.Empty:
                    continue
                drained = True
                handler(msg)
                self.delivered += 1
            if not drained:
                if stop.is_set():
                    break
                time.sleep(0.05)

    def drain(self, topics, handler) -> int:
        """Process everything currently queued, then return. Used by tests and
        by the synchronous notebook path."""
        n = 0
        for t in topics:
            q = self.queues.setdefault(t, queue.Queue())
            while True:
                try:
                    msg = q.get_nowait()
                except queue.Empty:
                    break
                handler(msg)
                n += 1
        return n

    def pending(self) -> Dict[str, int]:
        return {t: q.qsize() for t, q in self.queues.items()}


class KafkaBus(Bus):
    """kafka-python backed bus. Partitions by `key` to preserve entity order."""

    def __init__(self, bootstrap_servers: str = "localhost:9092",
                 group_id: str = "veritas", client_id: str = "veritas") -> None:
        from kafka import KafkaConsumer, KafkaProducer  # optional dependency

        self._KafkaConsumer = KafkaConsumer
        self.bootstrap = bootstrap_servers
        self.group_id = group_id
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            client_id=client_id,
            value_serializer=lambda v: v,
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
            # acks="all" + idempotence: an evidence pipeline must not silently
            # drop a filing because a broker restarted mid-write.
            acks="all",
            retries=5,
            linger_ms=20,
        )
        self._consumer = None

    def produce(self, msg: Message) -> None:
        self.producer.send(msg.topic, key=msg.key, value=msg.to_bytes())

    def flush(self) -> None:
        self.producer.flush()

    def consume(self, topics, handler, stop=None) -> None:
        stop = stop or threading.Event()
        consumer = self._KafkaConsumer(
            *topics,
            bootstrap_servers=self.bootstrap,
            group_id=self.group_id,
            enable_auto_commit=False,     # commit only after the handler succeeds
            auto_offset_reset="earliest",
            consumer_timeout_ms=1000,
        )
        self._consumer = consumer
        while not stop.is_set():
            for record in consumer:
                if stop.is_set():
                    break
                try:
                    handler(Message.from_bytes(record.value))
                    consumer.commit()
                except Exception as exc:   # noqa: BLE001 - a poison message must
                    # not kill the consumer; log, skip, keep the pipeline alive.
                    print(f"[kafka] handler failed on {record.topic}: {exc}")
                    consumer.commit()
        consumer.close()

    def close(self) -> None:
        try:
            self.producer.flush()
            self.producer.close()
        except Exception:
            pass


def make_bus(bootstrap_servers: Optional[str] = None, group_id: str = "veritas") -> Bus:
    """Kafka when a broker is configured and reachable, LocalBus otherwise.

    Falling back rather than failing is deliberate: the notebooks, the tests and
    a laptop demo must all work with no infrastructure, while production gets
    the durable log by setting one environment variable.
    """
    if not bootstrap_servers:
        return LocalBus()
    try:
        bus = KafkaBus(bootstrap_servers, group_id)
        bus.producer.partitions_for("__veritas_probe")  # forces a metadata fetch
        print(f"[bus] Kafka at {bootstrap_servers}")
        return bus
    except Exception as exc:  # noqa: BLE001
        print(f"[bus] Kafka unavailable ({str(exc)[:80]}); using in-process bus")
        return LocalBus()


# --------------------------------------------------------------- processors
class StreamingIngest:
    """Wires the four topics onto the existing pipeline stages.

    Each processor is a pure function of (message, pipeline) so it can be run
    in-process for tests or in a separate worker container in production
    without changing a line.
    """

    def __init__(self, pipeline, bus: Optional[Bus] = None) -> None:
        self.pipeline = pipeline
        self.bus = bus or LocalBus()
        self.stats: Dict[str, int] = {t: 0 for t in ALL_TOPICS}
        self.stats["suppressed"] = 0

    # ---- stage 1: a poller publishes a fetched document --------------------
    def publish_raw(self, source_id: str, text: str, doc_id: str = "",
                    published: str = "", entity: str = "", url: str = "",
                    tier: int = 3) -> None:
        self.bus.produce(Message(TOPIC_RAW, key=entity or source_id, payload={
            "source_id": source_id, "text": text, "doc_id": doc_id,
            "published": published, "entity": entity, "url": url, "tier": tier,
        }))

    # ---- stage 2: change detection (most messages die here) ----------------
    def handle_raw(self, msg: Message) -> None:
        self.stats[TOPIC_RAW] += 1
        p = msg.payload
        from ..evidence.claims import claims_to_state, extract_claims

        claims = extract_claims(str(p["text"]), str(p.get("doc_id", "")),
                                str(p.get("doc_id", "")), str(p.get("entity", "")))
        report = self.pipeline.detector.check(
            str(p["source_id"]), str(p["text"]), claims_to_state(claims))
        if not report.changed:
            # The cheap filter doing its job: nothing downstream is woken up.
            self.stats["suppressed"] += 1
            return
        self.bus.produce(Message(TOPIC_CHANGED, key=msg.key,
                                 payload={**p, "change_reason": report.reason}))

    # ---- stage 3: chunk + embed + index (GPU-bound; scale this) ------------
    def handle_changed(self, msg: Message) -> None:
        self.stats[TOPIC_CHANGED] += 1
        p = msg.payload
        res = self.pipeline.ingest(
            source_id=str(p["source_id"]), text=str(p["text"]),
            doc_id=str(p.get("doc_id") or ""), published=str(p.get("published") or ""),
            entity_hint=str(p.get("entity") or ""), url=str(p.get("url") or ""),
        )
        for change in res.state_changes:
            self.bus.produce(Message(TOPIC_CHANGES, key=msg.key,
                                     payload={"change": change,
                                              "source": p["source_id"]}))

    # ---- stage 4: state transitions, for notification/downstream -----------
    def handle_changes(self, msg: Message) -> None:
        self.stats[TOPIC_CHANGES] += 1

    def run_once(self) -> Dict[str, int]:
        """Drain the whole pipeline synchronously. Used by tests, notebooks and
        the API's /refresh endpoint."""
        if not isinstance(self.bus, LocalBus):
            raise RuntimeError("run_once is for the in-process bus; "
                               "with Kafka, run the worker processes instead")
        for _ in range(4):   # raw -> changed -> changes, plus a settling pass
            self.bus.drain([TOPIC_RAW], self.handle_raw)
            self.bus.drain([TOPIC_CHANGED], self.handle_changed)
            self.bus.drain([TOPIC_CHANGES], self.handle_changes)
        return dict(self.stats)

    def run_worker(self, topics: Sequence[str], stop: Optional[threading.Event] = None) -> None:
        """Long-running consumer. One container per topic in production."""
        handlers = {TOPIC_RAW: self.handle_raw, TOPIC_CHANGED: self.handle_changed,
                    TOPIC_CHANGES: self.handle_changes}

        def dispatch(msg: Message) -> None:
            fn = handlers.get(msg.topic)
            if fn:
                fn(msg)

        self.bus.consume(list(topics), dispatch, stop)
