"""VERITAS HTTP API.

    uvicorn api.main:app --reload --port 8000
    open http://localhost:8000

Design notes
------------
* **The model loads once, at startup.** Loading a checkpoint per request would
  dominate latency. The whole system (store, graph, indices, agents) is a
  single process-level singleton in `STATE`.
* **`/ask` streams.** An agentic investigation takes seconds, and the trace is
  part of the product -- the user should watch `PLAN -> SEARCH -> VERIFY`
  happen rather than stare at a spinner. Server-Sent Events, not WebSockets:
  the flow is one-directional and SSE reconnects for free.
* **The answer is a structure, not a string.** Every response carries claims
  with verdicts, citations, the timeline, conflicts and the support label, so
  the UI can make each sentence click-traceable to its evidence.
* **Voice is handled in the browser** (Web Speech API), so the API only ever
  sees text. That keeps the server stateless with respect to audio and avoids
  shipping a 150 MB model; `/ask` is identical whether the question was typed
  or spoken.
* **Both time axes are queryable.** `/entity/{name}/as_of` answers "what did we
  believe on date K about date V", and `/changes` is the feed of every state
  change, restatement and dispute the store has recorded.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="VERITAS", version="0.2",
              description="Verified Evolving Reality & Intelligence Tracking System")

# CORS is pinned by environment. The dev default is the local origins only
# (the API itself on 8000, the standalone frontend on 3000); a wildcard would
# let any site drive this API from a victim's browser.
_ORIGINS = [o.strip() for o in os.environ.get(
    "VERITAS_ALLOWED_ORIGIN",
    "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000",
).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# ------------------------------------------------------------ auth + limits
# VERITAS_API_KEY unset => open, for local development. Set it and every write
# endpoint requires the header. Reads stay public because the answers are
# meant to be shared; writes mutate the knowledge base.
_API_KEY = os.environ.get("VERITAS_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Optional[str] = Depends(_api_key_header)) -> None:
    if not _API_KEY:
        return
    # compare_digest: a plain == leaks key length and prefix through timing.
    if not key or not hmac.compare_digest(key, _API_KEY):
        raise HTTPException(401, "invalid or missing X-API-Key")


# Fixed-window limiter, per client IP. Deliberately in-process and dependency
# free: it protects a single instance from one abusive client. Behind more than
# one replica, move the counter to Redis or the platform's edge limiter.
_RATE_LIMIT = int(os.environ.get("VERITAS_RATE_LIMIT_PER_MIN", "60"))
# X-Forwarded-For is client-controlled unless a proxy you run overwrites it.
# Trusting it by default lets any client dodge the limit with a fake header, so
# it is honoured only when the deployment says a proxy sits in front.
_TRUST_PROXY = os.environ.get("VERITAS_TRUST_PROXY", "").lower() in ("1", "true", "yes")
_MAX_TRACKED_CLIENTS = 10_000
_hits: Dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    if _RATE_LIMIT <= 0:
        return
    ip = request.client.host if request.client else "unknown"
    if _TRUST_PROXY:
        ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or ip
    now = time.time()
    window = _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= _RATE_LIMIT:
        raise HTTPException(429, f"rate limit {_RATE_LIMIT}/min exceeded")
    window.append(now)
    # Idle clients are swept so the table cannot grow without bound.
    if len(_hits) > _MAX_TRACKED_CLIENTS:
        for k in [k for k, w in _hits.items() if not w or now - w[-1] > 60]:
            del _hits[k]


STATE: Dict[str, Any] = {"ready": False, "error": None}

# One lock around every read and write of the knowledge base. Answers iterate
# the chunk metadata and the store while /ingest and /refresh append to them;
# unsynchronised, a live ingest during an answer raises "dictionary changed
# size during iteration". One model on one device serialises the work anyway.
_KB_LOCK = threading.RLock()


# --------------------------------------------------------------- lifecycle
def _boot() -> None:
    """Load checkpoints and rebuild the system from real data."""
    import torch

    from veritas.agents.orchestrator import VeritasConfig
    from veritas.ingest.real_sources import build_real_system
    from veritas.model.transformer import ModelConfig, VeritasLM
    from veritas.pipeline import VeritasSystemBuilder
    from veritas.tokenizer.bpe import BPETokenizer

    t0 = time.time()
    # VERITAS_DEVICE pins the device (the CPU image sets "cpu"); unset = auto.
    wanted = os.environ.get("VERITAS_DEVICE", "").strip().lower()
    if wanted == "cuda" and not torch.cuda.is_available():
        print("[api] VERITAS_DEVICE=cuda but no GPU is visible; using cpu")
        wanted = "cpu"
    device = wanted or ("cuda" if torch.cuda.is_available() else "cpu")
    tok_path = ROOT / "checkpoints" / "tokenizer.json"
    if not tok_path.exists():
        raise RuntimeError("checkpoints/tokenizer.json missing -- run notebook 01 first")
    tok = BPETokenizer.load(tok_path)

    ckpt = ROOT / "checkpoints" / "sft.pt"
    if not ckpt.exists():
        ckpt = ROOT / "checkpoints" / "best.pt"
    if ckpt.exists():
        model = VeritasLM.load(str(ckpt), device)
    else:
        # The retrieval/verification stack works without a trained LM; only
        # generative synthesis degrades. Better to serve than to refuse.
        model = VeritasLM(ModelConfig(vocab_size=tok.vocab_size, d_model=256,
                                      n_layers=4, n_heads=4, n_kv_heads=2,
                                      max_seq_len=256)).to(device)
    model.eval()

    data_dir = _data_dir()
    builder = VeritasSystemBuilder(model, tok, device=device, domain="corporate")
    report = build_real_system(builder, data_dir, verbose=True)
    veritas = builder.build(config=VeritasConfig(k=8, max_iterations=3,
                                                 domain="corporate", verbose=False))
    STATE.update({
        "ready": True, "builder": builder, "veritas": veritas, "model": model,
        "tok": tok, "device": device, "boot_seconds": round(time.time() - t0, 2),
        "checkpoint": ckpt.name if ckpt.exists() else "untrained",
        "report": report, "data_dir": str(data_dir),
    })
    # The broker connection runs in its own thread. Inside startup, its retries
    # (up to two minutes while a broker boots) held uvicorn's startup hook, so
    # the API answered nothing -- not even /health -- until Kafka was reachable.
    STATE["bus"] = {"kind": "connecting" if os.environ.get("VERITAS_KAFKA_BOOTSTRAP", "").strip()
                    else "none"}
    threading.Thread(target=_start_bus, args=(builder,), name="veritas-bus-connect",
                     daemon=True).start()
    print(f"[api] ready in {STATE['boot_seconds']}s on {device} using {STATE['checkpoint']}, "
          f"data={data_dir}, bus={STATE['bus']['kind']}")


def _data_dir() -> Path:
    """The first real-data directory that holds the SEC facts.

    VERITAS_DATA_DIR, then data/real (a mount or a fresh fetch), then the seed
    copy baked into the image. Without the fallback, an empty volume mounted
    over data/ boots a system that knows nothing and still reports ready.
    """
    candidates = [os.environ.get("VERITAS_DATA_DIR", ""), ROOT / "data" / "real",
                  ROOT / "seed" / "real"]
    for c in candidates:
        if c and (Path(c) / "sec_facts.jsonl").exists():
            return Path(c)
    raise RuntimeError("no real data found (looked for sec_facts.jsonl in "
                       f"{[str(c) for c in candidates if c]}) -- run "
                       "python scripts/fetch_real_data.py")


def _start_bus(builder) -> None:
    """With VERITAS_KAFKA_BOOTSTRAP set, consume veritas.raw in this process.

    The knowledge base lives in this process's memory, so this process must be
    the consumer: a document is only answerable once it is in THIS store. Each
    replica consumes the whole log under its own group id, and a restart replays
    from the earliest retained offset -- the in-memory store is rebuilt from the
    log, which is what the log's retention is for.
    """
    from veritas.ingest.streaming import (TOPIC_CHANGED, TOPIC_CHANGES, TOPIC_RAW,
                                          KafkaBus, StreamingIngest, make_bus)

    bootstrap = os.environ.get("VERITAS_KAFKA_BOOTSTRAP", "").strip()
    if not bootstrap:
        STATE["bus"] = {"kind": "none"}
        return
    import socket
    import uuid

    # A NEW consumer group on every boot. Offsets are committed per group, so a
    # fixed name ("veritas-api-<host>") resumed after the last committed offset
    # on restart -- skipping everything already consumed, which this process's
    # fresh in-memory store no longer holds. A group with no committed offsets
    # reads from the earliest retained message: the restart replays the log.
    # Set VERITAS_KAFKA_GROUP only if you want resume-from-offset instead.
    group = os.environ.get("VERITAS_KAFKA_GROUP") or \
        f"veritas-api-{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    bus = make_bus(bootstrap, group_id=group, attempts=24, wait_s=5.0)
    if not isinstance(bus, KafkaBus):
        # Configured but unreachable: say so on /health instead of pretending.
        STATE["bus"] = {"kind": "unavailable", "bootstrap": bootstrap}
        return
    # inline: this process runs every stage itself and consumes ONLY veritas.raw.
    # Consuming the stage topics too made a replay re-ingest the veritas.changed
    # messages a previous run wrote AND write fresh ones from the raw replay, so
    # every restart duplicated work and grew the log.
    stream = StreamingIngest(builder.ingest, bus, lock=_KB_LOCK, inline=True)
    thread, stop = stream.start_background((TOPIC_RAW,))
    STATE["bus"] = {"kind": "kafka", "bootstrap": bootstrap, "group": group}
    STATE.update(stream=stream, bus_thread=thread, bus_stop=stop)


@app.on_event("startup")
async def startup() -> None:
    try:
        await asyncio.get_event_loop().run_in_executor(None, _boot)
    except Exception as exc:  # surface the reason via /health instead of dying
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[api] STARTUP FAILED: {STATE['error']}")


def _require_ready():
    if STATE.get("error"):
        raise HTTPException(503, f"system not ready: {STATE['error']}")
    if not STATE.get("ready"):
        raise HTTPException(503, "system still loading, retry shortly")
    return STATE["veritas"]


def _answer(question: str, history: List[str], mode: str):
    # The mode is passed per call. Writing it into the shared config let one
    # request's "generative" leak into a concurrent request's answer.
    with _KB_LOCK:
        return STATE["veritas"].answer(question, history, mode)


def _parse_time(value: str, name: str):
    from veritas.temporal.versioning import _dt

    try:
        return _dt(value, strict=True)
    except ValueError:
        raise HTTPException(422, f"'{name}' is not a date: {value!r} (use YYYY-MM-DD)")


def _version_json(v) -> Dict[str, Any]:
    return {
        "value": v.value,
        "valid_from": v.valid_from.date().isoformat(),
        "valid_to": "present" if v.valid_to.year > 9000 else v.valid_to.date().isoformat(),
        "recorded_at": v.recorded_at.date().isoformat(),
        "superseded_at": v.superseded_at.date().isoformat() if v.superseded_at else None,
        "source": v.source_id,
        "change_kind": v.change_kind,
        "previous_value": v.previous_value,
        "reason": v.reason,
    }


# ------------------------------------------------------------------ models
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    history: List[str] = Field(default_factory=list)
    mode: str = Field("extractive", pattern="^(extractive|generative)$")


class IngestRequest(BaseModel):
    source_id: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=200_000)
    entity: str = Field("", max_length=200)
    published: str = ""
    url: str = Field("", max_length=2000)
    tier: int = Field(3, ge=1, le=4)


# ------------------------------------------------------------------ routes
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ready": STATE.get("ready", False),
        "error": STATE.get("error"),
        "device": STATE.get("device"),
        "checkpoint": STATE.get("checkpoint"),
        "boot_seconds": STATE.get("boot_seconds"),
        # "none" = no broker configured; "unavailable" = configured but
        # unreachable, which a deploy must not mistake for a working pipeline.
        "bus": STATE.get("bus", {"kind": "starting"}),
    }


@app.get("/stats")
def stats() -> Dict[str, Any]:
    _require_ready()
    b = STATE["builder"]
    stream = STATE.get("stream")
    with _KB_LOCK:
        return {
            "bus": dict(STATE.get("bus", {}),
                        stats=dict(stream.stats) if stream is not None else None),
            "corpus_chunks": len(b.corpus),
            "entities": b.store.entities()[:200],
            "entity_count": len(b.store.entities()),
            "fact_versions": sum(len(v) for v in b.store._index.values()),
            "state_changes": len([c for c in b.store.changes
                                  if c.kind in ("CHANGED", "CORRECTED")]),
            "graph": b.graph.stats(),
            "sources": sorted({m.get("source", "") for m in b.metadata.values()
                               if m.get("source")}),
            "load_report": STATE.get("report", {}),
        }


@app.post("/ask", dependencies=[Depends(rate_limit)])
def ask(req: AskRequest) -> Dict[str, Any]:
    _require_ready()
    t0 = time.time()
    ans = _answer(req.question, req.history, req.mode)
    out = ans.to_dict()
    out["elapsed_seconds"] = round(time.time() - t0, 3)
    return out


@app.post("/ask/stream", dependencies=[Depends(rate_limit)])
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """SSE: emit progress events, then the final answer.

    The agent's own `trace` is replayed as events, so what the UI shows is the
    real decision path, not a decorative progress bar.
    """
    _require_ready()

    async def gen():
        yield _sse("status", {"stage": "planning", "message": "Understanding the question"})
        loop = asyncio.get_event_loop()
        t0 = time.time()
        try:
            ans = await loop.run_in_executor(None, _answer, req.question, req.history, req.mode)
        except Exception as exc:
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})
            return
        for line in ans.trace:
            yield _sse("trace", {"line": line})
            await asyncio.sleep(0.04)   # let the UI render the path
        payload = ans.to_dict()
        payload["elapsed_seconds"] = round(time.time() - t0, 3)
        yield _sse("answer", payload)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.get("/entity/{name}/timeline")
def timeline(name: str, attribute: str = "") -> Dict[str, Any]:
    """Every version of every attribute, superseded originals included."""
    _require_ready()
    store = STATE["builder"].store
    with _KB_LOCK:
        attrs = [attribute] if attribute else sorted(
            {a for (e, a) in store._index if e == store.canonical(name)})
        if not attrs or not any(store.history(name, a, include_superseded=True) for a in attrs):
            raise HTTPException(404, f"no facts recorded for '{name}'")
        out = {}
        for attr in attrs:
            cur = store.current(name, attr)
            out[attr] = {
                "current": cur.value if cur else None,
                "versions": [dict(_version_json(v),
                                  is_current=bool(cur and v.version_id == cur.version_id))
                             for v in store.history(name, attr, include_superseded=True)],
            }
        return {"entity": store.canonical(name), "attributes": out}


@app.get("/entity/{name}/as_of")
def as_of(name: str, attribute: str, valid: str = "", known: str = "") -> Dict[str, Any]:
    """Bitemporal time travel: the value valid on `valid`, as believed on `known`.

    `known` omitted means "as we believe now". The belief history lists every
    value held for that valid time in recording order, so a restatement reads
    as "reported X, restated to Y" rather than as a silent overwrite.
    """
    _require_ready()
    from veritas.temporal.versioning import _norm, now_utc

    store = STATE["builder"].store
    t = _parse_time(valid, "valid") if valid else now_utc()
    k = _parse_time(known, "known") if known else None
    with _KB_LOCK:
        if not store.history(name, attribute, include_superseded=True):
            raise HTTPException(404, f"no '{attribute}' recorded for '{name}'")
        believed = store.as_of(name, attribute, t, k)
        trail = [v for v in store.belief_history(name, attribute, t)
                 if k is None or v.recorded_at <= k]
        if believed is None:
            note = ("Nothing was on record yet for that time." if k is not None
                    and trail == [] else "No value is recorded as valid at that time.")
        elif len({_norm(v.value) for v in trail}) > 1:
            note = "The recorded value for this time was revised; see belief_history."
        else:
            note = ""
        return {
            "entity": store.canonical(name), "attribute": attribute,
            "valid_at": t.date().isoformat(),
            "known_at": k.date().isoformat() if k else "latest",
            "value": believed.value if believed else None,
            "version": _version_json(believed) if believed else None,
            # As believed on `known`, a replacement filed later had not happened yet.
            "belief_history": [
                dict(_version_json(v), superseded_at=None)
                if k is not None and v.superseded_at is not None and v.superseded_at > k
                else _version_json(v) for v in trail],
            "note": note,
        }


@app.get("/changes")
def changes(since: str = "", entity: str = "", kind: str = "",
            limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    """The world-state change feed, newest recording first.

    Defaults to the events that alter an answer -- CHANGED (the world moved),
    CORRECTED (a source restated itself) and CONFLICT (sources disagree about
    the same time) -- and leaves out mere reaffirmations.
    """
    _require_ready()
    store = STATE["builder"].store
    kinds = {x.strip().upper() for x in kind.split(",") if x.strip()} or {
        "CHANGED", "CORRECTED", "CONFLICT"}
    start = _parse_time(since, "since") if since else None
    with _KB_LOCK:
        ent = store.canonical(entity) if entity else None
        rows = [c for c in store.changes
                if c.kind in kinds and (ent is None or c.entity == ent)
                and (start is None or c.detected_at >= start)]
        rows.sort(key=lambda c: (c.detected_at, c.effective_at), reverse=True)
        return {
            "total": len(rows), "kinds": sorted(kinds),
            "changes": [{
                "entity": c.entity, "attribute": c.attribute, "kind": c.kind,
                "old_value": c.old_value, "new_value": c.new_value,
                "effective_at": c.effective_at.date().isoformat(),
                "recorded_at": c.detected_at.date().isoformat(),
                "source": c.source_id, "note": c.note,
            } for c in rows[:limit]],
        }


@app.get("/entity/{name}/provenance")
def provenance(name: str) -> Dict[str, Any]:
    _require_ready()
    g = STATE["builder"].graph
    with _KB_LOCK:
        claims = g.claims_about(name)
        if not claims:
            raise HTTPException(404, f"no claims recorded about '{name}'")
        return {"entity": name, "claims": [g.provenance_path(c) for c in claims[:40]]}


@app.post("/ingest", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def ingest(req: IngestRequest) -> Dict[str, Any]:
    """Add a document live. The answer to affected questions changes
    immediately -- no re-index, no retraining."""
    _require_ready()
    b = STATE["builder"]
    doc_id = f"api:{time.time_ns()}"
    published = req.published or time.strftime("%Y-%m-%d")
    with _KB_LOCK:
        b.add_source(req.source_id, tier=req.tier)
        res = b.add_document(req.source_id, doc_id, req.text, published, req.entity,
                             url=req.url)
    # With Kafka on, the document is also appended to veritas.raw, so a replay
    # of the log rebuilds a store that includes it. This process's own consumer
    # sees the same text again and change detection suppresses it, so the
    # write-through is idempotent rather than a double ingest.
    logged = False
    stream = STATE.get("stream")
    if stream is not None:
        stream.publish_raw(source_id=req.source_id, text=req.text, doc_id=doc_id,
                           published=published, entity=req.entity, url=req.url,
                           tier=req.tier)
        logged = True
    return {
        "changed": res.changed, "reason": res.reason, "chunks": res.n_chunks,
        "claims": res.n_claims, "state_changes": res.state_changes,
        "invalidated_cache_keys": res.invalidated,
        "elapsed_seconds": round(res.elapsed_s, 4), "retrain_required": False,
        "logged_to_kafka": logged,
    }


@app.post("/refresh", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def refresh() -> Dict[str, Any]:
    """Re-poll the live feeds now and ingest anything new."""
    _require_ready()
    from scripts.fetch_real_data import FEEDS  # single source of truth for feeds
    from veritas.ingest.real_sources import refresh_feeds

    with _KB_LOCK:
        return refresh_feeds(STATE["builder"], FEEDS, verbose=False)


@app.get("/")
def index():
    f = STATIC / "index.html"
    if not f.exists():
        raise HTTPException(404, "frontend not built")
    return FileResponse(f)


@app.get("/config.js")
def config_js() -> Response:
    # The page loads config.js to learn where the API is. Served from here, the
    # API is this same origin, so the value is empty. (On Vercel the file is
    # generated by frontend/build-config.mjs instead.)
    return Response('window.VERITAS_API = window.VERITAS_API || "";\n',
                    media_type="application/javascript",
                    headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
