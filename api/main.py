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
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="VERITAS", version="0.1",
              description="Verified Evolving Reality & Intelligence Tracking System")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE: Dict[str, Any] = {"ready": False, "error": None}


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    builder = VeritasSystemBuilder(model, tok, device=device, domain="corporate")
    report = build_real_system(builder, ROOT / "data" / "real", verbose=True)
    veritas = builder.build(config=VeritasConfig(k=8, max_iterations=3,
                                                 domain="corporate", verbose=False))
    STATE.update({
        "ready": True, "builder": builder, "veritas": veritas, "model": model,
        "tok": tok, "device": device, "boot_seconds": round(time.time() - t0, 2),
        "checkpoint": ckpt.name if ckpt.exists() else "untrained",
        "report": report,
    })
    print(f"[api] ready in {STATE['boot_seconds']}s on {device} using {STATE['checkpoint']}")


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


# ------------------------------------------------------------------ models
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    history: List[str] = Field(default_factory=list)
    mode: str = Field("extractive", pattern="^(extractive|generative)$")


class IngestRequest(BaseModel):
    source_id: str
    text: str
    entity: str = ""
    published: str = ""
    url: str = ""
    tier: int = 3


# ------------------------------------------------------------------ routes
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ready": STATE.get("ready", False),
        "error": STATE.get("error"),
        "device": STATE.get("device"),
        "checkpoint": STATE.get("checkpoint"),
        "boot_seconds": STATE.get("boot_seconds"),
    }


@app.get("/stats")
def stats() -> Dict[str, Any]:
    _require_ready()
    b = STATE["builder"]
    return {
        "corpus_chunks": len(b.corpus),
        "entities": b.store.entities()[:200],
        "entity_count": len(b.store.entities()),
        "fact_versions": sum(len(v) for v in b.store._index.values()),
        "state_changes": len([c for c in b.store.changes
                              if c.kind in ("CHANGED", "CORRECTED")]),
        "graph": b.graph.stats(),
        "sources": sorted({m.get("source", "") for m in b.metadata.values() if m.get("source")}),
        "load_report": STATE.get("report", {}),
    }


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    veritas = _require_ready()
    veritas.cfg.synthesis_mode = req.mode
    t0 = time.time()
    ans = veritas.answer(req.question, req.history)
    out = ans.to_dict()
    out["elapsed_seconds"] = round(time.time() - t0, 3)
    return out


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """SSE: emit progress events, then the final answer.

    The agent's own `trace` is replayed as events, so what the UI shows is the
    real decision path, not a decorative progress bar.
    """
    veritas = _require_ready()

    async def gen():
        yield _sse("status", {"stage": "planning", "message": "Understanding the question"})
        loop = asyncio.get_event_loop()
        t0 = time.time()
        try:
            ans = await loop.run_in_executor(None, veritas.answer, req.question, req.history)
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
    _require_ready()
    store = STATE["builder"].store
    attrs = [attribute] if attribute else sorted(
        {a for (e, a) in store._index if e == store.canonical(name)})
    if not attrs:
        raise HTTPException(404, f"no facts recorded for '{name}'")
    out = {}
    for attr in attrs:
        cur = store.current(name, attr)
        out[attr] = {
            "current": cur.value if cur else None,
            "versions": [{
                "value": v.value,
                "valid_from": v.valid_from.date().isoformat(),
                "valid_to": "present" if v.valid_to.year > 9000 else v.valid_to.date().isoformat(),
                "recorded_at": v.recorded_at.date().isoformat(),
                "source": v.source_id,
                "change_kind": v.change_kind,
                "previous_value": v.previous_value,
                "is_current": bool(cur and v.version_id == cur.version_id),
            } for v in store.history(name, attr)],
        }
    return {"entity": store.canonical(name), "attributes": out}


@app.get("/entity/{name}/provenance")
def provenance(name: str) -> Dict[str, Any]:
    _require_ready()
    g = STATE["builder"].graph
    claims = g.claims_about(name)
    if not claims:
        raise HTTPException(404, f"no claims recorded about '{name}'")
    return {"entity": name, "claims": [g.provenance_path(c) for c in claims[:40]]}


@app.post("/ingest")
def ingest(req: IngestRequest) -> Dict[str, Any]:
    """Add a document live. The answer to affected questions changes
    immediately -- no re-index, no retraining."""
    _require_ready()
    b = STATE["builder"]
    b.add_source(req.source_id, tier=req.tier)
    res = b.add_document(req.source_id, f"api:{int(time.time()*1000)}", req.text,
                         req.published or time.strftime("%Y-%m-%d"), req.entity, url=req.url)
    return {
        "changed": res.changed, "reason": res.reason, "chunks": res.n_chunks,
        "claims": res.n_claims, "state_changes": res.state_changes,
        "invalidated_cache_keys": res.invalidated,
        "elapsed_seconds": round(res.elapsed_s, 4), "retrain_required": False,
    }


@app.post("/refresh")
def refresh() -> Dict[str, Any]:
    """Re-poll the live feeds now and ingest anything new."""
    _require_ready()
    from scripts.fetch_real_data import FEEDS  # single source of truth for feeds
    from veritas.ingest.real_sources import refresh_feeds

    return refresh_feeds(STATE["builder"], FEEDS, verbose=False)


@app.get("/")
def index():
    f = STATIC / "index.html"
    if not f.exists():
        raise HTTPException(404, "frontend not built")
    return FileResponse(f)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
