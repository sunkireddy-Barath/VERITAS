"""Deployment preflight. Run before shipping anything.

    python scripts/preflight.py

Checks the things that actually break a first deploy: missing artifacts, an
unreachable data source, a config that would work locally and fail in a
container, and the security defaults that are fine in development and wrong in
production. Exits non-zero if any BLOCKER fails.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BLOCK, WARN, OK = "BLOCK", "WARN", "OK"
results: list[tuple[str, str, str]] = []


def check(level: str, name: str, passed: bool, detail: str = "") -> bool:
    results.append((OK if passed else level, name, detail))
    return passed


def main() -> int:
    print("VERITAS deployment preflight\n" + "=" * 60)

    # ---- artifacts ---------------------------------------------------------
    tok = ROOT / "checkpoints" / "tokenizer.json"
    check(BLOCK, "tokenizer.json present", tok.exists(),
          "run notebook 01" if not tok.exists() else f"{tok.stat().st_size/1024:.0f} KB")
    ckpts = list((ROOT / "checkpoints").glob("*.pt"))
    check(BLOCK, "model checkpoint present", bool(ckpts),
          "run notebooks 03-04" if not ckpts else ", ".join(c.name for c in ckpts))

    sec = ROOT / "data" / "real" / "sec_facts.jsonl"
    n_facts = sum(1 for _ in sec.open(encoding="utf-8")) if sec.exists() else 0
    check(BLOCK, "real SEC data present", n_facts > 0,
          f"{n_facts} facts" if n_facts else "run scripts/fetch_real_data.py --sec-only")

    # ---- the model actually loads -----------------------------------------
    try:
        from veritas.model.transformer import VeritasLM
        from veritas.tokenizer.bpe import BPETokenizer

        t = BPETokenizer.load(tok)
        ck = ROOT / "checkpoints" / "sft.pt"
        if not ck.exists() and ckpts:
            ck = ckpts[0]
        m = VeritasLM.load(str(ck), "cpu")
        vocab_match = m.cfg.vocab_size == t.vocab_size
        check(BLOCK, "checkpoint loads on CPU", True,
              f"{m.num_params():,} params, vocab {m.cfg.vocab_size}")
        # A mismatch here produces garbage answers rather than an error, which
        # is far worse than failing loudly.
        check(BLOCK, "tokenizer/model vocab match", vocab_match,
              f"model={m.cfg.vocab_size} tokenizer={t.vocab_size}")
    except Exception as exc:  # noqa: BLE001
        check(BLOCK, "checkpoint loads on CPU", False, f"{type(exc).__name__}: {exc}")

    # ---- deployment config -------------------------------------------------
    for f, lvl in [("docker/Dockerfile.api", BLOCK), ("docker-compose.yml", BLOCK),
                   ("fly.toml", WARN), ("frontend/vercel.json", WARN),
                   ("docker/initdb/01_schema.sql", WARN), (".env.example", WARN)]:
        check(lvl, f"{f} present", (ROOT / f).exists())

    try:
        vercel = json.load((ROOT / "frontend" / "vercel.json").open(encoding="utf-8"))
        check(WARN, "vercel.json parses", True)
        check(BLOCK, "Vercel build writes config.js",
              "build-config.mjs" in str(vercel.get("buildCommand", ""))
              and (ROOT / "frontend" / "build-config.mjs").exists(),
              "without it the deployed page calls itself instead of the API")
    except Exception as exc:  # noqa: BLE001
        check(WARN, "vercel.json parses", False, str(exc)[:60])

    # ---- things that broke real deploys of this repo ------------------------
    page = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    check(BLOCK, "frontend loads config.js", 'src="config.js"' in page,
          "window.VERITAS_API is never set on a static host")
    check(BLOCK, "served UI matches frontend/",
          (ROOT / "api" / "static" / "index.html").read_text(encoding="utf-8") == page,
          "cp frontend/index.html api/static/index.html")
    dockerfile = (ROOT / "docker" / "Dockerfile.api").read_text(encoding="utf-8")
    for needed in ("checkpoints/tokenizer.json", "checkpoints/sft.pt", "data/real"):
        check(BLOCK, f"image bakes {needed}", needed in dockerfile and (ROOT / needed).exists(),
              "a Fly machine has nothing mounted; the API would boot without it")
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8") if (ROOT / ".dockerignore").exists() else ""
    check(WARN, ".dockerignore excludes training checkpoints", "checkpoints/last.pt" in ignore,
          "otherwise every deploy uploads ~280 MB of unused weights")
    check(BLOCK, "requirements-api.txt present", (ROOT / "requirements-api.txt").exists())
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    check(BLOCK, "Kafka image is pullable",
          not re.search(r"^\s*image:\s*bitnami/kafka", compose, re.M),
          "bitnami/kafka was withdrawn from Docker Hub's free tier")
    fly = (ROOT / "fly.toml").read_text(encoding="utf-8")
    check(WARN, "fly.toml has no empty data volume", "[[mounts]]" not in fly,
          "a new volume is empty; the image's seed data covers it, but it must be created first")

    check(BLOCK, ".env not committed", not (ROOT / ".env").exists()
          or ".env" in (ROOT / ".gitignore").read_text(encoding="utf-8"),
          "secrets must never be committed")

    # ---- security defaults -------------------------------------------------
    api = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
    check(WARN, "CORS restricted", 'allow_origins=["*"]' not in api,
          "dev default is open; pin VERITAS_ALLOWED_ORIGIN before public deploy")
    check(WARN, "auth on write endpoints", "Depends(" in api,
          "/ingest is unauthenticated -- add an API key before exposing it")
    check(WARN, "rate limiting", "slowapi" in api or "limiter" in api.lower(),
          "no limiter; add one before a public endpoint")

    # ---- politeness --------------------------------------------------------
    ua = os.environ.get("VERITAS_USER_AGENT", "")
    check(WARN, "SEC User-Agent configured", "@" in ua,
          "SEC EDGAR requires a UA with contact details; set VERITAS_USER_AGENT")

    # ---- live sources reachable -------------------------------------------
    try:
        import urllib.request

        from veritas.ingest.real_sources import UA, _SSL

        req = urllib.request.Request("https://data.sec.gov/api/xbrl/companyconcept/"
                                     "CIK0000320193/us-gaap/Revenues.json",
                                     headers={"User-Agent": ua or UA})
        with urllib.request.urlopen(req, timeout=15, context=_SSL) as r:
            check(WARN, "SEC EDGAR reachable", r.status == 200, f"HTTP {r.status}")
    except Exception as exc:  # noqa: BLE001
        check(WARN, "SEC EDGAR reachable", False, str(exc)[:70])

    # ---- report ------------------------------------------------------------
    print()
    blockers = 0
    for level, name, detail in results:
        mark = {OK: "  OK  ", WARN: " WARN ", BLOCK: "BLOCK "}[level]
        print(f"[{mark}] {name}" + (f"  --  {detail}" if detail else ""))
        blockers += level == BLOCK
    warns = sum(1 for lvl, _, _ in results if lvl == WARN)

    print("\n" + "=" * 60)
    if blockers:
        print(f"NOT READY: {blockers} blocker(s), {warns} warning(s)")
        return 1
    print(f"READY TO DEPLOY: 0 blockers, {warns} warning(s)")
    if warns:
        print("Warnings are acceptable for a private/demo deploy and must be")
        print("resolved before exposing the API publicly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
