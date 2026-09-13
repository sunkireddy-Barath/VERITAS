"""Deploy the backend (API + model + UI) to a Hugging Face Docker Space.

NOTE: Hugging Face requires a PRO subscription for Docker Spaces on cpu-basic
(a free account gets HTTP 402 at create_repo). Static Spaces remain free.

    python scripts/deploy_hf_space.py --space <hf-user>/veritas --dry-run    # assemble only
    HF_TOKEN=hf_... python scripts/deploy_hf_space.py --space <hf-user>/veritas
    HF_TOKEN=hf_... python scripts/deploy_hf_space.py --space <hf-user>/veritas \
        --allow-origin https://<project>.vercel.app --config-only           # after Vercel

Why a Space: cpu-basic runs a Docker container with 16 GB of RAM over HTTPS,
which fits a model plus in-memory indices. Free web tiers elsewhere cap memory
around 512 MB, below what PyTorch and the store need.

The Space is the same image as every other deployment: docker/Dockerfile.api
becomes its Dockerfile, with the serving checkpoint and the real data baked in.
The token needs WRITE access (https://huggingface.co/settings/tokens) and is
read from HF_TOKEN; it is never written anywhere. A generated VERITAS_API_KEY
(for /ingest and /refresh) is kept in .deploy/secrets.json, which git ignores.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".deploy" / "hf-space"
SECRETS = ROOT / ".deploy" / "secrets.json"

#: Exactly what Dockerfile.api COPYs. Nothing else is uploaded.
FILES = ["veritas", "api", "scripts", "run_veritas.py", "requirements-api.txt",
         "checkpoints/tokenizer.json", "checkpoints/sft.pt", "data/real"]

SPACE_README = """---
title: VERITAS
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
short_description: Time-aware, evidence-verified answers over SEC and Wikidata
---

# VERITAS

Verified Evolving Reality & Intelligence Tracking System: a from-scratch
tokenizer, transformer, retrieval stack and bitemporal evidence store that
answers questions about companies with dated, cited evidence, and refuses when
the evidence is not there.

* UI: this Space's root page
* API: `/ask`, `/entity/{name}/as_of`, `/changes`, `/health`, `/docs`

Deployed from `scripts/deploy_hf_space.py`.
"""


def space_url(repo_id: str) -> str:
    user, name = repo_id.split("/", 1)
    sub = f"{user}-{name}".lower().replace("_", "-").replace(".", "-")
    return f"https://{sub}.hf.space"


def assemble(checkpoints: Path | None = None) -> int:
    """Stage the Space. `checkpoints` overrides where tokenizer.json and sft.pt
    come from -- e.g. a verified backup while notebooks are retraining the
    working copies."""
    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir(parents=True)
    total = 0
    for rel in FILES:
        src = ROOT / rel
        if checkpoints is not None and rel.startswith("checkpoints/"):
            src = checkpoints / Path(rel).name
        if not src.exists():
            sys.exit(f"missing {rel}: run the notebooks / scripts/fetch_real_data.py first")
        dst = STAGE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".gitkeep"))
        else:
            shutil.copy2(src, dst)
    shutil.copy2(ROOT / "docker" / "Dockerfile.api", STAGE / "Dockerfile")
    (STAGE / "README.md").write_text(SPACE_README, encoding="utf-8")
    for f in STAGE.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def api_key() -> str:
    """Reuse the key from a previous deploy so clients keep working."""
    if SECRETS.exists():
        key = json.loads(SECRETS.read_text(encoding="utf-8")).get("VERITAS_API_KEY")
        if key:
            return key
    key = secrets.token_urlsafe(24)
    SECRETS.parent.mkdir(parents=True, exist_ok=True)
    SECRETS.write_text(json.dumps({"VERITAS_API_KEY": key}, indent=2), encoding="utf-8")
    return key


def wait_until_ready(api, repo_id: str, url: str, timeout_s: int = 1800) -> bool:
    """Follow the Space through build and boot, then require /health ready."""
    deadline, last = time.time() + timeout_s, None
    while time.time() < deadline:
        stage = str(api.get_space_runtime(repo_id).stage)
        if stage != last:
            print(f"[space] {stage}", flush=True)
            last = stage
        if stage.endswith("ERROR"):
            print(f"[space] failed: see https://huggingface.co/spaces/{repo_id}?logs=build")
            return False
        if stage.endswith("RUNNING"):
            try:
                health = json.load(urllib.request.urlopen(f"{url}/health", timeout=15))
                if health.get("ready"):
                    print(f"[space] ready: {health}", flush=True)
                    return True
                if health.get("error"):
                    print(f"[space] boot error: {health['error']}")
                    return False
            except Exception:  # noqa: BLE001 - the app is still loading the model
                pass
        time.sleep(15)
    print("[space] timed out waiting for the Space to become ready")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<hf-user>/<space-name>")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--allow-origin", default="",
                    help="comma-separated browser origins, e.g. https://<project>.vercel.app")
    ap.add_argument("--dry-run", action="store_true", help="assemble the Space locally, upload nothing")
    ap.add_argument("--config-only", action="store_true", help="update variables/secrets, no upload")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--checkpoints", default="",
                    help="directory holding tokenizer.json and sft.pt (default: checkpoints/)")
    args = ap.parse_args()
    if "/" not in args.space:
        sys.exit("--space must be <hf-user>/<space-name>")

    url = space_url(args.space)
    if not args.config_only:
        size = assemble(Path(args.checkpoints) if args.checkpoints else None)
        print(f"[space] staged {STAGE} ({size / 1e6:.1f} MB) for {args.space} -> {url}")
    if args.dry_run:
        for rel in sorted(p.relative_to(STAGE).as_posix() for p in STAGE.iterdir()):
            print(f"  {rel}")
        return 0

    from huggingface_hub import HfApi, get_token

    # HF_TOKEN, else the login cached by `huggingface-cli login` / `hf auth login`.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()
    if not token:
        sys.exit("log in with `hf auth login` or set HF_TOKEN (a token with WRITE access)")
    api = HfApi(token=token)
    print(f"[space] authenticated as {api.whoami()['name']}")
    api.create_repo(args.space, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)

    origins = ",".join(o for o in [url, *args.allow_origin.split(",")] if o.strip())
    for key, value in {"VERITAS_DEVICE": "cpu", "VERITAS_TRUST_PROXY": "1",
                       "VERITAS_ALLOWED_ORIGIN": origins}.items():
        api.add_space_variable(args.space, key, value)
    api.add_space_secret(args.space, "VERITAS_API_KEY", api_key())
    print(f"[space] variables set; CORS origins: {origins}; API key in {SECRETS}")

    if not args.config_only:
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(STAGE),
                          commit_message="Deploy VERITAS")
        print(f"[space] uploaded: https://huggingface.co/spaces/{args.space}")
    if args.no_wait:
        return 0
    return 0 if wait_until_ready(api, args.space, url) else 1


if __name__ == "__main__":
    sys.exit(main())
