# Deployment

## The one constraint that decides the topology

VERITAS holds a PyTorch model, a vector index, a BM25 index, a bitemporal store
and an evidence graph **in memory**, and builds them once at startup: ~10–35 s
on a GPU, and minutes on CPU, where embedding every chunk dominates (382 s
measured on a loaded machine). It is a **stateful, long-lived process**, and
every health-check grace period in this repo is 600 s to cover the CPU case.

That rules out Vercel, Netlify Functions or Lambda for the backend: serverless
has no GPU, a bundle cap in the hundreds of MB, and would pay the startup on
every cold start. So:

```
Browser ── Vercel (static page + config.js) ──HTTPS──▶ Fly.io machine (FastAPI + model + UI)
                                                           ▲
                                  feed poller ──▶ Kafka ───┘  (optional; the API consumes it)
```

| component | where | notes |
|---|---|---|
| frontend | Vercel (or any static host) | `build-config.mjs` writes the API URL into `config.js` |
| API + model | Fly.io (or Railway, Render, a VM) | image bakes the serving checkpoint and seed data |
| Kafka | local: `docker compose --profile kafka`; hosted: Confluent / Redpanda / Aiven | optional |
| Postgres, Redis | **not used yet** | `docker/initdb/01_schema.sql` is the target schema only |

Run `python scripts/preflight.py` before any deploy. It blocks on the failures
that actually broke this repo's deploys: an image without the checkpoint, a
page that never loads `config.js`, an unpullable Kafka image.

---

## 1. Local

Without Docker:

```bash
python run_veritas.py              # API + UI on http://localhost:8000
```

With Docker:

```bash
docker compose up -d --build       # the same, in a container
make kafka                         # + Kafka broker + feed poller
make kafka-smoke                   # proves a Kafka message changes an answer
```

### How Kafka is wired, and why this way

The knowledge base lives in the API process's memory, so **the API is the
consumer**. With `VERITAS_KAFKA_BOOTSTRAP` set it starts a background thread
that drains `veritas.raw` → change detection → `veritas.changed` → chunk, embed,
index, assert facts → `veritas.changes`, holding the same lock as request
handlers. A separate consumer container would build its own private knowledge
base that no question can reach; the first version of this stack did exactly
that, and nothing that went through Kafka ever appeared in an answer.

| process | role | replicas |
|---|---|---|
| `api` | serves questions **and** consumes the log | one per knowledge-base copy |
| `worker-poller` | fetches the monitored feeds, publishes to `veritas.raw`; loads no model | exactly one |

* Every API boot joins a **new** consumer group (`veritas-api-<hostname>-<random>`),
  so it has no committed offsets and reads from the earliest retained message:
  a restarted replica **replays the log** and rebuilds its store — that is what
  the 7-day retention is for. A fixed group name would resume after the last
  committed offset and silently lose everything consumed before the restart.
  Setting `VERITAS_KAFKA_GROUP` opts into resume-from-offset.
* `/ingest` still answers synchronously and also writes the document to
  `veritas.raw`, so a replay includes it. The replica's own consumer sees the
  text again and change detection suppresses it: idempotent, not a double ingest.
* `/health` reports `bus.kind`: `none` (no broker configured), `kafka`
  (consuming), or `unavailable` (configured but unreachable after two minutes
  of retries) — a deploy must not mistake the last for a working pipeline.
  `/stats` adds per-topic counters, suppressed duplicates and handler errors.

Two settings are correctness requirements, not tuning:

* **Partition by entity.** `TemporalStore.assert_fact` classifies an assertion
  against what it already holds, so out-of-order delivery turns a succession
  into a spurious `CONFLICT`. Messages are keyed by entity for per-entity order.
* **Commit after the handler.** A crash mid-handler redelivers the message;
  redelivery lands as `REAFFIRMED` or is suppressed by change detection.

The Compose broker is the official `apache/kafka` image in KRaft mode (no
ZooKeeper). `bitnami/kafka` was withdrawn from Docker Hub's free tier in 2025.

---

## 2. Free public link — Cloudflare quick tunnel from your own machine

The only hosting in this guide that costs nothing and needs no account or card.
The app runs where it already runs (your GPU), and Cloudflare gives it a public
HTTPS URL. It is online only while the machine and the tunnel are running, and
the URL changes each time the tunnel restarts.

```bash
# 1. the app, bound to localhost only, with writes protected
export VERITAS_API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(24))")
export VERITAS_TRUST_PROXY=1          # safe: only the local tunnel can reach 127.0.0.1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# 2. the tunnel (download cloudflared from github.com/cloudflare/cloudflared/releases)
cloudflared tunnel --url http://127.0.0.1:8000    # prints https://<words>.trycloudflare.com

# 3. prove it works over the real network path
python scripts/e2e_public.py https://<words>.trycloudflare.com
```

`scripts/e2e_public.py` checks the UI, every read endpoint, SSE streaming through
the proxy, API-key enforcement, CORS, input validation, refusals and live
ingestion. Keep `VERITAS_API_KEY` set: the link is public, and without a key
anyone could write to the knowledge base.

---

## 2b. Backend — Hugging Face Spaces (Docker; needs HF PRO)

A Docker Space on `cpu-basic` runs the container with 16 GB of RAM over HTTPS,
which fits the model and the in-memory indices. **As of September 2026 Hugging
Face requires a PRO subscription for Docker and Gradio Spaces on cpu-basic**;
creating one on a free account returns `402 Payment Required`. (Static Spaces
stay free, but this backend is not static.) Free web tiers that cap memory near
512 MB cannot hold PyTorch plus the store, and a 0.1-vCPU instance would take
far longer than any health check allows to embed the corpus at boot.

```bash
# token with WRITE access: https://huggingface.co/settings/tokens
export HF_TOKEN=hf_...
python scripts/deploy_hf_space.py --space <hf-user>/veritas --dry-run    # see what uploads
python scripts/deploy_hf_space.py --space <hf-user>/veritas              # create, upload, wait for /health
# -> https://<hf-user>-veritas.hf.space  (API + UI)
```

The script stages exactly what `Dockerfile.api` copies (code, `tokenizer.json`,
`sft.pt`, `data/real`, ~58 MB), uses that Dockerfile unchanged, sets
`VERITAS_DEVICE=cpu`, `VERITAS_TRUST_PROXY=1` and the CORS origins as Space
variables, stores a generated `VERITAS_API_KEY` as a Space secret (and in
`.deploy/secrets.json`, git-ignored), then follows the build until `/health`
reports ready. The container runs as UID 1000, which Spaces require.

After deploying the frontend, allow its origin:

```bash
python scripts/deploy_hf_space.py --space <hf-user>/veritas \
  --allow-origin https://<project>.vercel.app --config-only
```

---

## 2c. Backend, paid — Fly.io

```bash
# once
fly auth login
fly launch --copy-config --no-deploy --name <unique-app-name>
fly secrets set VERITAS_API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(24))")

# every deploy
python scripts/preflight.py
fly deploy --remote-only                     # builds on Fly's builders; no local Docker needed
curl https://<unique-app-name>.fly.dev/health
```

What `fly.toml` and the image guarantee:

* **The image is self-contained.** `Dockerfile.api` copies `checkpoints/tokenizer.json`,
  `checkpoints/sft.pt` and `data/real/` in. `.dockerignore` keeps the other
  training checkpoints and the raw corpus out of the upload. The build context is
  your local directory, so run `fly deploy` from a checkout that has the
  checkpoints (they are git-ignored and not on GitHub).
* **No volume.** The store is rebuilt in memory at boot, and the API falls back to
  the baked `seed/real` data if `data/real` is empty.
* **Never scale to zero** (`auto_stop_machines = "off"`, `min_machines_running = 1`):
  a stopped machine costs the next user a full model load.
* **Health grace period 120 s**, longer than a CPU boot.
* `VERITAS_TRUST_PROXY=1` so the rate limiter sees the real client behind Fly's edge.
* CPU torch is installed from PyTorch's CPU index into a virtualenv; installing
  it with `pip --prefix` would pull the 2.5 GB CUDA build instead.

Kafka on Fly — point the API at a managed cluster:

```bash
fly secrets set VERITAS_KAFKA_BOOTSTRAP=<host:9092> \
  VERITAS_KAFKA_SECURITY_PROTOCOL=SASL_SSL VERITAS_KAFKA_SASL_MECHANISM=SCRAM-SHA-256 \
  VERITAS_KAFKA_USERNAME=<user> VERITAS_KAFKA_PASSWORD=<password>
```

Run the poller as a second Fly app from the same image with
`python -m scripts.run_worker --poll`, and exactly one machine.

---

## 3. Frontend — Vercel

```bash
cd frontend
vercel link                                        # once
vercel env add VERITAS_API_URL production          # https://<unique-app-name>.fly.dev
vercel deploy --prod
```

`vercel.json` runs `node build-config.mjs`, which writes `config.js` from
`VERITAS_API_URL` and **fails the build** for a non-HTTPS URL (an HTTPS page
cannot call an HTTP API). `index.html` loads `config.js`; served by FastAPI, the
same file comes from the `/config.js` route with an empty value (same origin).

Then allow the Vercel origin on the backend, or the browser blocks every call:

```bash
fly secrets set VERITAS_ALLOWED_ORIGIN=https://<project>.vercel.app
```

**Voice needs HTTPS.** Vercel and Fly both provide it.

---

## 4. Postgres and Redis — not wired yet

No code reads `VERITAS_DATABASE_URL` or `VERITAS_REDIS_URL`; earlier versions of
the Compose file started both services anyway. They are out of the stack until
the store has a Postgres backend. `docker/initdb/01_schema.sql` is the design
for it:

* `valid_range TSTZRANGE` + GiST index — the as-of query is an index scan.
* `EXCLUDE USING gist (entity =, attribute =, source_id =, valid_range &&)` — one
  source cannot assert overlapping values for the same fact.
* `pgvector` beside the facts, so retrieval and provenance share a transaction.

---

## 5. Before you call it production

| item | status | action |
|---|---|---|
| Auth on writes | ✅ `VERITAS_API_KEY` | set it as a Fly secret |
| Rate limiting | ✅ per-client, in-process | Redis or the edge limiter beyond one replica |
| CORS | ✅ pinned by env | add the Vercel origin |
| Secrets | ✅ env / `fly secrets` | never commit `.env` |
| Persistence | ⚠️ in memory; Kafka replay rebuilds ingested docs | Postgres backend for the store |
| Observability | ⚠️ `/health`, `/stats`, logs | `/metrics` + Prometheus |
| HTTPS | ✅ via platform | required for voice |

Domain metrics matter more than latency: evidence coverage, abstention rate,
freshness lag, conflicts per day, consumer errors. A system can get faster and
quietly stop citing anything.

---

## 6. Cost, roughly

| tier | setup | ~monthly |
|---|---|---|
| demo | Fly shared-cpu 2x / 4 GB + Vercel free | ~$25 |
| with Kafka | + managed Kafka (smallest tier) + poller machine | ~$50–100 |
| GPU | Fly GPU machine | billed per hour while running |

CPU inference is fine for the 14M-parameter model.
