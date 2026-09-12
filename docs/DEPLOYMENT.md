# Deployment

## The one constraint that decides the topology

VERITAS holds a PyTorch model, a vector index, a BM25 index and a bitemporal
store **in memory**, and loads them once at startup (~20–30 s). It is a
**stateful, long-lived process**.

That rules out putting the backend on Vercel, Netlify Functions or Lambda:
serverless has no GPU, a bundle-size cap in the hundreds of MB, and pays the
30-second load on every cold start. So:

```
   Browser ── Vercel (static React) ──HTTPS──▶ Container host (FastAPI + model)
                                                    │
                                    ┌───────────────┼───────────────┐
                                Postgres          Redis           Kafka
                            (tstzrange+pgvector) (cache)      (optional bus)
```

| component | where | why |
|---|---|---|
| frontend | Vercel / Cloudflare Pages | static, global CDN, free tier is enough |
| API + model | Fly.io, Railway, Render, or a VM | needs persistent memory, ideally a GPU |
| Postgres | Neon, Supabase, RDS | `tstzrange` + GiST + `pgvector` |
| Redis | Upstash, ElastiCache | answer cache only — losing it costs latency, never correctness |
| Kafka | Redpanda Cloud, Confluent, self-hosted | optional; the bus falls back in-process |

---

## 1. Local — everything in Docker

```bash
cp .env.example .env            # set POSTGRES_PASSWORD
docker compose up -d            # API + Postgres + Redis
curl localhost:8000/health      # {"ready":true,...}
open http://localhost:8000
```

With the streaming pipeline:

```bash
docker compose --profile kafka up -d
docker compose logs -f worker-ingest
```

Kafka sits behind a profile because the system is **correct without it** —
`make_bus()` returns an in-process queue when `VERITAS_KAFKA_BOOTSTRAP` is
unset. Turn it on when you want a replayable ingestion log and independent
scaling of the embedding stage.

Prerequisite either way: `checkpoints/tokenizer.json` and a `.pt` file, i.e.
run notebooks 01–04 first. The compose file mounts `./checkpoints` read-only
rather than baking weights into the image, so you can swap a model without a
rebuild.

---

## 2. Backend — Fly.io (recommended)

Fly gives persistent machines, a real filesystem, and optional GPUs, which is
exactly the shape of this workload.

```bash
fly launch --no-deploy --name veritas-api
fly volumes create veritas_data --size 10 --region iad
fly secrets set POSTGRES_PASSWORD=... VERITAS_DATABASE_URL=postgres://...
fly deploy
```

`fly.toml`:

```toml
app = "veritas-api"
primary_region = "iad"

[build]
  dockerfile = "docker/Dockerfile.api"

[env]
  VERITAS_DEVICE = "cpu"          # "cuda" on a GPU machine

[[mounts]]
  source = "veritas_data"
  destination = "/app/data"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = false      # NEVER auto-stop: a stop costs a 30s reload
  min_machines_running = 1

  [http_service.concurrency]
    type = "requests"
    soft_limit = 20
    hard_limit = 40

[[vm]]
  memory = "4gb"                  # model + indices; 8gb once the corpus grows
  cpu_kind = "shared"
  cpus = 2

[checks.health]
  type = "http"
  path = "/health"
  interval = "30s"
  timeout = "10s"
  grace_period = "90s"            # must exceed model load time
```

**`auto_stop_machines = false` matters.** Scale-to-zero is the default on most
platforms and it is wrong here: every cold start reloads the model and re-indexes
the corpus, so the first user after an idle period waits 30 seconds.

Railway/Render equivalents: set a health-check grace period ≥ 90 s, disable
scale-to-zero, attach a volume at `/app/data`.

---

## 3. Frontend — Vercel

```bash
cd frontend
echo 'window.VERITAS_API="https://veritas-api.fly.dev";' > config.js
vercel --prod
```

Add `<script src="/config.js"></script>` before the React bundle in
`index.html`. The page reads `window.VERITAS_API` and falls back to
same-origin, so the identical file works locally behind FastAPI and on Vercel
with a remote backend.

Then lock CORS down — the dev default is `allow_origins=["*"]`:

```python
# api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("VERITAS_ALLOWED_ORIGIN", "http://localhost:8000")],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

**Voice needs HTTPS.** The Web Speech API is blocked on plain HTTP except on
`localhost`. Vercel gives you TLS automatically; a self-hosted frontend needs a
certificate or the microphone button silently does nothing.

---

## 4. Postgres

```bash
psql "$VERITAS_DATABASE_URL" -f docker/initdb/01_schema.sql
```

The schema is where the bitemporal design becomes a database guarantee rather
than a convention:

* `valid_range TSTZRANGE` + a GiST index — the as-of query
  (`valid_range @> $t`) is an index scan, not a table scan.
* `EXCLUDE USING gist (entity =, attribute =, source_id =, valid_range &&)` —
  one source physically cannot assert two overlapping values for the same fact.
  Application code can be bypassed; a constraint cannot.
* `pgvector` keeps embeddings beside the facts they came from, so retrieval and
  provenance stay in one transaction and cannot drift apart.

Build the IVFFlat index **after** the bulk load — it needs data to choose
sensible centroids:

```sql
CREATE INDEX chunk_embedding_ivf ON chunk
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
ANALYZE chunk;
```

Managed options: **Neon** (branching is genuinely useful for testing a
migration against production data), **Supabase** (pgvector preinstalled).

---

## 5. Kafka — when and why

Turn it on when any of these is true:

* more than ~50 monitored sources, so one slow fetch stalls the cycle;
* you need to **replay** ingestion — rebuilding the store from the raw log is
  the ingestion-side counterpart of the append-only evidence guarantee;
* embedding throughput needs more than one machine.

```
veritas.raw     ──▶ change detection  (most messages die here)
veritas.changed ──▶ chunk + embed + index   ← scale this consumer group
veritas.facts   ──▶ bitemporal assert + graph + cache invalidation
veritas.changes ──▶ notifications, dashboards
```

Two settings are correctness requirements, not tuning:

* **Partition by entity.** `TemporalStore.assert_fact` classifies an assertion
  by comparing it with what it already holds, so out-of-order delivery turns a
  real succession into a spurious `CONFLICT`. Keying by entity gives per-entity
  ordering.
* **Manual commit after the handler.** With auto-commit, a crash mid-handler
  loses the message. Re-delivery is safe — a repeat lands as `REAFFIRMED` —
  so at-least-once is the right trade.

```bash
kafka-topics.sh --create --topic veritas.raw --partitions 6 \
  --config retention.ms=604800000 --bootstrap-server $BOOTSTRAP
```

Six partitions caps useful parallelism at six consumers; size it to your
expected entity count, not your current one — repartitioning later reshuffles
keys and breaks ordering guarantees during the migration.

Managed: **Redpanda Cloud** (Kafka-compatible, no ZooKeeper, cheaper at this
scale) or **Confluent Cloud**.

---

## 6. Before you call it production

| item | status | action |
|---|---|---|
| Auth | ❌ none | API keys or OAuth on `/ask`, `/ingest` |
| Rate limiting | ❌ none | `slowapi` or the platform's limiter |
| CORS | ⚠️ `*` | pin to the frontend origin |
| Secrets | ✅ env vars | never commit `.env` |
| Persistence | ⚠️ in-memory | run the Postgres migration |
| Observability | ⚠️ logs only | `/metrics` + Prometheus |
| Backups | ❌ | `pg_dump` on a schedule; the store is the system of record |
| HTTPS | ✅ via platform | required for voice |

**Domain metrics matter more than latency here.** Export evidence coverage,
abstention rate, freshness lag, conflicts detected per day and ingest queue
depth. p99 latency tells you nothing about whether answers are still
well-evidenced — a system can get faster and quietly stop citing anything.

---

## 7. Cost, roughly

| tier | setup | ~monthly |
|---|---|---|
| demo | Fly shared-1x 2 GB + Neon free + Vercel free | ~$5 |
| small prod | Fly 4 GB + Neon Launch + Upstash | ~$50 |
| with Kafka | + Redpanda Serverless + 2 workers | ~$150 |
| GPU | Fly A10 (`a10` machine) | ~$1.50/hr while running |

CPU inference is fine for the 14M-parameter model — GPU only becomes worthwhile
once you scale the model up or need to re-embed a large corpus frequently.
