# VERITAS frontend

Static React (no build step) that talks to the VERITAS API.

## Why the backend cannot go on Vercel

Vercel's serverless functions have no GPU, a 250 MB unzipped bundle limit and
cold starts measured in seconds. VERITAS loads a PyTorch model plus the real
SEC corpus at startup (~20-30 s) and holds the indices in memory across
requests. That is a *stateful, long-lived* process — the opposite of what
serverless is for.

So: **frontend on Vercel, backend on a container host.**

| piece | where | why |
|---|---|---|
| this page | Vercel / Netlify / Cloudflare Pages | static, global CDN, free |
| API + model | Fly.io, Railway, Render, or any VM | needs persistent memory and (ideally) a GPU |
| Postgres | Neon, Supabase, or RDS | `tstzrange` + GiST + pgvector |
| Redis | Upstash | answer cache |
| Kafka | Redpanda Cloud / Confluent / self-hosted | optional; the bus falls back in-process |

## Deploy

```bash
# 1. backend first — you need its URL
fly deploy                      # or: railway up

# 2. point the frontend at it and ship
cd frontend
echo 'window.VERITAS_API="https://your-api.fly.dev";' > config.js
vercel --prod
```

`index.html` reads `window.VERITAS_API` when present and falls back to
same-origin, so the identical file works both behind the FastAPI server
(`python run_veritas.py`) and on Vercel.

## Migrating to Next.js

The components port unchanged; wrap them in `app/page.tsx` and generate the
API types from the FastAPI OpenAPI schema:

```bash
npx openapi-typescript https://your-api.fly.dev/openapi.json -o types/api.ts
```

That makes a backend field rename a frontend compile error rather than a blank
panel in production — worth doing once the `Answer` shape stops changing.
