.PHONY: help data test verify audit eval notebooks api web docker kafka deploy-api deploy-web clean

help:
	@echo "VERITAS"
	@echo "  make data        fetch real data (SEC, Wikidata CEOs, Gutenberg, live feeds)"
	@echo "  make test        23 unit/integration/API/streaming tests"
	@echo "  make verify      22 answer checks against real SEC filings and Wikidata"
	@echo "  make audit       ground-truth audit against a running API (VERITAS_API=...)"
	@echo "  make eval        baseline ablation ladder"
	@echo "  make notebooks   execute all 8 notebooks headless"
	@echo "  make api         run the API + UI locally (http://localhost:8000)"
	@echo "  make web         serve the UI alone on http://localhost:3000 (needs make api)"
	@echo "  make docker      build + run the full stack"
	@echo "  make kafka       full stack including the streaming pipeline"
	@echo "  make deploy-api  deploy the backend to Fly.io"
	@echo "  make deploy-web  deploy the frontend to Vercel"

data:
	python scripts/fetch_real_data.py

test:
	python tests/test_smoke.py

verify:
	python scripts/verify_real.py

audit:
	python scripts/audit.py

eval:
	python scripts/run_eval.py --synthetic 5 --checkpoint checkpoints/sft.pt

notebooks:
	python scripts/run_notebook.py notebooks/*.ipynb

api:
	python run_veritas.py

web:
	python -m http.server 3000 --directory frontend

docker:
	docker compose up -d --build
	@echo "waiting for health..." && sleep 40 && curl -sf localhost:8000/health

kafka:
	VERITAS_KAFKA_BOOTSTRAP=kafka:9092 docker compose --profile kafka up -d --build

kafka-smoke:
	python scripts/kafka_smoke.py

# Preflight first: it blocks on a missing checkpoint, a page that never loads
# config.js, and an unpullable Kafka image -- each of which broke a deploy.
deploy-api:
	python scripts/preflight.py
	fly deploy --remote-only

# Backend on Hugging Face Spaces (Docker, 16 GB RAM). Docker Spaces on cpu-basic
# need HF PRO (a free account gets 402). Uses `hf auth login` or HF_TOKEN (write).
#   make deploy-hf SPACE=<hf-user>/veritas
deploy-hf:
	python scripts/preflight.py
	python scripts/deploy_hf_space.py --space $(SPACE)

# Free frontend: Vercel. VERITAS_API_URL is the backend's https URL; it is
# written into config.js at build time. Needs VERCEL_TOKEN or `vercel login`.
#   make deploy-web VERITAS_API_URL=https://<hf-user>-veritas.hf.space
deploy-web:
	python scripts/preflight.py
	cd frontend && npx --yes vercel@latest deploy --prod --yes \
	  --build-env VERITAS_API_URL=$(VERITAS_API_URL) $(if $(VERCEL_TOKEN),--token $(VERCEL_TOKEN),)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
