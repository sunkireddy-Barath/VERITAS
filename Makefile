.PHONY: help data test verify eval train-check api docker deploy-api deploy-web clean

help:
	@echo "VERITAS"
	@echo "  make data        fetch real data (SEC, Gutenberg, live feeds)"
	@echo "  make test        11 unit/integration tests"
	@echo "  make verify      7 behaviour checks against real SEC filings"
	@echo "  make eval        baseline ablation ladder"
	@echo "  make notebooks   execute all 8 notebooks headless"
	@echo "  make api         run the API locally (http://localhost:8000)"
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

eval:
	python scripts/run_eval.py --synthetic 5 --checkpoint checkpoints/sft.pt

notebooks:
	python scripts/run_notebook.py notebooks/*.ipynb

api:
	python run_veritas.py

docker:
	docker compose up -d --build
	@echo "waiting for health..." && sleep 40 && curl -sf localhost:8000/health

kafka:
	docker compose --profile kafka up -d --build

deploy-api:
	fly deploy

deploy-web:
	cd frontend && vercel --prod

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
