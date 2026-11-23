.PHONY: install dev api web test lint eval eval-dense docker

install:            ## Python + web dependencies
	pip install -r requirements-dev.txt
	cd frontend && npm ci

api:                ## FastAPI on :8000 (hot reload)
	uvicorn backend.main:app --reload --port 8000

web:                ## Vite dev server on :3000 (proxies /api to :8000)
	cd frontend && npm run dev

test:               ## Full offline test suite
	pytest --cov=backend --cov-report=term-missing:skip-covered

lint:
	ruff check . && ruff format --check backend evals tests
	cd frontend && npm run typecheck

eval:               ## Retrieval ablations on the held-out split
	python -m evals.retrieval_eval --split test --failures

eval-dense:         ## Same, including the Gemini embedding channel (needs GOOGLE_API_KEY)
	python -m evals.retrieval_eval --split test --dense

docker:             ## Single container: API + web app on :8000
	docker compose up --build
