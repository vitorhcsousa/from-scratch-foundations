.PHONY: help init install install-dev lint format type-check test test-cov \
       run-demo \
       pre-commit-install pre-commit-run \
       docker-build docker-run \
       docs docs-build docs-deploy \
       ci clean

PYTHON := uv run python
APP    := uv run foundations

# ── Default ──────────────────────────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────────

init: ## Bootstrap dev environment (uv sync + pre-commit hooks)
	uv sync
	uv run pre-commit install

install: ## Install production dependencies
	uv sync --no-dev

install-dev: ## Install all dependencies (including dev)
	uv sync

# ── Code Quality ─────────────────────────────────────────────────────────────

lint: ## Run ruff linter
	uv run ruff check src/ tests/

format: ## Format code with ruff
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

type-check: ## Run ty type checker
	uv run ty check src/

# ── Tests ────────────────────────────────────────────────────────────────────

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage report
	uv run pytest --cov=foundations --cov-report=term-missing --cov-report=html

# ── Run ──────────────────────────────────────────────────────────────────────

run-demo: ## Demo: list packaged project modules
	$(APP) projects list

# ── Pre-commit ───────────────────────────────────────────────────────────────

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

pre-commit-run: ## Run pre-commit on all files
	uv run pre-commit run --all-files

# ── Docker (optional) ────────────────────────────────────────────────────────

IMAGE_NAME := from-scratch-foundations
IMAGE_TAG  := latest

docker-build: ## Build Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run: ## Run Docker container (CLI)
	docker run --rm -it \
		-v $(PWD)/docs:/app/docs \
		-v $(PWD)/projects:/app/projects \
		-v $(PWD)/experiments:/app/experiments \
		-v $(PWD)/anki:/app/anki \
		-v $(PWD)/artifacts:/app/artifacts \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		projects list

# ── CI ───────────────────────────────────────────────────────────────────────

ci: lint type-check test ## Run full CI pipeline locally

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove generated artifacts
	rm -rf .pytest_cache htmlcov .ruff_cache __pycache__ dist .mypy_cache site/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ── Docs ─────────────────────────────────────────────────────────────────

docs: ## Serve docs locally (live-reload)
	uv run mkdocs serve

docs-build: ## Build static docs site
	uv run mkdocs build --strict

docs-deploy: ## Deploy docs to GitHub Pages
	uv run mkdocs gh-deploy --force