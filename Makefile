.PHONY: help install install-dev lint format test test-cov docs-build docs-serve api-run docker-build docker-up docker-down clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and core dependencies
	pip install -e .

install-dev:  ## Install with development dependencies
	pip install -e ".[dev,docs]"
	pre-commit install

lint:  ## Run linters (ruff + mypy)
	ruff check api/src/
	mypy api/src/

format:  ## Auto-format code with black and ruff
	black api/src/ api/tests/
	ruff check --fix api/src/

test:  ## Run test suite
	pytest api/tests/ -v

test-cov:  ## Run tests with coverage report
	pytest api/tests/ --cov=api/src --cov-report=html --cov-report=term-missing -v

docs-build:  ## Build documentation
	mkdocs build

docs-serve:  ## Serve documentation locally
	mkdocs serve

api-run:  ## Run API development server
	uvicorn api.src.main:app --reload --host 0.0.0.0 --port 8000

docker-build:  ## Build Docker images
	docker-compose build

docker-up:  ## Start Docker services
	docker-compose up -d

docker-down:  ## Stop Docker services
	docker-compose down

clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/ site/
