.PHONY: help setup up down pull-data train serve logs clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================
# Setup
# ============================================================

setup: ## Create .env from example and build containers
	@test -f .env || cp .env.example .env
	@echo "Edit .env with your API tokens, then run: make up"
	docker compose build

# ============================================================
# Services
# ============================================================

up: ## Start Prefect + MLflow
	docker compose up -d prefect mlflow

down: ## Stop all services
	docker compose down

serve: ## Start inference API
	docker compose up -d inference

logs: ## Tail logs from all running services
	docker compose logs -f

# ============================================================
# Pipeline Commands
# ============================================================

pull-data: ## Run data ingestion pipeline
	docker compose run --rm prefect python -m src.pipelines.data_flow

train: ## Run model training (CPU)
	docker compose --profile training run --rm trainer

train-gpu: ## Run model training (GPU)
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
		--profile training run --rm trainer

full-pipeline: ## Run end-to-end pipeline (pull data + train)
	docker compose run --rm prefect python -m src.pipelines.full_flow

# ============================================================
# Monitoring (Phase 2)
# ============================================================

monitoring-up: ## Start Prometheus + Grafana
	docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d prometheus grafana

# ============================================================
# Development
# ============================================================

test-api: ## Test inference API health endpoint
	@curl -s http://localhost:8000/health | python -m json.tool

clean: ## Remove data, models, and MLflow artifacts
	rm -rf data/raw/* data/processed/* models/checkpoints/* mlflow/
	@echo "Cleaned data, models, and MLflow artifacts"
