.PHONY: help setup demo up down serve logs clean train train-gpu train-local test-api setup-data

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================
# Setup
# ============================================================

setup: ## Create .env from example and build all containers
	@test -f .env || cp .env.example .env
	docker compose build
	@echo ""
	@echo "Done! Next: run 'make setup-data' then 'make demo'"

setup-data: ## Copy local data from DeepBiocharge into this project
	./scripts/setup_local_data.sh

# ============================================================
# Demo (Full Stack)
# ============================================================

demo: ## Start ALL services (Prefect + MLflow + Inference + Prometheus + Grafana)
	docker compose up -d
	@echo ""
	@echo "=== All Services Running ==="
	@echo "Prefect UI:    http://localhost:4200"
	@echo "MLflow UI:     http://localhost:5000"
	@echo "FastAPI Docs:  http://localhost:8000/docs"
	@echo "Grafana:       http://localhost:3000  (admin/admin)"
	@echo "Prometheus:    http://localhost:9090"
	@echo ""
	@echo "Next: run 'make train-local' to train with local data"

# ============================================================
# Services
# ============================================================

up: ## Start core services only (Prefect + MLflow)
	docker compose up -d prefect mlflow

down: ## Stop all services
	docker compose down

serve: ## Start inference API
	docker compose up -d inference

logs: ## Tail logs from all running services
	docker compose logs -f

# ============================================================
# Training
# ============================================================

train-local: ## Train model using local data (no API pull)
	docker compose run --rm prefect python -m src.pipelines.local_train_flow

train: ## Run model training via Prefect (CPU)
	docker compose --profile training run --rm trainer

train-gpu: ## Run model training with GPU
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
		--profile training run --rm trainer

# ============================================================
# Testing
# ============================================================

test-api: ## Test inference API endpoints
	@echo "=== Health Check ==="
	@curl -s http://localhost:8000/health | python3 -m json.tool
	@echo ""
	@echo "=== Model Info ==="
	@curl -s http://localhost:8000/model-info | python3 -m json.tool
	@echo ""
	@echo "=== Metrics ==="
	@curl -s http://localhost:8000/metrics

# ============================================================
# Cleanup
# ============================================================

clean: ## Remove generated data, models, and MLflow artifacts
	rm -rf models/checkpoints/* mlflow/
	@echo "Cleaned checkpoints and MLflow artifacts"
