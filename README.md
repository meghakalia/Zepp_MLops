# Biocharge MLOps

End-to-end MLOps pipeline for biocharge (wellness) prediction. Pulls wearable data from Xiaomi/Huami APIs, trains a PyTorch neural network, serves predictions via REST API, and monitors results in Grafana/MLflow.

---

## Quick Start

### 1. Prerequisites

- Docker + Docker Compose
- Model checkpoint in `models/production/dual_head_no_weighted_loss/`
- Z-score file at `data/source/updated_z_score_user.json`
- Processed user Excel files in `data/source/z_norm_rhr_hrv_7_14_corrected_complete_original/`

### 2. Start all services

```bash
docker compose up -d
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Frontend | open `frontend/index.html` in browser | — |
| FastAPI | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Prefect | http://localhost:4200 | — |

### 3. Open the frontend

Just open the file directly in your browser — no server needed:

```bash
open frontend/index.html
```

### 4. Use the UI

1. Enter **User IDs** as a JSON array, e.g. `["1000836634", "1006877628"]`
2. Set **From Date** and **To Date** (YYYY-MM-DD)
3. Click **Pull Data** — fetches raw wearable data for those users and dates
4. Click **Run Inference** — runs the ML model on the pulled data and logs results to MLflow
5. Check **Save plots to MLflow** before running inference to also attach trajectory plot PNGs to the MLflow run
6. Click **MLflow** or **Grafana** to open those dashboards

> Pull and inference are separate steps. Always pull first, then run inference.

---

## Starting the Backend and Frontend (Step by Step)

### Backend (FastAPI inference server)

**Option A — Docker (recommended)**

1. Copy the environment file and fill in any required values:
   ```bash
   cp .env.example .env
   ```
2. Start the inference backend (and supporting services):
   ```bash
   docker compose up -d
   ```
3. Confirm the server is healthy:
   ```bash
   curl http://localhost:8000/health
   ```
   You should see `{"status":"ok", ...}`.

**Option B — Local Python (no Docker)**

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set required environment variables (or copy `.env.example` to `.env` and source it):
   ```bash
   cp .env.example .env
   # edit .env with your model path and data paths, then:
   source .env
   ```
3. Start the FastAPI server:
   ```bash
   uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
   ```
4. Confirm the server is healthy:
   ```bash
   curl http://localhost:8000/health
   ```

---

### Frontend

The frontend is a single static HTML file — no build step or server required.

1. Make sure the backend is running on `http://localhost:8000` (see above).
2. Open the frontend in your browser:
   ```bash
   open frontend/index.html        # macOS
   # or double-click the file in Finder / File Explorer
   ```
3. The UI will load and connect to the backend automatically.

> If you see CORS or connection errors, check that the backend is running and accessible on port 8000.

---

## Project Structure

```
biocharge-mlops/
├── frontend/
│   └── index.html                  # Single-page dashboard UI
│
├── src/
│   ├── data_ingestion/             # Pull raw data from Xiaomi/Huami API
│   │   ├── pull_biocharge_data.py
│   │   └── puller.py
│   │
│   ├── training/                   # Model architecture, dataset, training loop
│   │   ├── model.py
│   │   ├── dataset_delta_v2.py
│   │   ├── trainer.py
│   │   └── utils.py
│   │
│   ├── inference/                  # Prediction serving
│   │   ├── api.py                  # FastAPI app (all REST endpoints)
│   │   ├── predictor.py            # Model loading + feature normalisation
│   │   ├── fresh_inference.py      # End-to-end pipeline: pull → ground truth → predict
│   │   ├── run_inference.py        # Autoregressive day-by-day inference
│   │   ├── plotting.py             # Trajectory plot generation
│   │   └── charge_analytics/       # Domain ground truth computation
│   │
│   └── pipelines/                  # Prefect orchestration flows
│       ├── local_train_flow.py
│       ├── train_flow.py
│       ├── data_flow.py
│       ├── inference_flow.py
│       └── full_flow.py
│
├── models/
│   └── production/                 # Model checkpoints and config
│
├── data/
│   ├── raw/                        # Raw pulled data (CSV per user/date)
│   ├── source/                     # Processed Excel files + z-score JSON
│   └── plots/                      # Saved trajectory PNGs
│
├── monitoring/
│   ├── prometheus/                 # Scrape config
│   └── grafana/                    # Dashboard + provisioning
│
├── config/
│   └── local_data.yaml             # Local training config
│
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## Component Overview

| Component | Description |
|-----------|-------------|
| **frontend/index.html** | Minimal single-page UI. Pull data and run inference for a set of users over a date range. Links to MLflow and Grafana. |
| **src/data_ingestion** | Pulls raw health metrics (HR, HRV, sleep, activity) from the Xiaomi/Huami API. Supports online (API) and offline (local file) modes. |
| **src/training/model.py** | PyTorch neural networks. `MLP_delta` is the primary model; `GatedDualHeadMLP` adds separate heads for physical and mental biocharge. |
| **src/training/dataset_delta_v2.py** | Dataset pipeline — z-score normalisation, windowed features, positional encoding, augmentation. |
| **src/training/trainer.py** | Training loop with MLflow logging, early stopping, and checkpointing. |
| **src/inference/api.py** | FastAPI server. Key endpoints: `/pull` (fetch data), `/batch-inference` (run model on pulled data), `/mlflow`, `/grafana`. |
| **src/inference/predictor.py** | Loads a model checkpoint and handles feature normalisation for inference. |
| **src/inference/fresh_inference.py** | Orchestrates the full per-user pipeline: pull → compute ground truth → run ML inference. |
| **src/inference/charge_analytics** | Analytical ground truth engine — computes mental/physical battery, HRV scores, sleep metrics from raw wearable data. |
| **src/pipelines** | Prefect flows for scheduled data ingestion, training, and inference. |
| **MLflow** | Tracks every inference run — metrics (MAE, RMSE per region), params, and optional trajectory plot artifacts. Experiment name: `biocharge-inference`. |
| **Grafana** | Live dashboards fed by Prometheus. Shows service health, aggregate region MAE, quality over time, and per-user breakdown. |
| **Prometheus** | Scrapes `/metrics` on the inference API every 15 s. Stores time-series for Grafana. |
| **Prefect** | Pipeline orchestration with retry logic for API calls. UI at port 4200. |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/pull` | POST | Fetch raw wearable data for a list of users |
| `/batch-inference` | POST | Run inference on already-pulled data (no re-pull) |
| `/batch-fresh-inference` | POST | Pull + inference in one call |
| `/fresh-inference` | POST | Full pipeline for a single user |
| `/fresh-inference/plot` | POST | Same, returns trajectory PNG |
| `/autoregressive` | POST | Autoregressive inference for a user + date range |
| `/model-info` | GET | Loaded model type, device, config |
| `/metrics` | GET | Prometheus metrics |
| `/mlflow` | GET | Redirects to MLflow UI |
| `/grafana` | GET | Redirects to Grafana UI |

---

## Training

```bash
# Train from local data (no API keys needed)
make train-local

# Train via Prefect (requires API credentials in .env)
make train

# Train with GPU
make train-gpu
```

---

## Common Commands

```bash
make demo          # Start full stack
make down          # Stop all services
make serve         # Start inference API only
make test-api      # Smoke test all endpoints
make setup-data    # Copy local data from DeepBiocharge project
```

---

## Rebuild After Code Changes

Since `src/` is mounted as a live volume, most Python changes are picked up by restarting the container — no rebuild needed:

```bash
docker compose restart inference
```

Only rebuild if you change `Dockerfile` or add new dependencies:

```bash
docker compose build inference && docker compose up -d inference
```
