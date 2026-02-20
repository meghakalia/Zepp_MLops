# Biocharge MLOps — Inference Guide

Step-by-step instructions for running biocharge inference using the dual-head model.

## Prerequisites

1. **Python 3.12+**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Model checkpoint** at `models/production/dual_head_no_weighted_loss/` containing:
   - `model_config.json`
   - `best_model_*.pt`
4. **Z-score file** at `data/source/updated_z_score_user.json`

---

## Option A: Docker (recommended)

Runs the full stack: FastAPI + Prometheus + Grafana.

### Start

```bash
docker compose build inference
docker compose up -d
```

| Service | URL | Credentials |
|---------|-----|-------------|
| FastAPI | http://localhost:8000/docs | — |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| MLflow | http://localhost:5000 | — |

### Stored data (processed Excel already exists)

Place `{user_id}_processed.xlsx` in `./data/source/z_norm_rhr_hrv_7_14_corrected_complete_original/` (mounted to `/app/data/` inside the container), then call:

```bash
# Single user
curl -X POST http://localhost:8000/fresh-inference \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "1000836634",
    "from_date": "2025-03-01",
    "to_date": "2025-03-03",
    "data_dir": "/app/data/source/z_norm_rhr_hrv_7_14_corrected_complete_original",
    "skip_pull": true,
    "skip_ground_truth": true
  }'

# Up to 3 users (batch)
curl -X POST http://localhost:8000/batch-fresh-inference \
  -H "Content-Type: application/json" \
  -d '{
    "user_ids": ["1000836634", "1000263101", "1001234567"],
    "from_date": "2025-03-01",
    "to_date": "2025-03-03",
    "data_dir": "/app/data/source/z_norm_rhr_hrv_7_14_corrected_complete_original",
    "skip_pull": true,
    "skip_ground_truth": true,
    "per_user_metrics": true
  }'
```

`per_user_metrics: true` exposes per-user region errors in Grafana (see [Grafana section](#grafana-dashboard)).

### Rebuild after code changes

```bash
docker compose build inference && docker compose up -d inference
```

---

## Option B: CLI (local)

### Full pipeline (pulls data from API)

```bash
python -m src.inference.fresh_inference \
    --user_id 1000836634 \
    --from_date 2025-03-01 \
    --to_date 2025-03-15 \
    --model_dir models/production/dual_head_no_weighted_loss \
    --pull_mode ONLINE
```

Requires `XIAOMI_SUPERTOKEN` (and optionally `XIAOMI_TOKEN`) in your environment.

### Skip data pull (raw data already in `./data/raw/`)

```bash
python -m src.inference.fresh_inference \
    --user_id 1000836634 \
    --from_date 2025-03-15 \
    --to_date 2025-03-16 \
    --model_dir models/production/dual_head_no_weighted_loss \
    --skip_pull
```

### Skip pull + ground truth (processed Excel already exists)

```bash
python -m src.inference.fresh_inference \
    --user_id 1000836634 \
    --from_date 2025-03-15 \
    --to_date 2025-03-16 \
    --model_dir models/production/dual_head_no_weighted_loss \
    --data_dir data/source/z_norm_rhr_hrv_7_14_corrected_complete_original \
    --skip_pull \
    --skip_ground_truth
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--user_id` | `1000836634` | User ID |
| `--from_date` | `2025-03-15` | Start date (YYYY-MM-DD) |
| `--to_date` | `2025-03-16` | End date (YYYY-MM-DD) |
| `--model_dir` | `models/production/dual_head_no_weighted_loss` | Model checkpoint directory |
| `--data_dir` | `./data` | Directory containing `{user_id}_processed.xlsx` |
| `--zscores_file` | `data/source/updated_z_score_user.json` | Z-score normalization JSON |
| `--pull_mode` | `OFFLINE` | `ONLINE` (API) or `OFFLINE` (local files) |
| `--skip_pull` | off | Skip data pull stage |
| `--skip_ground_truth` | off | Skip ground truth computation |
| `--plot` | off | Save trajectory plot PNG to `--output_dir` |
| `--output_dir` | `outputs` | Plot output directory |
| `--verbose` | off | Enable debug logging |

---

## Option C: Python API

```python
from src.inference.fresh_inference import run_fresh_inference

results = run_fresh_inference(
    user_id="1000836634",
    from_date="2025-03-01",
    to_date="2025-03-15",
    model_dir="models/production/dual_head_no_weighted_loss",
    data_dir="data/source/z_norm_rhr_hrv_7_14_corrected_complete_original",
    skip_pull=True,
    skip_ground_truth=True,
)

print(results["mae"], results["rmse"])
print(results["region_errors"])   # exercise, sleep, nap, non_wear, day_start, day_end
```

Returned dict keys: `user_id`, `dates`, `num_samples`, `preds`, `targets`, `mae`, `mse`, `rmse`, `pred_range`, `target_range`, `processed_excel_path`, `region_errors`, `plot_path` (if plot=True).

---

## FastAPI Endpoints

Start locally (outside Docker):

```bash
MODEL_CHECKPOINT_DIR=models/production/dual_head_no_weighted_loss \
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model loaded status |
| `/model-info` | GET | Model type, device, config |
| `/metrics` | GET | Prometheus metrics |
| `/predict` | POST | Single feature vector → delta |
| `/batch` | POST | Batch feature vectors → deltas |
| `/autoregressive` | POST | Autoregressive inference for user + dates |
| `/autoregressive/plot` | POST | Same, returns PNG |
| `/fresh-inference` | POST | Full pipeline: pull → ground truth → inference (single user) |
| `/fresh-inference/plot` | POST | Same, returns trajectory PNG |
| `/batch-fresh-inference` | POST | Full pipeline for up to 3 users, returns aggregate + per-user metrics |

### Get trajectory plot

```bash
curl -X POST http://localhost:8000/fresh-inference/plot \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "1000836634",
    "from_date": "2025-03-15",
    "to_date": "2025-03-16",
    "data_dir": "data/source/z_norm_rhr_hrv_7_14_corrected_complete_original",
    "skip_pull": true,
    "skip_ground_truth": true
  }' --output trajectory_plot.png
```

---

## Grafana Dashboard

Open http://localhost:3000 (admin / admin). The **Biocharge Inference Dashboard** has four sections:

1. **Service Health** — model status, call counts, errors, latency
2. **Aggregate Region MAE** — bargauge showing mean MAE across all users and all dates for each event type (exercise, sleep, nap, non-wear, day-start, day-end, overall trajectory)
3. **Aggregate Quality Over Time** — timeseries of overall MAE/RMSE and key region MAEs
4. **Per-User Region MAE** — use the **User** dropdown at the top to select one or more users; the bargauge updates to show per-user breakdown

The per-user section is populated after a `/batch-fresh-inference` call with `per_user_metrics: true`.

### Prometheus metrics emitted

**Labelled (multi-user):**

```
biocharge_region_mae{region="exercise|sleep|nap|non_wear|day_start|day_end|overall_traj", user_id="<id>|all"}
biocharge_user_inference_mae{user_id="<id>|all"}
biocharge_user_inference_rmse{user_id="<id>|all"}
```

`user_id="all"` is the mean across all users in the last batch run.

**Scalar (single-user / backward compat):**

```
biocharge_fresh_inference_mae/rmse/mse
biocharge_exercise_mae, biocharge_sleep_mae, biocharge_nap_mae
biocharge_nonwear_mae, biocharge_overall_traj_mae
biocharge_day_start_mae, biocharge_day_end_mae
biocharge_fresh_data_pull_seconds, biocharge_fresh_ground_truth_seconds, biocharge_fresh_ml_inference_seconds
```

---

## Pipeline Stages

The fresh inference pipeline runs up to 4 stages:

1. **Data Pull** — Fetches raw wearable data from Xiaomi/Huami API (skippable with `--skip_pull` / `skip_pull: true`)
2. **Ground Truth** — Computes analytical biocharge values via the charge analytics engine (skippable with `--skip_ground_truth` / `skip_ground_truth: true`)
3. **ML Inference** — Runs autoregressive prediction with GatedDualHeadMLP, computes region-based errors
4. **Plot** (optional) — Generates trajectory curves with exercise/sleep/nap/non-wear annotations

---

## Helper Commands

### Check what's running

```bash
# All Docker services and their status
docker compose ps

# Which ports are in use on the host
lsof -i -P -n | grep LISTEN | grep -E "8000|9090|3000|5000|4200"

# Logs for a specific service (last 50 lines)
docker compose logs inference --tail=50
docker compose logs grafana --tail=50
docker compose logs prometheus --tail=50
```

### Health checks

```bash
# FastAPI
curl -s http://localhost:8000/health

# Which model is loaded
curl -s http://localhost:8000/model-info | python3 -m json.tool

# Check Prometheus is scraping correctly
curl -s http://localhost:9090/api/v1/targets | python3 -c "
import sys, json
targets = json.load(sys.stdin)['data']['activeTargets']
for t in targets: print(t['labels']['job'], '->', t['health'])
"
```

### Stop services

```bash
# Stop everything (keeps volumes)
docker compose down

# Stop a single service
docker compose stop inference

# Stop and remove ALL containers + volumes (full reset)
docker compose down -v
```

### Restart a service

```bash
# Restart just inference (picks up env var changes, no rebuild needed)
docker compose restart inference

# Restart the full monitoring stack
docker compose restart prometheus grafana
```

### When to rebuild vs restart

| Situation | Command |
|-----------|---------|
| Changed Python code (`src/`) | `docker compose build inference && docker compose up -d inference` |
| Changed `docker-compose.yml` env vars only | `docker compose up -d inference` (no rebuild) |
| Changed `monitoring/grafana/dashboards/biocharge.json` | Just reload in browser — volume is mounted live |
| Changed `monitoring/prometheus/prometheus.yml` | `docker compose restart prometheus` |
| Changed `Dockerfile` | `docker compose build inference && docker compose up -d inference` |
| Added new Python dependency | Add to `Dockerfile` RUN block, then rebuild |

### Rebuild a single service

```bash
# Rebuild inference only (faster than rebuilding everything)
docker compose build inference

# Rebuild + restart in one step
docker compose build inference && docker compose up -d inference

# Force clean rebuild (no cache)
docker compose build --no-cache inference
```

### Start individual services

```bash
# Start just the inference API + monitoring (skip Prefect + MLflow)
docker compose up -d inference prometheus grafana

# Start full stack
docker compose up -d

# Start with logs attached (Ctrl+C to stop)
docker compose up inference
```

### Kill a local uvicorn process (if running outside Docker)

```bash
# Find and kill the process on port 8000
kill $(lsof -ti :8000)

# Or by process name
pkill -f "uvicorn src.inference.api"
```

---

## Directory Structure

```
models/production/dual_head_no_weighted_loss/
    model_config.json           # Model architecture + training config
    best_model_*.pt             # Trained weights
data/
    source/
        updated_z_score_user.json                          # Z-score normalization
        z_norm_rhr_hrv_7_14_corrected_complete_original/   # Processed Excel files
            {user_id}_processed.xlsx
    raw/                                                   # Raw pulled data (Stage 1)
        user_score_data/
        user_sleep_data/
outputs/                        # Plot output directory
monitoring/
    prometheus/prometheus.yml   # Scrapes inference:8000/metrics every 15s
    grafana/dashboards/         # Auto-provisioned dashboard
```
