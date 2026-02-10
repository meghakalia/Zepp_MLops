"""
FastAPI inference server for biocharge prediction.

Endpoints:
    GET  /health              - Health check
    POST /predict             - Single prediction
    POST /batch               - Batch predictions
    POST /autoregressive      - Full autoregressive inference for a user+dates
    POST /autoregressive/plot - Same as above, returns PNG plot
    GET  /model-info          - Current model info
    GET  /metrics             - Prometheus metrics (for Grafana)
"""

# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()
# print("Debugger attached")


import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel

from src.inference.predictor import BiochargePredictor

# import pdb; pdb.set_trace()

logger = logging.getLogger(__name__)

# Global predictor instance
# predictor: BiochargePredictor | None = None

# Simple metrics counters (no prometheus_client dependency needed)
_metrics = {
    "predictions_total": 0,
    "predictions_errors": 0,
    "prediction_latency_sum": 0.0,
    "prediction_latency_count": 0,
    "batch_predictions_total": 0,
    "autoregressive_predictions_total": 0,
    "autoregressive_latency_sum": 0.0,
    "autoregressive_latency_count": 0,
    "last_autoregressive_mae": 0.0,
    "last_autoregressive_rmse": 0.0,
    "last_autoregressive_mse": 0.0,
}
# Default paths (overridable via env vars or request body)
DEFAULT_DATA_DIR = os.environ.get(
    "BIOCHARGE_DATA_DIR",
    "data/source/z_norm_rhr_hrv_7_14_corrected_complete_original",
)
DEFAULT_TORCH_DIR = os.environ.get(
    "BIOCHARGE_TORCH_DIR",
    "data/source/torch_oversampled_hrr_rhr_ema_waso_original/torch",
)
DEFAULT_ZSCORES_FILE = os.environ.get(
    "BIOCHARGE_ZSCORES_FILE",
    "data/source/updated_z_score_user.json",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    print("Lifespan startup: attempting to load model...")  # DEBUG

    checkpoint_dir = os.environ.get("MODEL_CHECKPOINT_DIR")
    mlflow_uri = os.environ.get("MLFLOW_MODEL_URI")

    predictor = None
    if checkpoint_dir:
        print(f"MODEL_CHECKPOINT_DIR set: {checkpoint_dir}")  # DEBUG
        try:
            predictor = BiochargePredictor(checkpoint_dir=checkpoint_dir)
            print("Model loaded successfully.")  # DEBUG
        except Exception as e:
            logger.error("Failed to load model from %s: %s", checkpoint_dir, e)
            print(f"Failed to load model from {checkpoint_dir}: {e}")  # DEBUG
    elif mlflow_uri:
        print(f"MLFLOW_MODEL_URI set: {mlflow_uri}")  # DEBUG
        try:
            predictor = BiochargePredictor(mlflow_model_uri=mlflow_uri)
            print("Model loaded successfully from MLflow.")  # DEBUG
        except Exception as e:
            logger.error("Failed to load model from MLflow %s: %s", mlflow_uri, e)
            print(f"Failed to load model from MLflow {mlflow_uri}: {e}")  # DEBUG
    else:
        logger.warning("No model configured. Set MODEL_CHECKPOINT_DIR or MLFLOW_MODEL_URI.")
        print("No model configured. Set MODEL_CHECKPOINT_DIR or MLFLOW_MODEL_URI.")  # DEBUG

    app.state.predictor = predictor

    yield

    app.state.predictor = None
    print("Lifespan shutdown: predictor set to None.")  # DEBUG


app = FastAPI(
    title="Biocharge Prediction API",
    description="MLP_delta model serving for biocharge delta predictions",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    delta_charge: float


class BatchPredictRequest(BaseModel):
    features: list[list[float]]


class BatchPredictResponse(BaseModel):
    delta_charges: list[float]


class AutoregressiveRequest(BaseModel):
    user_id: str
    dates: list[str]
    data_dir: Optional[str] = None
    torch_dir: Optional[str] = None
    zscores_file: Optional[str] = None


class AutoregressiveResponse(BaseModel):
    user_id: str
    dates: list[str]
    num_samples: int
    preds: list[float]
    targets: list[float]
    mae: float
    mse: float
    rmse: float
    pred_range: list[float]
    target_range: list[float]


class ModelInfo(BaseModel):
    status: str
    model_type: str
    device: str
    config: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(request: Request):
    # breakpoint()
    predictor = request.app.state.predictor
    print(f"/health called, predictor is: {predictor}", flush = True)  # DEBUG
    # breakpoint()
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, fastapi_request: Request):
    predictor = fastapi_request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        features = np.array(request.features, dtype=np.float32)
        delta = predictor.predict_single(features)
        _metrics["predictions_total"] += 1
        _metrics["prediction_latency_sum"] += time.time() - start
        _metrics["prediction_latency_count"] += 1
        return PredictResponse(delta_charge=delta)
    except Exception as e:
        _metrics["predictions_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest, fastapi_request: Request):
    predictor = fastapi_request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        features = np.array(request.features, dtype=np.float32)
        deltas = predictor.predict(features)
        _metrics["batch_predictions_total"] += 1
        _metrics["predictions_total"] += len(request.features)
        _metrics["prediction_latency_sum"] += time.time() - start
        _metrics["prediction_latency_count"] += 1
        return BatchPredictResponse(delta_charges=deltas.tolist())
    except Exception as e:
        _metrics["predictions_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


def _run_autoregressive(request: AutoregressiveRequest, predictor):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data_dir = request.data_dir or DEFAULT_DATA_DIR
    torch_dir = request.torch_dir or DEFAULT_TORCH_DIR
    zscores_file = request.zscores_file or DEFAULT_ZSCORES_FILE

    return predictor.run_inference_for_user(
        user_id=request.user_id,
        dates=request.dates,
        data_dir=data_dir,
        torch_dir=torch_dir,
        zscores_file=zscores_file,
    )


@app.post("/autoregressive", response_model=AutoregressiveResponse)
async def autoregressive(request: AutoregressiveRequest, fastapi_request: Request):
    predictor = fastapi_request.app.state.predictor
    start = time.time()
    try:
        results = _run_autoregressive(request, predictor)
        _metrics["autoregressive_predictions_total"] += 1
        _metrics["autoregressive_latency_sum"] += time.time() - start
        _metrics["autoregressive_latency_count"] += 1
        _metrics["last_autoregressive_mae"] = results["mae"]
        _metrics["last_autoregressive_rmse"] = results["rmse"]
        _metrics["last_autoregressive_mse"] = results["mse"]
        return AutoregressiveResponse(**{
            k: results[k]
            for k in AutoregressiveResponse.model_fields
        })
    except HTTPException:
        raise
    except Exception as e:
        _metrics["predictions_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/autoregressive/plot")
async def autoregressive_plot(request: AutoregressiveRequest, fastapi_request: Request):
    predictor = fastapi_request.app.state.predictor
    start = time.time()
    try:
        results = _run_autoregressive(request, predictor)
        png_bytes = BiochargePredictor.generate_plot(
            preds=results["preds"],
            targets=results["targets"],
            user_id=results["user_id"],
            dates=results["dates"],
            mae=results["mae"],
            mse=results["mse"],
        )
        _metrics["autoregressive_predictions_total"] += 1
        _metrics["autoregressive_latency_sum"] += time.time() - start
        _metrics["autoregressive_latency_count"] += 1
        _metrics["last_autoregressive_mae"] = results["mae"]
        _metrics["last_autoregressive_rmse"] = results["rmse"]
        _metrics["last_autoregressive_mse"] = results["mse"]
        return Response(content=png_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        _metrics["predictions_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", response_model=ModelInfo)
async def model_info(request: Request):
    predictor = request.app.state.predictor
    if predictor is None:
        return ModelInfo(status="not_loaded", model_type="none", device="none", config={})
    return ModelInfo(
        status="loaded",
        model_type="mlp_delta",
        device=str(predictor.device),
        config=predictor.config,
    )


@app.get("/metrics")
async def metrics(request: Request):
    """Prometheus-format metrics endpoint for Grafana scraping."""
    predictor = request.app.state.predictor
    model_loaded = 1 if predictor is not None else 0

    avg_latency = (
        _metrics["prediction_latency_sum"] / _metrics["prediction_latency_count"]
        if _metrics["prediction_latency_count"] > 0
        else 0
    )
    avg_ar_latency = (
        _metrics["autoregressive_latency_sum"] / _metrics["autoregressive_latency_count"]
        if _metrics["autoregressive_latency_count"] > 0
        else 0
    )
    lines = [
        "# HELP biocharge_predictions_total Total number of predictions made",
        "# TYPE biocharge_predictions_total counter",
        f'biocharge_predictions_total {_metrics["predictions_total"]}',
        "",
        "# HELP biocharge_predictions_errors_total Total prediction errors",
        "# TYPE biocharge_predictions_errors_total counter",
        f'biocharge_predictions_errors_total {_metrics["predictions_errors"]}',
        "",
        "# HELP biocharge_prediction_latency_seconds Average prediction latency",
        "# TYPE biocharge_prediction_latency_seconds gauge",
        f"biocharge_prediction_latency_seconds {avg_latency:.6f}",
        "",
        "# HELP biocharge_batch_predictions_total Total batch prediction calls",
        "# TYPE biocharge_batch_predictions_total counter",
        f'biocharge_batch_predictions_total {_metrics["batch_predictions_total"]}',
        "",
        "# HELP biocharge_autoregressive_predictions_total Total autoregressive inference calls",
        "# TYPE biocharge_autoregressive_predictions_total counter",
        f'biocharge_autoregressive_predictions_total {_metrics["autoregressive_predictions_total"]}',
        "",
        "# HELP biocharge_autoregressive_latency_seconds Average autoregressive latency",
        "# TYPE biocharge_autoregressive_latency_seconds gauge",
        f"biocharge_autoregressive_latency_seconds {avg_ar_latency:.6f}",
        "",
        "# HELP biocharge_model_loaded Whether model is loaded",
        "# TYPE biocharge_model_loaded gauge",
        f"biocharge_model_loaded {model_loaded}",
        "",
        "# HELP biocharge_autoregressive_mae Last autoregressive inference MAE",
        "# TYPE biocharge_autoregressive_mae gauge",
        f'biocharge_autoregressive_mae {_metrics["last_autoregressive_mae"]:.6f}',
        "",
        "# HELP biocharge_autoregressive_rmse Last autoregressive inference RMSE",
        "# TYPE biocharge_autoregressive_rmse gauge",
        f'biocharge_autoregressive_rmse {_metrics["last_autoregressive_rmse"]:.6f}',
        "",
        "# HELP biocharge_autoregressive_mse Last autoregressive inference MSE",
        "# TYPE biocharge_autoregressive_mse gauge",
        f'biocharge_autoregressive_mse {_metrics["last_autoregressive_mse"]:.6f}',
        "",
    ]
    return Response(content="\n".join(lines), media_type="text/plain")
