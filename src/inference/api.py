"""
FastAPI inference server for biocharge prediction.

Endpoints:
    GET  /health     - Health check
    POST /predict    - Single prediction
    POST /batch      - Batch predictions
    GET  /model-info - Current model info
    GET  /metrics    - Prometheus metrics (for Grafana)
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from src.inference.predictor import BiochargePredictor

logger = logging.getLogger(__name__)

# Global predictor instance
predictor: BiochargePredictor | None = None

# Simple metrics counters (no prometheus_client dependency needed)
_metrics = {
    "predictions_total": 0,
    "predictions_errors": 0,
    "prediction_latency_sum": 0.0,
    "prediction_latency_count": 0,
    "batch_predictions_total": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global predictor

    checkpoint_dir = os.environ.get("MODEL_CHECKPOINT_DIR")
    mlflow_uri = os.environ.get("MLFLOW_MODEL_URI")

    if checkpoint_dir:
        try:
            predictor = BiochargePredictor(checkpoint_dir=checkpoint_dir)
        except Exception as e:
            logger.error("Failed to load model from %s: %s", checkpoint_dir, e)
    elif mlflow_uri:
        try:
            predictor = BiochargePredictor(mlflow_model_uri=mlflow_uri)
        except Exception as e:
            logger.error("Failed to load model from MLflow %s: %s", mlflow_uri, e)
    else:
        logger.warning("No model configured. Set MODEL_CHECKPOINT_DIR or MLFLOW_MODEL_URI.")

    yield

    predictor = None


app = FastAPI(
    title="Biocharge Prediction API",
    description="MLP_delta model serving for biocharge delta predictions",
    version="0.1.0",
    lifespan=lifespan,
)


# Request/Response models
class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    delta_charge: float


class BatchPredictRequest(BaseModel):
    features: list[list[float]]


class BatchPredictResponse(BaseModel):
    delta_charges: list[float]


class ModelInfo(BaseModel):
    status: str
    model_type: str
    device: str
    config: dict


# Endpoints
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
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
async def batch_predict(request: BatchPredictRequest):
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


@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    if predictor is None:
        return ModelInfo(status="not_loaded", model_type="none", device="none", config={})
    return ModelInfo(
        status="loaded",
        model_type="mlp_delta",
        device=str(predictor.device),
        config=predictor.config,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus-format metrics endpoint for Grafana scraping."""
    avg_latency = (
        _metrics["prediction_latency_sum"] / _metrics["prediction_latency_count"]
        if _metrics["prediction_latency_count"] > 0
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
        "# HELP biocharge_model_loaded Whether model is loaded",
        "# TYPE biocharge_model_loaded gauge",
        f"biocharge_model_loaded {1 if predictor is not None else 0}",
        "",
    ]
    return Response(content="\n".join(lines), media_type="text/plain")
