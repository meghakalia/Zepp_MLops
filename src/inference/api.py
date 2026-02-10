"""
FastAPI inference server for biocharge prediction.

Endpoints:
    GET  /health              - Health check
    POST /predict             - Single prediction
    POST /batch               - Batch predictions
    POST /autoregressive      - Full autoregressive inference for a user+dates
    POST /autoregressive/plot - Same as above, returns PNG plot
    POST /fresh-inference     - Full pipeline: pull data -> ground truth -> inference
    POST /fresh-inference/plot - Same as above, returns PNG plot
    GET  /model-info          - Current model info
    GET  /metrics             - Prometheus metrics (for Grafana)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel

from src.inference.predictor import BiochargePredictor

logger = logging.getLogger(__name__)

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
    "fresh_inference_total": 0,
    "fresh_inference_errors": 0,
    "fresh_inference_latency_sum": 0.0,
    "fresh_inference_latency_count": 0,
    "last_fresh_inference_mae": 0.0,
    "last_fresh_inference_rmse": 0.0,
    "last_fresh_inference_mse": 0.0,
    "last_fresh_data_pull_seconds": 0.0,
    "last_fresh_ground_truth_seconds": 0.0,
    "last_fresh_inference_seconds": 0.0,
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
DEFAULT_FRESH_DATA_DIR = os.environ.get(
    "BIOCHARGE_FRESH_DATA_DIR",
    "./data",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Lifespan startup: attempting to load model...")

    checkpoint_dir = os.environ.get("MODEL_CHECKPOINT_DIR")
    mlflow_uri = os.environ.get("MLFLOW_MODEL_URI")

    predictor = None
    if checkpoint_dir:
        logger.info("MODEL_CHECKPOINT_DIR set: %s", checkpoint_dir)
        try:
            predictor = BiochargePredictor(checkpoint_dir=checkpoint_dir)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load model from %s: %s", checkpoint_dir, e)
    elif mlflow_uri:
        logger.info("MLFLOW_MODEL_URI set: %s", mlflow_uri)
        try:
            predictor = BiochargePredictor(mlflow_model_uri=mlflow_uri)
            logger.info("Model loaded successfully from MLflow.")
        except Exception as e:
            logger.error("Failed to load model from MLflow %s: %s", mlflow_uri, e)
    else:
        logger.warning("No model configured. Set MODEL_CHECKPOINT_DIR or MLFLOW_MODEL_URI.")

    app.state.predictor = predictor

    yield

    app.state.predictor = None
    logger.info("Lifespan shutdown: predictor set to None.")


app = FastAPI(
    title="Biocharge Prediction API",
    description="MLP_delta model serving for biocharge delta predictions",
    version="0.3.0",
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


class FreshInferenceRequest(BaseModel):
    user_id: str
    from_date: str
    to_date: str
    model_dir: Optional[str] = None
    data_dir: Optional[str] = None
    zscores_file: Optional[str] = None
    pull_mode: str = "ONLINE"
    skip_pull: bool = False
    skip_ground_truth: bool = False
    plot: bool = False


class FreshInferenceResponse(BaseModel):
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
    processed_excel_path: str
    timings: dict
    total_seconds: float
    stages_completed: list[str]


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
    predictor = request.app.state.predictor
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


@app.post("/fresh-inference", response_model=FreshInferenceResponse)
async def fresh_inference(request: FreshInferenceRequest, fastapi_request: Request):
    """
    Full pipeline: pull data -> compute ground truth -> run ML inference.
    This is the API equivalent of running fresh_inference.py from CLI.
    """
    from src.inference.fresh_inference import run_fresh_inference

    predictor = fastapi_request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        results = run_fresh_inference(
            user_id=request.user_id,
            from_date=request.from_date,
            to_date=request.to_date,
            model_dir=request.model_dir or os.environ.get("MODEL_CHECKPOINT_DIR", "models/production"),
            data_dir=request.data_dir or DEFAULT_FRESH_DATA_DIR,
            zscores_file=request.zscores_file or DEFAULT_ZSCORES_FILE,
            pull_mode=request.pull_mode,
            skip_pull=request.skip_pull,
            skip_ground_truth=request.skip_ground_truth,
            plot=request.plot,
            predictor=predictor,
        )
        elapsed = time.time() - start
        _metrics["fresh_inference_total"] += 1
        _metrics["fresh_inference_latency_sum"] += elapsed
        _metrics["fresh_inference_latency_count"] += 1
        _metrics["last_fresh_inference_mae"] = results["mae"]
        _metrics["last_fresh_inference_rmse"] = results["rmse"]
        _metrics["last_fresh_inference_mse"] = results["mse"]
        if "timings" in results:
            _metrics["last_fresh_data_pull_seconds"] = results["timings"].get("data_pull_seconds", 0)
            _metrics["last_fresh_ground_truth_seconds"] = results["timings"].get("ground_truth_seconds", 0)
            _metrics["last_fresh_inference_seconds"] = results["timings"].get("inference_seconds", 0)

        return FreshInferenceResponse(**{
            k: results[k]
            for k in FreshInferenceResponse.model_fields
        })
    except HTTPException:
        raise
    except Exception as e:
        _metrics["fresh_inference_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fresh-inference/plot")
async def fresh_inference_plot(request: FreshInferenceRequest, fastapi_request: Request):
    """Same as /fresh-inference but returns a PNG plot."""
    from src.inference.fresh_inference import run_fresh_inference

    predictor = fastapi_request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = run_fresh_inference(
            user_id=request.user_id,
            from_date=request.from_date,
            to_date=request.to_date,
            model_dir=request.model_dir or os.environ.get("MODEL_CHECKPOINT_DIR", "models/production"),
            data_dir=request.data_dir or DEFAULT_FRESH_DATA_DIR,
            zscores_file=request.zscores_file or DEFAULT_ZSCORES_FILE,
            pull_mode=request.pull_mode,
            skip_pull=request.skip_pull,
            skip_ground_truth=request.skip_ground_truth,
            plot=False,
            predictor=predictor,
        )
        png_bytes = BiochargePredictor.generate_plot(
            preds=results["preds"],
            targets=results["targets"],
            user_id=results["user_id"],
            dates=results["dates"],
            mae=results["mae"],
            mse=results["mse"],
        )
        _metrics["fresh_inference_total"] += 1
        if "timings" in results:
            _metrics["last_fresh_data_pull_seconds"] = results["timings"].get("data_pull_seconds", 0)
            _metrics["last_fresh_ground_truth_seconds"] = results["timings"].get("ground_truth_seconds", 0)
            _metrics["last_fresh_inference_seconds"] = results["timings"].get("inference_seconds", 0)
        return Response(content=png_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        _metrics["fresh_inference_errors"] += 1
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
    avg_fresh_latency = (
        _metrics["fresh_inference_latency_sum"] / _metrics["fresh_inference_latency_count"]
        if _metrics["fresh_inference_latency_count"] > 0
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
        # Fresh inference metrics
        "# HELP biocharge_fresh_inference_total Total fresh inference pipeline calls",
        "# TYPE biocharge_fresh_inference_total counter",
        f'biocharge_fresh_inference_total {_metrics["fresh_inference_total"]}',
        "",
        "# HELP biocharge_fresh_inference_errors_total Total fresh inference errors",
        "# TYPE biocharge_fresh_inference_errors_total counter",
        f'biocharge_fresh_inference_errors_total {_metrics["fresh_inference_errors"]}',
        "",
        "# HELP biocharge_fresh_inference_latency_seconds Average fresh inference latency",
        "# TYPE biocharge_fresh_inference_latency_seconds gauge",
        f"biocharge_fresh_inference_latency_seconds {avg_fresh_latency:.6f}",
        "",
        "# HELP biocharge_fresh_inference_mae Last fresh inference MAE",
        "# TYPE biocharge_fresh_inference_mae gauge",
        f'biocharge_fresh_inference_mae {_metrics["last_fresh_inference_mae"]:.6f}',
        "",
        "# HELP biocharge_fresh_inference_rmse Last fresh inference RMSE",
        "# TYPE biocharge_fresh_inference_rmse gauge",
        f'biocharge_fresh_inference_rmse {_metrics["last_fresh_inference_rmse"]:.6f}',
        "",
        "# HELP biocharge_fresh_inference_mse Last fresh inference MSE",
        "# TYPE biocharge_fresh_inference_mse gauge",
        f'biocharge_fresh_inference_mse {_metrics["last_fresh_inference_mse"]:.6f}',
        "",
        "# HELP biocharge_fresh_data_pull_seconds Last data pull duration",
        "# TYPE biocharge_fresh_data_pull_seconds gauge",
        f'biocharge_fresh_data_pull_seconds {_metrics["last_fresh_data_pull_seconds"]:.6f}',
        "",
        "# HELP biocharge_fresh_ground_truth_seconds Last ground truth computation duration",
        "# TYPE biocharge_fresh_ground_truth_seconds gauge",
        f'biocharge_fresh_ground_truth_seconds {_metrics["last_fresh_ground_truth_seconds"]:.6f}',
        "",
        "# HELP biocharge_fresh_ml_inference_seconds Last ML inference duration",
        "# TYPE biocharge_fresh_ml_inference_seconds gauge",
        f'biocharge_fresh_ml_inference_seconds {_metrics["last_fresh_inference_seconds"]:.6f}',
        "",
    ]
    return Response(content="\n".join(lines), media_type="text/plain")
