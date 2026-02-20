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
    POST /batch-inference     - Inference only (no pull) for multiple users — call /pull first
    GET  /model-info          - Current model info
    GET  /metrics             - Prometheus metrics (for Grafana)
    GET  /mlflow              - Redirect to MLflow UI (port 5000)
    GET  /grafana             - Redirect to Grafana UI (port 3000)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import RedirectResponse
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
    # Region-based error metrics
    "last_exercise_mae": float("nan"),
    "last_exercise_mse": float("nan"),
    "last_sleep_mae": float("nan"),
    "last_sleep_mse": float("nan"),
    "last_nap_mae": float("nan"),
    "last_nap_mse": float("nan"),
    "last_nonwear_mae": float("nan"),
    "last_nonwear_mse": float("nan"),
    "last_overall_traj_mae": float("nan"),
    "last_overall_traj_mse": float("nan"),
    "last_day_start_mae": float("nan"),
    "last_day_end_mae": float("nan"),
    # Multi-user: per-user and aggregate region metrics
    # { user_id: {"mae": float, "rmse": float, "exercise_mae": float, ...} }
    "per_user": {},
    # mean across users in last batch run
    "aggregate": {},
    # Last pull metadata
    "last_pull_user_ids": [],
    "last_pull_from_date": "",
    "last_pull_to_date": "",
    "last_pull_timestamp": 0.0,
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
    description="Dual-head model serving for biocharge delta predictions",
    version="0.4.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    save_plots_mlflow: bool = False  # log trajectory plot to MLflow as artifact


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
    timings: dict = {}
    total_seconds: float = 0.0
    stages_completed: list[str] = []
    region_errors: dict = {}


class BatchFreshInferenceRequest(BaseModel):
    user_ids: list[str]           # any number of user IDs
    from_date: str
    to_date: str
    model_dir: Optional[str] = None
    data_dir: Optional[str] = None
    zscores_file: Optional[str] = None
    pull_mode: str = "ONLINE"
    skip_pull: bool = False
    skip_ground_truth: bool = False
    per_user_metrics: bool = True  # store per-user breakdown in Prometheus
    save_plots_mlflow: bool = False  # log per-user trajectory plots to MLflow as artifacts



class BatchFreshInferenceResponse(BaseModel):
    user_ids: list[str]
    from_date: str
    to_date: str
    results: list[dict]           # per-user result dicts
    aggregate: dict               # mean metrics across all successful users
    num_users_success: int
    num_users_failed: int


class PullRequest(BaseModel):
    user_ids: list[str]
    from_date: str
    to_date: str
    mode: str = "ONLINE"


class PullResponse(BaseModel):
    user_ids: list[str]
    from_date: str
    to_date: str
    results: dict[str, bool]  # user_id -> success
    num_success: int
    num_failed: int


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

        # Capture region-based error metrics
        region_errors = results.get("region_errors", {})
        if region_errors:
            _metrics["last_exercise_mae"] = region_errors.get("exercise_mae", float("nan"))
            _metrics["last_exercise_mse"] = region_errors.get("exercise_mse", float("nan"))
            _metrics["last_sleep_mae"] = region_errors.get("sleep_mae", float("nan"))
            _metrics["last_sleep_mse"] = region_errors.get("sleep_mse", float("nan"))
            _metrics["last_nap_mae"] = region_errors.get("nap_mae", float("nan"))
            _metrics["last_nap_mse"] = region_errors.get("nap_mse", float("nan"))
            _metrics["last_nonwear_mae"] = region_errors.get("non_wear_mae", float("nan"))
            _metrics["last_nonwear_mse"] = region_errors.get("non_wear_mse", float("nan"))
            _metrics["last_overall_traj_mae"] = region_errors.get("overall_traj_mae", float("nan"))
            _metrics["last_overall_traj_mse"] = region_errors.get("overall_traj_mse", float("nan"))
            _metrics["last_day_start_mae"] = region_errors.get("day_start_mae", float("nan"))
            _metrics["last_day_end_mae"] = region_errors.get("day_end_mae", float("nan"))

        # Track last pull metadata and per-user stats (single user)
        user_metrics = _collect_user_metrics(results)
        _metrics["per_user"] = {request.user_id: user_metrics}
        _metrics["aggregate"] = user_metrics
        _metrics["last_pull_user_ids"] = [request.user_id]
        _metrics["last_pull_from_date"] = request.from_date
        _metrics["last_pull_to_date"] = request.to_date
        _metrics["last_pull_timestamp"] = time.time()

        if request.save_plots_mlflow:
            try:
                _log_plots_to_mlflow(
                    run_name=f"fresh_inference_{request.user_id}_{request.from_date}_{request.to_date}",
                    from_date=request.from_date,
                    to_date=request.to_date,
                    per_user_plots={request.user_id: results["plot_bytes"]} if results.get("plot_bytes") else {},
                    aggregate=user_metrics,
                    per_user_data={request.user_id: user_metrics},
                )
            except Exception as plot_err:
                logger.warning("MLflow plot logging failed for user %s: %s", request.user_id, plot_err)

        return FreshInferenceResponse(**{
            k: results[k]
            for k in FreshInferenceResponse.model_fields
            if k in results
        })
    except HTTPException:
        raise
    except Exception as e:
        _metrics["fresh_inference_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fresh-inference/plot")
async def fresh_inference_plot(request: FreshInferenceRequest, fastapi_request: Request):
    """Same as /fresh-inference but returns a PNG trajectory plot with event annotations."""
    from src.inference.fresh_inference import run_fresh_inference
    from src.inference.plotting import generate_trajectory_plot

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

        # Generate trajectory plot with event annotations
        from src.inference.fresh_inference import _generate_date_list
        dates = _generate_date_list(request.from_date, request.to_date)
        data_dir = results.get("processed_excel_path", "")
        if data_dir:
            data_dir = os.path.dirname(data_dir)

        png_bytes = generate_trajectory_plot(
            preds=results["preds"],
            targets=results["targets"],
            user_id=request.user_id,
            dates=dates,
            data_dir=data_dir,
            error_dict=results.get("region_errors", {}),
        )

        _metrics["fresh_inference_total"] += 1
        _metrics["last_fresh_inference_mae"] = results["mae"]
        _metrics["last_fresh_inference_rmse"] = results["rmse"]
        _metrics["last_fresh_inference_mse"] = results["mse"]
        if "timings" in results:
            _metrics["last_fresh_data_pull_seconds"] = results["timings"].get("data_pull_seconds", 0)
            _metrics["last_fresh_ground_truth_seconds"] = results["timings"].get("ground_truth_seconds", 0)
            _metrics["last_fresh_inference_seconds"] = results["timings"].get("inference_seconds", 0)

        # Capture region errors for metrics
        region_errors = results.get("region_errors", {})
        if region_errors:
            _metrics["last_exercise_mae"] = region_errors.get("exercise_mae", float("nan"))
            _metrics["last_exercise_mse"] = region_errors.get("exercise_mse", float("nan"))
            _metrics["last_sleep_mae"] = region_errors.get("sleep_mae", float("nan"))
            _metrics["last_sleep_mse"] = region_errors.get("sleep_mse", float("nan"))
            _metrics["last_nap_mae"] = region_errors.get("nap_mae", float("nan"))
            _metrics["last_nap_mse"] = region_errors.get("nap_mse", float("nan"))
            _metrics["last_nonwear_mae"] = region_errors.get("non_wear_mae", float("nan"))
            _metrics["last_nonwear_mse"] = region_errors.get("non_wear_mse", float("nan"))
            _metrics["last_overall_traj_mae"] = region_errors.get("overall_traj_mae", float("nan"))
            _metrics["last_overall_traj_mse"] = region_errors.get("overall_traj_mse", float("nan"))
            _metrics["last_day_start_mae"] = region_errors.get("day_start_mae", float("nan"))
            _metrics["last_day_end_mae"] = region_errors.get("day_end_mae", float("nan"))

        # Track last pull metadata and per-user stats (single user)
        user_metrics = _collect_user_metrics(results)
        _metrics["per_user"] = {request.user_id: user_metrics}
        _metrics["aggregate"] = user_metrics
        _metrics["last_pull_user_ids"] = [request.user_id]
        _metrics["last_pull_from_date"] = request.from_date
        _metrics["last_pull_to_date"] = request.to_date
        _metrics["last_pull_timestamp"] = time.time()

        return Response(content=png_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        _metrics["fresh_inference_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


def _collect_user_metrics(result: dict) -> dict:
    """Extract the metrics we want to track per user from a fresh_inference result."""
    region_errors = result.get("region_errors", {})
    return {
        "mae": result.get("mae", float("nan")),
        "rmse": result.get("rmse", float("nan")),
        "mse": result.get("mse", float("nan")),
        "exercise_mae": region_errors.get("exercise_mae", float("nan")),
        "sleep_mae": region_errors.get("sleep_mae", float("nan")),
        "nap_mae": region_errors.get("nap_mae", float("nan")),
        "non_wear_mae": region_errors.get("non_wear_mae", float("nan")),
        "day_start_mae": region_errors.get("day_start_mae", float("nan")),
        "day_end_mae": region_errors.get("day_end_mae", float("nan")),
        "overall_traj_mae": region_errors.get("overall_traj_mae", float("nan")),
    }


def _compute_aggregate(per_user: dict) -> dict:
    """Compute mean over all users, skipping NaN values."""
    if not per_user:
        return {}
    import math
    keys = list(next(iter(per_user.values())).keys())
    agg = {}
    for k in keys:
        vals = [v[k] for v in per_user.values() if not (isinstance(v[k], float) and math.isnan(v[k]))]
        agg[k] = sum(vals) / len(vals) if vals else float("nan")
    return agg


def _log_plots_to_mlflow(
    run_name: str,
    from_date: str,
    to_date: str,
    per_user_plots: dict,
    aggregate: dict,
    per_user_data: dict,
) -> None:
    """Log per-user trajectory plots and metrics to an MLflow run.

    Args:
        run_name: Display name for the MLflow run.
        from_date: Inference start date (YYYY-MM-DD).
        to_date: Inference end date (YYYY-MM-DD).
        per_user_plots: Mapping of user_id -> PNG bytes.
        aggregate: Aggregate metrics dict (mean across users).
        per_user_data: Per-user metrics dicts from _collect_user_metrics.
    """
    import math
    import tempfile

    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed; skipping MLflow plot logging")
        return

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

    try:
        mlflow.set_tracking_uri(mlflow_uri)
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "biocharge-inference")
        try:
            mlflow.set_experiment(experiment_name)
        except mlflow.exceptions.MlflowException as e:
            if "deleted experiment" in str(e).lower():
                # get_experiment_by_name only finds active exps; search deleted ones explicitly
                client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
                from mlflow.entities import ViewType
                deleted_exps = client.search_experiments(
                    view_type=ViewType.DELETED_ONLY,
                    filter_string=f"name = '{experiment_name}'",
                )
                if deleted_exps:
                    client.restore_experiment(deleted_exps[0].experiment_id)
                mlflow.set_experiment(experiment_name)
            else:
                raise
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("from_date", from_date)
            mlflow.log_param("to_date", to_date)
            mlflow.log_param("num_users", len(per_user_plots))

            # Aggregate metrics
            for k, v in aggregate.items():
                if isinstance(v, float) and not math.isnan(v):
                    mlflow.log_metric(f"aggregate_{k}", v)

            # Per-user metrics (prefixed with user id)
            for uid, udata in per_user_data.items():
                for k, v in udata.items():
                    if isinstance(v, float) and not math.isnan(v):
                        mlflow.log_metric(f"user_{uid}_{k}", v)

            # Plot artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                for uid, png_bytes in per_user_plots.items():
                    plot_path = os.path.join(tmpdir, f"user_{uid}_trajectory.png")
                    with open(plot_path, "wb") as f:
                        f.write(png_bytes)
                    mlflow.log_artifact(plot_path, artifact_path="plots")
    except Exception as e:
        logger.error("MLflow logging failed: %s", e)


@app.post("/batch-fresh-inference", response_model=BatchFreshInferenceResponse)
async def batch_fresh_inference(request: BatchFreshInferenceRequest, fastapi_request: Request):
    """
    Run fresh inference for any number of users sequentially.
    Aggregates region-based errors as mean across all users.
    When per_user_metrics=True, per-user breakdown is emitted to Prometheus.
    When save_plots_mlflow=True, trajectory plots are logged as MLflow artifacts.
    """
    from src.inference.fresh_inference import run_fresh_inference

    predictor = fastapi_request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    user_ids = request.user_ids
    per_user_results = []
    per_user_data = {}
    per_user_plots: dict = {}
    failed = 0

    for uid in user_ids:
        try:
            result = run_fresh_inference(
                user_id=uid,
                from_date=request.from_date,
                to_date=request.to_date,
                model_dir=request.model_dir or os.environ.get("MODEL_CHECKPOINT_DIR", "models/production"),
                data_dir=request.data_dir or DEFAULT_FRESH_DATA_DIR,
                zscores_file=request.zscores_file or DEFAULT_ZSCORES_FILE,
                pull_mode=request.pull_mode,
                skip_pull=request.skip_pull,
                skip_ground_truth=request.skip_ground_truth,
                plot=request.save_plots_mlflow,  # generate plot while Excel is still alive
                output_dir=os.path.join(request.data_dir or DEFAULT_FRESH_DATA_DIR, "plots"),
                predictor=predictor,
            )
            per_user_results.append({
                "user_id": uid,
                "mae": result["mae"],
                "rmse": result["rmse"],
                "mse": result["mse"],
                "num_samples": result["num_samples"],
                "region_errors": result.get("region_errors", {}),
            })
            per_user_data[uid] = _collect_user_metrics(result)
            _metrics["fresh_inference_total"] += 1

            if request.save_plots_mlflow:
                png_bytes = result.get("plot_bytes")
                if png_bytes:
                    per_user_plots[uid] = png_bytes
                else:
                    logger.warning("No plot bytes returned for user %s", uid)
        except Exception as e:
            logger.error("Batch inference failed for user %s: %s", uid, e)
            _metrics["fresh_inference_errors"] += 1
            failed += 1

    aggregate = _compute_aggregate(per_user_data)
    _metrics["aggregate"] = aggregate

    # Record last pull metadata
    _metrics["last_pull_user_ids"] = list(user_ids)
    _metrics["last_pull_from_date"] = request.from_date
    _metrics["last_pull_to_date"] = request.to_date
    _metrics["last_pull_timestamp"] = time.time()

    if request.per_user_metrics:
        _metrics["per_user"] = per_user_data
    else:
        _metrics["per_user"] = {}

    # Update the flat metrics with aggregate values for backward compat
    if aggregate:
        _metrics["last_fresh_inference_mae"] = aggregate.get("mae", float("nan"))
        _metrics["last_fresh_inference_rmse"] = aggregate.get("rmse", float("nan"))
        _metrics["last_fresh_inference_mse"] = aggregate.get("mse", float("nan"))
        _metrics["last_exercise_mae"] = aggregate.get("exercise_mae", float("nan"))
        _metrics["last_sleep_mae"] = aggregate.get("sleep_mae", float("nan"))
        _metrics["last_nap_mae"] = aggregate.get("nap_mae", float("nan"))
        _metrics["last_nonwear_mae"] = aggregate.get("non_wear_mae", float("nan"))
        _metrics["last_day_start_mae"] = aggregate.get("day_start_mae", float("nan"))
        _metrics["last_day_end_mae"] = aggregate.get("day_end_mae", float("nan"))
        _metrics["last_overall_traj_mae"] = aggregate.get("overall_traj_mae", float("nan"))

    if request.save_plots_mlflow and per_user_plots:
        try:
            _log_plots_to_mlflow(
                run_name=f"batch_inference_{request.from_date}_{request.to_date}",
                from_date=request.from_date,
                to_date=request.to_date,
                per_user_plots=per_user_plots,
                aggregate=aggregate,
                per_user_data=per_user_data,
            )
        except Exception as e:
            logger.error("MLflow plot logging failed: %s", e)

    return BatchFreshInferenceResponse(
        user_ids=user_ids,
        from_date=request.from_date,
        to_date=request.to_date,
        results=per_user_results,
        aggregate=aggregate,
        num_users_success=len(per_user_results),
        num_users_failed=failed,
    )


@app.post("/batch-inference", response_model=BatchFreshInferenceResponse)
async def batch_inference(request: BatchFreshInferenceRequest, fastapi_request: Request):
    """
    Run inference for multiple users using already-pulled local data (never pulls).
    Identical to /batch-fresh-inference but always skips the data pull step.
    Use /pull first to fetch data, then call this endpoint.
    """
    from src.inference.fresh_inference import run_fresh_inference

    predictor = fastapi_request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    user_ids = request.user_ids
    per_user_results = []
    per_user_data = {}
    per_user_plots: dict = {}
    failed = 0

    for uid in user_ids:
        try:
            result = run_fresh_inference(
                user_id=uid,
                from_date=request.from_date,
                to_date=request.to_date,
                model_dir=request.model_dir or os.environ.get("MODEL_CHECKPOINT_DIR", "models/production"),
                data_dir=request.data_dir or DEFAULT_FRESH_DATA_DIR,
                zscores_file=request.zscores_file or DEFAULT_ZSCORES_FILE,
                pull_mode=request.pull_mode,
                skip_pull=True,  # always skip pull — use /pull endpoint to fetch data
                skip_ground_truth=request.skip_ground_truth,
                plot=request.save_plots_mlflow,
                output_dir=os.path.join(request.data_dir or DEFAULT_FRESH_DATA_DIR, "plots"),
                predictor=predictor,
            )
            per_user_results.append({
                "user_id": uid,
                "mae": result["mae"],
                "rmse": result["rmse"],
                "mse": result["mse"],
                "num_samples": result["num_samples"],
                "region_errors": result.get("region_errors", {}),
            })
            per_user_data[uid] = _collect_user_metrics(result)
            _metrics["fresh_inference_total"] += 1

            if request.save_plots_mlflow:
                png_bytes = result.get("plot_bytes")
                if png_bytes:
                    per_user_plots[uid] = png_bytes
                else:
                    logger.warning("No plot bytes returned for user %s", uid)
        except Exception as e:
            logger.error("Batch inference failed for user %s: %s", uid, e)
            _metrics["fresh_inference_errors"] += 1
            failed += 1

    aggregate = _compute_aggregate(per_user_data)
    _metrics["aggregate"] = aggregate

    if request.per_user_metrics:
        _metrics["per_user"] = per_user_data
    else:
        _metrics["per_user"] = {}

    if aggregate:
        _metrics["last_fresh_inference_mae"] = aggregate.get("mae", float("nan"))
        _metrics["last_fresh_inference_rmse"] = aggregate.get("rmse", float("nan"))
        _metrics["last_fresh_inference_mse"] = aggregate.get("mse", float("nan"))
        _metrics["last_exercise_mae"] = aggregate.get("exercise_mae", float("nan"))
        _metrics["last_sleep_mae"] = aggregate.get("sleep_mae", float("nan"))
        _metrics["last_nap_mae"] = aggregate.get("nap_mae", float("nan"))
        _metrics["last_nonwear_mae"] = aggregate.get("non_wear_mae", float("nan"))
        _metrics["last_day_start_mae"] = aggregate.get("day_start_mae", float("nan"))
        _metrics["last_day_end_mae"] = aggregate.get("day_end_mae", float("nan"))
        _metrics["last_overall_traj_mae"] = aggregate.get("overall_traj_mae", float("nan"))

    if request.save_plots_mlflow and per_user_plots:
        try:
            _log_plots_to_mlflow(
                run_name=f"batch_inference_{request.from_date}_{request.to_date}",
                from_date=request.from_date,
                to_date=request.to_date,
                per_user_plots=per_user_plots,
                aggregate=aggregate,
                per_user_data=per_user_data,
            )
        except Exception as e:
            logger.error("MLflow plot logging failed: %s", e)

    return BatchFreshInferenceResponse(
        user_ids=user_ids,
        from_date=request.from_date,
        to_date=request.to_date,
        results=per_user_results,
        aggregate=aggregate,
        num_users_success=len(per_user_results),
        num_users_failed=failed,
    )


@app.post("/pull", response_model=PullResponse)
async def pull_data(request: PullRequest):
    """
    Pull data for multiple users.
    """
    from src.data_ingestion.puller import pull_multiple_users


    # Prepare configs for puller
    configs = [
        {
            "userid": uid,
            "from_date": request.from_date,
            "to_date": request.to_date
        }
        for uid in request.user_ids
    ]

    try:
        # We can use the default data dir from api.py or let puller decide
        # puller.py signature: pull_multiple_users(user_configs, data_dir, mode, max_workers)
        # We'll use DEFAULT_FRESH_DATA_DIR from this file
        
        # Note: DEFAULT_FRESH_DATA_DIR might be defined globally in api.py
        # If not visible here, request.app.state... or just import/use constant
        # It is defined near top of api.py
        
        results_map = pull_multiple_users(
            user_configs=configs,
            data_dir=DEFAULT_FRESH_DATA_DIR,
            mode=request.mode,
        )

        # Convert results_map (uid -> data or None) to success bools
        results_bool = {uid: (data is not None) for uid, data in results_map.items()}
        num_success = sum(results_bool.values())
        num_failed = len(results_bool) - num_success

        return PullResponse(
            user_ids=request.user_ids,
            from_date=request.from_date,
            to_date=request.to_date,
            results=results_bool,
            num_success=num_success,
            num_failed=num_failed,
        )

    except Exception as e:
        logger.error("Pull failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/model-info", response_model=ModelInfo)
async def model_info(request: Request):
    predictor = request.app.state.predictor
    if predictor is None:
        return ModelInfo(status="not_loaded", model_type="none", device="none", config={})
    model_type = predictor.config.get("model_type", "mlp_delta")
    return ModelInfo(
        status="loaded",
        model_type=model_type,
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

    # Region-based error metrics (only emit if not NaN)
    region_metrics = [
        ("biocharge_exercise_mae", "Last exercise region MAE", "last_exercise_mae"),
        ("biocharge_exercise_mse", "Last exercise region MSE", "last_exercise_mse"),
        ("biocharge_sleep_mae", "Last sleep region MAE", "last_sleep_mae"),
        ("biocharge_sleep_mse", "Last sleep region MSE", "last_sleep_mse"),
        ("biocharge_nap_mae", "Last nap region MAE", "last_nap_mae"),
        ("biocharge_nap_mse", "Last nap region MSE", "last_nap_mse"),
        ("biocharge_nonwear_mae", "Last non-wear region MAE", "last_nonwear_mae"),
        ("biocharge_nonwear_mse", "Last non-wear region MSE", "last_nonwear_mse"),
        ("biocharge_overall_traj_mae", "Last overall trajectory MAE", "last_overall_traj_mae"),
        ("biocharge_overall_traj_mse", "Last overall trajectory MSE", "last_overall_traj_mse"),
        ("biocharge_day_start_mae", "Last day-start boundary MAE", "last_day_start_mae"),
        ("biocharge_day_end_mae", "Last day-end boundary MAE", "last_day_end_mae"),
    ]
    for metric_name, help_text, key in region_metrics:
        val = _metrics[key]
        if not (isinstance(val, float) and val != val):  # skip NaN
            lines.extend([
                f"# HELP {metric_name} {help_text}",
                f"# TYPE {metric_name} gauge",
                f"{metric_name} {val:.6f}",
                "",
            ])

    # ---- Labelled multi-user metrics ----
    # biocharge_region_mae{region="...", user_id="..."} and
    # biocharge_user_inference_mae{user_id="..."}
    import math

    # region key → label value mapping
    region_label_map = [
        ("exercise_mae", "exercise"),
        ("sleep_mae", "sleep"),
        ("nap_mae", "nap"),
        ("non_wear_mae", "non_wear"),
        ("day_start_mae", "day_start"),
        ("day_end_mae", "day_end"),
        ("overall_traj_mae", "overall_traj"),
    ]

    aggregate = _metrics.get("aggregate", {})
    per_user = _metrics.get("per_user", {})

    if aggregate or per_user:
        lines.extend([
            "# HELP biocharge_region_mae Region MAE by event type and user",
            "# TYPE biocharge_region_mae gauge",
        ])
        for field_key, region_label in region_label_map:
            # aggregate row (user_id="all")
            agg_val = aggregate.get(field_key, float("nan"))
            if not math.isnan(agg_val):
                lines.append(f'biocharge_region_mae{{region="{region_label}",user_id="all"}} {agg_val:.6f}')
            # per-user rows
            for uid, udata in per_user.items():
                u_val = udata.get(field_key, float("nan"))
                if not math.isnan(u_val):
                    lines.append(f'biocharge_region_mae{{region="{region_label}",user_id="{uid}"}} {u_val:.6f}')
        lines.append("")

        lines.extend([
            "# HELP biocharge_user_inference_mae Fresh inference MAE per user",
            "# TYPE biocharge_user_inference_mae gauge",
        ])
        agg_mae = aggregate.get("mae", float("nan"))
        if not math.isnan(agg_mae):
            lines.append(f'biocharge_user_inference_mae{{user_id="all"}} {agg_mae:.6f}')
        for uid, udata in per_user.items():
            u_mae = udata.get("mae", float("nan"))
            if not math.isnan(u_mae):
                lines.append(f'biocharge_user_inference_mae{{user_id="{uid}"}} {u_mae:.6f}')
        lines.append("")

        lines.extend([
            "# HELP biocharge_user_inference_rmse Fresh inference RMSE per user",
            "# TYPE biocharge_user_inference_rmse gauge",
        ])
        agg_rmse = aggregate.get("rmse", float("nan"))
        if not math.isnan(agg_rmse):
            lines.append(f'biocharge_user_inference_rmse{{user_id="all"}} {agg_rmse:.6f}')
        for uid, udata in per_user.items():
            u_rmse = udata.get("rmse", float("nan"))
            if not math.isnan(u_rmse):
                lines.append(f'biocharge_user_inference_rmse{{user_id="{uid}"}} {u_rmse:.6f}')
        lines.append("")

    # ---- Last pull metadata ----
    last_pull_user_ids = _metrics.get("last_pull_user_ids", [])
    last_pull_from = _metrics.get("last_pull_from_date", "")
    last_pull_to = _metrics.get("last_pull_to_date", "")
    last_pull_ts = _metrics.get("last_pull_timestamp", 0.0)

    if last_pull_user_ids:
        user_ids_str = ",".join(str(u) for u in last_pull_user_ids)
        lines.extend([
            "# HELP biocharge_last_pull_info Metadata labels for the most recent data pull",
            "# TYPE biocharge_last_pull_info gauge",
            f'biocharge_last_pull_info{{user_ids="{user_ids_str}",from_date="{last_pull_from}",to_date="{last_pull_to}"}} 1',
            "",
            "# HELP biocharge_last_pull_timestamp Unix timestamp of the most recent data pull",
            "# TYPE biocharge_last_pull_timestamp gauge",
            f"biocharge_last_pull_timestamp {last_pull_ts:.3f}",
            "",
            "# HELP biocharge_last_pull_user_count Number of users in the most recent data pull",
            "# TYPE biocharge_last_pull_user_count gauge",
            f"biocharge_last_pull_user_count {len(last_pull_user_ids)}",
            "",
            "# HELP biocharge_last_pull_user Whether a user was included in the last pull",
            "# TYPE biocharge_last_pull_user gauge",
        ])
        for uid in last_pull_user_ids:
            lines.append(f'biocharge_last_pull_user{{user_id="{uid}"}} 1')
        lines.append("")

    return Response(content="\n".join(lines), media_type="text/plain")


@app.get("/mlflow")
async def mlflow_redirect():
    """Redirect to MLflow UI (port 5000)."""
    return RedirectResponse(url="http://localhost:5000", status_code=302)


@app.get("/grafana")
async def grafana_redirect():
    """Redirect to Grafana UI (port 3000)."""
    return RedirectResponse(url="http://localhost:3000", status_code=302)
