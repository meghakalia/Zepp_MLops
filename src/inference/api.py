"""
FastAPI inference server for biocharge prediction.

Endpoints:
    GET  /health     - Health check
    POST /predict    - Single prediction
    POST /batch      - Batch predictions
    GET  /model-info - Current model info
"""

import logging
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference.predictor import BiochargePredictor

logger = logging.getLogger(__name__)

# Global predictor instance
predictor: BiochargePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global predictor

    checkpoint_dir = os.environ.get("MODEL_CHECKPOINT_DIR")
    mlflow_uri = os.environ.get("MLFLOW_MODEL_URI")

    if checkpoint_dir:
        predictor = BiochargePredictor(checkpoint_dir=checkpoint_dir)
    elif mlflow_uri:
        predictor = BiochargePredictor(mlflow_model_uri=mlflow_uri)
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

    features = np.array(request.features, dtype=np.float32)
    delta = predictor.predict_single(features)
    return PredictResponse(delta_charge=delta)


@app.post("/batch", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array(request.features, dtype=np.float32)
    deltas = predictor.predict(features)
    return BatchPredictResponse(delta_charges=deltas.tolist())


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
