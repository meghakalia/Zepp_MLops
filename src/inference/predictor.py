"""
Model loading and prediction for inference.

Loads a trained MLP_delta model from a checkpoint directory or MLflow
model registry and runs predictions.
"""

import json
import logging
import os
from pathlib import Path

import mlflow.pytorch
import numpy as np
import torch

from src.training.model import MLP_delta

logger = logging.getLogger(__name__)


class BiochargePredictor:
    """Loads a trained MLP_delta model and runs inference."""

    def __init__(
        self,
        checkpoint_dir: str | None = None,
        mlflow_model_uri: str | None = None,
        device: str | None = None,
    ):
        """
        Initialize predictor from either a local checkpoint or MLflow.

        Args:
            checkpoint_dir: Path to directory containing model_config.json and best_model_*.pt
            mlflow_model_uri: MLflow model URI (e.g. "models:/biocharge-mlp/Production")
            device: "cpu" or "cuda" (auto-detected if None)
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if mlflow_model_uri:
            self.model = mlflow.pytorch.load_model(mlflow_model_uri, map_location=self.device)
            self.config = {}
            logger.info("Loaded model from MLflow: %s", mlflow_model_uri)
        elif checkpoint_dir:
            self.config = self._load_config(checkpoint_dir)
            self.model = self._load_from_checkpoint(checkpoint_dir)
            logger.info("Loaded model from checkpoint: %s", checkpoint_dir)
        else:
            raise ValueError("Provide either checkpoint_dir or mlflow_model_uri")

        self.model.eval()
        self.model.to(self.device)

    def _load_config(self, checkpoint_dir: str) -> dict:
        config_path = os.path.join(checkpoint_dir, "model_config.json")
        with open(config_path) as f:
            return json.load(f)

    def _load_from_checkpoint(self, checkpoint_dir: str) -> MLP_delta:
        config = self.config
        model = MLP_delta(
            input_dim=config["input_dim"],
            hidden=config.get("hidden_dim", config.get("hidden", 128)),
            layers=config.get("num_layers", config.get("layers", 3)),
            dropout=config.get("dropout", 0.1),
            norm=config.get("norm_type", config.get("norm", "layer")),
        )

        # Find the latest best_model_*.pt file
        ckpt_files = sorted(Path(checkpoint_dir).glob("best_model_*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

        latest_ckpt = ckpt_files[-1]
        model.load_state_dict(torch.load(latest_ckpt, map_location=self.device, weights_only=True))
        return model

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of feature vectors.

        Args:
            features: numpy array of shape (batch_size, input_dim)

        Returns:
            numpy array of shape (batch_size,) with delta charge predictions
        """
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.device)
            yhat = self.model(x)
            return yhat.cpu().numpy().flatten()

    def predict_single(self, features: np.ndarray) -> float:
        """Predict delta charge for a single feature vector."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return float(self.predict(features)[0])
