"""
Prefect flow for model training with MLflow experiment tracking.

Trains the MLP_delta model on available data and registers the model
in MLflow for versioning and deployment.
"""

import os
import logging
from datetime import datetime

import mlflow
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger

load_dotenv()


@task
def setup_mlflow(experiment_name: str = "biocharge-mlp-delta") -> str:
    """Configure MLflow tracking."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


@task
def prepare_dataloaders(data_dir: str, config: dict):
    """
    Build training and validation DataLoaders.

    This task wraps the DeepBiocharge dataset utilities to create
    DataLoaders compatible with the training loop.
    """
    from src.training.dataset_delta_v2 import DatasetConfig, build_dataloaders

    logger = get_run_logger()
    logger.info("Building dataloaders from %s", data_dir)

    cfg = DatasetConfig(
        data_dir=config.get("data_dir", data_dir),
        index_csv=config.get("train_index_csv", os.path.join(data_dir, "train_indices.csv")),
        zscores_file=config.get("zscores_file"),
        charge_col=config.get("charge_col", "charge.timeseries"),
        static_cols=config.get("static_cols", ["height", "weight"]),
        sleep_cols=config.get("sleep_cols", [
            "sleep.waso_score", "sleep.start_time_score",
            "sleep.duration_score", "stress.fitness_fatigue_difference",
            "rhrScore", "hrv_factor",
        ]),
        ts_cols=config.get("ts_cols", []),
        use_past_charge=True,
    )

    train_loader, val_loader, input_dim = build_dataloaders(
        cfg,
        batch_size=config.get("batch_size", 32),
    )

    return train_loader, val_loader, input_dim


@task
def run_training(train_loader, val_loader, config: dict, checkpoint_dir: str):
    """Execute the training loop."""
    from src.training.trainer import train

    logger = get_run_logger()
    logger.info("Starting training with config: %s", config)

    model, metrics = train(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=checkpoint_dir,
    )

    logger.info("Training complete. Best metrics: %s", metrics)
    return model, metrics


@task
def register_model(run_id: str, model_name: str = "biocharge-mlp-delta"):
    """Register the trained model in MLflow model registry."""
    logger = get_run_logger()

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)

    logger.info("Registered model %s version %s", model_name, result.version)
    return result.version


@flow(name="training-flow", log_prints=True)
def training_flow(
    data_dir: str = "./data/processed",
    config: dict | None = None,
):
    """
    Model training flow.

    Trains the MLP_delta model, logs to MLflow, and registers the model.
    """
    logger = get_run_logger()

    if config is None:
        config = {
            "hidden": 128,
            "layers": 3,
            "dropout": 0.1,
            "norm": "layer",
            "learning_rate": 1e-3,
            "loss_type": "mse",
            "epochs": 30,
            "batch_size": 32,
            "patience": 10,
        }

    tracking_uri = setup_mlflow()
    logger.info("MLflow tracking at: %s", tracking_uri)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("models", "checkpoints", timestamp)

    with mlflow.start_run(run_name=f"mlp_delta_{timestamp}") as run:
        train_loader, val_loader, input_dim = prepare_dataloaders(data_dir, config)
        config["input_dim"] = input_dim

        model, metrics = run_training(train_loader, val_loader, config, checkpoint_dir)

        model_version = register_model(run.info.run_id)

        logger.info("Training flow complete. Model version: %s, Metrics: %s", model_version, metrics)

    return {"run_id": run.info.run_id, "model_version": model_version, "metrics": metrics}


if __name__ == "__main__":
    training_flow()
