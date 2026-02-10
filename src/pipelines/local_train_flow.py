"""
Prefect flow for local training — skips API pull, uses local data.

This is the "mock" pipeline for demoing the full stack without VPN/API keys.
It reads data from data/source/ and trains the MLP_delta model with MLflow.

Usage:
    python -m src.pipelines.local_train_flow
"""

import os
import logging
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import yaml
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger

load_dotenv()


def load_config(config_path: str = "config/local_data.yaml") -> dict:
    """Load local data configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


@task
def setup_mlflow(experiment_name: str = "biocharge-mlp-delta") -> str:
    """Configure MLflow tracking."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


@task
def build_dataloaders_from_local(cfg_yaml: dict):
    """
    Build DataLoaders from local data files.

    Reads the Excel files and index CSVs directly from data/source/.
    """
    logger = get_run_logger()

    from src.training.dataset_delta_v2 import DatasetConfig, build_dataloaders

    data_cfg = cfg_yaml["data"]
    model_cfg = cfg_yaml["model"]

    logger.info("Building dataloaders from local data: %s", data_cfg["data_dir"])

    # z-norm data dir for per-user normalization
    z_data_dir = os.path.join(data_cfg["data_dir"], "z_norm_rhr_hrv_7_14_corrected_complete_original")
    actual_data_dir = z_data_dir if os.path.isdir(z_data_dir) else data_cfg["data_dir"]

    cfg_train = DatasetConfig(
        data_dir=actual_data_dir,
        index_csv=data_cfg["train_index_csv"],
        zscores_file=data_cfg.get("zscores_file"),
        charge_col=data_cfg.get("charge_col", "charge.timeseries"),
        static_cols=data_cfg.get("static_cols", ["height", "weight"]),
        sleep_cols=data_cfg.get("sleep_cols", [
            "sleep.waso_score", "sleep.start_time_score",
            "sleep.duration_score", "stress.fitness_fatigue_difference",
            "rhrScore", "hrv_factor",
        ]),
        ts_cols=data_cfg.get("ts_cols", []),
        use_past_charge=True,
        enable_augmentation=True,
        augment_prob=0.3,
    )

    cfg_val = DatasetConfig(
        data_dir=actual_data_dir,
        index_csv=data_cfg["val_index_csv"],
        zscores_file=data_cfg.get("zscores_file"),
        charge_col=data_cfg.get("charge_col", "charge.timeseries"),
        static_cols=data_cfg.get("static_cols", ["height", "weight"]),
        sleep_cols=data_cfg.get("sleep_cols", [
            "sleep.waso_score", "sleep.start_time_score",
            "sleep.duration_score", "stress.fitness_fatigue_difference",
            "rhrScore", "hrv_factor",
        ]),
        ts_cols=data_cfg.get("ts_cols", []),
        use_past_charge=True,
        enable_augmentation=False,
    )

    train_loader, val_loader, train_ds, val_ds = build_dataloaders(
        cfg_train, cfg_val,
        batch_size=model_cfg.get("batch_size", 32),
        data_fraction=model_cfg.get("data_fraction", 0.1),  # Use 10% for quick demo
    )

    # Infer input_dim from dataset
    sample = train_ds[0]
    if isinstance(sample, dict):
        input_dim = sample["features"].shape[-1]
    else:
        input_dim = sample[0].shape[-1]

    logger.info("DataLoaders ready. Train: %d batches, Val: %d batches, input_dim: %d",
                len(train_loader), len(val_loader), input_dim)

    return train_loader, val_loader, input_dim


@task
def run_training(train_loader, val_loader, config: dict, checkpoint_dir: str):
    """Run the training loop with MLflow logging."""
    from src.training.trainer import train

    logger = get_run_logger()
    logger.info("Starting training: %d epochs, lr=%s", config["epochs"], config["learning_rate"])

    model, metrics = train(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=checkpoint_dir,
    )

    logger.info("Training complete. Best metrics: %s", metrics)
    return metrics


@task
def register_model_task(run_id: str, model_name: str = "biocharge-mlp-delta"):
    """Register trained model in MLflow registry."""
    logger = get_run_logger()
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)
    logger.info("Registered model %s version %s", model_name, result.version)
    return result.version


@flow(name="local-training-flow", log_prints=True)
def local_train_flow(config_path: str = "config/local_data.yaml"):
    """
    Local training flow — uses data from data/source/, no API pull needed.

    This is the demo pipeline. When VPN is available, use full_flow.py instead.
    """
    logger = get_run_logger()
    logger.info("=== Local Training Flow (No API Pull) ===")

    # Load config
    cfg = load_config(config_path)
    model_cfg = cfg["model"]

    # Setup MLflow
    tracking_uri = setup_mlflow()
    logger.info("MLflow tracking at: %s", tracking_uri)

    # Build dataloaders from local data
    train_loader, val_loader, input_dim = build_dataloaders_from_local(cfg)

    # Prepare training config
    train_config = {
        "input_dim": input_dim,
        **model_cfg,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("models", "checkpoints", timestamp)

    # Train with MLflow
    with mlflow.start_run(run_name=f"local_mlp_delta_{timestamp}") as run:
        metrics = run_training(train_loader, val_loader, train_config, checkpoint_dir)

        try:
            model_version = register_model_task(run.info.run_id)
        except Exception as e:
            logger.warning("Could not register model (MLflow registry may not be available): %s", e)
            model_version = "N/A"

    result = {
        "status": "success",
        "run_id": run.info.run_id,
        "model_version": model_version,
        "metrics": metrics,
        "checkpoint_dir": checkpoint_dir,
    }
    logger.info("=== Flow Complete ===")
    logger.info("Result: %s", result)
    return result


if __name__ == "__main__":
    local_train_flow()
