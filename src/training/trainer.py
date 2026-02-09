"""
Training loop for MLP_delta with MLflow experiment tracking.

Simplified from DeepBiocharge/train_data_synth_delta_v2.py to focus on
the core training loop with MLflow integration instead of WandB.
"""

import os
import sys
import json
import logging
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

from src.training.model import MLP_delta

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["features"].to(device)
        y = batch["target"].to(device)
        mask = batch.get("mask")

        optimizer.zero_grad()
        yhat = model(x)

        if mask is not None:
            mask = mask.to(device)
            valid = mask.bool()
            if valid.sum() == 0:
                continue
            loss = criterion(yhat[valid], y[valid])
        else:
            loss = criterion(yhat, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model, return loss and MAE."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y = batch["target"].to(device)
            mask = batch.get("mask")

            yhat = model(x)

            if mask is not None:
                mask = mask.to(device)
                valid = mask.bool()
                if valid.sum() == 0:
                    continue
                loss = criterion(yhat[valid], y[valid])
                all_preds.extend(yhat[valid].cpu().numpy().flatten())
                all_targets.extend(y[valid].cpu().numpy().flatten())
            else:
                loss = criterion(yhat, y)
                all_preds.extend(yhat.cpu().numpy().flatten())
                all_targets.extend(y.cpu().numpy().flatten())

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    mae = mean_absolute_error(all_targets, all_preds) if all_targets else float("inf")
    return {"loss": avg_loss, "mae": mae}


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    checkpoint_dir: str | None = None,
) -> tuple[nn.Module, dict]:
    """
    Full training loop with MLflow logging.

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Dict with keys: input_dim, hidden, layers, dropout, norm,
                epochs, learning_rate, loss_type, patience
        checkpoint_dir: Where to save model checkpoints (optional)

    Returns:
        (trained_model, best_metrics)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    model = MLP_delta(
        input_dim=config["input_dim"],
        hidden=config.get("hidden", 128),
        layers=config.get("layers", 3),
        dropout=config.get("dropout", 0.1),
        norm=config.get("norm", "layer"),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=1e-4,
    )

    loss_type = config.get("loss_type", "mse")
    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "mae":
        criterion = nn.L1Loss()
    elif loss_type == "huber":
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    epochs = config.get("epochs", 30)
    patience = config.get("patience", 10)

    # Setup checkpoint dir
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # MLflow tracking
    mlflow.log_params({
        "model_type": "mlp_delta",
        "input_dim": config["input_dim"],
        "hidden": config.get("hidden", 128),
        "layers": config.get("layers", 3),
        "dropout": config.get("dropout", 0.1),
        "norm": config.get("norm", "layer"),
        "learning_rate": config.get("learning_rate", 1e-3),
        "loss_type": loss_type,
        "epochs": epochs,
        "batch_size": config.get("batch_size", 32),
    })

    best_val_loss = float("inf")
    best_metrics = {}
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
            },
            step=epoch,
        )

        logger.info(
            "Epoch %d/%d - train_loss: %.6f, val_loss: %.6f, val_mae: %.6f",
            epoch + 1, epochs, train_loss, val_metrics["loss"], val_metrics["mae"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = {**val_metrics, "epoch": epoch}
            epochs_without_improvement = 0

            if checkpoint_dir:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model_{epoch}.pt"))
                with open(os.path.join(checkpoint_dir, "model_config.json"), "w") as f:
                    json.dump(config, f, indent=2)
                logger.info("Saved checkpoint at epoch %d", epoch)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # Log final model to MLflow
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_metrics({"best_val_loss": best_metrics.get("loss", float("inf")), "best_val_mae": best_metrics.get("mae", float("inf"))})

    return model, best_metrics
