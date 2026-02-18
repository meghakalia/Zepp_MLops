"""
Model loading and prediction for inference.

Loads a trained model (MLP_delta or GatedDualHeadMLP) from a checkpoint
directory or MLflow model registry and runs predictions — including full
autoregressive inference with region-based error computation.
"""

from __future__ import annotations

import ast
import io
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import mlflow.pytorch

from src.training.model import MLP_delta, GatedDualHeadMLP
from src.training.dataset_delta_v2 import DatasetConfig, TorchBiochargeDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & helpers (ported from run_inference.py)
# ---------------------------------------------------------------------------

COLUMNS_TO_READ = [
    "sleep.waso_score", "sleep.start_time_score",
    "sleep.duration_score", "sleep.deep_sleep_score",
    "rhrScore", "hrvScore", "hrv_factor", "rhr_factor",
    "stress.fitness_fatigue_difference",
    "wakefulness_after_sleep_onset_duration", "rem_sleep_ratio",
    "deep_sleep_ratio", "sleep_duration",
    "z_rhr_7", "z_hrv_7", "sleep_stage", "timeseries.sleep_markers",
    "timeseries.nap_state", "timeseries.exercise",
    "timeseries.hr_filtered", "timeseries.acc",
    "age", "gender", "weight", "height",
    "timeseries.full_stress_list", "timeseries.hrr_minute_list",
    "timeseries.min_status_list",
    "timeseries.hrr_raw", "timeseries.minute_trimp_raw",
    "charge.timeseries",
    "timeseries.hr_available",
    "timeseries.hr_available_5",
    "timeseries.hr",  # fallback for hr_filtered
]


def safe_parse(x):
    """Parse data that may be string, numeric, or NaN into a list."""
    if pd.isna(x):
        return []
    if isinstance(x, (int, float)):
        return [x]
    try:
        return ast.literal_eval(x)
    except Exception:
        try:
            return json.loads(x)
        except Exception:
            return []


def excel_to_pt(user_id, data_dir, output_dir):
    """Convert a user's Excel file to .pt format for TorchBiochargeDataset."""
    user_id = str(user_id)
    user_file = os.path.join(data_dir, f"{user_id}_processed.xlsx")

    if not os.path.exists(user_file):
        raise FileNotFoundError(f"User file not found: {user_file}")

    logger.info("Loading Excel file: %s", user_file)
    df = pd.read_excel(user_file)

    available_cols = [col for col in COLUMNS_TO_READ if col in df.columns]
    df_subset = df[available_cols + ["date"]].copy()

    all_features_data = []
    feature_names = []
    numeric_cols = df_subset.select_dtypes(include=[np.number]).columns

    for col in COLUMNS_TO_READ:
        if col not in df_subset.columns:
            feature_names.append(col)
            all_features_data.append([0.0] * len(df_subset))
            continue

        feature_names.append(col)
        col_data = []
        for _, row in df_subset.iterrows():
            val = row[col]
            if pd.isna(val):
                col_data.append(0.0)
            elif col.startswith("timeseries.") or col.startswith("charge.") or col == "sleep_stage":
                parsed_val = safe_parse(val) if isinstance(val, str) else val
                col_data.append(parsed_val)
            else:
                col_data.append(float(val))
        all_features_data.append(col_data)

    column_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    # If hr_filtered data is empty/zeros but hr data exists, alias it
    if "timeseries.hr_filtered" in column_to_idx and "timeseries.hr" in column_to_idx:
        hr_filtered_idx = column_to_idx["timeseries.hr_filtered"]
        hr_idx = column_to_idx["timeseries.hr"]
        # Check if hr_filtered is all zeros/empty
        hr_filtered_data = all_features_data[hr_filtered_idx]
        has_hr_filtered = any(
            isinstance(d, (list, tuple)) and len(d) > 0
            for d in hr_filtered_data
        )
        if not has_hr_filtered:
            # Copy hr data into hr_filtered slot
            all_features_data[hr_filtered_idx] = all_features_data[hr_idx]
            logger.info("Mapped timeseries.hr -> timeseries.hr_filtered (no hr_filtered data found)")

    string_cols = df_subset.select_dtypes(include=["object", "string"]).columns
    string_data = {}
    for col in string_cols:
        if col != "date":
            parsed_values = []
            for val in df_subset[col]:
                parsed_val = safe_parse(val)
                if isinstance(parsed_val, list) and len(parsed_val) > 0:
                    parsed_values.append(parsed_val)
                elif pd.isna(val) or val is None:
                    parsed_values.append([])
                else:
                    parsed_values.append(str(val))
            string_data[col] = parsed_values

    dates = pd.to_datetime(df_subset["date"]).values.astype("datetime64[D]")

    os.makedirs(output_dir, exist_ok=True)
    torch_path = os.path.join(output_dir, f"{user_id}.pt")
    torch.save(
        {
            "features_data": all_features_data,
            "feature_names": feature_names,
            "column_to_idx": column_to_idx,
            "dates": torch.from_numpy(dates.astype("int64")),
            "string_data": string_data,
            "X": df_subset[numeric_cols].to_numpy(dtype=np.float32),
            "columns": list(numeric_cols),
        },
        torch_path,
    )
    logger.info("Saved .pt file: %s", torch_path)
    return torch_path


def create_temp_index(user_id, dates, data_dir, output_path):
    """Create a temporary index CSV for the given user and dates."""
    user_id = str(user_id)
    user_file = os.path.join(data_dir, f"{user_id}_processed.xlsx")
    df = pd.read_excel(user_file)

    rows = []
    for date_str in dates:
        df_row = df[df["date"] == date_str]
        if df_row.empty:
            logger.warning("No data for user %s on %s", user_id, date_str)
            continue

        charge_series = safe_parse(df_row.iloc[0]["charge.timeseries"])
        n_steps = len(charge_series)
        logger.info("Date %s: %d timesteps", date_str, n_steps)

        for idx in range(n_steps):
            rows.append({"userid": user_id, "date": date_str, "index": idx})

    index_df = pd.DataFrame(rows)
    index_df.to_csv(output_path, index=False)
    logger.info("Saved temp index CSV: %s (%d rows)", output_path, len(rows))
    return output_path


def create_inference_config(model_config_path, temp_pt_dir, temp_index_csv, zscores_file):
    """Build a DatasetConfig for inference from model_config.json."""
    config_file = os.path.join(model_config_path, "model_config.json")
    with open(config_file) as f:
        model_config = json.load(f)

    dataset_config = model_config.get("dataset_config", {})

    sleep_cols = (
        model_config.get("sleep_columns")
        or dataset_config.get("sleep_cols")
        or model_config.get("sleep_cols")
        or []
    )
    ts_cols = (
        model_config.get("ts_columns")
        or dataset_config.get("ts_cols")
        or model_config.get("ts_cols")
        or []
    )
    static_cols = dataset_config.get("static_cols") or []

    normalization_type = (
        model_config.get("normalization_type")
        or dataset_config.get("normalization_type")
        or "znorm"
    )

    use_acceleration = model_config.get("use_acceleration") or dataset_config.get("use_acceleration", False)
    use_hr = model_config.get("use_hr") or dataset_config.get("use_hr", False)
    use_past_charge = model_config.get("use_past_charge") or dataset_config.get("use_past_charge", True)
    use_positional_encoding = model_config.get("use_positional_encoding") or dataset_config.get("use_positional_encoding", False)
    pos_encoding_type = (
        model_config.get("positional_encoding_type")
        or dataset_config.get("pos_encoding_type")
        or model_config.get("pos_encoding_type")
        or "biocharge_circadian"
    )
    pos_encoding_dim = dataset_config.get("pos_encoding_dim", 2)
    use_windowed_features = model_config.get("use_windowed_features", False)
    window_size = model_config.get("window_size_minutes", 15)
    include_gradients = model_config.get("use_gradient_features") or dataset_config.get("include_gradients", False)
    use_recovery_rate_feature = model_config.get("use_recovery_rate_feature") or dataset_config.get("use_recovery_rate_feature", False)
    weighted_sleep_score = model_config.get("weighted_sleep_score") or dataset_config.get("weighted_sleep_score", False)

    if ts_cols:
        ts_cols = list(ts_cols)
        if "timeseries.ema_hrr" in ts_cols:
            ts_cols.remove("timeseries.ema_hrr")
            ts_cols.append("timeseries.hrr_raw")
        if any(c in ts_cols for c in ["timeseries.ema_hr", "timeseries.hr_filtered", "timeseries.hrr_raw"]):
            use_hr = False
        if "timeseries.acc" in ts_cols:
            use_acceleration = False

    if use_hr and "timeseries.hr_filtered" not in ts_cols:
        ts_cols.append("timeseries.hr_filtered")
    if use_acceleration and "timeseries.acc" not in ts_cols:
        ts_cols.append("timeseries.acc")

    cfg = DatasetConfig(
        data_dir=temp_pt_dir,
        index_csv=temp_index_csv,
        zscores_file=zscores_file,
        charge_col="charge.timeseries",
        static_cols=static_cols,
        sleep_cols=sleep_cols,
        ts_cols=ts_cols,
        use_zscores=True,
        augment_prob=0.0,
        enable_augmentation=False,
        use_acceleration=use_acceleration,
        use_hr=use_hr,
        use_past_charge=use_past_charge,
        use_positional_encoding=use_positional_encoding,
        pos_encoding_type=pos_encoding_type,
        pos_encoding_dim=pos_encoding_dim,
        use_windowed_features=use_windowed_features,
        include_gradients=include_gradients,
        use_windowed_zscore=False,
        window_size_minutes=window_size,
        z_data_norm="population",
        normalization_type=normalization_type,
        use_recovery_rate_feature=use_recovery_rate_feature,
        weighted_sleep_score=weighted_sleep_score,
    )

    return cfg


# ---------------------------------------------------------------------------
# Region-based error helpers
# ---------------------------------------------------------------------------

def find_transition_events_by_type(user_id, dates, data_dir):
    """
    Find start/end indices of physiological events (exercise, sleep, nap, non-wear)
    across multiple days, returning indices in the concatenated timeseries.

    Returns:
        dict: {event_type: {'start': [...], 'end': [...]}}
    """
    user_file = os.path.join(data_dir, f"{user_id}_processed.xlsx")
    if not os.path.exists(user_file):
        return {}

    df = pd.read_excel(user_file)
    event_cols = {
        'exercise': 'timeseries.exercise',
        'sleep': 'timeseries.sleep_markers',
        'nap': 'timeseries.nap_state',
        'non_wear': 'timeseries.min_status_list',
    }
    active_values = {
        'exercise': {1},
        'sleep': {1},
        'nap': {1},
        'non_wear': {3},
    }

    result = {etype: {'start': [], 'end': []} for etype in event_cols}
    cumulative_offset = 0

    for date_str in dates:
        df_row = df[df['date'] == date_str]
        if df_row.empty:
            continue

        for etype, col_name in event_cols.items():
            try:
                series = safe_parse(df_row.iloc[0].get(col_name, '[]'))
            except Exception:
                series = []
            if not series:
                continue

            active_set = active_values[etype]
            in_interval = False
            start_idx = None

            for i, val in enumerate(series):
                is_active = val in active_set
                if is_active and not in_interval:
                    start_idx = i
                    in_interval = True
                elif not is_active and in_interval:
                    result[etype]['start'].append(start_idx + cumulative_offset)
                    result[etype]['end'].append(i - 1 + cumulative_offset)
                    in_interval = False

            if in_interval and start_idx is not None:
                result[etype]['start'].append(start_idx + cumulative_offset)
                result[etype]['end'].append(len(series) - 1 + cumulative_offset)

        # Get day length from charge timeseries
        try:
            charge = safe_parse(df_row.iloc[0].get('charge.timeseries', '[]'))
            cumulative_offset += len(charge)
        except Exception:
            cumulative_offset += 1440  # default day length

    return result


def calculate_region_errors(preds, targets, meta_dict, data_dir):
    """
    Compute region-based autoregressive errors.

    Returns:
        dict with keys like exercise_mae, sleep_mse, overall_traj_mae, etc.
    """
    if not meta_dict:
        return {}

    user_id = list(meta_dict.keys())[0]
    dates = sorted(list(meta_dict[user_id]))

    # Find event transitions
    events = find_transition_events_by_type(user_id, dates, data_dir)

    np_preds = np.array(preds, dtype=np.float64)
    np_targets = np.array(targets, dtype=np.float64)
    min_len = min(len(np_preds), len(np_targets))

    error_dict = {}

    # Region-based errors
    for event_type, intervals in events.items():
        traj_errors = []
        start_end_errors = []

        for start_idx, end_idx in zip(intervals.get('start', []), intervals.get('end', [])):
            if start_idx >= min_len or end_idx >= min_len or start_idx > end_idx:
                continue

            seg_p = np_preds[start_idx:end_idx + 1]
            seg_t = np_targets[start_idx:end_idx + 1]
            traj_errors.append(seg_p - seg_t)

            # Start-end error
            diff_pred = abs(float(np_preds[end_idx] - np_preds[start_idx]))
            diff_gt = abs(float(np_targets[end_idx] - np_targets[start_idx]))
            start_end_errors.append(diff_gt - diff_pred)

        if traj_errors:
            all_errs = np.concatenate(traj_errors)
            error_dict[f"{event_type}_mae"] = float(np.mean(np.abs(all_errs)))
            error_dict[f"{event_type}_mse"] = float(np.mean(np.square(all_errs)))
        else:
            error_dict[f"{event_type}_mae"] = float('nan')
            error_dict[f"{event_type}_mse"] = float('nan')

        if start_end_errors:
            error_dict[f"{event_type}_start_end_error"] = float(np.mean(np.square(start_end_errors)))
        else:
            error_dict[f"{event_type}_start_end_error"] = float('nan')

    # Day boundary errors
    user_file = os.path.join(data_dir, f"{user_id}_processed.xlsx")
    day_boundaries = [0]
    if os.path.exists(user_file):
        df = pd.read_excel(user_file)
        current_length = 0
        for i, date_str in enumerate(dates):
            df_row = df[df['date'] == date_str]
            if df_row.empty:
                continue
            charge_series = safe_parse(df_row.iloc[0].get('charge.timeseries', '[]'))
            current_length += len(charge_series)
            if i < len(dates) - 1:
                day_boundaries.append(current_length)

    start_errors = []
    end_errors = []
    for day_idx in range(len(dates)):
        if day_idx < len(day_boundaries):
            day_start = day_boundaries[day_idx]
        else:
            break
        if day_idx < len(day_boundaries) - 1:
            day_end = day_boundaries[day_idx + 1] - 1
        else:
            day_end = min_len - 1

        if day_end >= min_len or day_start >= min_len:
            continue

        if day_idx > 0:
            start_errors.append(float(np_preds[day_start] - np_targets[day_start]))
        end_errors.append(float(np_preds[day_end] - np_targets[day_end]))

    error_dict['day_start_mae'] = float(np.mean(np.abs(start_errors))) if start_errors else float('nan')
    error_dict['day_start_mse'] = float(np.mean(np.square(start_errors))) if start_errors else float('nan')
    error_dict['day_end_mae'] = float(np.mean(np.abs(end_errors))) if end_errors else float('nan')
    error_dict['day_end_mse'] = float(np.mean(np.square(end_errors))) if end_errors else float('nan')

    # Overall trajectory error
    overall_errors = np_preds[:min_len] - np_targets[:min_len]
    error_dict['overall_traj_mae'] = float(np.mean(np.abs(overall_errors)))
    error_dict['overall_traj_mse'] = float(np.mean(np.square(overall_errors)))

    return error_dict


def less_than_5_biocharge_patch(charge_pct):
    """Cap biocharge to [5, 100] range."""
    return max(5.0, min(100.0, charge_pct))


# ---------------------------------------------------------------------------
# BiochargePredictor
# ---------------------------------------------------------------------------

class BiochargePredictor:
    """Loads a trained model and runs inference."""

    def __init__(
        self,
        checkpoint_dir: str | None = None,
        mlflow_model_uri: str | None = None,
        device: str | None = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_dir = checkpoint_dir

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

        # Determine scaling from normalization type
        normalization_type = self.config.get("normalization_type", "znorm")
        self.custom_scaling = (normalization_type == "max")

    def _load_config(self, checkpoint_dir: str) -> dict:
        config_path = os.path.join(checkpoint_dir, "model_config.json")
        with open(config_path) as f:
            return json.load(f)

    def _load_from_checkpoint(self, checkpoint_dir: str):
        config = self.config
        model_type = config.get("model_type", "mlp")

        if model_type == "gated_dual_head":
            model = GatedDualHeadMLP(
                input_dim=config["input_dim"],
            )
            logger.info("Created GatedDualHeadMLP: input_dim=%d", config["input_dim"])
        else:
            model = MLP_delta(
                input_dim=config["input_dim"],
                hidden=config.get("hidden_dim", config.get("hidden", 128)),
                layers=config.get("num_layers", config.get("layers", 3)),
                dropout=config.get("dropout", 0.1),
                norm=config.get("norm_type", config.get("norm", "layer")),
            )

        ckpt_files = sorted(Path(checkpoint_dir).glob("best_model_*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

        latest_ckpt = ckpt_files[-1]
        model.load_state_dict(torch.load(latest_ckpt, map_location=self.device, weights_only=True))
        logger.info("Loaded weights from: %s", latest_ckpt)
        return model

    # --- Simple prediction (existing) ---

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference on a batch of feature vectors."""
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.device)
            yhat = self.model(x)
            return yhat.cpu().numpy().flatten()

    def predict_single(self, features: np.ndarray) -> float:
        """Predict delta charge for a single feature vector."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return float(self.predict(features)[0])

    # --- Autoregressive inference ---

    def run_autoregressive(self, dataset: TorchBiochargeDataset) -> dict:
        """Run autoregressive inference over a TorchBiochargeDataset.

        At each timestep the model predicts a delta charge value. The predicted
        delta is accumulated into an absolute charge level. The past-charge
        feature in the input is replaced with the current predicted charge.

        During non-wear periods (mask < 0.5) the ground-truth delta is used.

        Returns:
            dict with keys: preds, targets, meta, mae, mse, rmse
        """
        self.model.eval()

        init_charge = dataset[1]["x"][-1]
        start_idx = 1

        current_pred_charge = init_charge.clone()
        current_target_charge = init_charge.clone()

        preds = [init_charge.item() * 100]
        targets = [init_charge.item() * 100]
        meta_dict = {}

        with torch.no_grad():
            for i in range(start_idx, len(dataset)):
                sample = dataset[i]

                uid = sample["meta"]["user_id"]
                date = sample["meta"]["date"]
                if uid not in meta_dict:
                    meta_dict[uid] = set()
                meta_dict[uid].add(date)

                x = sample["x"].clone()
                y = sample["y"]

                # Autoregressive: inject current predicted charge
                x[..., -1] = current_pred_charge

                x_in = x.unsqueeze(0).float().to(self.device)

                model_output = self.model(x_in)
                if isinstance(model_output, tuple):
                    yhat, _ = model_output
                else:
                    yhat = model_output

                if self.custom_scaling:
                    delta_pred = (1.1 * (yhat.cpu().item() + 1) / 2 - 0.4) / 100
                    delta_target = (1.1 * (y.cpu().item() + 1) / 2 - 0.4) / 100
                else:
                    if torch.isnan(yhat).any() or torch.isinf(yhat).any():
                        logger.warning("Model produced NaN/Inf at step %d, using 0", i)
                        delta_pred = 0.0
                    else:
                        delta_pred = yhat.cpu().item() * 0.01
                    delta_target = y.cpu().item() * 0.01

                mask = sample.get("mask", torch.tensor([1.0]))
                if mask.item() < 0.5:
                    delta_pred = delta_target

                current_pred_charge += delta_pred
                current_target_charge += delta_target

                # Apply biocharge capping
                pred_pct = less_than_5_biocharge_patch(current_pred_charge.item() * 100)
                current_pred_charge = torch.tensor(pred_pct / 100.0)

                preds.append(pred_pct)
                targets.append(current_target_charge.item() * 100)

        for uid in meta_dict:
            meta_dict[uid] = sorted(list(meta_dict[uid]))

        errors = [p - t for p, t in zip(preds, targets)]
        mae = float(np.mean(np.abs(errors)))
        mse = float(np.mean(np.square(errors)))
        rmse = float(np.sqrt(mse))

        return {
            "preds": preds,
            "targets": targets,
            "meta": meta_dict,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
        }

    def run_inference_for_user(
        self,
        user_id: str,
        dates: list[str],
        data_dir: str,
        torch_dir: str | None = None,
        zscores_file: str = "data/source/updated_z_score_user.json",
    ) -> dict:
        """Run the full autoregressive pipeline for a user and date(s).

        Handles .pt resolution, temp-index creation, dataset building,
        autoregressive prediction, and region-based error computation.

        Returns:
            dict with keys: user_id, dates, num_samples, preds, targets,
                            mae, mse, rmse, pred_range, target_range, meta,
                            region_errors
        """
        if not self.checkpoint_dir:
            raise ValueError(
                "run_inference_for_user requires a checkpoint_dir "
                "(needed for model_config.json dataset params)"
            )

        user_id = str(user_id)
        temp_dir = tempfile.mkdtemp(prefix="biocharge_inference_")

        try:
            # Step 1: resolve .pt source
            if torch_dir:
                existing_pt = os.path.join(torch_dir, f"{user_id}.pt")
                if os.path.exists(existing_pt):
                    pt_dir = torch_dir
                else:
                    excel_to_pt(user_id, data_dir, temp_dir)
                    pt_dir = temp_dir
            else:
                excel_to_pt(user_id, data_dir, temp_dir)
                pt_dir = temp_dir

            # Step 2: temp index CSV
            temp_index_csv = os.path.join(temp_dir, "temp_index.csv")
            create_temp_index(user_id, dates, data_dir, temp_index_csv)

            # Step 3: dataset config
            cfg = create_inference_config(
                model_config_path=self.checkpoint_dir,
                temp_pt_dir=pt_dir,
                temp_index_csv=temp_index_csv,
                zscores_file=zscores_file,
            )

            # Step 4: build dataset
            dataset = TorchBiochargeDataset(cfg)
            logger.info("Dataset size: %d samples", len(dataset))

            # Step 5: autoregressive inference
            results = self.run_autoregressive(dataset)

            # Step 6: region-based errors
            region_errors = calculate_region_errors(
                preds=results["preds"],
                targets=results["targets"],
                meta_dict=results["meta"],
                data_dir=data_dir,
            )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        preds = results["preds"]
        targets = results["targets"]

        return {
            "user_id": user_id,
            "dates": dates,
            "num_samples": len(preds),
            "preds": preds,
            "targets": targets,
            "mae": results["mae"],
            "mse": results["mse"],
            "rmse": results["rmse"],
            "pred_range": [float(min(preds)), float(max(preds))],
            "target_range": [float(min(targets)), float(max(targets))],
            "meta": results["meta"],
            "region_errors": region_errors,
        }

    # --- Plot generation ---

    @staticmethod
    def generate_plot(
        preds: list[float],
        targets: list[float],
        user_id: str,
        dates: list[str],
        mae: float,
        mse: float,
        output_path: str | None = None,
    ) -> bytes:
        """Create a prediction-vs-actual charge plot.

        Returns PNG bytes (for MLflow log_artifact / Grafana image panel).
        If *output_path* is provided the PNG is also saved to disk.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        date_str = "_".join(dates)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(targets, label="Ground Truth", color="blue", alpha=0.8)
        ax.plot(preds, label="Predicted (Autoregressive)", color="red", alpha=0.8)
        ax.set_xlabel("Timestep (minutes)")
        ax.set_ylabel("Charge (%)")
        ax.set_title(
            f"Biocharge Inference — User {user_id}, {date_str}\n"
            f"MAE={mae:.2f}, MSE={mse:.2f}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        png_bytes = buf.read()

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(png_bytes)
            logger.info("Plot saved to: %s", output_path)

        return png_bytes
