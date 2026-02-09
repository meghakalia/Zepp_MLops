
"""
Autoregressive inference script for biocharge prediction.

Reads a user's Excel data, converts to .pt format, builds a TorchBiochargeDataset,
and runs autoregressive prediction where the model predicts delta charge at each
timestep and accumulates into absolute charge.

Usage:
    python -m src.inference.run_inference \
        --user_id 1000836634 \
        --dates 2025-03-15 \
        --model_dir models/production \
        --plot
"""

import argparse
import ast
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from pathlib import Path

from src.training.model import MLP_delta
from src.training.dataset_delta_v2 import DatasetConfig, TorchBiochargeDataset


# Same columns used in DeepBiocharge/data_processing_utils.py for .pt conversion
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
    """
    Convert a user's Excel file to .pt format matching the structure
    expected by TorchBiochargeDataset.

    Args:
        user_id: User identifier (string or int)
        data_dir: Directory containing {user_id}_processed.xlsx files
        output_dir: Directory to write the .pt file

    Returns:
        Path to the created .pt file
    """
    user_id = str(user_id)
    user_file = os.path.join(data_dir, f"{user_id}_processed.xlsx")

    if not os.path.exists(user_file):
        raise FileNotFoundError(f"User file not found: {user_file}")

    print(f"Loading Excel file: {user_file}")
    df = pd.read_excel(user_file)

    # Filter to available columns
    available_cols = [col for col in COLUMNS_TO_READ if col in df.columns]
    df_subset = df[available_cols + ["date"]].copy()

    # Process all columns into features_data structure
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

    # Parse string columns for backward compatibility
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

    # Save as .pt
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
    print(f"Saved .pt file: {torch_path}")
    return torch_path


def create_temp_index(user_id, dates, data_dir, output_path):
    """
    Create a temporary index CSV for the given user and dates.

    Reads the user's Excel to determine the number of timesteps per date
    (from charge.timeseries length), then writes one row per timestep.

    Args:
        user_id: User identifier
        dates: List of date strings (YYYY-MM-DD)
        data_dir: Directory containing the Excel files
        output_path: Path to write the index CSV

    Returns:
        Path to the created CSV
    """
    user_id = str(user_id)
    user_file = os.path.join(data_dir, f"{user_id}_processed.xlsx")
    df = pd.read_excel(user_file)

    rows = []
    for date_str in dates:
        df_row = df[df["date"] == date_str]
        if df_row.empty:
            print(f"Warning: No data for user {user_id} on {date_str}")
            continue

        charge_series = safe_parse(df_row.iloc[0]["charge.timeseries"])
        n_steps = len(charge_series)
        print(f"  Date {date_str}: {n_steps} timesteps")

        for idx in range(n_steps):
            rows.append({"userid": user_id, "date": date_str, "index": idx})

    index_df = pd.DataFrame(rows)
    index_df.to_csv(output_path, index=False)
    print(f"Saved temp index CSV: {output_path} ({len(rows)} rows)")
    return output_path


def create_inference_config(model_config_path, temp_pt_dir, temp_index_csv, zscores_file):
    """
    Build a DatasetConfig for inference from model_config.json.

    Args:
        model_config_path: Path to directory with model_config.json
        temp_pt_dir: Directory containing the .pt file
        temp_index_csv: Path to the temporary index CSV
        zscores_file: Path to z-score JSON file

    Returns:
        DatasetConfig instance
    """
    config_file = os.path.join(model_config_path, "model_config.json")
    with open(config_file) as f:
        model_config = json.load(f)

    print(f"Loaded model config: {config_file}")

    # Extract dataset params from model config
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

    # Handle ts_cols overrides (same logic as validation_errors.py)
    if ts_cols:
        ts_cols = list(ts_cols)  # copy
        if "timeseries.ema_hrr" in ts_cols:
            ts_cols.remove("timeseries.ema_hrr")
            ts_cols.append("timeseries.hrr_raw")
        if any(c in ts_cols for c in ["timeseries.ema_hr", "timeseries.hr_filtered", "timeseries.hrr_raw"]):
            use_hr = False
        if "timeseries.acc" in ts_cols:
            use_acceleration = False

    # Append HR/ACC if enabled and not already in ts_cols
    if use_hr and "timeseries.hr_filtered" not in ts_cols:
        ts_cols.append("timeseries.hr_filtered")
    if use_acceleration and "timeseries.acc" not in ts_cols:
        ts_cols.append("timeseries.acc")

    print(f"  sleep_cols: {sleep_cols}")
    print(f"  ts_cols: {ts_cols}")
    print(f"  static_cols: {static_cols}")
    print(f"  normalization: {normalization_type}")

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


def load_model(model_dir, device=None):
    """
    Load MLP_delta model from a checkpoint directory.

    Args:
        model_dir: Directory containing model_config.json and best_model_*.pt
        device: torch device (auto-detected if None)

    Returns:
        tuple: (model, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path) as f:
        config = json.load(f)

    model = MLP_delta(
        input_dim=config["input_dim"],
        hidden=config.get("hidden_dim", config.get("hidden", 128)),
        layers=config.get("num_layers", config.get("layers", 3)),
        dropout=config.get("dropout", 0.1),
        norm=config.get("norm_type", config.get("norm", "layer")),
    )

    # Find the latest best_model_*.pt file
    ckpt_files = sorted(Path(model_dir).glob("best_model_*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    latest_ckpt = ckpt_files[-1]
    model.load_state_dict(torch.load(latest_ckpt, map_location=device, weights_only=True))
    model.eval()
    model.to(device)
    print(f"Loaded model from {latest_ckpt} (input_dim={config['input_dim']})")

    return model, device


def run_autoregressive(model, dataset, device):
    """
    Run autoregressive inference over the dataset.

    At each timestep, the model predicts a delta charge value. The predicted
    delta is accumulated into an absolute charge level. The past charge feature
    in the input is replaced with the current predicted charge (autoregressive).

    During non-wear periods (mask < 0.5), the ground truth delta is used instead
    of the model prediction.

    Args:
        model: Loaded MLP_delta model
        dataset: TorchBiochargeDataset instance
        device: torch device

    Returns:
        tuple: (preds, targets, meta_dict)
            - preds: list of predicted absolute charge percentages (0-100)
            - targets: list of target absolute charge percentages (0-100)
            - meta_dict: {user_id: [list of dates]}
    """
    model.to(device)
    model.eval()

    # Initialize from the second sample (index 1) — past charge feature
    init_charge = dataset[1]["x"][-1]
    start_idx = 1

    current_pred_charge = init_charge.clone()
    current_target_charge = init_charge.clone()

    preds = [init_charge.item() * 100]
    targets = [init_charge.item() * 100]
    meta_dict = {}

    print(f"Running autoregressive inference over {len(dataset)} samples...")

    with torch.no_grad():
        for i in range(start_idx, len(dataset)):
            sample = dataset[i]

            # Track metadata
            uid = sample["meta"]["user_id"]
            date = sample["meta"]["date"]
            if uid not in meta_dict:
                meta_dict[uid] = set()
            meta_dict[uid].add(date)

            x = sample["x"].clone()
            y = sample["y"]

            # Autoregressive: inject current predicted charge as the past charge feature
            x[..., -1] = current_pred_charge

            x_in = x.unsqueeze(0).float().to(device)
            y_in = y.unsqueeze(0).float().to(device)

            model_output = model(x_in)
            if isinstance(model_output, tuple):
                yhat, _ = model_output
            else:
                yhat = model_output

            # Check for NaN/Inf
            if torch.isnan(yhat).any() or torch.isinf(yhat).any():
                print(f"Warning: Model produced NaN/Inf at step {i}, using 0")
                delta_pred = 0.0
            else:
                delta_pred = yhat.cpu().item() * 0.01

            delta_target = y_in.cpu().item() * 0.01

            # During non-wear, use GT delta
            mask = sample.get("mask", torch.tensor([1.0]))
            if mask.item() < 0.5:
                delta_pred = delta_target

            current_pred_charge += delta_pred
            current_target_charge += delta_target

            preds.append(current_pred_charge.item() * 100)
            targets.append(current_target_charge.item() * 100)

    # Sort dates
    for uid in meta_dict:
        meta_dict[uid] = sorted(list(meta_dict[uid]))

    return preds, targets, meta_dict


def main():
    parser = argparse.ArgumentParser(
        description="Run autoregressive biocharge inference for a user and date(s)"
    )
    parser.add_argument("--user_id", required=True, help="User ID (e.g., 1000836634)")
    parser.add_argument(
        "--dates", required=True, nargs="+",
        help="Date(s) in YYYY-MM-DD format (e.g., 2025-03-15)"
    )
    parser.add_argument(
        "--model_dir", default="models/production",
        help="Path to model directory with model_config.json and best_model_*.pt"
    )
    parser.add_argument(
        "--data_dir",
        default="data/source/z_norm_rhr_hrv_7_14_corrected_complete_original",
        help="Path to directory with user Excel files"
    )
    parser.add_argument(
        "--torch_dir",
        default="data/source/torch_oversampled_hrr_rhr_ema_waso_original/torch",
        help="Path to pre-processed .pt files (used if available, otherwise converts from Excel)"
    )
    parser.add_argument(
        "--zscores_file",
        default="data/source/updated_z_score_user.json",
        help="Path to z-score JSON file"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save a matplotlib plot of predicted vs actual charge"
    )
    parser.add_argument(
        "--output_dir", default="outputs",
        help="Directory to save plot output"
    )

    args = parser.parse_args()

    # Create temp directory for index files (and .pt if needed)
    temp_dir = tempfile.mkdtemp(prefix="biocharge_inference_")
    print(f"Temp directory: {temp_dir}")

    try:
        # Step 1: Determine .pt source
        # Prefer pre-processed .pt files; fall back to Excel conversion
        existing_pt = os.path.join(args.torch_dir, f"{args.user_id}.pt")
        if os.path.exists(existing_pt):
            print(f"\n=== Step 1: Using existing .pt file ===")
            print(f"  {existing_pt}")
            pt_dir = args.torch_dir
        else:
            print(f"\n=== Step 1: Converting Excel to .pt ===")
            excel_to_pt(args.user_id, args.data_dir, temp_dir)
            pt_dir = temp_dir

        # Step 2: Create temp index CSV
        print(f"\n=== Step 2: Creating temp index CSV ===")
        temp_index_csv = os.path.join(temp_dir, "temp_index.csv")
        create_temp_index(args.user_id, args.dates, args.data_dir, temp_index_csv)

        # Step 3: Build DatasetConfig
        print(f"\n=== Step 3: Building DatasetConfig ===")
        cfg = create_inference_config(
            model_config_path=args.model_dir,
            temp_pt_dir=pt_dir,
            temp_index_csv=temp_index_csv,
            zscores_file=args.zscores_file,
        )

        # Step 4: Build dataset
        print(f"\n=== Step 4: Building TorchBiochargeDataset ===")
        dataset = TorchBiochargeDataset(cfg)
        print(f"Dataset size: {len(dataset)} samples")

        # Step 5: Load model
        print(f"\n=== Step 5: Loading model ===")
        model, device = load_model(args.model_dir)

        # Step 6: Run autoregressive inference
        print(f"\n=== Step 6: Autoregressive Inference ===")
        preds, targets, meta = run_autoregressive(model, dataset, device)

        # Step 7: Print results
        print(f"\n{'='*50}")
        print(f"RESULTS")
        print(f"{'='*50}")
        print(f"User: {args.user_id}")
        print(f"Dates: {args.dates}")
        print(f"Samples: {len(preds)}")
        print(f"Predicted charge range: [{min(preds):.1f}, {max(preds):.1f}]")
        print(f"Target charge range:    [{min(targets):.1f}, {max(targets):.1f}]")

        # Calculate errors
        errors = [p - t for p, t in zip(preds, targets)]
        mae = np.mean(np.abs(errors))
        mse = np.mean(np.square(errors))
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}")

        # Step 8: Optional plot
        if args.plot:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                os.makedirs(args.output_dir, exist_ok=True)
                date_str = "_".join(args.dates)
                plot_path = os.path.join(
                    args.output_dir,
                    f"inference_{args.user_id}_{date_str}.png",
                )

                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(targets, label="Ground Truth", color="blue", alpha=0.8)
                ax.plot(preds, label="Predicted (Autoregressive)", color="red", alpha=0.8)
                ax.set_xlabel("Timestep (minutes)")
                ax.set_ylabel("Charge (%)")
                ax.set_title(
                    f"Biocharge Inference — User {args.user_id}, {date_str}\n"
                    f"MAE={mae:.2f}, MSE={mse:.2f}"
                )
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)
                print(f"\nPlot saved to: {plot_path}")
            except ImportError:
                print("\nWarning: matplotlib not available, skipping plot")

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up temp directory")


if __name__ == "__main__":
    main()
