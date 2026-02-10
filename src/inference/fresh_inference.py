"""
Fresh data inference pipeline.

Pulls raw wearable data for a user, computes biocharge analytical ground truth
using the charge analytics engine, then runs autoregressive ML inference.

Usage (CLI):
    python -m src.inference.fresh_inference \
        --user_id 1000836634 \
        --from_date 2025-03-01 \
        --to_date 2025-03-15 \
        --model_dir models/production \
        --plot

Usage (Python):
    from src.inference.fresh_inference import run_fresh_inference
    results = run_fresh_inference(
        user_id="1000836634",
        from_date="2025-03-01",
        to_date="2025-03-15",
        model_dir="models/production",
    )
"""

import argparse
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta

from src.data_ingestion.puller import pull_user_data
from src.inference.charge_analytics import compute_biocharge_ground_truth
from src.inference.predictor import BiochargePredictor

logger = logging.getLogger(__name__)


def _generate_date_list(from_date: str, to_date: str) -> list[str]:
    """Generate list of YYYY-MM-DD strings between from_date and to_date inclusive."""
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def _validate_processed_columns(excel_path: str) -> list[str]:
    """
    Check that the processed Excel contains the columns required by inference.
    Returns list of missing column names.
    """
    import pandas as pd
    from src.inference.predictor import COLUMNS_TO_READ

    df = pd.read_excel(excel_path)
    missing = [col for col in COLUMNS_TO_READ if col not in df.columns]
    return missing


def run_fresh_inference(
    user_id: str,
    from_date: str,
    to_date: str,
    model_dir: str = "models/production",
    data_dir: str = "./data",
    zscores_file: str = "data/source/updated_z_score_user.json",
    pull_mode: str = "ONLINE",
    skip_pull: bool = False,
    skip_ground_truth: bool = False,
    plot: bool = False,
    output_dir: str = "outputs",
    predictor: "BiochargePredictor | None" = None,
) -> dict:
    """
    End-to-end fresh data inference pipeline.

    Steps:
        1. Pull fresh wearable data from API (or skip if data already exists)
        2. Compute biocharge analytical ground truth via charge analytics engine
        3. Run autoregressive ML inference
        4. Optionally generate a plot

    Args:
        user_id: User ID string
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        model_dir: Path to trained model directory (model_config.json + best_model_*.pt)
        data_dir: Base data directory (puller writes to data_dir/raw/)
        zscores_file: Path to z-score normalization JSON
        pull_mode: "ONLINE" for API pull, "OFFLINE" for local files
        skip_pull: If True, skip data pull (assume raw data already exists)
        skip_ground_truth: If True, skip ground truth computation
                           (assume {user_id}_processed.xlsx already exists in data_dir)
        plot: If True, save a PNG plot of predicted vs actual
        output_dir: Directory for plot output
        predictor: Optional pre-loaded BiochargePredictor instance (avoids reloading model)

    Returns:
        dict with keys: user_id, dates, num_samples, preds, targets,
                        mae, mse, rmse, pred_range, target_range,
                        processed_excel_path, timings, total_seconds,
                        stages_completed, plot_path (if plot=True)
    """
    user_id = str(user_id)
    temp_dir = tempfile.mkdtemp(prefix="biocharge_fresh_")
    timings = {}
    stages_completed = []

    try:
        # ---- Stage 1: Pull fresh data ----
        logger.info("=" * 60)
        logger.info("STAGE 1: Data Pull")
        logger.info("=" * 60)
        t0 = time.time()
        if skip_pull:
            logger.info("Skipping data pull (--skip_pull)")
        else:
            logger.info(
                "Pulling data for user %s from %s to %s (mode=%s)",
                user_id, from_date, to_date, pull_mode,
            )
            pull_result = pull_user_data(
                userid=user_id,
                from_date=from_date,
                to_date=to_date,
                data_dir=data_dir,
                mode=pull_mode,
            )
            if pull_result is None:
                raise RuntimeError(
                    f"Data pull returned no data for user {user_id} "
                    f"({from_date} to {to_date}). Check API tokens and user ID."
                )
            logger.info("Data pull complete.")
        timings["data_pull_seconds"] = time.time() - t0
        stages_completed.append("pull")

        # data_folder is what charge_old expects: parent of user_score_data/ and user_sleep_data/
        raw_data_folder = os.path.join(data_dir, "raw")

        # Validate that the expected files exist
        score_file = os.path.join(raw_data_folder, "user_score_data", f"{user_id}.csv")
        sleep_dir = os.path.join(raw_data_folder, "user_sleep_data", str(user_id))
        if not os.path.exists(score_file):
            raise FileNotFoundError(
                f"Score data not found: {score_file}\n"
                f"Ensure data was pulled successfully or check data_dir ({data_dir})."
            )
        if not os.path.isdir(sleep_dir):
            raise FileNotFoundError(
                f"Sleep data directory not found: {sleep_dir}\n"
                f"Ensure data was pulled successfully or check data_dir ({data_dir})."
            )
        logger.info("  Score data: %s", score_file)
        logger.info("  Sleep data: %s", sleep_dir)

        # ---- Stage 2: Compute biocharge ground truth ----
        logger.info("=" * 60)
        logger.info("STAGE 2: Biocharge Ground Truth Computation")
        logger.info("=" * 60)
        t0 = time.time()

        if skip_ground_truth:
            # Expect processed Excel already in data_dir
            result_path = data_dir
            processed_excel = os.path.join(result_path, f"{user_id}_processed.xlsx")
            if not os.path.exists(processed_excel):
                raise FileNotFoundError(
                    f"--skip_ground_truth set but processed file not found: {processed_excel}"
                )
            logger.info("Skipping ground truth (--skip_ground_truth), using: %s", processed_excel)
        else:
            result_path = os.path.join(temp_dir, "processed")
            logger.info("Computing biocharge analytical values for user %s...", user_id)
            processed_excel = compute_biocharge_ground_truth(
                user_id=user_id,
                data_folder=raw_data_folder,
                result_path=result_path,
            )
            logger.info("Ground truth saved: %s", processed_excel)
        timings["ground_truth_seconds"] = time.time() - t0
        stages_completed.append("ground_truth")

        # Validate columns
        missing_cols = _validate_processed_columns(processed_excel)
        if missing_cols:
            logger.warning(
                "Processed Excel is missing %d expected columns: %s",
                len(missing_cols), missing_cols,
            )
        else:
            logger.info("All required inference columns present.")

        # ---- Stage 3: Run inference ----
        logger.info("=" * 60)
        logger.info("STAGE 3: ML Inference")
        logger.info("=" * 60)
        t0 = time.time()
        dates = _generate_date_list(from_date, to_date)
        logger.info("Running autoregressive inference for %d dates...", len(dates))

        if predictor is None:
            predictor = BiochargePredictor(checkpoint_dir=model_dir)
        results = predictor.run_inference_for_user(
            user_id=user_id,
            dates=dates,
            data_dir=result_path,  # directory containing {user_id}_processed.xlsx
            zscores_file=zscores_file,
        )
        timings["inference_seconds"] = time.time() - t0
        stages_completed.append("inference")

        results["processed_excel_path"] = processed_excel
        results["timings"] = timings
        results["total_seconds"] = sum(timings.values())
        results["stages_completed"] = stages_completed

        # ---- Stage 4: Optional plot ----
        if plot:
            logger.info("=" * 60)
            logger.info("STAGE 4: Plot")
            logger.info("=" * 60)
            try:
                date_str = f"{from_date}_to_{to_date}"
                plot_path = os.path.join(output_dir, f"fresh_inference_{user_id}_{date_str}.png")
                os.makedirs(output_dir, exist_ok=True)
                png_bytes = BiochargePredictor.generate_plot(
                    preds=results["preds"],
                    targets=results["targets"],
                    user_id=user_id,
                    dates=dates,
                    mae=results["mae"],
                    mse=results["mse"],
                )
                if png_bytes:
                    with open(plot_path, "wb") as f:
                        f.write(png_bytes)
                    results["plot_path"] = plot_path
                    stages_completed.append("plot")
                    logger.info("Plot saved: %s", plot_path)
            except Exception as e:
                logger.warning("Plot generation failed: %s", e)

        # ---- Summary ----
        logger.info("=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        logger.info("User: %s", results["user_id"])
        logger.info("Dates: %s", results["dates"])
        logger.info("Samples: %s", results["num_samples"])
        logger.info("MAE: %.4f", results["mae"])
        logger.info("MSE: %.4f", results["mse"])
        logger.info("RMSE: %.4f", results["rmse"])
        logger.info("Pred range: %s", results["pred_range"])
        logger.info("Target range: %s", results["target_range"])
        logger.info("Total time: %.2fs (pull=%.2fs, gt=%.2fs, inference=%.2fs)",
                     results["total_seconds"],
                     timings.get("data_pull_seconds", 0),
                     timings.get("ground_truth_seconds", 0),
                     timings.get("inference_seconds", 0))

        return results

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pull fresh data, compute biocharge ground truth, and run inference"
    )
    parser.add_argument("--user_id", default="1000836634", help="User ID (default: 1000836634)")
    parser.add_argument("--from_date", default="2025-03-15", help="Start date (YYYY-MM-DD, default: 2025-03-15)")
    parser.add_argument("--to_date", default="2025-03-16", help="End date (YYYY-MM-DD, default: 2025-03-15)")
    parser.add_argument(
        "--model_dir", default="models/production",
        help="Model checkpoint directory (default: models/production)"
    )
    parser.add_argument(
        "--data_dir", default="./data",
        help="Base data directory (default: ./data)"
    )
    parser.add_argument(
        "--zscores_file", default="data/source/updated_z_score_user.json",
        help="Path to z-score JSON file"
    )
    parser.add_argument(
        "--pull_mode", default="ONLINE", choices=["ONLINE", "OFFLINE"],
        help="Data pull mode: ONLINE (API) or OFFLINE (local files)"
    )
    parser.add_argument(
        "--skip_pull", action="store_true",
        help="Skip data pull (assume raw data already exists in data_dir/raw/)"
    )
    parser.add_argument(
        "--skip_ground_truth", action="store_true",
        help="Skip ground truth computation (assume {user_id}_processed.xlsx exists)"
    )
    parser.add_argument("--plot", action="store_true", help="Generate prediction plot")
    parser.add_argument("--output_dir", default="outputs", help="Plot output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_fresh_inference(
        user_id=args.user_id,
        from_date=args.from_date,
        to_date=args.to_date,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        zscores_file=args.zscores_file,
        pull_mode=args.pull_mode,
        skip_pull=args.skip_pull,
        skip_ground_truth=args.skip_ground_truth,
        plot=args.plot,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
