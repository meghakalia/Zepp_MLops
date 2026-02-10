"""
Prefect flow for running fresh inference pipeline.

Pulls data, computes ground truth, runs ML inference for configured users.
Can be scheduled as a daily Prefect deployment alongside data_flow.

Usage:
    python -m src.pipelines.inference_flow
"""

import os
import logging
from datetime import datetime, timedelta

import mlflow
from dotenv import load_dotenv
from prefect import flow, task, get_run_logger

load_dotenv()


@task(retries=2, retry_delay_seconds=30)
def run_fresh_inference_task(
    user_id: str,
    from_date: str,
    to_date: str,
    model_dir: str,
    data_dir: str,
    zscores_file: str,
    pull_mode: str = "ONLINE",
) -> dict:
    """Run the full fresh inference pipeline for one user."""
    logger = get_run_logger()
    logger.info("Running fresh inference for user %s (%s to %s)", user_id, from_date, to_date)

    from src.inference.fresh_inference import run_fresh_inference

    results = run_fresh_inference(
        user_id=user_id,
        from_date=from_date,
        to_date=to_date,
        model_dir=model_dir,
        data_dir=data_dir,
        zscores_file=zscores_file,
        pull_mode=pull_mode,
    )

    logger.info(
        "User %s: MAE=%.4f, RMSE=%.4f, samples=%d",
        user_id, results["mae"], results["rmse"], results["num_samples"],
    )
    return results


@task
def log_results_to_mlflow(results: list[dict], experiment_name: str = "biocharge-fresh-inference"):
    """Log per-user inference results to MLflow for tracking over time."""
    logger = get_run_logger()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"fresh_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        for r in results:
            mlflow.log_metrics({
                f"{r['user_id']}_mae": r["mae"],
                f"{r['user_id']}_rmse": r["rmse"],
                f"{r['user_id']}_mse": r["mse"],
                f"{r['user_id']}_num_samples": r["num_samples"],
            })
            if "timings" in r:
                for stage, duration in r["timings"].items():
                    mlflow.log_metric(f"{r['user_id']}_{stage}", duration)

        # Aggregate metrics
        if results:
            avg_mae = sum(r["mae"] for r in results) / len(results)
            avg_rmse = sum(r["rmse"] for r in results) / len(results)
            mlflow.log_metrics({"avg_mae": avg_mae, "avg_rmse": avg_rmse})

    logger.info("Logged %d user results to MLflow", len(results))


@flow(name="fresh-inference-flow", log_prints=True)
def fresh_inference_flow(
    user_ids: list[str] | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    model_dir: str = "models/production",
    data_dir: str = "./data",
    zscores_file: str = "data/source/updated_z_score_user.json",
    pull_mode: str = "ONLINE",
):
    """
    Fresh inference flow for multiple users.

    Runs the full pull -> ground truth -> inference pipeline for each user,
    then logs aggregate results to MLflow.
    """
    logger = get_run_logger()

    if user_ids is None:
        user_ids_str = os.environ.get("USER_IDS", "")
        user_ids = [uid.strip() for uid in user_ids_str.split(",") if uid.strip()]

    if from_date is None:
        from_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    if to_date is None:
        to_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    if not user_ids:
        logger.warning("No user IDs configured. Set USER_IDS env var.")
        return {"total": 0, "success": 0, "failed": 0, "results": []}

    logger.info("Running fresh inference for %d users: %s to %s", len(user_ids), from_date, to_date)

    all_results = []
    failures = 0
    for uid in user_ids:
        try:
            result = run_fresh_inference_task(
                user_id=uid,
                from_date=from_date,
                to_date=to_date,
                model_dir=model_dir,
                data_dir=data_dir,
                zscores_file=zscores_file,
                pull_mode=pull_mode,
            )
            all_results.append(result)
        except Exception as e:
            logger.error("Failed for user %s: %s", uid, e)
            failures += 1

    if all_results:
        log_results_to_mlflow(all_results)

    summary = {
        "total": len(user_ids),
        "success": len(all_results),
        "failed": failures,
        "results": all_results,
    }
    logger.info(
        "Fresh inference flow complete: total=%d, success=%d, failed=%d",
        summary["total"], summary["success"], summary["failed"],
    )
    return summary


if __name__ == "__main__":
    fresh_inference_flow()
