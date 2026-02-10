"""
End-to-end Prefect flow: data ingestion → training → deployment.

Orchestrates the full pipeline from pulling fresh data to training
a new model and making it available for inference.
"""

import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from prefect import flow, get_run_logger
from prefect.deployments import run_deployment

from src.pipelines.data_flow import data_ingestion_flow
from src.pipelines.train_flow import training_flow
from src.pipelines.inference_flow import fresh_inference_flow

load_dotenv()


@flow(name="full-pipeline", log_prints=True)
def full_pipeline(
    user_ids: list[str] | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    data_dir: str = "./data",
    training_config: dict | None = None,
    run_inference: bool = False,
):
    """
    End-to-end pipeline: pull data → train model → (optional) run inference.

    Args:
        user_ids: List of user IDs (or read from env)
        from_date: Data pull start date
        to_date: Data pull end date
        data_dir: Base data directory
        training_config: Model training hyperparameters
        run_inference: If True, run fresh inference after training to validate new model
    """
    logger = get_run_logger()

    # Step 1: Pull data
    logger.info("=== Step 1: Data Ingestion ===")
    ingestion_result = data_ingestion_flow(
        user_ids=user_ids,
        from_date=from_date,
        to_date=to_date,
        data_dir=data_dir,
    )
    logger.info("Ingestion result: %s", ingestion_result)

    if ingestion_result["success"] == 0:
        logger.error("No data was ingested. Skipping training.")
        return {"status": "failed", "reason": "no_data_ingested"}

    # Step 2: Train model
    logger.info("=== Step 2: Model Training ===")
    processed_dir = os.path.join(data_dir, "processed")
    training_result = training_flow(
        data_dir=processed_dir,
        config=training_config,
    )
    logger.info("Training result: %s", training_result)

    # Step 3 (optional): Run fresh inference to validate new model
    inference_result = None
    if run_inference:
        logger.info("=== Step 3: Fresh Inference Validation ===")
        inference_result = fresh_inference_flow(
            user_ids=user_ids,
            from_date=from_date,
            to_date=to_date,
            model_dir=training_result.get("checkpoint_dir", "models/production"),
            data_dir=data_dir,
        )
        logger.info(
            "Inference result: total=%d, success=%d, failed=%d",
            inference_result["total"], inference_result["success"], inference_result["failed"],
        )

    return {
        "status": "success",
        "ingestion": ingestion_result,
        "training": training_result,
        "inference": inference_result,
    }


if __name__ == "__main__":
    full_pipeline()
