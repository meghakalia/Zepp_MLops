"""
Prefect flow for data ingestion from Xiaomi/Huami APIs.

Pulls health data for configured users and saves to data/raw/.
Can be scheduled as a daily Prefect deployment.
"""

import os
import logging

from dotenv import load_dotenv
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import datetime, timedelta

load_dotenv()


@task(retries=3, retry_delay_seconds=60)
def pull_single_user(userid: str, from_date: str, to_date: str, data_dir: str) -> bool:
    """Pull data for a single user with retries."""
    logger = get_run_logger()
    logger.info("Pulling data for user %s (%s to %s)", userid, from_date, to_date)

    from src.data_ingestion.puller import pull_user_data

    result = pull_user_data(
        userid=userid,
        from_date=from_date,
        to_date=to_date,
        data_dir=data_dir,
        mode="ONLINE",
    )

    success = result is not None
    if success:
        logger.info("Successfully pulled data for user %s", userid)
    else:
        logger.warning("No data found for user %s", userid)

    return success


@task
def get_user_configs() -> list[dict]:
    """Read user configurations from environment variables."""
    user_ids = os.environ.get("USER_IDS", "").split(",")
    from_date = os.environ.get("PULL_FROM_DATE", (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
    to_date = os.environ.get("PULL_TO_DATE", datetime.now().strftime("%Y-%m-%d"))

    configs = []
    for uid in user_ids:
        uid = uid.strip()
        if uid:
            configs.append({"userid": uid, "from_date": from_date, "to_date": to_date})

    return configs


@flow(name="data-ingestion-flow", log_prints=True)
def data_ingestion_flow(
    user_ids: list[str] | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    data_dir: str = "./data",
):
    """
    Daily data ingestion flow.

    Pulls health data from Xiaomi/Huami APIs for all configured users.
    Can be triggered manually with specific user_ids/dates, or uses
    environment variable defaults for scheduled runs.
    """
    logger = get_run_logger()

    if user_ids and from_date and to_date:
        configs = [{"userid": uid, "from_date": from_date, "to_date": to_date} for uid in user_ids]
    else:
        configs = get_user_configs()

    if not configs:
        logger.warning("No user configurations found. Set USER_IDS env var.")
        return {"total": 0, "success": 0, "failed": 0}

    logger.info("Starting data ingestion for %d users", len(configs))

    results = []
    for cfg in configs:
        success = pull_single_user(
            userid=cfg["userid"],
            from_date=cfg["from_date"],
            to_date=cfg["to_date"],
            data_dir=data_dir,
        )
        results.append(success)

    n_success = sum(results)
    summary = {
        "total": len(configs),
        "success": n_success,
        "failed": len(configs) - n_success,
    }
    logger.info("Data ingestion complete: %s", summary)
    return summary


if __name__ == "__main__":
    data_ingestion_flow()
