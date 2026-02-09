"""
Data ingestion wrapper for pulling biocharge data from Xiaomi/Huami APIs.

Wraps the standalone copy of pull_biocharge_data.py with:
- Environment variable-based token management (no hardcoded secrets)
- Structured logging
- Return values suitable for pipeline orchestration
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor

from src.data_ingestion.pull_biocharge_data import (
    fetch_all_data_for_period,
    process_fetched_data,
    fetch_all_data_from_local,
)

logger = logging.getLogger(__name__)


def get_tokens() -> tuple[str, str]:
    """Read API tokens from environment variables."""
    supertoken = os.environ.get("XIAOMI_SUPERTOKEN")
    token = os.environ.get("XIAOMI_TOKEN")
    if not supertoken:
        raise ValueError("XIAOMI_SUPERTOKEN environment variable is required. See .env.example")
    if not token:
        token = supertoken
    return supertoken, token


def pull_user_data(
    userid: str,
    from_date: str,
    to_date: str,
    data_dir: str,
    mode: str = "ONLINE",
) -> dict | None:
    """
    Pull and process data for a single user.

    Args:
        userid: User ID string
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        data_dir: Base data directory (will create raw/, score/, sleep/ subdirs)
        mode: "ONLINE" to fetch from API, "OFFLINE" to read from local files

    Returns:
        dict with raw data DataFrames, or None if no data found
    """
    supertoken, token = get_tokens()

    raw_path = os.path.join(data_dir, "raw", "user_raw_data")
    score_path = os.path.join(data_dir, "raw", "user_score_data")
    sleep_path = os.path.join(data_dir, "raw", "user_sleep_data")

    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(score_path, exist_ok=True)
    os.makedirs(sleep_path, exist_ok=True)

    logger.info("Pulling data for user %s from %s to %s (mode=%s)", userid, from_date, to_date, mode)

    if mode == "ONLINE":
        fetched = fetch_all_data_for_period(
            userid, from_date, to_date, supertoken, token, raw_path
        )
    elif mode == "OFFLINE":
        fetched = fetch_all_data_from_local(userid, raw_path)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if fetched is None:
        logger.warning("No data returned for user %s", userid)
        return None

    process_fetched_data(userid, fetched, score_path, sleep_path)
    logger.info("Data processing complete for user %s", userid)
    return fetched


def pull_multiple_users(
    user_configs: list[dict],
    data_dir: str,
    mode: str = "ONLINE",
    max_workers: int = 6,
) -> dict[str, dict | None]:
    """
    Pull data for multiple users in parallel.

    Args:
        user_configs: List of dicts with keys: userid, from_date, to_date
        data_dir: Base data directory
        mode: "ONLINE" or "OFFLINE"
        max_workers: Max parallel threads

    Returns:
        Dict mapping userid -> fetched data (or None)
    """
    results = {}

    def _pull_one(cfg):
        uid = cfg["userid"]
        try:
            data = pull_user_data(uid, cfg["from_date"], cfg["to_date"], data_dir, mode)
            return uid, data
        except Exception as e:
            logger.error("Failed to pull data for user %s: %s", uid, e)
            return uid, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for uid, data in pool.map(_pull_one, user_configs):
            results[uid] = data

    logger.info("Pulled data for %d/%d users", sum(1 for v in results.values() if v), len(user_configs))
    return results
