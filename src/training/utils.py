import ast
import json
import math
import random
from typing import Any, List, Optional

import numpy as np
import pandas as pd

def parse_series(cell: Any) -> Optional[np.ndarray]:
    """
    Convert a cell value into a numpy array.
    Supports: already-list/array, JSON string, python-literal string, comma-separated string.
    Returns None if cannot parse.
    """
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return None
    if isinstance(cell, (list, tuple, np.ndarray, pd.Series)):
        try:
            return np.asarray(cell, dtype=float)
        except Exception:
            return np.asarray(cell)
    if isinstance(cell, (int, float)):
        return np.asarray([cell], dtype=float)

    s = str(cell).strip()
    if len(s) == 0:
        return None

    # Try JSON
    try:
        v = json.loads(s)
        return np.asarray(v, dtype=float)
    except Exception:
        pass

    # Try Python literal
    try:
        v = ast.literal_eval(s)
        return np.asarray(v, dtype=float)
    except Exception:
        pass

    # Try comma-separated
    if "," in s:
        try:
            v = [float(x) for x in s.split(",")]
            return np.asarray(v, dtype=float)
        except Exception:
            pass

    return None


def safe_get_series(row: pd.Series, col: str, aliases: dict) -> Optional[np.ndarray]:
    """
    Fetch a timeseries column with alias fallbacks, parsed to ndarray.
    """
    cand = col
    if cand not in row.index and col in aliases:
        for alt in aliases[col]:
            if alt in row.index:
                cand = alt
                break
    if cand not in row.index:
        return None
    return parse_series(row[cand])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def balanced_sample_indices(
    idx_pool: np.ndarray,
    awake_mask: np.ndarray,
    sleep_mask: np.ndarray,
    target_total: int,
) -> np.ndarray:
    """
    Sample up to target_total indices, balancing awake vs sleep when possible.
    """
    awake_idxs = idx_pool[awake_mask]
    sleep_idxs = idx_pool[sleep_mask]

    half = target_total // 2
    take_awake = min(half, len(awake_idxs))
    take_sleep = min(half, len(sleep_idxs))

    sampled = []
    if take_awake > 0:
        sampled.extend(np.random.choice(awake_idxs, size=take_awake, replace=False))
    if take_sleep > 0:
        sampled.extend(np.random.choice(sleep_idxs, size=take_sleep, replace=False))

    remaining = target_total - len(sampled)
    if remaining > 0:
        remaining_pool = np.setdiff1d(idx_pool, np.asarray(sampled))
        if len(remaining_pool) > 0:
            remaining_take = min(remaining, len(remaining_pool))
            sampled.extend(np.random.choice(remaining_pool, size=remaining_take, replace=False))

    return np.sort(np.asarray(sampled, dtype=int))


# Column alias map to adapt to your file
COLUMN_ALIASES = {
    # Requested -> possible alternatives seen in your file
    "sleep.duration_score": ["sleep_duration", "sleep_score"],
    "sleep.start_time_score": ["sleep_start_time"],
    "sleep.deep_sleep_score": ["deep_sleep_duration"],
    "sleep.waso_score": ["wake_up_scenario"],  # weak proxy
    "sleep.total_score": ["sleep_score"],
    "health.score": ["fitness.fitness_percentage"],  # proxy
    "chronic_daily_stress": ["param.updated_mental_final_value"],  # proxy

    "sleep_rhr": ["heart.rhr"],
    "sleep_hrv": ["timeseries.hrr_minute_list", "timeseries.normalized_hrr"],  # proxy

    "timeseries.exercise": ["timeseries.minute_exertion", "timeseries.exercise"],
    "timeseries.nap_state": ["timeseries.min_status_list", "timeseries.nap_state"],
    "timeseries.sleep_markers": ["timeseries.min_status_list", "timeseries.sleep_markers"],
    "timeseries.minute_trimp": ["timeseries.minute_trimp", "timeseries.nornalized_minute_trimp", "timeseries.normalized_minute_trimp"],
    "timeseries.hrr_filtered": ["timeseries.hrr_minute_list", "timeseries.normalized_hrr"],
    "timeseries.full_stress_list": ["timeseries.full_stress_list"],
    "heart.hr_max": ["heart.hr_max"],
    "charge.timeseries": ["charge.timeseries", "biocharge.timeseries"],
}

