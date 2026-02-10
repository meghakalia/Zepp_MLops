import pandas as pd
import numpy as np
import math
from typing import List, Dict, Tuple

"""
This module provides utility functions for calculating physiological metrics
related to mental and physical charge, stress, exertion, and overall health scores.
These functions process time-series data (like heart rate, activity) and daily
summary data to generate advanced insights.
"""

# ==============================================================================
# Public API Functions
# ==============================================================================

def update_fitness_score(
    physical_battery_morning_score: float,
    mental_battery_morning_score: float,
    physical_battery_current: float,
    physical_battery_previous: float,
    mental_battery_current: float,
    mental_battery_previous: float,
    total_score_previous: float,
    physical_weight: float = 0.5,
    mental_weight: float = 0.5,
    fitness_baseline_score: int = 30,
) -> float:
    """
    Updates the total fitness score based on changes in physical and mental battery scores.
    The contribution of each battery is attenuated by its morning score.

    Args:
        physical_battery_morning_score (float): Morning physical fitness score from the device.
        mental_battery_morning_score (float): Morning mental fitness score from the device.
        physical_battery_current (float): Current physical battery score.
        physical_battery_previous (float): Physical battery score at the previous time step.
        mental_battery_current (float): Current mental battery score.
        mental_battery_previous (float): Mental battery score at the previous time step.
        total_score_previous (float): Total score at the previous time step.
        physical_weight (float): Weight for the physical battery's contribution.
        mental_weight (float): Weight for the mental battery's contribution.
        fitness_baseline_score (int): The baseline score below which attenuation is capped.

    Returns:
        float: The newly calculated total fitness score.
    """
    # Calculate the attenuation factor for the physical battery.
    # The factor is 1 if the morning score is below the baseline, otherwise it decreases.
    physical_ratio = (fitness_baseline_score - physical_battery_morning_score) / 100
    physical_attenuation = math.exp(min(0, physical_ratio))

    # Calculate the attenuation factor for the mental battery.
    mental_ratio = (fitness_baseline_score - mental_battery_morning_score) / 100
    mental_attenuation = math.exp(min(0, mental_ratio))

    # Calculate the change contribution from each battery, scaled by weight and attenuation.
    physical_delta = (physical_battery_current - physical_battery_previous) * physical_weight * physical_attenuation
    mental_delta = (mental_battery_current - mental_battery_previous) * mental_weight * mental_attenuation

    # Update the total score by adding the deltas.
    return total_score_previous + physical_delta + mental_delta


def calculate_exertion_and_stress_metrics(
    daily_data: pd.DataFrame,
    current_day_index: int,
    timeseries_df: pd.DataFrame,
    hr_column: str,
    resting_hr: int,
    age: int,
    sex: int,
    sleep_start_index: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates various exertion and stress metrics for a given day's time-series data.
    This function's logic is aligned with the original, correct implementation.

    Args:
        daily_data (pd.DataFrame): DataFrame containing historical daily summaries.
        current_day_index (int): The index for the current day in `daily_data`.
        timeseries_df (pd.DataFrame): DataFrame with minute-by-minute data for the current day.
        hr_column (str): The name of the heart rate column in `timeseries_df`.
        resting_hr (int): The user's resting heart rate.
        age (int): The user's age.
        sex (int): The user's sex (0 for female, otherwise male).
        sleep_start_index (int): The minute index when sleep starts.

    Returns:
        A tuple containing the updated `daily_data` and `timeseries_df`.
    """
    # --- Step 1: Preprocessing and TRIMP Calculation ---
    timeseries_df = _preprocess_and_calculate_trimp(timeseries_df, hr_column, resting_hr, age, sex)
    if 'minute_trimp' not in timeseries_df or timeseries_df['minute_trimp'].sum() == 0:
        # Handle cases where preprocessing determines no valid data exists.
        zero_cols = ['minute_stress_ratio', 'minute_exertion', 'exertion_score', 'exertion_score_cumsum', 'daily_stress_accumulation']
        for col in zero_cols:
            timeseries_df[col] = 0
        return daily_data, timeseries_df
    
    timeseries_df['daily_stress_accumulation'] = timeseries_df['minute_trimp'].cumsum()

    # --- Step 2: Get Previous Day's Chronic Stress (Logic from original file) ---
    if current_day_index > 0:
        yesterday_chronic_stress = daily_data.loc[current_day_index - 1, "chronic_daily_stress"]
        if pd.isna(yesterday_chronic_stress):
            yesterday_chronic_stress = 220.0
    else:
        yesterday_chronic_stress = 220.0

    # --- Step 3: Calculate and Update Today's Chronic Stress (Logic from original file) ---
    # Get total stress for today at sleep onset and update the daily summary table.
    total_stress_today = timeseries_df['daily_stress_accumulation'].iloc[sleep_start_index]
    daily_data.loc[current_day_index, 'daily_stress_accumulation'] = total_stress_today

    # Calculate chronic stress using a 14-day window ending today (inclusive of today's data).
    window_start = max(0, current_day_index - 13)
    window_end = current_day_index
    stress_window = daily_data.loc[window_start:window_end, "daily_stress_accumulation"]
    today_chronic_stress = stress_window.rolling(window=14, min_periods=1).mean().iloc[-1]
    daily_data.loc[current_day_index, 'chronic_daily_stress'] = today_chronic_stress

    # --- Step 4: Calculate Final Minute-by-Minute Metrics (Logic from original file) ---
    # Assign the correct chronic stress value for each minute of the day.
    # Use yesterday's chronic stress before sleep onset, and today's value after.
    timeseries_df['chronic_stress_ref'] = yesterday_chronic_stress
    if sleep_start_index < len(timeseries_df):
        timeseries_df.loc[sleep_start_index:, 'chronic_stress_ref'] = today_chronic_stress
    
    # Calculate minute-level stress ratios and exertion.
    chronic_per_minute = timeseries_df['chronic_stress_ref'] / 1440
    timeseries_df['minute_stress_ratio'] = timeseries_df['minute_trimp'].divide(chronic_per_minute).fillna(0)
    
    def calculate_minute_exertion(stress_ratio: float) -> float:
        """Calculates minute-by-minute exertion using a sigmoid function."""
        sigmoid_coefficient = -0.15
        return round(100 * (1 / (1 + np.exp(sigmoid_coefficient * (stress_ratio - 1)))), 4)
    
    timeseries_df['minute_exertion'] = timeseries_df['minute_stress_ratio'].apply(calculate_minute_exertion)
    
    # --- Step 5: Calculate Final Cumulative Exertion Score ---
    timeseries_df['exertion_score'] = timeseries_df['minute_exertion'].apply(lambda x: x - 50 if x > 50 else 0.001)
    
    total_minutes = timeseries_df['exertion_score'].shape[0]
    timeseries_df['exertion_score_cumsum'] = round(
        100 * timeseries_df['exertion_score'].cumsum() / (50 * total_minutes), 2
    ).ffill()

    # Store the final exertion score for the day (value just before sleep onset).
    if sleep_start_index > 0:
        daily_data.loc[current_day_index, 'exertion_score'] = timeseries_df['exertion_score_cumsum'].iloc[sleep_start_index-1]
    else:
        # Handle case where sleep starts at the very beginning of the data.
        daily_data.loc[current_day_index, 'exertion_score'] = 0

    # Clean up temporary column.
    timeseries_df.drop(columns=['chronic_stress_ref'], inplace=True, errors='ignore')
    
    return daily_data, timeseries_df


def aggregate_scores_nonlinearly(score_dict: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Aggregates multiple scores using a nonlinear weighting scheme.
    Scores are weighted based on an exponential decay function, giving lower scores higher weights.

    Args:
        score_dict (Dict[str, float]): A dictionary of score names and their values.

    Returns:
        A tuple containing:
        - The final aggregated score.
        - A dictionary of the weights applied to each score.
    """
    if not score_dict:
        return 0.0, {}

    score_values = np.array(list(score_dict.values()))
    
    # Weights are calculated such that lower scores get more weight.
    weights = [np.exp(-(score / 100) ** 0.5) for score in score_values]
    total_weight = np.sum(weights)
    
    if total_weight == 0:
        return 0.0, {}
        
    normalized_weights = weights / total_weight
    
    weighted_score_sum = np.sum(score_values * normalized_weights)
    
    weight_dict = {key: weight for key, weight in zip(score_dict.keys(), normalized_weights)}
    
    return weighted_score_sum, weight_dict


def detect_user_status(minute_mode_list: List[int]) -> Tuple[List[int], List[int]]:
    """
    Detects user status (awake, sleep, shutdown) from minute-level firmware data.

    Args:
        minute_mode_list (List[int]): A list of minute-level status codes from firmware.

    Returns:
        A tuple containing:
        - A list of parsed statuses (0: awake, 1: sleep, 3: shutdown/not worn).
        - A list of cumulative shutdown durations in minutes.
    """
    status_list = []
    shutdown_duration_list = []
    shutdown_counter = 0

    for status_code in minute_mode_list:
        # Check the lower 4 bits for shutdown status (codes 3 or 6).
        if (status_code & 0x0F) in [3, 6]:
            shutdown_counter += 1
            status_list.append(3)  # Shutdown or not worn
        else:
            shutdown_counter = 0
            # Check bit 3 to determine if the user is sleeping.
            if (status_code & (1 << 3)) == 8:
                status_list.append(1)  # Sleep
            else:
                status_list.append(0)  # Awake
        
        shutdown_duration_list.append(shutdown_counter)

    return status_list, shutdown_duration_list


def normalize_and_fill_stress(
    raw_stress_list: List[float],
    status_list: List[int],
    heart_rate_list: List[float],
    activity_list: List[int],
    resting_hr: float,
    max_hr: float,
) -> List[float]:
    """
    Normalizes raw stress values and fills in missing data points.

    The process involves:
    1. Normalizing valid stress values based on user status (sleep/awake).
    2. Interpolating short gaps of missing stress data using HR and activity changes.
    3. Filling long gaps of missing stress data using a rolling median of recent valid stress.

    Args:
        raw_stress_list (List[float]): Raw stress values from firmware (255 for missing).
        status_list (List[int]): User status (awake, sleep, etc.) for each minute.
        heart_rate_list (List[float]): Heart rate for each minute.
        activity_list (List[int]): Activity level for each minute.
        resting_hr (float): User's resting heart rate.
        max_hr (float): User's max heart rate.

    Returns:
        List[float]: A fully processed list of stress values, normalized and with no gaps.
    """
    # 1. Normalize raw values to a common scale, marking gaps with -1.
    normalized_stress = _normalize_raw_stress(raw_stress_list, status_list)

    # 2. Fill the gaps in the normalized list.
    filled_stress = _fill_stress_gaps(
        normalized_stress=normalized_stress,
        heart_rate_list=heart_rate_list,
        activity_list=activity_list,
        resting_hr=resting_hr,
        max_hr=max_hr
    )

    return filled_stress


def calculate_hr_max(age: int) -> float:
    """
    Calculates the maximum heart rate using the Tanaka formula.
    NOTE: The constant is 207 to match the original, validated algorithm.

    Args:
        age (int): The user's age.

    Returns:
        float: The estimated maximum heart rate.
    """
    return 207 - 0.7 * age


def calculate_hrr_percentage_series(
    heart_rate_list: List[float],
    max_hr: float,
    resting_hr: float,
    default_value: int = 10,
) -> List[float]:
    """
    Calculates Heart Rate Reserve (HRR) percentage for a time series.
    It also fills short gaps of missing HR data (coded as 255 or 254)
    with the last known valid HRR percentage for up to 9 minutes.

    Args:
        heart_rate_list (List[float]): A list of minute-by-minute heart rates.
        max_hr (float): The user's maximum heart rate.
        resting_hr (float): The user's resting heart rate.
        default_value (int): The value to use for long gaps of missing data.

    Returns:
        List[float]: A list of HRR percentages.
    """
    result = []
    last_valid_hrr = 0
    fill_countdown = 0

    hr_reserve_range = max_hr - resting_hr
    if hr_reserve_range <= 0: # Avoid division by zero
        return [float(default_value)] * len(heart_rate_list)

    for hr_value in heart_rate_list:
        if hr_value != 255 and hr_value != 254:
            # Calculate and clamp the HRR percentage.
            last_valid_hrr = max(0, min(100, (hr_value - resting_hr) / hr_reserve_range * 100))
            result.append(last_valid_hrr)
            fill_countdown = 9  # Set the countdown to fill the next 9 missing values.
        elif fill_countdown > 0:
            # Use the last valid value to fill a short gap.
            result.append(last_valid_hrr)
            fill_countdown -= 1
        else:
            # Use the default value for long gaps.
            result.append(float(default_value))
            
    return result

# ==============================================================================
# Internal Helper Functions
# ==============================================================================

# --- Helpers for calculate_exertion_and_stress_metrics ---

def _calculate_minute_trimp(heart_rate: float, age: int, resting_hr: float, sex: int) -> float:
    """Calculates the Training Impulse (TRIMP) for a single minute."""
    max_hr = calculate_hr_max(age)
    
    if sex == 0:  # Female
        coeff_a, coeff_b = 0.64, 1.92
    else:  # Male
        coeff_a, coeff_b = 0.86, 1.67

    if max_hr > resting_hr:
        hr_reserve_ratio = max(0, min(1, (heart_rate - resting_hr) / (max_hr - resting_hr)))
    else:
        hr_reserve_ratio = 0
    
    trimp = float(1 * hr_reserve_ratio * coeff_a * np.exp(coeff_b * hr_reserve_ratio))
    return max(0, trimp)


def _preprocess_and_calculate_trimp(
    timeseries_df: pd.DataFrame, hr_column: str, resting_hr: int, age: int, sex: int
) -> pd.DataFrame:
    """Preprocesses heart rate data and calculates minute-by-minute TRIMP."""
    if (timeseries_df[hr_column] > 250).all():
        for col in ['minute_trimp', 'minute_stress_ratio', 'minute_exertion', 'daily_stress_accumulation']:
            timeseries_df[col] = 0
        return timeseries_df

    valid_hr_mask = (timeseries_df[hr_column] <= 200) & (timeseries_df[hr_column] > 30)
    processed_hr = timeseries_df[hr_column].where(valid_hr_mask).interpolate(method='linear', limit_direction='both').fillna(resting_hr).astype(int)
    
    timeseries_df['minute_trimp'] = processed_hr.apply(
        lambda hr: _calculate_minute_trimp(hr, age, resting_hr, sex)
    )
    return timeseries_df

# --- Helpers for normalize_and_fill_stress ---

def _normalize_raw_stress(raw_stress_list: List[float], status_list: List[int]) -> List[float]:
    """Normalizes raw stress values, marking missing values as -1."""
    normalized_stress = []
    for i, raw_stress in enumerate(raw_stress_list):
        if raw_stress == 255:
            normalized_stress.append(-1)
        else:
            if status_list[i] in [1, 3]:  # Sleep or shutdown
                stress = raw_stress / 115.0
            else:  # Awake
                stress = (raw_stress - 15.5) / 115.0
            normalized_stress.append(stress)
    return normalized_stress

def _fill_stress_gaps(
    normalized_stress: List[float], heart_rate_list: List[float], activity_list: List[int], resting_hr: float, max_hr: float
) -> List[float]:
    """Fills gaps in a normalized stress list using interpolation or median fill."""
    # First, backfill any missing values at the beginning of the list.
    try:
        first_valid_index = next(i for i, v in enumerate(normalized_stress) if v != -1)
        if first_valid_index > 0:
            normalized_stress[:first_valid_index] = [normalized_stress[first_valid_index]] * first_valid_index
    except StopIteration: # Handle case where all values are missing.
        return [0.0] * len(normalized_stress)

    # Now, fill the remaining gaps.
    filled_stress = list(normalized_stress)
    stress_buffer = []
    missing_streak = 0
    
    for i in range(len(filled_stress)):
        if filled_stress[i] != -1:
            missing_streak = 0  # Reset counter
            stress_buffer.append(filled_stress[i])
            if len(stress_buffer) > 20:
                stress_buffer.pop(0)
            continue

        missing_streak += 1
        last_valid_idx = i - missing_streak
        
        if last_valid_idx >= 0:
            if missing_streak < 10:  # Short gap: Interpolate using HR and activity.
                estimated_stress = _estimate_stress_from_delta(
                    base_stress=filled_stress[last_valid_idx],
                    base_hr=heart_rate_list[last_valid_idx],
                    base_activity=activity_list[last_valid_idx],
                    current_hr=heart_rate_list[i],
                    current_activity=activity_list[i],
                    resting_hr=resting_hr,
                    max_hr=max_hr
                )
                filled_stress[i] = estimated_stress
            else:  # Long gap: Use the median of the recent stress buffer.
                filled_stress[i] = np.median(stress_buffer) if stress_buffer else filled_stress[last_valid_idx]
    
    return [max(0, s) for s in filled_stress]  # Ensure no negative stress values


def _estimate_stress_from_delta(
    base_stress: float, base_hr: float, base_activity: float,
    current_hr: float, current_activity: float, resting_hr: float, max_hr: float
) -> float:
    """Internal helper to estimate a new stress value based on changes in HR and activity."""
    hr_change_ratio = (current_hr - base_hr) / base_hr if base_hr != 0 else 1
    hr_reserve_ratio = (current_hr - resting_hr) / (max_hr - resting_hr) if (max_hr - resting_hr) != 0 else 1
    activity_change_ratio = (current_activity - base_activity) / base_activity if base_activity != 0 else 1

    if base_stress <= 0.2:
        ratio_hr = 1.2 / (1 + np.exp(-2 * hr_change_ratio)) - 0.6
        ratio_act = 1.0 / (1 + np.exp(-1 * activity_change_ratio)) - 0.5
    elif 0.2 < base_stress <= 0.3:
        ratio_hr = 1.4 / (1 + np.exp(-2 * hr_change_ratio)) - 0.7
        ratio_act = 0.6 / (1 + np.exp(-1.5 * activity_change_ratio)) - 0.3
    else:
        ratio_hr = 1.2 / (1 + np.exp(-2 * hr_change_ratio)) - 0.7
        ratio_act = 0.6 / (1 + np.exp(-2 * activity_change_ratio)) - 0.3

    activity_press_delta = (2 if current_hr == 255 else 1) * ratio_act * base_stress
    hr_press_delta = 0 if current_hr == 255 else (ratio_hr * base_stress) + (hr_reserve_ratio * 0.1 * base_stress)
    
    total_delta = 0.6 * hr_press_delta + 0.4 * activity_press_delta
    return base_stress + total_delta

# ==============================================================================
# Legacy Functions (Kept for reference)
# ==============================================================================

def get_sleep_inertia(si: float, time_interval: float, inertia_max: float = 5, i_val: float = 0.04) -> float:
    """Calculates sleep inertia effect."""
    return -inertia_max * np.exp(-i_val * time_interval / si)

def get_circadian_output(time_in_hour: float, major_peak: float = 18, minor_peak_shift: float = 3, beta: float = 0.5) -> float:
    """Models the circadian rhythm output."""
    cos1 = np.cos(2 * math.pi * (time_in_hour - major_peak) / 24)
    cos2 = beta * np.cos(2 * math.pi * (time_in_hour - major_peak - minor_peak_shift) / 12)
    return cos1 + cos2

def get_circadian_factor(ct: float, Rt: float, Rc: float = 2880, a1: float = 7, a2: float = 5) -> float:
    """Calculates the circadian factor."""
    return ct * (a1 + a2 * (Rc - Rt) / Rc)

def main_synthetic_process(Rt: float, Rc: float) -> float:
    """Converts a raw value (Rt) to a score (Et) based on a max value (Rc)."""
    return 100 * (Rt / Rc)

def Et2Rt(Et: float, Rc: float) -> float:
    """Converts a score (Et) back to a raw value (Rt)."""
    return (Et * Rc) / 100
