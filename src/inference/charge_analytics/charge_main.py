"""
Biocharge analytical ground truth computation.

Trimmed from charge_old/charge_logic/charge_main.py — contains only
calculate_one_user() and calculate_one_day() with their helper functions.
Finetune functions and DiagnoseAndFinetune dependency removed.
"""

import ast
import json
import os
from datetime import datetime, timedelta
import copy
from tqdm import tqdm

from .sleep_metric_new import *

from .mental_battery_new import *
from .physical_battery_new import *
from . import charge_utils
from .charge_utils import *
from .readiness import *
from .online_glm import *
from .dynamic_ewma_decompose_reconstruct import *
from .t_digest_wrapper import *
from .dynamic_stats import *
from .offline_guidance import *
from .HRV_score_calculation import get_hrv_factor_for_day

import warnings
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial

warnings.filterwarnings("ignore")

np.random.seed(25)


def estimates(values, function):
    result = function(values)
    if ~np.isnan(result) and ~np.isinf(result):
        return round(result)
    else:
        return result
    

def get_last_element(list):
    result = list[-1]
    if ~np.isnan(result) and ~np.isinf(result):
        return round(result)
    else:
        return result
    

def get_previous_value(daily_data, prev_idx, column_name, default_value):
    """
    从上一行中获取指定列的值，如果无效则返回默认值。

    参数:
        daily_data (DataFrame): 包含历史数据的 DataFrame。
        prev_idx (int): 上一行的索引。
        column_name (str): 列名。
        default_value (int): 默认值。

    返回:
        int: 上一行的值或默认值。
    """
    if prev_idx >= 0 and not np.isnan(daily_data.at[prev_idx, column_name]):
        # print(f"prev_idx: {prev_idx}, column_name: {column_name}, value: {daily_data.at[prev_idx, column_name]}")
        return daily_data.at[prev_idx, column_name]
    return default_value


def process_exercise_data(timeseries_sheet, mental_result_new, physical_result_new, charge_result_new, checkpoint_times, sheet):
    """Process exercise/charge timestamps and checkpoints for mental/physical/charge metrics."""
    
    # Vectorized transition detection for both exercise and charge
    def find_transitions(series):
        shifted = series.astype(int).shift(1)
        starts = series[(shifted == 0) & (series == 1)].index.tolist()
        ends = series[(shifted == 1) & (series == 0)].index.tolist()

        if len(starts) != len(ends):
            ends.append(series.index[-1])
        return {
            'starts': starts,
            'ends': ends
        }

    # Detect transitions for both exercise and charge
    exercise_trans = find_transitions(timeseries_sheet['exercise'])
    nap_trans = find_transitions(timeseries_sheet['nap_state'])  # 复用transition检测函数[[3]]
    sleep_trans = find_transitions(timeseries_sheet['sleep_markers'])

    # Create timestamp-value entries from indices
    def create_entries(indices, *results):
        return [
            # MODIFIED: Ensured rounding to 2 decimal places
            [{str(timeseries_sheet['time'].iloc[idx]): round(result[idx], 2)} for idx in indices]
            for result in results
        ]

    # Generate all result sets
    mental_exercise_starts, physical_exercise_starts, charge_exercise_starts = create_entries(
        exercise_trans['starts'], 
        mental_result_new, 
        physical_result_new, 
        charge_result_new
    )
    
    mental_exercise_ends, physical_exercise_ends, charge_exercise_ends = create_entries(
        exercise_trans['ends'],
        mental_result_new,
        physical_result_new,
        charge_result_new
    )

    mental_nap_starts, physical_nap_starts, charge_nap_starts = create_entries(
    nap_trans['starts'], 
    mental_result_new, 
    physical_result_new, 
    charge_result_new
    )

    mental_nap_ends, physical_nap_ends, charge_nap_ends = create_entries(
        nap_trans['ends'],
        mental_result_new,
        physical_result_new,
        charge_result_new
    )

    # Checkpoint processing with date adjustment
    time_str = timeseries_sheet['time'].astype(str)
    correct_date = (datetime.strptime(sheet, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    
    checkpoints = {
        cp: time_str[time_str == f"{correct_date} {cp}"].index[0] 
        if any(time_str == f"{correct_date} {cp}") else None
        for cp in checkpoint_times
    }

    # Create all checkpoint metrics
    metric_checkpoints = {
        f"{metric}_checkpoints": {
            # MODIFIED: Ensured rounding to 2 decimal places
            cp: round(data[idx], 2) if idx is not None else None
            for cp, idx in checkpoints.items()
        }
        for metric, data in zip(
            ['mental', 'physical', 'charge'],
            [mental_result_new, physical_result_new, charge_result_new]
        )
    }
    exercise_list = [exercise_trans['starts'], exercise_trans['ends']]
    nap_list = [nap_trans['starts'], nap_trans['ends']]
    sleep_list = [sleep_trans['starts'], sleep_trans['ends']]

    return {
        "exercise_index": exercise_list,
        "nap_index": nap_list,
        "sleep_index": sleep_list,

        # exercise字段...
        "mental_exercise_starts": mental_exercise_starts,
        "physical_exercise_starts": physical_exercise_starts,
        "charge_exercise_starts": charge_exercise_starts,
        "mental_exercise_ends": mental_exercise_ends,
        "physical_exercise_ends": physical_exercise_ends,
        "charge_exercise_ends": charge_exercise_ends,

        # nap字段...
        "mental_nap_starts": mental_nap_starts,
        "physical_nap_starts": physical_nap_starts,
        "charge_nap_starts": charge_nap_starts,
        "mental_nap_ends": mental_nap_ends,
        "physical_nap_ends": physical_nap_ends,
        "charge_nap_ends": charge_nap_ends,

        **metric_checkpoints
    }


def calculate_normal_time_charge_expenditure(
    charge_dynamic, timeseries_sheet, nap_starts, nap_ends, exercise_starts, exercise_ends
):
    """
    Calculate the total charge expenditure during normal periods (excluding nap and exercise).

    Args:
        charge_dynamic (list): Charge time series.
        timeseries_sheet (DataFrame): DataFrame containing 'time' column.
        nap_starts (list): List of dicts for nap start times.
        nap_ends (list): List of dicts for nap end times.
        exercise_starts (list): List of dicts for exercise start times.
        exercise_ends (list): List of dicts for exercise end times.

    Returns:
        float: Total charge expenditure during normal time.
    """
    # Get all nap intervals
    nap_intervals = []
    if nap_starts and nap_ends and isinstance(nap_starts, list) and isinstance(nap_ends, list):
        for start_dict, end_dict in zip(nap_starts, nap_ends):
            if isinstance(start_dict, dict) and isinstance(end_dict, dict):
                if start_dict and end_dict:
                    start_time = next(iter(start_dict.keys()))
                    end_time = next(iter(end_dict.keys()))
                    try:
                        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
                    try:
                        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
                    nap_intervals.append((start_dt, end_dt))

    # Get all exercise intervals
    exercise_intervals = []
    if exercise_starts and exercise_ends and isinstance(exercise_starts, list) and isinstance(exercise_ends, list):
        for start_dict, end_dict in zip(exercise_starts, exercise_ends):
            if isinstance(start_dict, dict) and isinstance(end_dict, dict):
                if start_dict and end_dict:
                    start_time = next(iter(start_dict.keys()))
                    end_time = next(iter(end_dict.keys()))
                    try:
                        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
                    try:
                        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
                    exercise_intervals.append((start_dt, end_dt))

    # Merge all intervals
    all_intervals = nap_intervals + exercise_intervals

    def is_in_intervals(dt, intervals):
        """
        Check if dt is within any interval in intervals.

        Args:
            dt (datetime): Current time.
            intervals (list): List of (start, end) tuples.

        Returns:
            bool: True if dt is in any interval, else False.
        """
        for start, end in intervals:
            if start <= dt <= end:
                return True
        return False

    # Traverse charge_dynamic and sum expenditure during normal time
    normal_time_charge_expenditure = 0
    prev_val = None
    for idx, val in enumerate(charge_dynamic):
        # Get current time
        try:
            current_time = timeseries_sheet['time'].iloc[idx]
            if isinstance(current_time, str):
                try:
                    dt = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    dt = datetime.strptime(current_time, "%Y-%m-%d %H:%M")
            else:
                dt = pd.to_datetime(current_time)
        except Exception:
            continue

        # Skip nap and exercise intervals
        if is_in_intervals(dt, all_intervals):
            prev_val = val
            continue

        # Only count charge decrease
        if prev_val is not None and val < prev_val:
            normal_time_charge_expenditure += round(prev_val - val, 2)
        prev_val = val
    return normal_time_charge_expenditure


def calculate_sleep_recovery(charge_dynamic, sleep_state):
    """
    Calculate the total charge recovery during the main sleep period (not naps, only main sleep).

    Args:
        charge_dynamic (list): Charge time series.
        sleep_state (list): List indicating sleep state (1 for sleep, 0 for awake).

    Returns:
        float: Total charge recovery during main sleep.
    """
    # Find the main sleep interval (the longest continuous segment where sleep_state==1)
    max_len = 0
    main_sleep_start = None
    main_sleep_end = None
    current_start = None
    current_len = 0

    for idx, state in enumerate(sleep_state):
        if state == 1:
            if current_start is None:
                current_start = idx
                current_len = 1
            else:
                current_len += 1
        else:
            if current_start is not None:
                if current_len > max_len:
                    max_len = current_len
                    main_sleep_start = current_start
                    main_sleep_end = idx - 1
                current_start = None
                current_len = 0
    # Check if the last segment is the longest
    if current_start is not None and current_len > max_len:
        max_len = current_len
        main_sleep_start = current_start
        main_sleep_end = len(sleep_state) - 1

    sleep_recovery = 0.0
    # Only calculate recovery during the main sleep interval
    if main_sleep_start is not None and main_sleep_end is not None and main_sleep_end > main_sleep_start:
        prev_val = charge_dynamic[main_sleep_start]
        for idx in range(main_sleep_start + 1, main_sleep_end + 1):
            if charge_dynamic[idx] > prev_val:
                sleep_recovery += charge_dynamic[idx] - prev_val
            prev_val = charge_dynamic[idx]
    return round(sleep_recovery, 2)


# ------------------------------------------------------------------------------
# 计算正常时间的消耗和恢复（不包括小睡、锻炼、睡眠期间，且用户佩戴设备）
# ------------------------------------------------------------------------------

def calculate_normal_time_expenditure_and_recovery(
    charge_dynamic, timeseries_sheet, nap_starts, nap_ends, exercise_starts, exercise_ends, sleep_state
):
    """
    计算正常时间（用户佩戴设备，且不在小睡、锻炼、睡眠期间）的电量消耗和恢复。

    Args:
        charge_dynamic (list): 电量时间序列
        timeseries_sheet (DataFrame): 包含'time'列
        nap_starts (list): 小睡开始时间字典列表
        nap_ends (list): 小睡结束时间字典列表
        exercise_starts (list): 锻炼开始时间字典列表
        exercise_ends (list): 锻炼结束时间字典列表
        sleep_state (list): 睡眠状态序列（1为睡眠，0为清醒）

    Returns:
        tuple: (正常时间消耗总和, 正常时间恢复总和)
    """
    # 获取所有小睡区间
    nap_intervals = []
    if nap_starts and nap_ends and isinstance(nap_starts, list) and isinstance(nap_ends, list):
        for start_dict, end_dict in zip(nap_starts, nap_ends):
            if isinstance(start_dict, dict) and isinstance(end_dict, dict):
                if start_dict and end_dict:
                    start_time = next(iter(start_dict.keys()))
                    end_time = next(iter(end_dict.keys()))
                    try:
                        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
                    try:
                        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
                    nap_intervals.append((start_dt, end_dt))

    # 获取所有锻炼区间
    exercise_intervals = []
    if exercise_starts and exercise_ends and isinstance(exercise_starts, list) and isinstance(exercise_ends, list):
        for start_dict, end_dict in zip(exercise_starts, exercise_ends):
            if isinstance(start_dict, dict) and isinstance(end_dict, dict):
                if start_dict and end_dict:
                    start_time = next(iter(start_dict.keys()))
                    end_time = next(iter(end_dict.keys()))
                    try:
                        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
                    try:
                        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
                    exercise_intervals.append((start_dt, end_dt))

    # 合并所有区间
    all_intervals = nap_intervals + exercise_intervals

    def is_in_intervals(dt, intervals):
        """
        检查dt是否在任一区间内

        Args:
            dt (datetime): 当前时间
            intervals (list): (start, end)元组列表

        Returns:
            bool: True-在区间内，False-不在
        """
        for start, end in intervals:
            if start <= dt <= end:
                return True
        return False

    normal_time_expenditure = 0
    normal_time_recovery = 0
    prev_val = None

    for idx, val in enumerate(charge_dynamic):
        # 获取当前时间
        try:
            current_time = timeseries_sheet['time'].iloc[idx]
            if isinstance(current_time, str):
                try:
                    dt = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    dt = datetime.strptime(current_time, "%Y-%m-%d %H:%M")
            else:
                dt = pd.to_datetime(current_time)
        except Exception:
            prev_val = val
            continue

        # 跳过小睡、锻炼、睡眠期间
        if is_in_intervals(dt, all_intervals) or (sleep_state[idx] == 1):
            prev_val = val
            continue

        # 只统计用户佩戴设备的时段（假设charge_dynamic有值即为佩戴）
        if prev_val is not None:
            if val < prev_val:
                # 电量下降，计为消耗
                normal_time_expenditure += round(prev_val - val, 2)
            elif val > prev_val:
                # 电量上升，计为恢复
                normal_time_recovery += round(val - prev_val, 2)
        prev_val = val

    return normal_time_expenditure, normal_time_recovery


def calculate_one_day(timeseries_sheet, sheet, daily_data, daily_row, yesterday_row, current_index,
                       mental_parameters_tuned, physical_parameters_tuned, current_dir, df_guidance, 
                       model, minute_tracker, decomposer, signal_processor, original=False, press_thresh=12/115, 
                       plot=False, dynamic=False, exertion_growth_rate= 0.05, CP_threshold=15):
    mode_values = {4: 2, 8: 1, 5: 0}
    timeseries_sheet['sleep_stage'] =  timeseries_sheet['sleep_stage'].apply(lambda x: mode_values.get(x, -1))
    # Set sleep_state to 1 if sleep_stage in [0, 1, 2], else keep original value
    timeseries_sheet['sleep_state'] = timeseries_sheet.apply(
        lambda row: 1 if row['sleep_stage'] in [0, 1, 2] else row.get('sleep_state', 0),
        axis=1
    )

    hrv_factor_today = get_hrv_factor_for_day(daily_data, current_index, decay_missing=False)
    hrv_factor_yesterday = get_hrv_factor_for_day(daily_data, current_index - 1, decay_missing=False)

    if yesterday_row.shape[0] != 0:
        yesterday_wakeup_time = datetime.strptime(f"{yesterday_row.date.iloc[0]} {yesterday_row.sleep_end_time.iloc[0]}", '%Y-%m-%d %H:%M')
        yesterday_date_str = yesterday_row.date.iloc[0]
        yesterday_sleep_data_valid = not pd.isna(yesterday_row.sleep_start_time.iloc[0])
        yesterday_heart_parameters = [
        hrv_factor_yesterday,
        yesterday_row.rhrScore.iloc[0]
        ]
        yesterday_fitness_fatigue_score = yesterday_row['fitness.fitness_fatigue_difference'].iloc[0]
        yesterday_stress_fitness_fatigue_score = yesterday_row['stress.fitness_fatigue_difference'].iloc[0]
    else:
        # If no yesterday data, set default sleep start time to 11 PM yesterday
        yesterday_date = datetime.strptime(daily_row.date.iloc[0], '%Y-%m-%d') - timedelta(days=1)
        yesterday_date_str = yesterday_date.strftime('%Y-%m-%d')
        yesterday_wakeup_time = datetime.strptime(f"{yesterday_date_str} 08:00", '%Y-%m-%d %H:%M')  # Default wakeup time 8 AM
        yesterday_sleep_data_valid = False
        yesterday_heart_parameters = [1, 100]
        yesterday_fitness_fatigue_score = 0
        yesterday_stress_fitness_fatigue_score = 0

    sleep_start_time = daily_row.sleep_start_time.iloc[0]
    # Check if sleep_start_time is a string or NaN
    if isinstance(sleep_start_time, str):
        pass  # It's already a string, no conversion needed
    else:
        # If it's not a string (likely NaN or None), set default
        sleep_start_time = "22:00"
    if sleep_start_time.split(":")[0] > '12':
        today_date_str = yesterday_date_str
    else:
        today_date_str = daily_row.date.iloc[0]
    # If sleep_start_time is negative (e.g., "-22:00"), drop the negative sign # Bug fix
    if isinstance(sleep_start_time, str) and sleep_start_time.startswith('-'):
        sleep_start_time = sleep_start_time[1:]
    today_sleep_start_time = datetime.strptime(f"{today_date_str} {sleep_start_time}", '%Y-%m-%d %H:%M')
    sleep_start_time_index = int((today_sleep_start_time - yesterday_wakeup_time).total_seconds() / 60)


    current_data_index = daily_row.index[0]
    if current_data_index != current_index:
        print("Index mismatch", daily_data.userid[current_data_index], daily_data.date[current_data_index])
    age = daily_data.age[current_data_index]
    sex = daily_data.gender[current_data_index]
    wake_up_scenario = daily_row.get('wake_up_scenario', pd.Series([None])).iloc[0]

    # Create flags to indicate if sleep-related data and scores are valid
    today_sleep_data_valid = not pd.isna(sleep_start_time) and sleep_start_time_index > 0
    
    
    # Check if scores are valid (not NaN or missing)
    # Checking HRV, RHR, temperature, and AHI scores
    scores_valid = (
        not np.isnan(daily_data.hrvScore[current_data_index]) and
        not np.isnan(daily_data.rhrScore[current_data_index]) and
        not np.isnan(daily_data.skinTempScore[current_data_index]) and
        not np.isnan(daily_data.ahiScore[current_data_index])
    )
    try:
        # 尝试获取值
        prior_health_score = yesterday_row['health.score'].iloc[0]
        # 如果获取到的值是 NaN，则赋默认值 1.0
        if pd.isna(prior_health_score):
            prior_health_score = 1.0
    except (IndexError, KeyError):
        # 如果 yesterday_row 为空或没有 'health.score' 列，则直接赋默认值 1.0
        prior_health_score = 1.0

    # Initialize default values
    if current_index == 1 and yesterday_row.empty is False:
        # Get initial values from the dataframe columns. Online charge is morning charge. So we use yesterday's data.
        initial_mental_expenditure = yesterday_row.mentalWake.iloc[0] if not pd.isna(yesterday_row.mentalWake.iloc[0]) else 69
        initial_physical_expenditure = yesterday_row.physicalWake.iloc[0] if not pd.isna(yesterday_row.physicalWake.iloc[0]) else 69
        # chronic_daily_stress = daily_row.chronicWeightDaily.iloc[0] if not pd.isna(daily_row.chronicWeightDaily.iloc[0]) else 1.0
    # according to the wake_up_scenario, get the initial_physical_expenditure and initial_mental_expenditure
    elif wake_up_scenario == "Actual" and not yesterday_row.empty:
        # Try to get the value from the previous day. If it's missing or NaN, use a default.
        prev_physical_val = yesterday_row['physical.final_value'].iloc[0]
        initial_physical_expenditure = prev_physical_val if not pd.isna(prev_physical_val) else 69

        prev_mental_val = yesterday_row['mental.final_value'].iloc[0]
        initial_mental_expenditure = prev_mental_val if not pd.isna(prev_mental_val) else 69
    else:
        # 否则，执行重置逻辑
        initial_physical_expenditure = 69
        initial_mental_expenditure = 69

    if today_sleep_data_valid:
        sleep_score = calculate_sleep_score(age,
                                        daily_data.sleep_duration[current_data_index],
                                        sleep_start_time,
                                        daily_data.deep_sleep_ratio[current_data_index],
                                        daily_data.wakefulness_after_sleep_onset_frequency[current_data_index],
                                        daily_data.wakefulness_after_sleep_onset_duration[current_data_index]
                                        )

        sleep_duration_score = sleep_score[1]["sleep_duration_score"] / 100
        daily_data.loc[current_data_index, "sleep.duration_score_temp"] = sleep_duration_score # Temp storage

        sleep_start_time_score = sleep_score[1]["sleep_start_time_score"] / 100
        daily_data.loc[current_data_index, "sleep.start_time_score_temp"] = sleep_start_time_score # Temp storage

        deep_sleep_score = sleep_score[1]["deep_sleep_ratio_score"] / 100
        daily_data.loc[current_data_index, "sleep.deep_sleep_score_temp"] = deep_sleep_score # Temp storage

        WASO_score = (sleep_score[1]["WASO_frequency_score"] + sleep_score[1]["WASO_duration_score"]) / (2 * 100)
        daily_data.loc[current_data_index, "sleep.waso_score_temp"] = WASO_score # Temp storage
   
        today_sleep_parameters = [sleep_duration_score, sleep_start_time_score, deep_sleep_score, WASO_score]
    else:
        sleep_score = [1, {}] # Ensure sleep_score has a default structure
        sleep_score[1] = {"sleep_duration_score":100, "sleep_start_time_score":100, "deep_sleep_ratio_score":100, "WASO_frequency_score":100, "WASO_duration_score":100}
        today_sleep_parameters = [1, 1, 1, 1]

    if not yesterday_sleep_data_valid:
        yesterday_sleep_parameters = [1, 1, 1, 1]
    else:
        yesterday_sleep_score = calculate_sleep_score(age,
                                            yesterday_row.sleep_duration.iloc[0],
                                            yesterday_row.sleep_start_time.iloc[0],
                                            yesterday_row.deep_sleep_ratio.iloc[0],
                                            yesterday_row.wakefulness_after_sleep_onset_frequency.iloc[0],
                                            yesterday_row.wakefulness_after_sleep_onset_duration.iloc[0]
                                            )
    
        yesterday_sleep_parameters = [yesterday_sleep_score[1]["sleep_duration_score"]/100,
                                    yesterday_sleep_score[1]["sleep_start_time_score"]/100,
                                    yesterday_sleep_score[1]["deep_sleep_ratio_score"]/100,
                                    (yesterday_sleep_score[1]["WASO_frequency_score"]/100 +yesterday_sleep_score[1]["WASO_duration_score"]/100)/2]

    if 'hr' in timeseries_sheet.columns:
        hr_var = 'hr'
    else:
        hr_var = 'heartratedata'
        
    hr_list = list(timeseries_sheet[hr_var])
    rhr = daily_data.sleep_rhr[current_data_index]
    # c logic
    if not 30 <= rhr <= 150:
        rhr = 55

    daily_data.loc[current_data_index, "sleep.refined_rhr"] = rhr
    # UPDATED: Use the new function from charge_utils
    hr_max = calculate_hr_max(age)

    # UPDATED: Use the new function and a clearer variable name
    hrr_minute_list = calculate_hrr_percentage_series(heart_rate_list=hr_list, max_hr=hr_max, resting_hr=rhr, default_value=10)
    timeseries_sheet['hrr'] = hrr_minute_list

    
    timeseries_sheet['time'] = pd.to_datetime(timeseries_sheet['time'])
    # UPDATED: Use the new function name for exertion metrics
    daily_data, timeseries_sheet = calculate_exertion_and_stress_metrics(
        daily_data, current_data_index, timeseries_sheet, hr_var, 
        resting_hr=rhr, age=age, sex=sex, 
        sleep_start_index = sleep_start_time_index
    )

    # UPDATED: Use the new function for status detection
    min_status_list, shutdown_count_list = detect_user_status(list(timeseries_sheet['mode']))
    timeseries_sheet['min_status_list'] = min_status_list
    
   #=================================================================================#
   # START OF UPDATED STRESS PROCESSING
   #=================================================================================#
    if original:
        # UPDATED: Replace stress_mapping and stress_full with the new single function    
        raw_stress_list = list(timeseries_sheet['stress'])
        
        full_stress_list = normalize_and_fill_stress(
            raw_stress_list=raw_stress_list,
            status_list=min_status_list,
            heart_rate_list=hr_list,
            activity_list=list(timeseries_sheet['active']),
            resting_hr=rhr,
            max_hr=hr_max
        )
        
        mental_mode = [0] * len(raw_stress_list)
        # Check the raw stress list for the "all 255" condition
        if all(x == 255 for x in raw_stress_list):
            mental_mode = [3] * len(raw_stress_list)
        else:
            mental_mode = min_status_list
            
    else:
        
        serial_minute_of_day = (timeseries_sheet['time'].dt.hour * 60) + timeseries_sheet['time'].dt.minute
        features = pd.DataFrame(zip(list(timeseries_sheet[hr_var]), list(timeseries_sheet['active']), serial_minute_of_day), columns=['hr', 'activity', 'minute_of_day']) # ['hr', 'activity', 'minute_of_day']
        indices_to_process = timeseries_sheet[(timeseries_sheet['min_status_list'] < 2)].index.to_list()
        is_missing_mask =  (timeseries_sheet['stress'].isna()) | (timeseries_sheet['stress'] > 250) | (timeseries_sheet['stress'] < 0)

        imputed_stress = run_online_training(model, decomposer, minute_tracker, features_df=features, stress_signal=timeseries_sheet['stress'], is_missing_mask=is_missing_mask, indices_to_process=indices_to_process)

        signal_tuples = [(x,y,z,n) for x,y,z,n in zip(imputed_stress, hrr_minute_list, list(timeseries_sheet['active']), list(timeseries_sheet['sleep_stage']))]
        full_stress_list = signal_processor.run_online_training(y=signal_tuples, indices_to_process=indices_to_process)
        full_stress_list = [i/120 for i in full_stress_list]
        mental_mode = min_status_list
        
        
        
        fitness_threshold = 40
        stress_threshold = 99
        temperature = 0.1
            
        if full_stress_list:
            
            stress_array = np.array(full_stress_list)
            stress_array = stress_array[~np.isnan(stress_array)]

            val_fitness_percentile = 1 - np.percentile(stress_array, fitness_threshold)
            val_stress_percentile = - np.percentile(stress_array, stress_threshold)
            mean_fitness = 1 - np.mean(stress_array)
            
            score_list = [val_fitness_percentile, val_stress_percentile, mean_fitness]
            
            scores = np.array(score_list).__abs__() / temperature
            exps = np.exp(scores - np.max(scores))
            softmax_wts = exps / np.sum(exps)

            stress_score = sum([i*e for i,e in zip(softmax_wts, score_list)])/sum(score_list)
            if stress_score < 0 or stress_score > 1:
                sum([i*e for i,e in zip(softmax_wts, score_list)])/sum(np.array(score_list).__abs__())
                if stress_score < 0 or stress_score > 1:
                    print(daily_row.userid.iloc[0], stress_score, current_index)
                                    
            daily_data.loc[current_index, "stress.fitness_daily_score"] = max(0, min(1, stress_score))
            daily_data.loc[current_index, "stress.fitness_mean_7"] = daily_data.loc[:current_index, "stress.fitness_daily_score"].rolling(window=7, min_periods=1).mean().values.tolist()[-1]
            daily_data.loc[current_index, "stress.fatigue_daily_score"] = (1 - daily_data.loc[current_index, "stress.fitness_daily_score"])
            daily_data.loc[current_index, "stress.fatigue_mean_3"] = daily_data.loc[:current_index, "stress.fatigue_daily_score"].rolling(window=3, min_periods=1).mean().values.tolist()[-1]
            daily_data.loc[current_index, "stress.fitness_fatigue_difference"] = max(0, min(1, (daily_data.loc[current_index, "stress.fitness_mean_7"] - daily_data.loc[current_index, "stress.fatigue_mean_3"])))   
   
   #=================================================================================#
   # END OF UPDATED STRESS PROCESSING
   #=================================================================================#
    
    if not scores_valid:
        # 当天的分数数据不可用时，执行以下回溯逻辑
        
        # 1. 创建一个临时的、仅包含所需历史数据的副本，以避免修改原始DataFrame
        #    我们只复制到当前行的数据，以提高效率
        temp_data = daily_data.iloc[:current_data_index].copy()
        
        # 2. 在这个临时副本上，安全地将 'date' 列转换为 datetime 对象
        temp_data['date'] = pd.to_datetime(temp_data['date'])
        
        # 3. 获取今天的日期，并计算三天前回溯期的开始日期
        today_date = pd.to_datetime(daily_row.date.iloc[0])
        start_date_lookback = today_date - timedelta(days=3)
        
        # 4. 使用日期范围在临时副本上精确筛选出过去三天的数据
        prev_3_days_scores = temp_data[
            (temp_data['date'] >= start_date_lookback) & 
            (temp_data['date'] < today_date)
        ]
        
        # 5. 定义需要处理的指标列
        score_columns = ['skinTempScore', 'hrvScore', 'rhrScore', 'ahiScore']
        imputed_scores = {}

        # 6. 循环处理每个指标
        for col in score_columns:
            # 检查回溯期内该指标是否完全没有数据
            if prev_3_days_scores.empty or prev_3_days_scores[col].isna().all():
                # 如果是，则赋默认值 100
                imputed_scores[col] = 100
            else:
                # 如果至少有一个有效数据，则计算有效数据的均值
                mean_val = prev_3_days_scores[col].mean()
                imputed_scores[col] = round(mean_val, 2)
        
        # 7. 将最终计算出的值赋给局部变量，供后续算法使用
        #    原始的 daily_data DataFrame 在此过程中未发生任何改变
        temperature_score = imputed_scores['skinTempScore']
        HRV_score = imputed_scores['hrvScore']
        RHR_score = imputed_scores['rhrScore']
        AHI_score = imputed_scores['ahiScore']

    else:
        # 当天分数数据可用时，逻辑保持不变，直接从当天数据中取值
        temperature_score = daily_data.skinTempScore[current_data_index]
        HRV_score = daily_data.hrvScore[current_data_index]
        RHR_score = daily_data.rhrScore[current_data_index]
        AHI_score = daily_data.ahiScore[current_data_index]

    score_dict = {"RHR": RHR_score,
                    "HRV": HRV_score,
                    "AHI": AHI_score,
                    "TEMP": temperature_score}
    
    today_heart_parameters = [
    hrv_factor_today,
    daily_data.rhrScore[current_data_index] 
    ]

    filtered_score_dict = {k: v for k, v in score_dict.items() if 0 <= v <= 100}
    # UPDATED: Use the new function for score aggregation
    health_score, _ = aggregate_scores_nonlinearly(filtered_score_dict)
    health_score /= 100

    health_score_list = [prior_health_score] * len(timeseries_sheet)
    health_score_difference = health_score - prior_health_score

    if abs(health_score_difference) > 0.01 and len(timeseries_sheet) > 0:
        steps = min(round(health_score_difference / 0.005), 240, len(health_score_list))
        if steps > 0:
            ladder_values = np.linspace(prior_health_score, health_score, steps)
            ladeer_end_idx = sleep_start_time_index + steps
            health_score_list[sleep_start_time_index: ladeer_end_idx] = ladder_values
            remaining_steps = len(timeseries_sheet) - (ladeer_end_idx)
            if remaining_steps > 0:
                health_score_list[ladeer_end_idx:] = [health_score] * remaining_steps
    else:
        health_score_list = [health_score] * len(timeseries_sheet)

    mask = (timeseries_sheet["sleep_state"] == 1) & (~timeseries_sheet["sleep_stage"].isin([0, 1, 2]))
    prior = timeseries_sheet['sleep_stage'].shift(1).ffill()
    next_ = timeseries_sheet['sleep_stage'].shift(-1).bfill()
    condition = (prior == next_) & prior.isin([0, 1, 2])
    timeseries_sheet.loc[mask, 'sleep_stage'] = np.where(
        condition[mask], 
        prior[mask], 
        2
    ).astype(int)


    recovery_input = {}
    recovery_input['age'] = age
    recovery_input['rhr'] = rhr
    recovery_input['init_mental_expenditure'] = initial_mental_expenditure
    recovery_input['init_physical_expenditure'] = initial_physical_expenditure
    # UPDATED: Use the new variable name
    recovery_input['hrr_minute'] = hrr_minute_list
    recovery_input['activity_minute'] = list(timeseries_sheet['active'])
    recovery_input['stress'] = full_stress_list
    recovery_input['sleep_state'] = list(timeseries_sheet['sleep_state'])
    recovery_input['sleep_stage'] = list(timeseries_sheet["sleep_stage"])
    recovery_input['shutdown_count_series'] = shutdown_count_list
    recovery_input['exertion'] = list(timeseries_sheet['minute_exertion'])
    recovery_input['physical_mode'] = min_status_list
    recovery_input['mental_mode'] = mental_mode
    recovery_input['health_score_list'] = health_score_list
    recovery_input['nap_state'] = list(timeseries_sheet['nap_state'])

    # ==================================================================================================
    # CALCULATE FITNESS AND FATIGUE SCORES
    # ==================================================================================================
    
    percentile_threshold = 60
    temperature = 0.1
    
    minute_trimp = timeseries_sheet.minute_trimp.values.tolist()
    
    hrr_list = [hrr_minute_list[i] for i in range(len(hrr_minute_list)) if 20 < hr_list[i] < 220 and min_status_list[i] == 0]
    trimp_list = [minute_trimp[i] for i in range(len(minute_trimp)) if 20 < hr_list[i] < 220 and min_status_list[i] == 0]
        
    if hrr_list and trimp_list:

        hrr_max = calculate_hrr_percentage_series([hr_max], 
                                                    hr_max, 
                                                    daily_data.loc[current_index, "sleep.refined_rhr"])[0]
        
        trimp_max = charge_utils._calculate_minute_trimp(heart_rate = hr_max, 
                                                            age = daily_data.loc[current_index, "age"], 
                                                        resting_hr = daily_data.loc[current_index, "sleep.refined_rhr"], 
                                                        sex = daily_data.loc[current_index, "gender"])
                
        normalized_hrr_list = np.array(hrr_list) / hrr_max
        normalized_trimp_list = np.array(trimp_list) / trimp_max

        val_hrr_percentile = min(1, np.percentile(normalized_hrr_list, percentile_threshold))
        val_trimp_percentile = min(1, np.percentile(normalized_trimp_list, percentile_threshold))

        mean_hrr = np.mean(normalized_hrr_list)
        mean_trimp = np.mean(normalized_trimp_list)

        sd_hrr = np.std(normalized_hrr_list)
        sd_trimp = np.std(normalized_trimp_list)
                    
        zscore_hrr = (val_hrr_percentile-mean_hrr)/sd_hrr
        zscore_trimp = (val_trimp_percentile-mean_trimp)/sd_trimp

        zscore_list = [zscore_hrr, zscore_trimp, mean_hrr/sd_hrr, mean_trimp/sd_trimp]
        
        scores = np.array(zscore_list) / temperature
        exps = np.exp(scores - np.max(scores))
        softmax_wts = exps / np.sum(exps)

        fitness_score = sum([i*e for i,e in zip(softmax_wts, zscore_list)])/sum(zscore_list)
        if fitness_score < 0 or fitness_score > 1:
            fitness_score = sum([i*e for i,e in zip(softmax_wts, zscore_list)])/sum(abs(np.array(zscore_list)))
            if fitness_score < 0 or fitness_score > 1:
                print(daily_row.userid.iloc[0], fitness_score, current_index)
                                
        daily_data.loc[current_index, "fitness.fitness_daily_score"] = fitness_score
        
        daily_data.loc[current_index, "fitness.fitness_mean_3"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=3, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fitness_mean_7"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=7, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fitness_mean_14"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=14, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fitness_mean_42"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=42, min_periods=1).mean().values.tolist()[-1]

        daily_data.loc[current_index, "fitness.fitness_sum_3"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=3, min_periods=1).sum().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fitness_sum_7"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=7, min_periods=1).sum().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fitness_sum_14"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=14, min_periods=1).sum().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fitness_sum_42"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=42, min_periods=1).sum().values.tolist()[-1]
        
        daily_data.loc[current_index, "fitness.fatigue_daily_score"] = min(1, max(0, (daily_data.loc[current_index, "fitness.fitness_daily_score"] - daily_data.loc[current_index, "fitness.fitness_mean_7"]) / (1 - daily_data.loc[current_index, "fitness.fitness_mean_7"])))
                    
        daily_data.loc[current_index, "fitness.fatigue_mean_3"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=3, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fatigue_mean_7"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=7, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fatigue_mean_14"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=14, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fatigue_mean_42"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=42, min_periods=1).mean().values.tolist()[-1]
        
        daily_data.loc[current_index, "fitness.fatigue_sum_3"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=3, min_periods=1).sum().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fatigue_sum_7"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=7, min_periods=1).sum().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fatigue_sum_14"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=14, min_periods=1).sum().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fatigue_sum_42"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=42, min_periods=1).sum().values.tolist()[-1]
        
        daily_data.loc[current_index, "fitness.fitness_fatigue_difference"] = min(1, max(0, (daily_data.loc[current_index, "fitness.fitness_mean_7"] - daily_data.loc[current_index, "fitness.fatigue_mean_3"])))

    # ==================================================================================================
    # END OF FITNESS AND FATIGUE SCORE CALCULATION
    # ==================================================================================================
    
    today_fitness_fatigue_score = daily_data.loc[current_index, "fitness.fitness_fatigue_difference"]
    today_stress_fitness_fatigue_score = daily_data.loc[current_index, "stress.fitness_fatigue_difference"]
    
    percentile_threshold = 60
    temperature = 0.1
    
    minute_trimp = timeseries_sheet.minute_trimp.values.tolist()
    
    hrr_list = [hrr_minute_list[i] for i in range(len(hrr_minute_list)) if 20 < hr_list[i] < 220 and min_status_list[i] == 0]
    trimp_list = [minute_trimp[i] for i in range(len(minute_trimp)) if 20 < hr_list[i] < 220 and min_status_list[i] == 0]
        
    if hrr_list and trimp_list:

        hrr_max = calculate_hrr_percentage_series([hr_max], 
                                                    hr_max, 
                                                    daily_data.loc[current_index, "sleep.refined_rhr"])[0]
        
        trimp_max = charge_utils._calculate_minute_trimp(heart_rate = hr_max, 
                                                            age = daily_data.loc[current_index, "age"], 
                                                        resting_hr = daily_data.loc[current_index, "sleep.refined_rhr"], 
                                                        sex = daily_data.loc[current_index, "gender"])
                
        normalized_hrr_list     = np.array(hrr_list) / hrr_max
        normalized_trimp_list   = np.array(trimp_list) / trimp_max

        val_hrr_percentile = min(1, np.percentile(normalized_hrr_list, percentile_threshold))
        val_trimp_percentile = min(1, np.percentile(normalized_trimp_list, percentile_threshold))

        mean_hrr = np.mean(normalized_hrr_list)
        mean_trimp = np.mean(normalized_trimp_list)

        sd_hrr = np.std(normalized_hrr_list)
        sd_trimp = np.std(normalized_trimp_list)
                    
        zscore_hrr = (val_hrr_percentile-mean_hrr)/sd_hrr
        zscore_trimp = (val_trimp_percentile-mean_trimp)/sd_trimp

        zscore_list = [zscore_hrr, zscore_trimp, mean_hrr/sd_hrr, mean_trimp/sd_trimp]
        
        scores = np.array(zscore_list) / temperature
        exps = np.exp(scores - np.max(scores))
        softmax_wts = exps / np.sum(exps)

        fitness_score = sum([i*e for i,e in zip(softmax_wts, zscore_list)])/sum(zscore_list)
        if fitness_score < 0 or fitness_score > 1:
            fitness_score = sum([i*e for i,e in zip(softmax_wts, zscore_list)])/sum(abs(np.array(zscore_list)))
            if fitness_score < 0 or fitness_score > 1:
                print(daily_row.userid.iloc[0], fitness_score, current_index)
                                
        daily_data.loc[current_index, "fitness.fitness_daily_score"] = fitness_score
        
        daily_data.loc[current_index, "fitness.fitness_mean_3"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=3, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fitness_mean_7"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=7, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fitness_mean_14"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=14, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fitness_mean_42"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=42, min_periods=1).mean().values.tolist()[-1]

        daily_data.loc[current_index, "fitness.fitness_sum_3"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=3, min_periods=1).sum().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fitness_sum_7"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=7, min_periods=1).sum().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fitness_sum_14"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=14, min_periods=1).sum().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fitness_sum_42"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=42, min_periods=1).sum().values.tolist()[-1]
        
        daily_data.loc[current_index, "fitness.fatigue_daily_score"] = min(1, max(0, (daily_data.loc[current_index, "fitness.fitness_daily_score"] - daily_data.loc[current_index, "fitness.fitness_mean_7"]) / (1 - daily_data.loc[current_index, "fitness.fitness_mean_7"])))
                    
        daily_data.loc[current_index, "fitness.fatigue_mean_3"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=3, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fatigue_mean_7"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=7, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fatigue_mean_14"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=14, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fatigue_mean_42"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=42, min_periods=1).mean().values.tolist()[-1]
        
        daily_data.loc[current_index, "fitness.fatigue_sum_3"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=3, min_periods=1).sum().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fatigue_sum_7"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=7, min_periods=1).sum().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fatigue_sum_14"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=14, min_periods=1).sum().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fatigue_sum_42"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=42, min_periods=1).sum().values.tolist()[-1]
        
        daily_data.loc[current_index, "fitness.fitness_fatigue_difference"] = min(1, max(0, (daily_data.loc[current_index, "fitness.fitness_mean_7"] - daily_data.loc[current_index, "fitness.fatigue_mean_3"])))

    # ==================================================================================================
    
    if 'fitness.fitness_fatigue_difference' not in daily_data.columns:
        daily_data['fitness.fitness_fatigue_difference'] = 0
    today_fitness_fatigue_score = daily_data.loc[current_index, "fitness.fitness_fatigue_difference"]
    mental_no_wear_fitting = list(pd.read_csv(f"{current_dir}/data_mbat_new.csv")['0'][:1440])
    if len(mental_no_wear_fitting) <= len(recovery_input['sleep_state']):
        mental_no_wear_fitting = mental_no_wear_fitting + [mental_no_wear_fitting[-1]] * (len(recovery_input['sleep_state']) - len(mental_no_wear_fitting) + 1)

    physical_no_wear_fitting = list(pd.read_csv(f"{current_dir}/data_pbat_new.csv")['0'][:1440])
    if len(physical_no_wear_fitting) <= len(recovery_input['hrr_minute']):
        physical_no_wear_fitting = physical_no_wear_fitting + [physical_no_wear_fitting[-1]] * (len(recovery_input['hrr_minute']) - len(physical_no_wear_fitting)+1)

    Rt = 2880 * (recovery_input['init_mental_expenditure'] / 100)
    
    initial_time = timeseries_sheet['time'].iloc[0]
    initial_hours_of_day = initial_time.hour + initial_time.minute/60

    if original:
        param_margin = 0.1
    else:
        param_margin = 0.25
    
    mental_new = mental_batteryFT(initial_hours_of_day, mental_parameters_tuned, Rt=Rt, 
                                    press_thresh=press_thresh, expand1=5, expand2=5,
                                    usual_awake_ts=8, usual_slp_ts=0, 
                                     mental_no_wear_fitting=mental_no_wear_fitting,
                                     )
    physical_new = physical_batteryFT(recovery_input['age'], CP=CP_threshold, acc_threshold=5, hr_rest=recovery_input['rhr'], allks=physical_parameters_tuned,
                                        init_expenditure=recovery_input['init_physical_expenditure'],
                                        physical_no_wear_fitting=physical_no_wear_fitting, exertion_growth_rate=exertion_growth_rate
                                        )
    
    increase_flag, decrease_flag = True, True
    hypothetical_mental_expenditure_ts = []
    hypothetical_physical_expenditure_ts = []
    charge_dynamic = []
    
    #============================================
    # Initializations for Dynamic Processing
    #============================================
    batch_size = 5
    
    hrr_main_digest_wrapper = TDigestWrapper()
    hrr_stats_tracker = DynamicStats()
    hrr_batch_data = []

    trimp_main_digest_wrapper = TDigestWrapper()
    trimp_stats_tracker = DynamicStats()
    trimp_batch_data = []

    stress_main_digest_wrapper = TDigestWrapper()
    stress_stats_tracker = DynamicStats()
    stress_batch_data = []
    #=============================================
    
    for m_i in range(len(recovery_input['sleep_state'])):
        if m_i < sleep_start_time_index:
            sleep_duration_score, sleep_start_time_score, deep_sleep_score, WASO_score = yesterday_sleep_parameters
            hrv_factor, rhr_score = yesterday_heart_parameters
            nap_pointer = True
            fitness_fatigue_score = yesterday_fitness_fatigue_score if not np.isnan(yesterday_fitness_fatigue_score) else 0
            if original:
                stress_fitness_fatigue_score = fitness_fatigue_score
            else: 
                stress_fitness_fatigue_score = yesterday_stress_fitness_fatigue_score if not np.isnan(yesterday_stress_fitness_fatigue_score) else 0

        else:
            sleep_duration_score, sleep_start_time_score, deep_sleep_score, WASO_score = today_sleep_parameters
            hrv_factor, rhr_score = today_heart_parameters
            nap_pointer = False
            fitness_fatigue_score = today_fitness_fatigue_score if not np.isnan(today_fitness_fatigue_score) else 0
            if original:
                stress_fitness_fatigue_score = fitness_fatigue_score
            else:
                stress_fitness_fatigue_score = today_stress_fitness_fatigue_score if not np.isnan(today_stress_fitness_fatigue_score) else 0

        if m_i > 0:
            current_mental_score = mental_new.Et[-1] if not np.isnan(mental_new.Et[-1]) else 0
            current_physical_score = physical_new.output_battery[-1] if not np.isnan(physical_new.output_battery[-1]) else 0
            current_total_score = (current_mental_score + current_physical_score)/2
        else:
            current_total_score = (recovery_input['init_mental_expenditure'] + recovery_input['init_physical_expenditure'])/2

        if current_total_score > 100:
            increase_flag = False
            decrease_flag = True
        elif current_total_score < 5:
            increase_flag = True
            decrease_flag = False
        else:
            increase_flag = True
            decrease_flag = True
        
        #============================================
        # Dynamic Processing
        #============================================        
        if dynamic and not original:
            
            fitness_fatigue_score = yesterday_fitness_fatigue_score if not np.isnan(yesterday_fitness_fatigue_score) else 0
            stress_fitness_fatigue_score = yesterday_stress_fitness_fatigue_score if not np.isnan(yesterday_stress_fitness_fatigue_score) else 0
            
            # PHYSICAL
            if 20 < hr_list[m_i] < 250 and min_status_list[m_i] == 0:
                                                    
                normalized_hrr_value = hrr_minute_list[m_i]/hrr_max
                normalized_trimp_value = minute_trimp[m_i]/trimp_max

                hrr_stats_tracker.push(normalized_hrr_value)
                trimp_stats_tracker.push(normalized_trimp_value)
                
                hrr_batch_data.append(normalized_hrr_value)
                trimp_batch_data.append(normalized_trimp_value)
                
                if len(hrr_batch_data) == batch_size and trimp_batch_data == batch_size:
                        
                    hrr_batch_wrapper = TDigestWrapper()
                    hrr_batch_wrapper.batch_update(hrr_batch_data)
                    hrr_main_digest_wrapper.merge(hrr_batch_wrapper)
                    
                    trimp_batch_wrapper = TDigestWrapper()
                    trimp_batch_wrapper.batch_update(trimp_batch_data)                 
                    trimp_main_digest_wrapper.merge(trimp_batch_wrapper)
                                        
                    hrr_batch_data = []
                    trimp_batch_data = []
            
            # MENTAL
            if 0 <= full_stress_list[m_i] <= 100:

                stress_value = full_stress_list[m_i]
                stress_stats_tracker.push(stress_value)
                stress_batch_data.append(stress_value)
                
                if len(stress_batch_data) == batch_size:
                        
                    stress_batch_wrapper = TDigestWrapper()
                    stress_batch_wrapper.batch_update(stress_batch_data)
                    stress_main_digest_wrapper.merge(stress_batch_wrapper)
                                        
                    stress_batch_data = []
        #============================================
        # End of Dynamic Processing
        #============================================                   
        
        mental_new.run_battery(recovery_input['sleep_state'],
                                recovery_input['shutdown_count_series'], 
                                recovery_input['stress'],
                                recovery_input['mental_mode'],
                                recovery_input['health_score_list'],
                                sleep_duration_score, sleep_start_time_score, WASO_score, hrv_factor, rhr_score,
                                recovery_input['sleep_stage'], recovery_input['exertion'], 
                                m_i,
                                nap_pointer, stress_fitness_fatigue_score,
                                original)
        
        physical_new.run_battery(recovery_input['hrr_minute'],
                                recovery_input['activity_minute'],
                                recovery_input['physical_mode'],
                                shutdown_count_list, 
                                recovery_input["sleep_state"], 
                                recovery_input["sleep_stage"],
                                recovery_input['health_score_list'],
                                sleep_duration_score, sleep_start_time_score, WASO_score,
                                recovery_input['exertion'], 
                                m_i, nap_pointer, fitness_fatigue_score, 
                                increase_flag = increase_flag, decrease_flag = decrease_flag)
        
        # --- ADDED: Hypothetical Calculation Logic ---
        # This calculates the "what if" expenditure when the battery is too low to actually drain.
        # NOTE: This assumes `calculate_hypothetical_expenditure` methods have been added
        # to the `mental_batteryFT` and `physical_batteryFT` classes.
        if current_total_score <= 5 and recovery_input['sleep_state'][m_i] != 1:
            # Only calculate for awake state, as depletion logic is primarily for wakefulness.
            try:
                # Hypothetical Mental Expenditure
                mental_drain = mental_new.calculate_hypothetical_expenditure(
                    curr_press=recovery_input['stress'][m_i],
                    exertion_score=recovery_input['exertion'][m_i]
                )
                hypothetical_mental_expenditure_ts.append(mental_drain)
            except AttributeError:
                hypothetical_mental_expenditure_ts.append(0)

            try:
                # Hypothetical Physical Expenditure
                physical_drain = physical_new.calculate_hypothetical_expenditure(
                    current_hrr=recovery_input['hrr_minute'][m_i],
                    current_acc=recovery_input['activity_minute'][m_i],
                    exertion_score=recovery_input['exertion'][m_i]
                )
                hypothetical_physical_expenditure_ts.append(physical_drain)
            except AttributeError:
                hypothetical_physical_expenditure_ts.append(0)
        else:
            # If not in the low-battery state or if sleeping, there is no hypothetical drain.
            hypothetical_mental_expenditure_ts.append(0)
            hypothetical_physical_expenditure_ts.append(0)

    mental_result_new = [float(val) if not np.isnan(val) else 0 for val in mental_new.Et]
    physical_result_new = [float(val) if not np.isnan(val) else 0 for val in physical_new.output_battery]
    charge_dynamic = [float(min(max((m_v+p_v)/2, 0), 100)) for m_v, p_v in zip(mental_result_new, physical_result_new)]
    health_score_list = [round(val, 5) if not np.isnan(val) else 0 for val in health_score_list]


    #============================================
    # Dynamic Processing Summary
    #============================================  
    if dynamic and not original:
        
        ## PHYSICAL: CALCUATE THE FINAL DYNAMIC FITNESS-FATIGUE SCORE
        if hrr_batch_data and trimp_batch_data:

            hrr_batch_wrapper = TDigestWrapper()
            hrr_batch_wrapper.batch_update(hrr_batch_data)
            hrr_main_digest_wrapper.merge(hrr_batch_wrapper)
            
            trimp_batch_wrapper = TDigestWrapper()
            trimp_batch_wrapper.batch_update(trimp_batch_data)                 
            trimp_main_digest_wrapper.merge(trimp_batch_wrapper)
                            
        if hrr_stats_tracker.get_std() == 0 or trimp_stats_tracker.get_std() == 0:
            hrr_zscore_dynamic_digest = 0
            trimp_zscore_dynamic_digest = 0
            hrr_mean_zscore_dynamic_digest = 0
            trimp_mean_zscore_dynamic_digest = 0
        else:
            hrr_zscore_dynamic_digest = (hrr_main_digest_wrapper.get_percentile(60) - hrr_stats_tracker.get_mean()) / hrr_stats_tracker.get_std()
            trimp_zscore_dynamic_digest = (trimp_main_digest_wrapper.get_percentile(60) - trimp_stats_tracker.get_mean()) / trimp_stats_tracker.get_std()
            hrr_mean_zscore_dynamic_digest = hrr_stats_tracker.get_mean() / hrr_stats_tracker.get_std()
            trimp_mean_zscore_dynamic_digest = trimp_stats_tracker.get_mean() / trimp_stats_tracker.get_std()
                    
        zscore_digest_list = [hrr_zscore_dynamic_digest, trimp_zscore_dynamic_digest, hrr_mean_zscore_dynamic_digest, trimp_mean_zscore_dynamic_digest]        
        # dynamic_fitness_score_digest = calculate_fitness_score(zscore_digest_list)
        scores_digest = np.array(zscore_digest_list) / temperature
        exps_digest = np.exp(scores_digest - np.max(scores_digest))
        softmax_wts_digest = exps_digest / np.sum(exps_digest)
        dynamic_fitness_score_digest = min(1, sum([i*e for i,e in zip(softmax_wts_digest, zscore_digest_list)])/sum(zscore_digest_list))
        if dynamic_fitness_score_digest < 0:
            dynamic_fitness_score_digest = sum([i*e for i,e in zip(softmax_wts_digest, zscore_digest_list)])/sum(abs(np.array(zscore_digest_list)))
            if dynamic_fitness_score_digest < 0 or dynamic_fitness_score_digest > 1:
                print(daily_row.userid.iloc[0], dynamic_fitness_score_digest, current_index)

        daily_data.loc[current_index, "fitness.fitness_daily_score"] = dynamic_fitness_score_digest
        daily_data.loc[current_index, "fitness.fitness_mean_7"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=7, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "fitness.fatigue_daily_score"] =  min(1, max(0, (dynamic_fitness_score_digest - daily_data.loc[current_index, "fitness.fitness_mean_7"]) / (1 - daily_data.loc[current_index, "fitness.fitness_mean_7"])))
        daily_data.loc[current_index, "fitness.fatigue_mean_3"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=3, min_periods=1).mean().values.tolist()[-1]        
        
        daily_data.loc[current_index, "fitness.fitness_fatigue_difference_digest"] = max(0, min(1, daily_data.loc[current_index, "fitness.fitness_mean_7"] - daily_data.loc[current_index, "fitness.fatigue_mean_3"]))


        ## MENTAL: CALCUATE THE FINAL DYNAMIC FITNESS-FATIGUE SCORE
        if stress_batch_data:
            stress_batch_wrapper = TDigestWrapper()
            stress_batch_wrapper.batch_update(stress_batch_data)
            stress_main_digest_wrapper.merge(stress_batch_wrapper)            
            
        val_fitness_percentile = 1 - stress_main_digest_wrapper.get_percentile(40)
        val_stress_percentile = - stress_main_digest_wrapper.get_percentile(99)
        mean_fitness = 1 - stress_stats_tracker.get_mean()
        
        score_list = [val_fitness_percentile, val_stress_percentile, mean_fitness]
        
        scores = np.array(score_list).__abs__() / temperature
        exps = np.exp(scores - np.max(scores))
        softmax_wts = exps / np.sum(exps)

        stress_score = sum([i*e for i,e in zip(softmax_wts, score_list)])/sum(score_list)
        if stress_score < 0 or stress_score > 1:
            sum([i*e for i,e in zip(softmax_wts, score_list)])/sum(np.array(score_list).__abs__())
            if stress_score < 0 or stress_score > 1:
                print(daily_row.userid.iloc[0], stress_score, current_index)
                                
        daily_data.loc[current_index, "stress.fitness_daily_score"] = stress_score
        daily_data.loc[current_index, "stress.fitness_mean_7"] = daily_data.loc[:current_index, "stress.fitness_daily_score"].rolling(window=7, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "stress.fatigue_daily_score"] = (1 - daily_data.loc[current_index, "stress.fitness_daily_score"])
        daily_data.loc[current_index, "stress.fatigue_mean_3"] = daily_data.loc[:current_index, "stress.fatigue_daily_score"].rolling(window=3, min_periods=1).mean().values.tolist()[-1]
        daily_data.loc[current_index, "stress.fitness_fatigue_difference"] = max(0, min(1, (daily_data.loc[current_index, "stress.fitness_mean_7"] - daily_data.loc[current_index, "stress.fatigue_mean_3"])))
    #============================================
    # End of Dynamic Processing Summary
    #============================================      

    #====================================================
    #    END OF CHARGE PROCESSING LOOP
    #====================================================
    
    mental_final_value = mental_result_new[-1] if len(mental_result_new) > 0 else None
    physical_final_value = physical_result_new[-1] if len(physical_result_new) > 0 else None
    charge_final_value = charge_dynamic[-1] if len(charge_dynamic) > 0 else None

    sleep_start_charge = charge_dynamic[sleep_start_time_index] if len(charge_dynamic) > sleep_start_time_index else None
    sleep_charge_recovery = (charge_final_value - sleep_start_charge) if charge_final_value is not None and sleep_start_charge is not None else None


    sleep_start_mental = mental_result_new[sleep_start_time_index] if len(mental_result_new) > sleep_start_time_index else None
    sleep_start_physical = physical_result_new[sleep_start_time_index] if len(physical_result_new) > sleep_start_time_index else None

    checkpoint_times = ['08:00:00', '12:00:00', '18:00:00']
    exercies_checkpoints_dict = process_exercise_data(timeseries_sheet, mental_result_new, physical_result_new, charge_dynamic, checkpoint_times, sheet)
    exercise_indexes = exercies_checkpoints_dict.get('exercise_index')
    nap_indexes = exercies_checkpoints_dict.get('nap_index')
    sleep_indexes = exercies_checkpoints_dict.get('sleep_index')


    mental_changes_above_50 = []
    mental_changes_below_50 = []

    if len(mental_result_new) > sleep_start_time_index :
        for i in range(sleep_start_time_index - 1):
            current_value = mental_result_new[i]
            next_value = mental_result_new[i+1]
            change = next_value - current_value

            if change < 0:
                if current_value >= 50:
                    mental_changes_above_50.append(change)
                else: 
                    mental_changes_below_50.append(change)

    average_speed_above_50 = -sum(mental_changes_above_50) / len(mental_changes_above_50) if mental_changes_above_50 else 0
    average_speed_below_50 = -sum(mental_changes_below_50) / len(mental_changes_below_50) if mental_changes_below_50 else 0

    str_9pm = f"{yesterday_date_str} 21:00"
    datetime_9pm = datetime.strptime(str_9pm, '%Y-%m-%d %H:%M')
    index_9pm = int((datetime_9pm - yesterday_wakeup_time).total_seconds() / 60)
    charge_9pm = charge_dynamic[index_9pm] if len(charge_dynamic) > index_9pm else None

    total_nap_charge_recovery = []
    total_exercise_charge_expenditure = []
    nap_starts = exercies_checkpoints_dict.get('charge_nap_starts', [])
    nap_ends = exercies_checkpoints_dict.get('charge_nap_ends', [])
    nap_rates = []
    nap_recovery_rate = 0
    if nap_starts and nap_ends and isinstance(nap_starts, list) and isinstance(nap_ends, list):
        for start_dict, end_dict in zip(nap_starts, nap_ends):
            if isinstance(start_dict, dict) and isinstance(end_dict, dict):
                if start_dict and end_dict:
                    start_time, start_val = next(iter(start_dict.items()))
                    end_time, end_val = next(iter(end_dict.items()))
                    if start_val is not None and end_val is not None:
                        # MODIFIED: Round the result
                        total_nap_charge_recovery.append(round(end_val - start_val, 2))
                        try:
                            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
                        try:
                            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
                        duration_minutes = (end_dt - start_dt).total_seconds() / 60
                        if duration_minutes > 0:
                            nap_rates.append((end_val - start_val) / duration_minutes)
        if nap_rates:
            nap_recovery_rate = sum(nap_rates) / len(nap_rates)
        else:
            nap_recovery_rate = 0


    exercise_starts = exercies_checkpoints_dict.get('charge_exercise_starts', [])
    exercise_ends = exercies_checkpoints_dict.get('charge_exercise_ends', [])
    exercise_expenditure_rate = 0
    exercise_rates = []

    if exercise_starts and exercise_ends and isinstance(exercise_starts, list) and isinstance(exercise_ends, list):
        for start_dict, end_dict in zip(exercise_starts, exercise_ends):
            if isinstance(start_dict, dict) and isinstance(end_dict, dict):
                if start_dict and end_dict:
                    start_time, start_val = next(iter(start_dict.items()))
                    end_time, end_val = next(iter(end_dict.items()))
                    if start_val is not None and end_val is not None:
                        # MODIFIED: Round the result
                        total_exercise_charge_expenditure.append(round(start_val - end_val, 2))
                        try:
                            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
                        try:
                            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
                        duration_minutes = (end_dt - start_dt).total_seconds() / 60
                        if duration_minutes > 0:
                            exercise_rates.append((start_val - end_val) / duration_minutes)
        if exercise_rates:
            exercise_expenditure_rate = sum(exercise_rates) / len(exercise_rates)
        else:
            exercise_expenditure_rate = 0


    # Calculate nap recovery value and exercise expenditure value based on nap_state and exercise in timeseries_sheet

    # --- Nap Recovery Calculation ---
    nap_recovery_value_new = 0
    if 'nap_state' in timeseries_sheet.columns:
        nap_state = timeseries_sheet['nap_state'].values
        # Find all nap periods (contiguous 1s in nap_state)
        in_nap = False
        nap_start_idx = None
        for idx, val in enumerate(nap_state):
            if val == 1 and not in_nap:
                in_nap = True
                nap_start_idx = idx
            elif val != 1 and in_nap:
                in_nap = False
                nap_end_idx = idx - 1
                # Calculate recovery as charge difference over nap period
                if nap_start_idx is not None and nap_end_idx > nap_start_idx:
                    start_charge = charge_dynamic[nap_start_idx]
                    end_charge = charge_dynamic[nap_end_idx]
                    nap_recovery_value_new += max(0, end_charge - start_charge)
        # Handle if nap goes till the end
        if in_nap and nap_start_idx is not None and nap_start_idx < len(charge_dynamic) - 1:
            start_charge = charge_dynamic[nap_start_idx]
            end_charge = charge_dynamic[-1]
            nap_recovery_value_new += max(0, end_charge - start_charge)
        nap_recovery_value_new = round(nap_recovery_value_new, 2)
    if nap_recovery_value_new > 0:
        nap_duration = len(timeseries_sheet['nap_state'].values.nonzero()[0])
        nap_recovery_rate_new = nap_recovery_value_new / nap_duration
    else:
        nap_duration = 0
        nap_recovery_rate_new = 0

    # --- Exercise Expenditure Calculation ---
    exercise_expenditure_value_new = 0
    if 'exercise' in timeseries_sheet.columns:
        exercise_state = timeseries_sheet['exercise'].values
        in_exercise = False
        exercise_start_idx = None
        for idx, val in enumerate(exercise_state):
            if val == 1 and not in_exercise:
                in_exercise = True
                exercise_start_idx = idx
            elif val != 1 and in_exercise:
                in_exercise = False
                exercise_end_idx = idx - 1
                # Calculate expenditure as charge difference over exercise period
                if exercise_start_idx is not None and exercise_end_idx > exercise_start_idx:
                    start_charge = charge_dynamic[exercise_start_idx]
                    end_charge = charge_dynamic[exercise_end_idx]
                    exercise_expenditure_value_new += max(0, start_charge - end_charge)
        # Handle if exercise goes till the end
        if in_exercise and exercise_start_idx is not None and exercise_start_idx < len(charge_dynamic) - 1:
            start_charge = charge_dynamic[exercise_start_idx]
            end_charge = charge_dynamic[-1]
            exercise_expenditure_value_new += max(0, start_charge - end_charge)
        exercise_expenditure_value_new = round(exercise_expenditure_value_new, 2)
    exercise_duration = len(timeseries_sheet['exercise'].values.nonzero()[0])
    if exercise_duration > 0:
        exercise_expenditure_rate_new = exercise_expenditure_value_new / exercise_duration
    else:
        exercise_duration = 0
        exercise_expenditure_rate_new = 0


    # ==============================================================================
    #           START: Calculate True & Observed Expenditure for Each Exercise Session
    # ==============================================================================
    exercise_expenditure_summary = {}
    # ADDED: Initialize a new dictionary for the observed expenditure
    observed_exercise_expenditure = {} 
    exercise_indices = exercies_checkpoints_dict.get('exercise_index')

    if exercise_indices and len(exercise_indices) == 2:
        start_indices, end_indices = exercise_indices
        
        mental_max_reserve = mental_new.Rc if mental_new.Rc > 0 else 2880

        for start_idx, end_idx in zip(start_indices, end_indices):
            if end_idx >= len(timeseries_sheet) or start_idx < 0:
                continue

            start_time_str = timeseries_sheet['time'].iloc[start_idx].strftime('%H:%M')
            end_time_str = timeseries_sheet['time'].iloc[end_idx].strftime('%H:%M')
            session_key = f"{start_time_str}-{end_time_str}"
            
            # --- START: New "Observed" Expenditure Calculation ---
            # This is the simple calculation based on the final charge values
            start_charge = charge_dynamic[start_idx]
            end_charge = charge_dynamic[end_idx]
            observed_expenditure = start_charge - end_charge
            observed_exercise_expenditure[session_key] = round(observed_expenditure, 2)
            # --- END: New "Observed" Expenditure Calculation ---

            # --- Existing "True" Expenditure Calculation ---
            total_mental_drain_pct = 0.0
            total_physical_drain_pct = 0.0

            for i in range(start_idx, end_idx):
                if charge_dynamic[i] < 5:
                    mental_drain_pct = (hypothetical_mental_expenditure_ts[i] / mental_max_reserve) * 100
                    physical_drain_pct = hypothetical_physical_expenditure_ts[i]
                else:
                    mental_drain_pct = max(0, mental_result_new[i] - mental_result_new[i+1])
                    physical_drain_pct = max(0, physical_result_new[i] - physical_result_new[i+1])
                
                total_mental_drain_pct += mental_drain_pct
                total_physical_drain_pct += physical_drain_pct

            # Ensure total_mental_drain_pct and total_physical_drain_pct are Python floats rounded to 2 decimals
            total_mental_drain_pct = round(float(total_mental_drain_pct), 2)
            total_physical_drain_pct = round(float(total_physical_drain_pct), 2)

            total_charge_expenditure = (total_mental_drain_pct + total_physical_drain_pct) / 2
            exercise_expenditure_summary[session_key] = round(total_charge_expenditure, 2)
    # ==============================================================================
    #            END: Calculate True & Observed Expenditure for Each Exercise Session
    # ==============================================================================
    summary_exercise_charge_expenditure = sum(total_exercise_charge_expenditure) if total_exercise_charge_expenditure else 0
    summary_nap_charge_recovery = sum(total_nap_charge_recovery) if total_nap_charge_recovery else 0

    # 计算正常时间的消耗和恢复
    summary_normal_time_charge_expenditure, summary_normal_time_charge_recovery = calculate_normal_time_expenditure_and_recovery(
        charge_dynamic, timeseries_sheet, nap_starts, nap_ends, exercise_starts, exercise_ends, recovery_input['sleep_state']
    )

    summary_sleep_recovery = calculate_sleep_recovery(charge_dynamic, recovery_input['sleep_state'])
    summary_is_total_charge_touch_5 = any(val <= 5 for val in charge_dynamic)

    # ==============================================================================
    #                      START: Store results in DataFrame
    # ==============================================================================
    
    # [START] Store Sleep Metrics
    # ------------------------------------------------------------------------------
    # MODIFIED: Round sleep scores
    daily_data.loc[current_data_index, 'sleep.total_score'] = round(sleep_score[0], 2) if sleep_score[0] is not None else None
    daily_data.loc[current_data_index, 'sleep.duration_score'] = round(sleep_score[1].get("sleep_duration_score", 0), 2)
    daily_data.loc[current_data_index, 'sleep.start_time_score'] = round(sleep_score[1].get("sleep_start_time_score", 0), 2)
    daily_data.loc[current_data_index, 'sleep.deep_sleep_score'] = round(sleep_score[1].get("deep_sleep_ratio_score", 0), 2)
    waso_freq_score = sleep_score[1].get("WASO_frequency_score", 0)
    waso_dur_score = sleep_score[1].get("WASO_duration_score", 0)
    waso_score = (waso_freq_score + waso_dur_score) / 2 if (waso_freq_score and waso_dur_score) else None
    daily_data.loc[current_data_index, 'sleep.waso_score'] = round(waso_score, 2) if waso_score is not None else None
    # ------------------------------------------------------------------------------
    # [END] Store Sleep Metrics

    # [START] Store Health, Readiness Scores, and New Readiness Score
    # ------------------------------------------------------------------------------
    # MODIFIED: Round health and readiness scores, and calculate new readiness score
    daily_data.at[current_data_index, "health.score"] = round(health_score, 2) if health_score is not None else None
    readiness_score_raw, _ = weights_for_items([
        mental_final_value,
        physical_final_value,
        temperature_score,
        HRV_score,
        RHR_score,
        AHI_score]) if mental_final_value is not None and physical_final_value is not None else (None, None)
    daily_data.loc[current_data_index, "readiness.score"] = round(readiness_score_raw, 2) if readiness_score_raw is not None else None
    daily_data.loc[current_data_index, "hrv_factor"] = round(hrv_factor_today, 4) if hrv_factor_today is not None else None
    
    # # New Readiness Score
    readiness_score_values = {
            "hrvScore": daily_data.loc[current_data_index,"hrvScore"],
            "rhrScore": daily_data.loc[current_data_index,"rhrScore"],
            "ahiScore": daily_data.loc[current_data_index,"ahiScore"],
            "skinTempScore": daily_data.loc[current_data_index,"skinTempScore"],
            "mentScore": daily_data.loc[current_data_index,"mentScore"],
            "phyScore": daily_data.loc[current_data_index,"phyScore"],
            "sleep.duration_score": daily_data.loc[current_data_index,"sleep.duration_score"],
            "sleep.deep_sleep_score": daily_data.loc[current_data_index,"sleep.deep_sleep_score"],
        }
    
    readiness_available_keys = []
    for key, value in readiness_score_values.items():
        if ~np.isnan(value) and 0 < value <= 100:
            readiness_available_keys.append(key)
        elif isinstance(value, list):
            if any(~np.isnan(item) for item in value):
                readiness_available_keys.append(key)
    readiness_score_values = {key: value for key, value in readiness_score_values.items() if key in readiness_available_keys}
    
    if readiness_score_values:
        personalized_score_weights = {}
        for k,v in readiness_score_values.items():
            component_average = daily_data.loc[:current_data_index,k][(daily_data.loc[:current_data_index,k] > 0) & (daily_data.loc[:current_data_index,k] <= 100)].rolling(window=7, min_periods=1).mean().values.tolist()
            if component_average:
                component_average = component_average[-1]
            else:
                component_average = v
            score_weight = calculate_personalized_weight(v, component_average)
            personalized_score_weights[k] = score_weight
        readiness_bias = calculate_bias(list(readiness_score_values.values()))
        readiness_softmax_weights = calculate_personalized_softmax_weights(list(personalized_score_weights.values()), temperature=0.1, bias=readiness_bias)
        daily_data.loc[current_data_index,"readiness.new_readiness_score"] = multiply_sum_weights_scores(list(readiness_score_values.values()), readiness_softmax_weights)     

    # ------------------------------------------------------------------------------
    # [END] Store Health, Readiness Scores, and New Readiness Score

    # [START] Store Calculation Parameters
    # ------------------------------------------------------------------------------
    # MODIFIED: Round initial values
    daily_data.loc[current_data_index, 'param.initial_mental_value'] = round(initial_mental_expenditure, 2)
    daily_data.loc[current_data_index, 'param.initial_physical_value'] = round(initial_physical_expenditure, 2)
    # ------------------------------------------------------------------------------
    # [END] Store Calculation Parameters
    
    # [START] Store Mental Battery Results
    # ------------------------------------------------------------------------------
    daily_data.at[current_data_index, "mental.timeseries"] = str([round(float(val), 4) for val in mental_result_new])
    daily_data.loc[current_data_index, "mental.final_value"] = round(float(mental_final_value), 4) if mental_final_value is not None else None
    daily_data.loc[current_data_index, 'mental.value_at_sleep_start'] = round(float(sleep_start_mental), 4) if sleep_start_mental is not None else None
    # MODIFIED: Round expenditure rates
    daily_data.loc[current_data_index, 'mental.expenditure_rate_above_50'] = round(average_speed_above_50, 2)
    daily_data.loc[current_data_index, 'mental.expenditure_rate_below_50'] = round(average_speed_below_50, 2)
    # ------------------------------------------------------------------------------
    # [END] Store Mental Battery Results

    # [START] Store Physical Battery Results
    # ------------------------------------------------------------------------------
    daily_data.at[current_data_index, "physical.timeseries"] = str([round(float(val), 4) for val in physical_result_new])
    daily_data.loc[current_data_index, "physical.final_value"] = round(float(physical_final_value), 4) if physical_final_value is not None else None
    daily_data.loc[current_data_index, 'physical.value_at_sleep_start'] = round(float(sleep_start_physical), 4) if sleep_start_physical is not None else None
    # ------------------------------------------------------------------------------
    # [END] Store Physical Battery Results

    # [START] Store Charge (Combined) Results
    # ------------------------------------------------------------------------------
    daily_data.at[current_data_index, "charge.timeseries"] = str([round(float(val), 4) for val in charge_dynamic])
    daily_data.loc[current_data_index, "charge.final_value"] = charge_final_value
    daily_data.loc[current_data_index, "charge.value_at_2100"] = charge_9pm
    # MODIFIED: Round recovery value
    daily_data.loc[current_data_index, "charge.recovery_during_sleep"] = round(sleep_charge_recovery, 2) if sleep_charge_recovery is not None else None
    daily_data.loc[current_data_index, "charge.value_at_sleep_start"] = sleep_start_charge
    # ------------------------------------------------------------------------------
    # [END] Store Charge (Combined) Results

    # [START] Store Summary Data
    # ------------------------------------------------------------------------------
    daily_data.at[current_data_index, "charge.awake_normal_expenditure_sum"] = str(summary_normal_time_charge_expenditure)
    daily_data.at[current_data_index, "charge.awake_normal_recovery_sum"] = str(summary_normal_time_charge_recovery)
    daily_data.at[current_data_index, "charge.awake_sleep_recovery_sum"] = str(summary_sleep_recovery)
    daily_data.at[current_data_index, "charge.awake_exercise_expenditure_sum"] = str(summary_exercise_charge_expenditure)
    daily_data.at[current_data_index, "charge.awake_nap_recovery_sum"] = str(summary_nap_charge_recovery)
    daily_data.at[current_data_index, "charge.is_total_charge_touch_5"] = str(summary_is_total_charge_touch_5)
    # ------------------------------------------------------------------------------
    # [END] Store Summary Data

    # [START] Store Checkpoint Data
    # ------------------------------------------------------------------------------
    daily_data.at[current_data_index, "checkpoints.mental_values"] = str(exercies_checkpoints_dict['mental_checkpoints'])
    daily_data.at[current_data_index, "checkpoints.physical_values"] = str(exercies_checkpoints_dict['physical_checkpoints'])
    daily_data.at[current_data_index, "checkpoints.charge_values"] = str(exercies_checkpoints_dict['charge_checkpoints'])
    # ------------------------------------------------------------------------------
    # [END] Store Checkpoint Data

    # [START] Store Event-Specific Data (Exercise & Nap)
    # ------------------------------------------------------------------------------

    daily_data.at[current_data_index, "event.exercise_indices"] = str(exercise_indexes)
    daily_data.at[current_data_index, "event.sleep_indices"] = str(sleep_indexes)
    daily_data.at[current_data_index, "event.nap_indices"] = str(nap_indexes)

    daily_data.at[current_data_index, "event.exercise_starts.mental"] = str(exercies_checkpoints_dict['mental_exercise_starts'])
    daily_data.at[current_data_index, "event.exercise_starts.physical"] = str(exercies_checkpoints_dict['physical_exercise_starts'])
    daily_data.at[current_data_index, "event.exercise_starts.charge"] = str(exercies_checkpoints_dict['charge_exercise_starts'])
    daily_data.at[current_data_index, "event.exercise_ends.mental"] = str(exercies_checkpoints_dict['mental_exercise_ends'])
    daily_data.at[current_data_index, "event.exercise_ends.physical"] = str(exercies_checkpoints_dict['physical_exercise_ends'])
    daily_data.at[current_data_index, "event.exercise_ends.charge"] = str(exercies_checkpoints_dict['charge_exercise_ends'])
    
    daily_data.at[current_data_index, "event.nap_starts.mental"] = str(exercies_checkpoints_dict['mental_nap_starts'])
    daily_data.at[current_data_index, "event.nap_starts.physical"] = str(exercies_checkpoints_dict['physical_nap_starts'])
    daily_data.at[current_data_index, "event.nap_starts.charge"] = str(exercies_checkpoints_dict['charge_nap_starts'])
    daily_data.at[current_data_index, "event.nap_ends.mental"] = str(exercies_checkpoints_dict['mental_nap_ends'])
    daily_data.at[current_data_index, "event.nap_ends.physical"] = str(exercies_checkpoints_dict['physical_nap_ends'])
    daily_data.at[current_data_index, "event.nap_ends.charge"] = str(exercies_checkpoints_dict['charge_nap_ends'])

    daily_data.at[current_data_index, "event.nap_total_charge_recovery"] = str(total_nap_charge_recovery)
    daily_data.at[current_data_index, "event.exercise_total_charge_expenditure"] = str(total_exercise_charge_expenditure)
    daily_data.at[current_data_index, "event.nap_charge_recovery_rate"] = round(nap_recovery_rate, 2)
    daily_data.at[current_data_index, "event.exercise_charge_expenditure_rate"] = round(exercise_expenditure_rate, 2)

    daily_data.at[current_data_index, "event.nap_recovery"] = str(nap_recovery_value_new)
    daily_data.at[current_data_index, "event.exercise_expenditure"] = str(exercise_expenditure_value_new)
    daily_data.at[current_data_index, "event.nap_recovery_rate"] = str(nap_recovery_rate_new)
    daily_data.at[current_data_index, "event.exercise_expenditure_rate"] = str(exercise_expenditure_rate_new)
    daily_data.at[current_data_index, "event.exercise_duration"] = str(exercise_duration)
    daily_data.at[current_data_index, "event.nap_duration"] = str(nap_duration)

    daily_data.at[current_data_index, "event.exercise_true_expenditure"] = str(exercise_expenditure_summary)
    daily_data.at[current_data_index, "event.exercise_observed_expenditure"] = str(observed_exercise_expenditure)

    # ------------------------------------------------------------------------------
    # [END] Store Event-Specific Data

    # [START] Additional timeseries data
    # ------------------------------------------------------------------------------
    daily_data.at[current_data_index, "timeseries.exercise"] = str(timeseries_sheet['exercise'].tolist())
    daily_data.at[current_data_index, "timeseries.nap_state"] = str(timeseries_sheet['nap_state'].tolist())
    daily_data.at[current_data_index, "timeseries.sleep_markers"] = str(timeseries_sheet['sleep_markers'].tolist())
    daily_data.at[current_data_index, "event.exercise_indices"] = str(exercise_indexes)

    hrr_list_filtered = [round(float(hrr_minute_list[i]), 4) if (20 < hr_list[i] < 220 and min_status_list[i] == 0) else 0 for i in range(len(hrr_minute_list))]
    trimp_list_filtered = [round(minute_trimp[i], 4) if (20 < hr_list[i] < 220 and min_status_list[i] == 0) else 0 for i in range(len(minute_trimp))]

    hrr_list_normalized = [round(float(hrr_minute_list[i]/hrr_max), 4) if (20 < hr_list[i] < 220 and min_status_list[i] == 0) else 0 for i in range(len(hrr_minute_list))]
    trimp_list_filtered_normalized = [round(minute_trimp[i]/trimp_max, 4) if (20 < hr_list[i] < 220 and min_status_list[i] == 0) else 0 for i in range(len(minute_trimp))]

    daily_data.at[current_data_index, "timeseries.minute_trimp"] = str([round(float(val), 4) for val in trimp_list_filtered])
    daily_data.at[current_data_index, "timeseries.hrr_filtered"] = str([round(float(val), 4) for val in hrr_list_filtered])

    daily_data.at[current_data_index, "timeseries.nornalized_minute_trimp"] = str([round(float(val), 4) for val in trimp_list_filtered_normalized])
    daily_data.at[current_data_index, "timeseries.normalized_hrr"] = str([round(float(val), 4) for val in hrr_list_normalized])

    minute_exertion = list(timeseries_sheet['minute_exertion'])
    daily_data.at[current_data_index, "timeseries.minute_exertion"] = str([round(me, 4) for me in minute_exertion])

    if 'fitness.fitness_mean_7' not in daily_data.columns:
        daily_data['fitness.fitness_mean_7'] = 0 
    
    if 'fitness.fatigue_mean_7' not in daily_data.columns:
        daily_data['fitness.fatigue_mean_7'] = 0

    if not "fitness.fatigue_daily_score":
        daily_data.at[current_data_index, "fitness.fatigue_daily_score"] = 0.0
    
    if not "fitness.fitness_daily_score":
        daily_data.at[current_data_index, "fitness.fitness_daily_score"] = 0.0

    fitness_mean_7 = daily_data.loc[current_index, "fitness.fitness_mean_7"]
    fitness_fatigue_mean_7 = daily_data.loc[current_index, "fitness.fatigue_mean_7"]

    if fitness_mean_7 != 0:
        daily_data.at[current_data_index, "fitness.fitness_percentage"] = round(float((daily_data.loc[current_index, "fitness.fitness_daily_score"] / fitness_mean_7)) * 100, 4)
    else:
        daily_data.at[current_data_index, "fitness.fitness_percentage"] = 0.0

    if fitness_fatigue_mean_7 != 0:
        daily_data.at[current_data_index, "fitness.fatigue_percentage"] = round(float((daily_data.loc[current_index, "fitness.fatigue_daily_score"] / fitness_fatigue_mean_7)) * 100, 4)
    else:
        daily_data.at[current_data_index, "fitness.fatigue_percentage"] = 0.0

    daily_data.loc[current_index, "fitness.fitness_mean_3"] = daily_data.loc[:current_index, "fitness.fitness_daily_score"].rolling(window=3, min_periods=1).mean().values.tolist()[-1]

    # Convert np.float to Python float before rounding and assignment
    physical_awake_charge_diff = float(abs(float(daily_data.at[current_data_index,"param.initial_physical_value"]) - float(daily_data.at[current_data_index,"physical.value_at_sleep_start"])))
    daily_data.loc[current_index, "physical.awake_charge_diff_daily"] = round(physical_awake_charge_diff, 4)

    # Ensure rolling mean is a Python float
    rolling_mean_physical = daily_data.loc[:current_index, "physical.awake_charge_diff_daily"].rolling(window=7, min_periods=1).mean().values.tolist()[-1]
    daily_data.at[current_data_index, "physical.awake_recharge_7_day_window"] = float(rolling_mean_physical)

    mental_awake_charge_diff = float(abs(float(daily_data.at[current_data_index,"param.initial_mental_value"]) - float(daily_data.at[current_data_index,"mental.value_at_sleep_start"])))
    daily_data.loc[current_index, "mental.awake_charge_diff_daily"] = round(mental_awake_charge_diff, 4)

    # Ensure rolling mean is a Python float
    rolling_mean_mental = daily_data.loc[:current_index, "mental.awake_charge_diff_daily"].rolling(window=7, min_periods=1).mean().values.tolist()[-1]
    daily_data.at[current_data_index, "mental.awake_recharge_7_day_window"] = float(rolling_mean_mental)

    daily_data.at[current_data_index, "timeseries.min_status_list"] = str(min_status_list)
    daily_data.at[current_data_index, "timeseries.hrr_minute_list"] = str([round(float(hrr), 4) for hrr in hrr_minute_list])
    daily_data.at[current_data_index, "timeseries.full_stress_list"] = str([round(float(val), 4) for val in full_stress_list])

    # hrr_raw is the same HRR% series as hrr_minute_list (name expected by the model/model_config)
    daily_data.at[current_data_index, "timeseries.hrr_raw"] = str([round(float(hrr), 4) for hrr in hrr_minute_list])

    # hr_filtered: valid HR values (20 < hr < 220, device worn); 255 for invalid/off-wrist minutes
    # The dataset layer converts values > 240 to the -6.0 missing sentinel during training/inference
    hr_filtered = [round(float(hr_list[i]), 2) if (20 < hr_list[i] < 220 and min_status_list[i] == 0) else 255 for i in range(len(hr_list))]
    daily_data.at[current_data_index, "timeseries.hr_filtered"] = str(hr_filtered)

    # acc: per-minute acceleration/activity from the raw 'active' column
    daily_data.at[current_data_index, "timeseries.acc"] = str(list(timeseries_sheet['active']))

    # sleep_stage: per-minute sleep stage after firmware code mapping {4→2 deep, 8→1 light, 5→0 REM, else→-1}
    daily_data.at[current_data_index, "sleep_stage"] = str([int(s) for s in timeseries_sheet['sleep_stage']])
    daily_data.at[current_data_index, "heart.hr_max"] = str(hr_max)
    daily_data.at[current_data_index, "heart.rhr"] = str(rhr)

    # Save only the time part (HH:MM) for each timestamp
    daily_data.at[current_data_index, "time"] = str([int(pd.to_datetime(t).timestamp()) for t in timeseries_sheet['time']])

    daily_data.loc[current_index, "awake_recharge.fatigue_sum_7"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=7, min_periods=1).sum().values.tolist()[-1]
    daily_data.loc[current_index, "fitness.fatigue_sum_7"] = daily_data.loc[:current_index, "fitness.fatigue_daily_score"].rolling(window=7, min_periods=1).sum().values.tolist()[-1]

    # ------------------------------------------------------------------------------
    # [END] Additional timeseries data


    # [START] Store Guidance and Insights
    # ------------------------------------------------------------------------------
    today_insights = generate_guidance_for_one_day(current_data_index, daily_data, df_guidance)
    if isinstance(today_insights, pd.DataFrame):
        insight_column_names = today_insights.columns.to_list()
        for col_name in insight_column_names:
            guidance_value = today_insights.loc[0,col_name]
            new_col_name = f"guidance.{col_name}"
            daily_data.at[current_data_index, new_col_name] = str(guidance_value)
    # ------------------------------------------------------------------------------
    # [END] Store Guidance and Insights

    # ==============================================================================
    #                        END: Store results in DataFrame
    # ==============================================================================
    
    temp_cols = [col for col in daily_data.columns if col.endswith('_temp')]
    daily_data.drop(columns=temp_cols, inplace=True, errors='ignore')

    res_dict = {
            "initial_mental_expenditure": initial_mental_expenditure,
            "initial_physical_expenditure": initial_physical_expenditure,
            "temperature_score": temperature_score,
            "HRV_score": HRV_score,
            "RHR_score": RHR_score,
            "AHI_score": AHI_score,
            "final_charge_score": charge_final_value,
            "rhr": rhr,
            "hr_max": hr_max,
            "hrr_minute_list": hrr_minute_list,
            "sleep_start_time": sleep_start_time_index,
            "sleep_duration_score": sleep_duration_score,
            "WASO_score": WASO_score,
            "sleep_start_time_score": sleep_start_time_score,
            "health_score": health_score,
            "mental_result_new": mental_result_new,
            "physical_result_new": physical_result_new,
            "charge_dynamic": charge_dynamic,
            "min_status_list": min_status_list,
            "full_stress_list": full_stress_list,
            "shutdown_count_list": shutdown_count_list,
            "physical_mode": min_status_list,
            "mental_mode": mental_mode,
            "prior_health_score": prior_health_score,
            "health_score_list": health_score_list,
            "mental_new": mental_new,
            "physical_new": physical_new
        }
    return daily_data, res_dict


def calculate_one_user(user_ID, physical_parameters_tuned, mental_parameters_tuned, data_folder, current_dir, result_path, df_guidance, original=False, plot=False, dynamic=False):
    print(f"Processing user {user_ID}")
    daily_data_user = pd.read_csv(f"{data_folder}/user_score_data/{user_ID}.csv", index_col=False)
    all_date = list(daily_data_user.date)

    daily_data = daily_data_user.copy()

    new_columns = {
        "sleep.refined_rhr": None, "sleep.duration_score": None, "sleep.start_time_score": None,
        "sleep.deep_sleep_score": None, "sleep.waso_score": None, "sleep.total_score": None,
        "readiness.score": None, "health.score": None,
        "param.initial_mental_value": None, "param.initial_physical_value": None,
        "daily_stress_accumulation": None, "chronic_daily_stress": None,
        "daily_stress_accumulation": None, "exertion_score": None, "chronic_daily_stress": None,
        "mental.timeseries": None, "mental.final_value": None, "mental.value_at_sleep_start": None,
        "mental.expenditure_rate_above_50": None, "mental.expenditure_rate_below_50": None,
        "physical.timeseries": None, "physical.final_value": None, "physical.value_at_sleep_start": None,
        "charge.timeseries": None, "charge.final_value": None, "charge.value_at_2100": None,
        "charge.recovery_during_sleep": None, "charge.value_at_sleep_start": None,
        "checkpoints.mental_values": None, "checkpoints.physical_values": None, "checkpoints.charge_values": None,

        "event.sleep_indices": None, "event.nap_indices": None, "event.exercise_indices": None,
        "event.exercise_true_expenditure": None, "event.exercise_observed_expenditure": None,
        "event.exercise_starts.mental": None, "event.exercise_starts.physical": None,
        "event.exercise_starts.charge": None, "event.exercise_ends.mental": None, "event.exercise_ends.physical": None,
        "event.exercise_ends.charge": None, "event.exercise_total_charge_expenditure": None, "event.exercise_charge_expenditure_rate": None,
        "event.nap_starts.mental": None, "event.nap_starts.physical": None,
        "event.nap_starts.charge": None, "event.nap_ends.mental": None, "event.nap_ends.physical": None,
        "event.nap_ends.charge": None, "event.nap_total_charge_recovery": None, "event.nap_charge_recovery_rate": None,
        "event.nap_recovery": None, "event.exercise_expenditure": None, 
        "event.nap_duration": None, "event.exercise_duration": None,
        "event.nap_recovery_rate": None, "event.exercise_expenditure_rate": None,

        "guidance.morning_metric_reading": None, "guidance.morning_metric_level": None, "guidance.morning_factor": None,
        "guidance.morning_factor_reading": None, "guidance.morning_factor_level": None, "guidance.morning_factor_zscore": None,
        "guidance.morning_all_factors_zscores": None, "guidance.morning_guidance_text": None,
        "guidance.evening_metric_reading": None, "guidance.evening_metric_level": None, "guidance.evening_factor": None,
        "guidance.evening_factor_reading": None, "guidance.evening_factor_level": None, "guidance.evening_factor_zscore": None,
        "guidance.evening_all_factors_zscores": None, "guidance.evening_guidance_text": None,
        "guidance.nap_replenishment_readings": None, "guidance.nap_guidance_texts": None,
        "guidance.exercise_expenditure_readings": None, "guidance.exercise_guidance_texts": None, 
        
        "fitness.fitness_daily_score": None, "fitness.fitness_daily_score_refined": None, 
        "fitness.fitness_mean_3": None, "fitness.fitness_mean_7": None, "fitness.fitness_mean_14": None, "fitness.fitness_mean_42": None,
        "fitness.fitness_sum_3": None, "fitness.fitness_sum_7": None, "fitness.fitness_sum_14": None, "fitness.fitness_sum_42": None,
        
        "fitness.fatigue_daily_score": None,
        "fitness.fatigue_mean_3": None, "fitness.fatigue_mean_7": None, "fitness.fatigue_mean_14": None, "fitness.fatigue_mean_42": None,
        "fitness.fatigue_sum_3": None, "fitness.fatigue_sum_7": None, "fitness.fatigue_sum_14": None, "fitness.fatigue_sum_42": None,
        
        "fitness.fitness_fatigue_difference": None,
                
        "charge.awake_depletion_sum": None, "charge.awake_recharging_sum": None, "charge.awake_depletion_count": None, "charge.awake_recharging_count": None,
        
        "hrv_factor": None,
        
        "readiness.new_readiness_score": None,
        
        "stress.fitness_daily_score": None, "stress.fitness_mean_7": None, "stress.fatigue_daily_score": None, "stress.fatigue_mean_3": None, "stress.fitness_fatigue_difference":None,
    }

    daily_data                          = pd.concat([daily_data, pd.DataFrame(columns=list(new_columns.keys()))], axis=1)
    daily_data['mental.final_value']    = pd.to_numeric(daily_data['mental.final_value'], errors='coerce')
    daily_data['physical.final_value']  = pd.to_numeric(daily_data['physical.final_value'], errors='coerce')
    daily_data['chronic_daily_stress']  = pd.to_numeric(daily_data['chronic_daily_stress'], errors='coerce')
    daily_data['health.score']          = pd.to_numeric(daily_data['health.score'], errors='coerce')
    
    if 'mentalWake' not in daily_data.columns:
        daily_data['mentalWake']= None
        daily_data['mentalWake'] = pd.to_numeric(daily_data['mentalWake'], errors='coerce')
    if 'physicalWake' not in daily_data.columns:
        daily_data['physicalWake']= None
        daily_data['physicalWake'] = pd.to_numeric(daily_data['physicalWake'], errors='coerce')
        
    # Initialize the model and the online decomposer
    model = OnlineGLM(); minute_tracker = MinuteOfDayAverage()
    decomposer = OnlineDecomposer()
    signal_processor = DynamicSignalProcessor()
    daily_data_ = daily_data.copy()
    
    if original:
        press_thresh = 12/115
    else:
        press_thresh = 0.20
        
    for current_index, sheet in tqdm(enumerate(all_date), total=len(all_date), desc="Sub-tasks", position=1, leave=False, colour='yellow'):
        yesterday_date = (datetime.strptime(sheet, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        correct_date = sheet
        if os.path.exists(f"{data_folder}/user_sleep_data/{user_ID}/{correct_date}.csv"):
            timeseries_sheet = pd.read_csv(f"{data_folder}/user_sleep_data/{user_ID}/{correct_date}.csv")
        else:
            # print(f"No data for {sheet}")
            continue
        daily_row = daily_data[(daily_data.userid == int(user_ID)) & (daily_data.date == correct_date)]
        yesterday_row = daily_data[(daily_data.userid == int(user_ID)) & (daily_data.date == yesterday_date)]

        daily_data = daily_data.replace(-1, np.nan)
        
        if daily_row.shape[0] > 0:
            
            daily_data, res_dict = calculate_one_day(timeseries_sheet, sheet,
                                            daily_data, daily_row, yesterday_row, current_index,
                                            mental_parameters_tuned, physical_parameters_tuned, current_dir, df_guidance,
                                            model=model, minute_tracker=minute_tracker, decomposer=decomposer, signal_processor=signal_processor, 
                                            original=original, press_thresh = press_thresh, plot=plot, dynamic=dynamic)
    
            # print(f"current_date: {correct_date}", f"today fitness and fatigue : {daily_data.loc[current_index, 'fitness.fitness_daily_score']}, {daily_data.loc[current_index, 'fitness.fatigue_daily_score']}, 7-day fitness and fatigue : {daily_data.loc[current_index, 'fitness.fitness_mean_7']}, {daily_data.loc[current_index, 'fitness.fatigue_mean_7']}")
            # print(f"exercise duration: {daily_data.loc[current_index, 'exercise_duration']}")
            
            # print(daily_row['fitness.fatigue_daily_score'])
            # print(daily_row['fitness.fatigue_mean_7'])
            charge_timeseries = daily_data.loc[current_index,"charge.timeseries"]
            if isinstance(charge_timeseries, str):
                charge_timeseries = ast.literal_eval(daily_data.loc[current_index,"charge.timeseries"])
                charge_timeseries_diff = pd.DataFrame(charge_timeseries).diff().values.tolist()
                charge_timeseries_diff_awake = [charge_timeseries_diff[i][0] for i,_ in enumerate(charge_timeseries_diff) if timeseries_sheet.sleep_state[i] == 0]
                charge_timeseries_diff_awak_depletion = [i for i in charge_timeseries_diff_awake if i < 0]
                charge_timeseries_diff_awak_recharging = [i for i in charge_timeseries_diff_awake if i >= 0]
                
                daily_data.loc[current_index, "charge.awake_depletion_sum"] = sum(charge_timeseries_diff_awak_depletion)
                daily_data.loc[current_index, "charge.awake_recharging_sum"] = sum(charge_timeseries_diff_awak_recharging)
                
                daily_data.loc[current_index, "charge.awake_depletion_count"] = len(charge_timeseries_diff_awak_depletion)
                daily_data.loc[current_index, "charge.awake_recharging_count"] = len(charge_timeseries_diff_awak_recharging)


    file_name = f"{result_path}/{user_ID}_processed.xlsx"

    # ==============================================================================
    #                      START: Write to Excel with Colors
    # ==============================================================================
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        daily_data.to_excel(writer, sheet_name='daily_data', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['daily_data']

        # 定义颜色格式
        colors = {
            'sleep': '#DDEBF7',   # Light Blue
            'readiness': '#FFF2CC', # Light Yellow
            'health': '#E2F0D9',   # Light Green
            'param': '#FCE4D6',   # Light Orange
            'mental': '#D9E1F2',   # Light Purple
            'physical': '#F2DCDB',  # Light Red
            'charge': '#C9DAF8',  # Periwinkle
            'checkpoints': '#D0CECE', # Grey
            'event': '#EAD1DC',   # Pink
            'guidance': '#D4EAE6'  # Teal
        }
        
        # 创建格式对象
        formats = {name: workbook.add_format({'bg_color': color, 'bold': True}) for name, color in colors.items()}
        
        # 为原始列添加默认颜色
        original_cols_format = workbook.add_format({'bg_color': '#FFFFFF', 'bold': True}) # White

        # 获取所有列名
        column_headers = daily_data.columns.values.tolist()

        # 遍历列并应用格式
        for col_idx, header in enumerate(column_headers):
            prefix = header.split('.')[0]
            if prefix in formats:
                worksheet.set_column(col_idx, col_idx, len(header) + 2, formats[prefix])
            else:
                # 应用默认格式到没有特定前缀的原始列
                worksheet.set_column(col_idx, col_idx, len(header) + 2, original_cols_format)
        # 2. 检查 'wake_up_scenario' 列是否存在，避免出错
        green_cell_format = workbook.add_format({'bg_color': '#C6EFCE'}) # 一个柔和的绿色
        target_columns = ['physical.final_value', 'charge.final_value', 'mental.final_value']

        target_col_indices = {
                col: column_headers.index(col) 
                for col in target_columns if col in column_headers
            }

        # 3. 检查必要的列是否存在
        if 'wake_up_scenario' in column_headers and target_col_indices:
            # 4. 遍历DataFrame的每一行
            for index, row_data in daily_data.iterrows():
                
                # 5. 检查条件：'wake_up_scenario' 是否为 'Actual'
                if row_data['wake_up_scenario'] == 'Actual':
                    
                    # 获取Excel中的实际行号
                    excel_row_num = index + 1
                    
                    # 6. 如果条件满足，则遍历目标列，并修改对应单元格的颜色
                    for col_name, col_num in target_col_indices.items():
                        cell_value = row_data[col_name]
                        
                        # 检查并处理 NaN 值，防止程序出错
                        if pd.isna(cell_value):
                            worksheet.write_string(excel_row_num, col_num, '', green_cell_format)
                        else:
                            worksheet.write(excel_row_num, col_num, cell_value, green_cell_format)
    # ==============================================================================
    #                       END: Write to Excel with Colors
    # ==============================================================================
    print(f"User {user_ID} processed successfully and saved to {file_name}")
    return daily_data
