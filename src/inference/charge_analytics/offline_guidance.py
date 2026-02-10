import pandas as pd
import numpy as np
import ast
from typing import List, Optional, Union
import os
import random

def factor_level_convertor_vectorized(factor_boundaries_df):
    """
    Calculates adjusted 'Reading' and assigns qualitative 'Level' for each
    row in the input DataFrame based on factor name, raw reading, and boundaries.

    This version uses vectorized operations for efficiency.

    Args:
        factor_boundaries_df (pd.DataFrame): DataFrame with columns including:
            - Factor (str): The name of the factor.
            - Boundaries (list/array): List/array of boundary thresholds.
            - raw_reading (float): The initial reading before adjustments.

    Returns:
        pd.DataFrame: The DataFrame with two new/updated columns:
            - Reading (float): The calculated (potentially adjusted) reading.
            - Level (str or np.nan): The qualitative level ('High', 'Low', '')
                                     or np.nan if inputs are invalid.
    """
    if not isinstance(factor_boundaries_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not {'Factor', 'Boundaries', 'Reading'}.issubset(factor_boundaries_df.columns):
         raise ValueError("Input DataFrame must contain 'Factor', 'Boundaries', and 'Reading' columns.")

    df = factor_boundaries_df.copy() # Work on a copy

    # --- 1. Calculate Adjusted 'Reading' Column ---
    # Use pd.to_numeric with errors='coerce' to safely handle non-numeric/sequence values
    df['Reading'] = pd.to_numeric(df['Reading'], errors='coerce') 

    # --- 2. Prepare for Level Calculation ---
    # --- 2a. Validate Boundaries and Reading ---
    # Check if Boundaries column contains valid lists/arrays & Reading is not NaN
    def is_valid_boundary_list(b):
        try:
            # Check it's list/array, has items, and no NaNs inside
            return isinstance(b, (list, np.ndarray)) and len(b) > 0 and not np.any(pd.isna(b))
        except Exception: # Catch potential errors during isnan check etc.
            return False

    valid_boundaries_mask = df['Boundaries'].apply(is_valid_boundary_list)
    valid_reading_mask = df['Reading'].notna()
    valid_input_mask = valid_boundaries_mask & valid_reading_mask

    # --- 2b. Extract Boundary Values Safely ---
    # Initialize columns for boundary values
    b0 = pd.Series(np.nan, index=df.index)
    b1 = pd.Series(np.nan, index=df.index)
    b2 = pd.Series(np.nan, index=df.index)

    # Populate only where boundaries are valid
    if valid_input_mask.any():
        valid_df = df[valid_input_mask]
        len_ge_1 = valid_df['Boundaries'].str.len() >= 1
        if len_ge_1.any():
            b0[valid_df[len_ge_1].index] = valid_df.loc[len_ge_1, 'Boundaries'].str[0]
        
        len_ge_2 = valid_df['Boundaries'].str.len() >= 2
        if len_ge_2.any():
            b1[valid_df[len_ge_2].index] = valid_df.loc[len_ge_2, 'Boundaries'].str[1]
        
        len_ge_3 = valid_df['Boundaries'].str.len() >= 3
        if len_ge_3.any():
            b2[valid_df[len_ge_3].index] = valid_df.loc[len_ge_3, 'Boundaries'].str[2]


    # --- 2c. Create Helper Masks ---
    len1_mask = valid_input_mask & (df['Boundaries'].str.len() == 1)
    len2_mask = valid_input_mask & (df['Boundaries'].str.len() == 2)
    len3_mask = valid_input_mask & (df['Boundaries'].str.len() >= 3) # >= 3 covers original > 2

    # Factor type hints (using valid_input_mask to avoid unnecessary checks)
    # Use case=False for case-insensitivity, na=False to treat NaN Factor as False
    is_waso_duration = valid_input_mask & (df['Factor'] == 'wakefulness after sleep onset (WASO) duration')
    is_ahi = valid_input_mask & (df['Factor'] == 'apnea/hyponia index (AHI)')
    # Re-calculate time/duration masks using valid_input_mask scope
    is_time = valid_input_mask & df['Factor'].str.contains('time', case=False, na=False)
    is_duration = valid_input_mask & df['Factor'].str.contains('duration', case=False, na=False)

    is_rest_charge_change = valid_input_mask & (
        (df['Factor'] == 'rest Charge replenishment') | 
        (df['Factor'] == 'rest Charge rate of change')
    )

    # --- 3. Assign Level using np.select ---
    # Define conditions (most specific first within a length group)
    condlist = [
        # --- Len 1 logic ---
        # Specific case: WASO duration with Reading >= b0 -> "Long"
        len1_mask & is_waso_duration & (df['Reading'] >= b0),
        # Specific case: Charge Change Factors with Reading >= b0 -> "High"
        len1_mask & is_rest_charge_change & (df['Reading'] >= b0),
        # Specific case: Charge Change Factors with Reading < b0 -> "Low"
        len1_mask & is_rest_charge_change & (df['Reading'] < b0),
        # General case for Len 1 where Reading >= b0 -> "High" (covers non-WASO, non-charge change factors)
        len1_mask & ~is_waso_duration & ~is_rest_charge_change & (df['Reading'] >= b0),
        # General case for Len 1 where Reading < b0 -> "" (covers non-WASO, non-charge change factors)
        len1_mask & ~is_rest_charge_change & (df['Reading'] < b0), # Note: WASO < b0 also falls here and gets ""


        # Len 2
        len2_mask & is_time & (df['Reading'] <= b0),
        len2_mask & is_time & (df['Reading'] >= b1),
        len2_mask & is_duration & ~is_time & (df['Reading'] <= b0), # Added ~is_time
        len2_mask & is_duration & ~is_time & (df['Reading'] >= b1), # Added ~is_time
        len2_mask & ~is_time & ~is_duration & (df['Reading'] <= b0),
        len2_mask & ~is_time & ~is_duration & (df['Reading'] >= b1),
        # Case for between b0 and b1 for len 2: remains "" (covered by default)

        # Len 3+
        len3_mask & ~is_ahi & (df['Reading'] < b0),
        len3_mask & ~is_ahi & (df['Reading'] >= b0) & (df['Reading'] < b1),
        len3_mask & ~is_ahi & (df['Reading'] >= b1), # reading >= b1 (including original b1-b2 range and >b2) maps to ""
        len3_mask & is_ahi & (df['Reading'] < b0), # Added is_ahi
        len3_mask & is_ahi & (df['Reading'] >= b0) & (df['Reading'] < b1), # Added is_ahi
        len3_mask & is_ahi & (df['Reading'] >= b1) & (df['Reading'] < b2), # Added is_ahi
        len3_mask & is_ahi & (df['Reading'] >= b2) # Added is_ahi
    ]

    # Define corresponding choices
    choicelist = [
        # Len 1
        "Long",      # WASO >= b0
        "High",      # Rest Charge Change >= b0
        "Low",       # Rest Charge Change < b0
        "High",      # General >= b0 (non-WASO, non-charge change)
        "",          # General < b0 (non-charge change, WASO also gets "")
        # Len 2
        "Early", "Late",
        "Short", "Long",
        "Low", "High",
        # Len 3+
        "Low", "Fair", "Optimal",
        "", "Mild", "Moderate", "Severe",
    ]

    # Apply np.select, default to empty string for valid inputs not meeting conditions
    df['Level'] = np.select(condlist, choicelist, default="")

    # --- 4. Handle Invalid Inputs ---
    # Overwrite 'Level' with np.nan where input was invalid
    df.loc[~valid_input_mask, 'Level'] = np.nan
    # Ensure the 'Level' column allows object dtype if it contains strings and nans
    if df['Level'].isnull().any() and not pd.api.types.is_object_dtype(df['Level']):
         df['Level'] = df['Level'].astype(object)
         df.loc[~valid_input_mask, 'Level'] = np.nan


    # --- 5. Return Result ---
    return df

def zscore_factor_readings_vectorized(factor_levels: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates normalized 'Index' and 'Effect' columns based on 'Reading',
    'Mean', 'SD', and 'Weight', conditional on 'Boundaries' being valid.

    The 'Index' is calculated as the absolute rounded Z-score equivalent:
    abs(round((Reading - Mean) / SD, 2)).
    The 'Effect' is Index * Weight, rounded, unless Index is 0 (then NaN).

    This vectorized version applies the calculation efficiently across the DataFrame.

    Args:
        factor_levels (pd.DataFrame): DataFrame requiring columns:
            - Reading (float): The value to normalize.
            - Mean (float): The mean for Z-score calculation.
            - SD (float): The standard deviation for Z-score calculation.
            - Weight (float): The weight for calculating the effect.
            - Boundaries (list/array): A list/array used only for validation
                                       (must not contain NaNs). Content/length
                                       does not affect calculations in this version.

    Returns:
        pd.DataFrame: The DataFrame with two new/updated columns:
            - Index (float): The calculated normalized index, or np.nan.
            - Effect (float): The calculated effect, or np.nan.

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If required columns are missing.
    """
    if not isinstance(factor_levels, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    required_cols = {'Reading', 'Mean', 'SD', 'Weight', 'Boundaries'}
    if not required_cols.issubset(factor_levels.columns):
         missing_cols = required_cols - set(factor_levels.columns)
         raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")

    df = factor_levels.copy() # Work on a copy

    # Initialize output columns
    df['Index'] = np.nan
    df['Effect'] = np.nan

    # --- Create a mask for rows where calculation is possible ---

    # 1. Check if Boundaries is list/array-like and contains no NaNs
    #    Adapt validation if 'Boundaries' can have other types but still needs NaN check
    def is_valid_boundary_list(b):
        try:
            # Check for list/array type and no NaNs inside
            # Allow empty lists based on original `all` check behavior
            return isinstance(b, (list, np.ndarray)) and not np.any(pd.isna(b))
        except Exception:
            # Handle cases where isnan check might fail on non-numeric contents
            return False # Treat as invalid if check fails

    valid_boundaries_mask = df['Boundaries'].apply(is_valid_boundary_list)

    # 2. Check required numeric inputs are valid and SD is non-zero
    valid_inputs_mask = (
        pd.to_numeric(df['Reading'], errors='coerce').notna() &
        pd.to_numeric(df['Mean'], errors='coerce').notna() &
        pd.to_numeric(df['SD'], errors='coerce').notna() &
        (pd.to_numeric(df['SD'], errors='coerce') != 0) & # Avoid division by zero
        pd.to_numeric(df['Weight'], errors='coerce').notna()
    )

    # Combine masks: calculation only happens if boundaries AND inputs are valid
    calculation_mask = valid_boundaries_mask & valid_inputs_mask

    # --- Perform Calculations only on valid rows ---
    if calculation_mask.any():
        # Get the relevant subset of the DataFrame for calculation
        valid_rows = df.loc[calculation_mask]

        # Calculate Z-score equivalent (Reading - Mean) / SD
        # Ensure types are float for division
        z_score = (
            valid_rows['Reading'].astype(float) - valid_rows['Mean'].astype(float)
        ) / valid_rows['SD'].astype(float)

        # Calculate and assign 'Index': absolute rounded Z-score
        calculated_index = z_score.round(2).abs()
        df.loc[calculation_mask, 'Index'] = calculated_index

        # Calculate and assign 'Effect': Index * Weight, rounded
        # Use the 'Index' values we just put into the DataFrame for the mask subset
        calculated_effect = (
            df.loc[calculation_mask, 'Index'] * valid_rows['Weight'].astype(float)
        ).round(2)
        df.loc[calculation_mask, 'Effect'] = calculated_effect

        # Apply the zero-index rule: If Index is 0, Effect should be NaN
        # Apply this globally AFTER calculation, as Index might be 0 even if inputs were valid
        df.loc[df['Index'] == 0, 'Effect'] = np.nan

    # Return the DataFrame with new/updated columns
    return df

def calculate_slope_regression_refactored(
    charge_timeseries: Union[List[float], np.ndarray],
    start_index: Optional[int] = None,
    end_index: Optional[int] = None
) -> float:
    """
    Calculates the slope of a linear regression line for a timeseries segment.

    Handles potential NaN values within the segment by excluding them from
    the regression calculation.

    Args:
        charge_timeseries: The input timeseries data (list or NumPy array).
        start_index: The starting index of the segment (inclusive).
                     If None, defaults to the start of the timeseries (index 0).
        end_index: The ending index of the segment (inclusive).
                   If None, defaults to the end of the timeseries.

    Returns:
        The calculated slope of the linear regression line as a float,
        or np.nan if calculation is not possible (e.g., fewer than 2
        valid non-NaN data points in the segment, or input indices invalid).

    Raises:
        TypeError: If start_index or end_index is provided but not an integer.
        IndexError: If provided indices are out of the bounds of the timeseries.
        ValueError: If start_index > end_index.
    """
    # Ensure input is a NumPy array for efficient processing and NaN handling
    ts = np.asarray(charge_timeseries)
    n_total = len(ts)

    if n_total == 0:
        # Handle empty input timeseries
        return np.nan

    # --- Validate and determine slice boundaries ---
    start = 0 if start_index is None else start_index
    end = n_total - 1 if end_index is None else end_index # Inclusive end index

    # --- Input Index Validation ---
    if not isinstance(start, int):
        raise TypeError(f"start_index must be an integer or None, got {type(start_index)}")
    if not isinstance(end, int):
        raise TypeError(f"end_index must be an integer or None, got {type(end_index)}")

    if not (0 <= start < n_total):
        raise IndexError(f"start_index ({start}) is out of bounds for timeseries of length {n_total}.")
    # end index is inclusive, so it can be n_total - 1
    if not (0 <= end < n_total):
         raise IndexError(f"end_index ({end}) is out of bounds for timeseries of length {n_total}.")

    if start > end:
        raise ValueError(f"start_index ({start}) cannot be greater than end_index ({end}).")
    # --- End Index Validation ---

    # Select the relevant slice (inclusive end index means slice up to end + 1)
    segment = ts[start : end + 1]

    # Handle NaNs: Create mask for valid (non-NaN) values
    valid_mask = ~np.isnan(segment)
    y_values = segment[valid_mask]

    n_valid = len(y_values)

    # Check if enough valid points remain for regression
    if n_valid < 2:
        return np.nan 

    # Create corresponding x values ONLY for the valid y_values
    x_values_full_segment = np.arange(len(segment))
    x_values = x_values_full_segment[valid_mask]

    # Perform linear regression using polyfit
    try:
        coeffs = np.polyfit(x_values, y_values, 1)
        slope = coeffs[0]

        if not np.isfinite(slope):
             return np.nan
        return float(slope)

    except np.linalg.LinAlgError:
        return np.nan
    except Exception as e:
        return np.nan

def find_difference_refactored(
    charge_timeseries: Union[List[float], np.ndarray],
    start_index: int,
    end_index: int
) -> float:
    """
    Calculates the difference between two elements in a timeseries.

    Difference = value at end_index - value at start_index.

    Args:
        charge_timeseries: The input timeseries data (list or NumPy array).
        start_index: The index of the starting element (value to subtract).
        end_index: The index of the ending element (value to subtract from).

    Returns:
        The calculated difference as a float. If either of the values at the
        specified indices is np.nan, the result will be np.nan.

    Raises:
        TypeError: If start_index or end_index is not an integer.
        IndexError: If either index is out of the bounds of the timeseries.
    """
    # Ensure input is a NumPy array for consistent indexing and NaN handling
    ts = np.asarray(charge_timeseries)
    n_total = len(ts)

    # --- Input Validation ---
    if not isinstance(start_index, int):
        raise TypeError(f"start_index must be an integer, got {type(start_index)}")
    if not isinstance(end_index, int):
        raise TypeError(f"end_index must be an integer, got {type(end_index)}")

    if not (0 <= start_index < n_total):
        raise IndexError(f"start_index ({start_index}) is out of bounds for timeseries of length {n_total}.")
    if not (0 <= end_index < n_total):
         raise IndexError(f"end_index ({end_index}) is out of bounds for timeseries of length {n_total}.")
    # --- End Validation ---

    value_at_end = ts[end_index]
    value_at_start = ts[start_index]
    difference = value_at_end - value_at_start

    return float(difference)

def filter_controller(factor_df: pd.DataFrame) -> pd.DataFrame:
  """
  Filters the guidance factors DataFrame based on the 'Controller' column.

  It keeps rows where the 'Controller' column value is:
  1. Less than 85.
  OR
  2. NaN (Not a Number).

  Args:
    factor_df: The input guidance factors DataFrame.

  Returns:
    A new guidance factors DataFrame containing only the rows that meet the criteria.
    If the 'Controller' column does not exist, the original guidance factors DataFrame is
    returned with a warning.
  """
  if "Controller" in factor_df.columns:
    factor_df_copy = factor_df.copy()

    factor_df_copy["Controller"] = pd.to_numeric(factor_df_copy["Controller"], errors='coerce')

    condition = (factor_df_copy["Controller"] < 85) | (factor_df_copy["Controller"].isna())
    filtered_df = factor_df_copy[condition].reset_index(drop=True)
    return filtered_df
  else:
    print("Warning: Column 'Controller' not found in DataFrame. Returning original DataFrame.")
    return factor_df


def update_statistics_boundaries_user_data(factor_boundaries_df, factor_index, factor_mean, factor_sd):
    """
    If the user has historical personal data, this function updates the statistics and boundaries columns:
    'Boundaries', 'Mean', and 'SD'.

    Args:
    factor_boundaries_df: The input guidance factors DataFrame.
    factor_index: The index of the factor to change in the factor_boundaries_df DataFrame.
    factor_mean: The factor mean based on user historical data.
    factor_sd: The factor standard deviation based on user historical data.

    Returns:
    A modified guidance factors DataFrame with a modified factor row based on new boundaries, mean,
    and standard deviation based on the user's historical data.
    """    

    if not isinstance(factor_boundaries_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not {'Factor', 'Boundaries', 'Mean', 'SD', 'Historical Data'}.issubset(factor_boundaries_df.columns):
        raise ValueError("Input DataFrame must contain 'Factor', 'Boundaries', 'Mean', 'SD', and 'Historical Data' columns.")
    
    factor_df_copy = factor_boundaries_df.copy()

    if factor_df_copy.loc[factor_index, "Historical Data"]:
        factor_df_copy.at[factor_index, 'Boundaries'] = [factor_mean, factor_mean]
        factor_df_copy.loc[factor_index, 'Mean'] = factor_mean
        factor_df_copy.loc[factor_index, 'SD'] = factor_sd
        print(f"Historical data statistics and boundaries added for {factor_boundaries_df.loc[factor_index, 'Factor']}.")        
        return factor_df_copy
    else:
        print("Warning: Historical data not included. Returning original DataFrame.")        
        return factor_boundaries_df


def generate_charge_metrics_df(factor_info_df: pd.DataFrame, readings_map: dict, today_morning_charge: float, today_evening_charge: float) -> pd.DataFrame:
    """
    根据预计算的 readings_map 和今日的 Charge 值，生成一个包含 Metric/Factor Readings 和 Values 的 DataFrame。

    Args:
        readings_map (dict): 包含所有 Metric_Factor_FactorPeriod 组合的读取值的字典。
        today_morning_charge (float): 今日的 morning Charge 值。
        today_evening_charge (float): 今日的 evening Charge 值。

    Returns:
        pd.DataFrame: 包含 Metric, Factor, Factor Period, Metric Reading,
                      Factor Reading, Metric Value, Factor Value 的结果 DataFrame。
    """

    drop_factors_level = [("sleep duration", "Long"), ("deep sleep ratio", "High"), ("nap duration", "Long"), ("exercise Charge at lower threshold", "Reached")]
    factor_info_df = factor_info_df[~factor_info_df.apply(lambda row: (row['Factor'], row['Factor Level']) in drop_factors_level, axis=1)]

    unique_factors = factor_info_df.copy()
    unique_factors['Factor_Time'] = factor_info_df['Metric'] + " - " + factor_info_df['Factor'] + " - " + factor_info_df["Factor Period"]
    unique_factors = unique_factors.drop_duplicates(subset=['Factor_Time']).dropna(subset=['Factor']).reset_index()
    unique_factors['Boundaries'] = unique_factors['Boundaries'].apply(ast.literal_eval)

    unique_factors = unique_factors[['Metric', 'Factor', 'Factor Period', 'Boundaries', 'Mean', 'SD', 'Factor_Time']].reset_index()

    unique_factors["Reading"] = np.nan
    unique_factors["Weight"] = 1
    unique_factors["Controller"] = np.nan
    unique_factors["Historical Data"] = None


    all_processed_rows = []

    for idx,template_row in unique_factors.iterrows():
        new_processed_row = template_row.copy() 

        metric_col_val = new_processed_row['Metric']
        factor_col_val = new_processed_row['Factor']
        period_col_val = new_processed_row['Factor Period']

        if metric_col_val == 'morning Charge':
            new_processed_row['Metric Reading'] = today_morning_charge 
        elif metric_col_val == 'evening Charge':
            new_processed_row['Metric Reading'] = today_evening_charge 
        else:
            new_processed_row['Metric Reading'] = np.nan

        lookup_key = f"{metric_col_val}_{factor_col_val}_{period_col_val}"
        factor_reading_value = readings_map.get(lookup_key, np.nan) 

        new_processed_row['Reading'] = factor_reading_value

        all_processed_rows.append(new_processed_row)

    final_output_df = pd.DataFrame(all_processed_rows)
    return final_output_df

def process_one_day_insights(unique_factors: pd.DataFrame, factor_info_df: pd.DataFrame, today_total_nap_charge_recovery: list, today_total_exercise_charge_expenditure: list,
                             yesterday_factors: dict) -> pd.DataFrame:
    if unique_factors['Metric Reading'].isna().all():
        return pd.DataFrame() # Return empty DataFrame if no metric readings
    
    unique_factors = factor_level_convertor_vectorized(unique_factors)
    unique_factors = zscore_factor_readings_vectorized(unique_factors)

    drop_factors_level = [("sleep duration", "Long"), ("deep sleep ratio", "High"), ("nap duration", "Long"), ("exercise Charge at lower threshold", "Reached")]
    unique_factors = unique_factors[~unique_factors.apply(lambda row: (row['Factor'], row['Level']) in drop_factors_level, axis=1)]

    metrics = ['morning Charge', 'nap', 'exercise', 'evening Charge']
    
    guidance_column_names = [
        "morning_metric_reading", "morning_metric_level", "morning_factor", "morning_factor_reading", "morning_factor_level", "morning_factor_zscore", "morning_all_factors_zscores", "morning_guidance_text",
        "evening_metric_reading", "evening_metric_level", "evening_factor", "evening_factor_reading", "evening_factor_level", "evening_factor_zscore", "evening_all_factors_zscores", "evening_guidance_text",
        "nap_replenishment_readings", "nap_guidance_texts",
        "exercise_expenditure_readings", "exercise_guidance_texts"]
    empty_data = {col: [None] for col in guidance_column_names}
    guidance_results_df = pd.DataFrame(empty_data)
    
    for metric in metrics:
        guidance_text = ""
        if "Charge" in metric:
            metric_reading = np.nan
            metric_boundaries = []
            metric_time = ""
            metric_df = pd.DataFrame(columns=["Factor", "Boundaries", "Reading"])
            if "morning" in metric:
                metric_boundaries = [70, 85, 100]
                metric_reading_series = unique_factors['Metric Reading'][unique_factors['Metric'] == metric]
                if not metric_reading_series.empty:
                    metric_reading = metric_reading_series.iloc[0]
                
                guidance_results_df["morning_metric_reading"] = metric_reading
                metric_time = "morning"
                
            elif "evening" in metric:
                metric_boundaries = [25, 40, 100]
                metric_reading_series = unique_factors['Metric Reading'][unique_factors['Metric'] == metric]
                if not metric_reading_series.empty:
                    metric_reading = metric_reading_series.iloc[0]
                
                guidance_results_df["evening_metric_reading"] = metric_reading
                metric_time = "evening"

            metric_df.loc[0, "Factor"] = metric
            metric_df.loc[0, "Boundaries"] = metric_boundaries
            metric_df.loc[0, "Reading"] = metric_reading
            metric_df = factor_level_convertor_vectorized(metric_df)
            metric_level = metric_df.Level[0]
            
            guidance_results_df[f"{metric_time}_metric_level"] = metric_level
            
            factor_readings = unique_factors[unique_factors.Metric == metric]
            factor_readings = factor_readings[(factor_readings["Level"] != "") & (factor_readings["Level"] != "Optimal")]
            
            if not factor_readings.empty and any(factor_readings.Level[factor_readings.Factor == "daily exercise stress"].str.contains("Low", na=False)):
                factor_readings.drop(factor_readings[(factor_readings.Factor == "daily exercise stress") & (factor_readings.Level == "Low")].index, inplace=True)
        else:
            factor_readings = pd.DataFrame()


        # factor_readings = factor_readings.dropna(subset=['Reading'])

        if "Charge" in metric:
            metric_guidance_time = metric.split(" ")[0]
            
            if metric_level == "Optimal":
                guidance_rows = factor_info_df[(factor_info_df["Metric"] == metric) & (factor_info_df["Metric Level"] == metric_level)]
                guidance_text = guidance_rows['Guidance_edited'].iloc[0] if not guidance_rows.empty else ""
            elif metric_level in ["Fair", "Low"] and factor_readings.shape[0] == 0:
                guidance_rows = factor_info_df[(factor_info_df["Metric"] == metric) & (factor_info_df["Metric Level"] == metric_level) & pd.isna(factor_info_df["Factor"])]
                guidance_text = guidance_rows['Guidance_edited'].iloc[0] if not guidance_rows.empty else ""
            elif factor_readings.shape[0] > 0 and 'Effect' in factor_readings.columns and factor_readings['Effect'].notna().any():
                
                max_effect_factor = factor_readings.Effect.idxmax()
                factor_name = factor_readings.Factor[max_effect_factor]
                factor_level = factor_readings.Level[max_effect_factor]
                factor_period = factor_readings['Factor Period'][(factor_readings.Factor == factor_name) & (factor_readings.Level == factor_level)].iloc[0]
                                
                if yesterday_factors.get(metric_guidance_time) == factor_name and factor_readings.shape[0] > 1:
                    factor_readings_modified = factor_readings.drop(max_effect_factor)
                    if factor_readings_modified['Effect'].notna().any():
                        max_effect_factor = factor_readings_modified.Effect.idxmax()
                        factor_name = factor_readings_modified.Factor[max_effect_factor]
                        factor_level = factor_readings_modified.Level[max_effect_factor]
                        factor_period = factor_readings_modified['Factor Period'][(factor_readings_modified.Factor == factor_name) & (factor_readings_modified.Level == factor_level)].iloc[0]

                guidance_rows = factor_info_df[
                    (factor_info_df["Metric"] == metric)
                    & (factor_info_df["Metric Level"] == metric_level)
                    &(factor_info_df["Factor"] == factor_name)
                    & (factor_info_df["Factor Level"] == factor_level)
                    & (factor_info_df["Factor Period"] == factor_period)
                ]
                
                guidance_text = guidance_rows.Guidance_edited.iloc[0] if not guidance_rows.empty else ""
                
                guidance_results_df.loc[0,f"{metric_guidance_time}_factor"] = factor_name
                guidance_results_df.loc[0,f"{metric_guidance_time}_factor_reading"] = factor_readings.Reading[max_effect_factor]
                guidance_results_df.loc[0,f"{metric_guidance_time}_factor_level"] = factor_level
                guidance_results_df.loc[0,f"{metric_guidance_time}_factor_zscore"] = factor_readings.Effect[max_effect_factor]
                
                key_col = 'Factor'
                reading_col = 'Reading'
                boundaries_col = 'Boundaries'
                zscore_col = 'Effect'
                
                factor_dict = factor_readings.set_index(key_col).apply(lambda row: [row[reading_col], row[boundaries_col],row[zscore_col]], axis=1).to_dict()
                
                guidance_results_df.at[0,f"{metric_guidance_time}_all_factors_zscores"] = factor_dict
            
            guidance_results_df.loc[0, f"{metric_guidance_time}_guidance_text"] = guidance_text
                
        elif metric == "exercise":
            if today_total_exercise_charge_expenditure and len(today_total_exercise_charge_expenditure) > 0:
                exercise_guidance_texts = []
                exercise_guidance_expenditures = []
                for exercise_expenditure in today_total_exercise_charge_expenditure:
                    exercise_expenditure = round(exercise_expenditure, 0)
                    if exercise_expenditure > 1:
                        guidance_text = "Your exercise session consumed {diff:.0f} points of BioCharge."
                    elif exercise_expenditure == 1:
                        guidance_text = "Your exercise session consumed {diff:.0f} point of BioCharge."
                    elif exercise_expenditure == 0:
                        guidance_text = "Your exercise session consumed no BioCharge."
                    guidance_text = guidance_text.format(diff = exercise_expenditure)
                    exercise_guidance_texts.append(guidance_text)
                    exercise_guidance_expenditures.append(exercise_expenditure)

                guidance_results_df.at[0,f"{metric}_guidance_texts"] = exercise_guidance_texts
                guidance_results_df.at[0,f"{metric}_expenditure_readings"] = exercise_guidance_expenditures

            else:
                guidance_results_df.at[0,f"{metric}_guidance_texts"] = []
                guidance_results_df.at[0,f"{metric}_expenditure_readings"] = []
            
        elif metric == "nap":
            if today_total_nap_charge_recovery and len(today_total_nap_charge_recovery) > 0:
                nap_guidance_texts = []
                nap_guidance_expenditures = []
                for nap_recovery in today_total_nap_charge_recovery:
                    nap_recovery = round(nap_recovery, 0)
                    if nap_recovery > 1:
                        guidance_text = "Your rest replenished your BioCharge with {diff:.0f} points."
                    elif nap_recovery == 1:
                        guidance_text = "Your rest replenished your BioCharge with {diff:.0f} point."
                    elif nap_recovery == 0:
                        guidance_text = "Your rest did not replenish your BioCharge."                        
                    guidance_text = guidance_text.format(diff = nap_recovery)
                    nap_guidance_texts.append(guidance_text)
                    nap_guidance_expenditures.append(nap_recovery)
                
                guidance_results_df.at[0,f"{metric}_guidance_texts"] = nap_guidance_texts
                guidance_results_df.at[0,f"{metric}_replenishment_readings"] = nap_guidance_expenditures

            else:
                guidance_results_df.at[0,f"{metric}_guidance_texts"] = []
                guidance_results_df.at[0,f"{metric}_replenishment_readings"] = []

    return guidance_results_df


def convert_time_to_minutes(time_str):
    if isinstance(time_str, str):
        hours, minutes = map(int, time_str.split(":"))
        total_minutes = hours * 60 + minutes
        if total_minutes < 720:
            total_minutes += 1440
        return total_minutes
    else:
        return 0


def generate_guidance_for_one_day(today_index, df_result: pd.DataFrame, df_guidance: pd.DataFrame) -> pd.DataFrame:
    row = df_result.iloc[today_index]
    
    today_morning_charge = row.get('charge.final_value', np.nan)
    today_sleep_duration = row.get('sleep_duration', np.nan)
    today_sleep_duration = round(today_sleep_duration/60, 2) if pd.notna(today_sleep_duration) else np.nan
    today_deep_sleep_ratio = row.get('deep_sleep_ratio', np.nan)
    today_sleep_start_time = row.get('sleep_start_time', np.nan)
    today_sleep_start_time = convert_time_to_minutes(today_sleep_start_time)
    today_wakefulness_after_sleep_onset_frequency = row.get('wakefulness_after_sleep_onset_frequency', np.nan)
    today_wakefulness_after_sleep_onset_duration = row.get('wakefulness_after_sleep_onset_duration', np.nan)

    today_hrv = row.get('sleep_hrv', np.nan)
    today_rhr = row.get('sleep_rhr', np.nan)

    today_exertion = row.get('exertion_score', np.nan)
    yesterday_exertion = np.nan if today_index == 0 else df_result.loc[today_index-1, "exertion_score"]
    yesterday_exercise_stress = np.nan if today_index == 0 else df_result.loc[today_index-1, "daily_stress_accumulation"]
    tow_days_ago_exercise_stress = np.nan if today_index < 2 else df_result.loc[today_index-2, "daily_stress_accumulation"]
    three_days_ago_exercise_stress = np.nan if today_index < 3 else df_result.loc[today_index-3, "daily_stress_accumulation"]

    yesterday_evening_charge = np.nan if today_index == 0 else df_result.loc[today_index-1, "charge.value_at_2100"]
    today_evening_charge = row.get('charge.value_at_2100', np.nan)

    today_total_nap_charge_recovery = row.get('event.nap_total_charge_recovery', np.nan)
    try:
        today_total_nap_charge_recovery = ast.literal_eval(today_total_nap_charge_recovery) if isinstance(today_total_nap_charge_recovery, str) else today_total_nap_charge_recovery
    except (ValueError, SyntaxError):
        today_total_nap_charge_recovery = [] # or np.nan
    today_total_nap_charge_recovery_value = sum(today_total_nap_charge_recovery) if isinstance(today_total_nap_charge_recovery, list) else today_total_nap_charge_recovery
    
    today_total_exercise_charge_expenditure = row.get('event.exercise_total_charge_expenditure', np.nan)
    try:
        today_total_exercise_charge_expenditure = ast.literal_eval(today_total_exercise_charge_expenditure) if isinstance(today_total_exercise_charge_expenditure, str) else today_total_exercise_charge_expenditure
    except (ValueError, SyntaxError):
        today_total_exercise_charge_expenditure = [] # or np.nan
    today_total_exercise_charge_expenditure_value = sum(today_total_exercise_charge_expenditure) if isinstance(today_total_exercise_charge_expenditure, list) else today_total_exercise_charge_expenditure
    
    today_nap_recovery_rate = row.get('event.nap_charge_recovery_rate', np.nan)
    today_exercise_expenditure_rate = row.get('event.exercise_charge_expenditure_rate', np.nan)

    readings_map_for_today = {
        'morning Charge_sleep duration_Today': today_sleep_duration,
        'morning Charge_deep sleep ratio_Today': today_deep_sleep_ratio,
        'morning Charge_sleep start time_Today': today_sleep_start_time,
        'morning Charge_wakefulness after sleep onset (WASO) frequency_Today': today_wakefulness_after_sleep_onset_frequency,
        'morning Charge_wakefulness after sleep onset (WASO) duration_Today': today_wakefulness_after_sleep_onset_duration,
        'morning Charge_resting heart rate (RHR)_Today': today_rhr,
        'morning Charge_heart rate variability (HRV)_Today': today_hrv,
        'morning Charge_exertion score_Yesterday': yesterday_exertion,
        'morning Charge_daily exercise stress_Yesterday': yesterday_exercise_stress,
        'morning Charge_daily exercise stress_Two days ago': tow_days_ago_exercise_stress,
        'morning Charge_daily exercise stress_Three days ago': three_days_ago_exercise_stress,
        'morning Charge_evening Charge_Yesterday': yesterday_evening_charge,

        'evening Charge_morning Charge_Today': today_morning_charge,
        'evening Charge_exertion score_Today': today_exertion,
        'evening Charge_rest Charge replenishment_Today': today_total_nap_charge_recovery_value,
        'evening Charge_rest Charge rate of change_Today': today_nap_recovery_rate,
        'evening Charge_exercise Charge expenditure_Today': today_total_exercise_charge_expenditure_value,
        'evening Charge_exercise Charge rate of change_Today': today_exercise_expenditure_rate,
        'evening Charge_daily exercise stress_Yesterday': yesterday_exercise_stress,
        'evening Charge_daily exercise stress_Two days ago': tow_days_ago_exercise_stress,
        'evening Charge_daily exercise stress_Three days ago': three_days_ago_exercise_stress,
    }

    final_metrics_df = generate_charge_metrics_df(
        df_guidance,
        readings_map_for_today,
        today_morning_charge,
        today_evening_charge
    )

    yesterday_factors = {
        "morning": "" if today_index == 0 else df_result.get("guidance.morning_factor", {}).get(today_index-1, ""),
        "evening": "" if today_index == 0 else df_result.get("guidance.evening_factor", {}).get(today_index-1, "")
    }

    
    guidance_results = process_one_day_insights(final_metrics_df, df_guidance, today_total_nap_charge_recovery, today_total_exercise_charge_expenditure,
                                                yesterday_factors)
    
    return guidance_results

if __name__ == "__main__":
    userid = 1017856546
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(current_dir)
    result_path = os.path.join(main_dir, 'data', 'hrv_users', 'result_v3_2_optimized', f'{userid}_processed.xlsx')
    df_result = pd.read_excel(result_path)
    
    df_guidance = pd.read_excel(os.path.join(current_dir, 'guidance_online.xlsx'))

    guidance_column_names = [
        "morning_metric_reading", "morning_metric_level", "morning_factor", "morning_factor_reading", "morning_factor_level", "morning_factor_zscore", "morning_all_factors_zscores", "morning_guidance_text",
        "evening_metric_reading", "evening_metric_level", "evening_factor", "evening_factor_reading", "evening_factor_level", "evening_factor_zscore", "evening_all_factors_zscores", "evening_guidance_text",
        "nap_replenishment_readings", "nap_guidance_texts",
        "exercise_expenditure_readings", "exercise_guidance_texts"]
    guidance_rows = pd.DataFrame(columns=guidance_column_names)
    for idx, row in df_result.iterrows():
        guidance_results = generate_guidance_for_one_day(idx, df_result, df_guidance)
        if isinstance(guidance_results, pd.DataFrame):
            if guidance_rows.empty:
                guidance_rows = guidance_results.copy()
            elif not guidance_rows.empty:
                guidance_rows = pd.concat([guidance_rows, guidance_results], ignore_index=True)
    print(guidance_rows)
