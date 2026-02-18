"""
Dataset Delta V2 - Enhanced Biocharge Dataset Implementation

This file is the second iteration of the delta dataset implementation, building upon
dataset_delta.py and dataset.py with key improvements and modifications.

KEY DIFFERENCES FROM dataset.py:
1. **Data Sampling Strategy**:
   - dataset.py: Samples unique (userid, date) combinations for data fraction
   - dataset_delta_v2.py: Samples unique userids only (simpler approach)
   
2. **Configuration Parameters**:
   - Adds 'include_current_hr' parameter for better HR data control
   - Enhanced HR filtering and processing capabilities
   
3. **Data Processing**:
   - Improved user data caching mechanisms
   - Enhanced sleep start index caching
   - Better data length validation caching

KEY DIFFERENCES FROM dataset_delta.py:
1. **User Sampling Logic**:
   - Both use the same userid-only sampling approach (not userid+date pairs)
   - Similar caching strategies and performance optimizations
   
2. **Feature Processing**:
   - Enhanced HR data handling with include_current_hr parameter
   - Improved error handling and validation
   
3. **Code Organization**:
   - Better structured caching mechanisms
   - More robust data validation processes

MAIN FEATURES:
- Biocharge prediction dataset with time series physiological data
- Support for Z-score normalization and data augmentation
- Flexible feature selection (HR, acceleration, charge history, etc.)
- Windowed features and positional encoding support
- Sleep recharge pattern modeling
- LLM integration for biocharge column selection
- Comprehensive data caching for performance optimization

This version represents the most current and refined implementation of the delta
dataset approach, incorporating lessons learned from previous versions while
maintaining backward compatibility for most use cases.
"""

import os
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset, default_collate
import json
from src.training.utils import parse_series, safe_get_series, COLUMN_ALIASES
import ast

import numpy as np

class DatasetConfig:
    def __init__(
        self,
        data_dir: str,
        index_csv: str,
        zscores_file: Optional[str],
        charge_col: str,
        static_cols: List[str],
        sleep_cols: List[str],
        ts_cols: List[str],
        use_zscores: bool = True,
        augment_prob: float = 0.3,
        enable_augmentation: bool = False,
        trajectory: bool = False,
        use_acceleration: bool = True,
        use_past_charge: bool = True,
        use_hr : bool = True,
        use_positional_encoding: bool = False,
        pos_encoding_type: str = "biocharge_circadian",  # "time_of_day" or "sinusoidal"
        pos_encoding_dim: int = 2,
        use_windowed_features: bool = False,
        window_size_minutes: int = 15,
        include_gradients: bool = True,
        use_windowed_zscore: bool = False,
        include_current_hr: bool = True,
        z_data_norm='population',
        charge_col_llm: Optional[str] = None,
        llm_col_prob: float = 0.25, 
        llm_non_existent_userids_file: Optional[str] = None,
        data_dir_llm: Optional[str] = None, 
        charge_reconstruction: bool = False, 
        generate_trajectory: bool = False,
        drop_hr: bool = False, 
        add_hr_hrr: bool = False, 
        use_recovery_rate_feature: bool = True, 
        weighted_sleep_score: bool = True, 
        normalization_type = "znorm"
    ):
        self.data_dir = data_dir
        self.index_csv = index_csv
        self.zscores_file = zscores_file
        self.charge_col = charge_col
        self.static_cols = static_cols
        self.sleep_cols = sleep_cols
        self.ts_cols = ts_cols
        self.use_zscores = use_zscores
        self.augment_prob = augment_prob
        self.enable_augmentation = enable_augmentation
        self.trajectory  = trajectory
        self.use_past_charge = use_past_charge
        self.use_acceleration = use_acceleration
        self.use_hr = use_hr
        self.use_positional_encoding = use_positional_encoding
        self.pos_encoding_type = pos_encoding_type
        self.pos_encoding_dim = pos_encoding_dim
        self.use_windowed_features = use_windowed_features
        self.window_size_minutes = window_size_minutes
        self.include_gradients = include_gradients
        self.use_windowed_zscore = use_windowed_zscore
        self.include_current_hr = include_current_hr
        self.z_data_norm = z_data_norm
        self.charge_col_llm = charge_col_llm
        self.llm_col_prob = llm_col_prob
        self.llm_non_existent_userids_file = llm_non_existent_userids_file
        self.data_dir_llm = data_dir_llm
        self.charge_reconstruction = charge_reconstruction
        self.generate_trajectory = generate_trajectory
        self.drop_hr = drop_hr, 
        self.add_hr_hrr = add_hr_hrr,
        self.use_recovery_rate_feature = use_recovery_rate_feature,
        self.weighted_sleep_score = weighted_sleep_score, 
        self.normalization_type = normalization_type



class BiochargeDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, data_fraction: float = 1.0):
        super().__init__()
        self.cfg = cfg

        
        # Load index (contains [user_id, date, idx, ...])
        self.index = pd.read_csv(cfg.index_csv)
        # remove all indices start with 0, (off when plotting traejctory)
        # if not cfg.trajectory:
        #     self.index = self.index[self.index['index'] > 0]

        # Subsample data if requested (sample users, not indices, to keep user data together)
        if data_fraction < 1.0:
            # randomly sample a fraction of rows
            n_samples = int(len(self.index) * data_fraction)
            index_sub = self.index.sample(n=n_samples, random_state=42).reset_index(drop=True)
            self.index = index_sub

        self.sub_index = self.index.copy()
        # find all
        self.index["date"] = pd.to_datetime(self.index["date"])

        # Optional zscores
        self.zdf = None
        if cfg.zscores_file and cfg.use_zscores:
            with open(cfg.zscores_file, "r") as f:
                self.zdf = json.load(f)

        # Cache for user files (don't reload each time)
        self.user_cache: Dict[str, pd.DataFrame] = {}
        
        # Cache for processed dates (to avoid re-filtering)
        self.processed_dates_cache: Dict[str, set] = {}
        
        self.add_hr_hrr = cfg.add_hr_hrr
        
        # Cache for sleep start indices
        self.sleep_start_cache: Dict[str, Dict[str, int]] = {}
        
        # Cache for data length validation to avoid repeated warnings
        self.data_length_cache: Dict[str, Dict[str, int]] = {}

        if cfg.llm_non_existent_userids_file is not None:
            with open(cfg.llm_non_existent_userids_file, "r") as f:
                self.no_exist_userids = set(json.load(f).keys())

        # Select which (userid, date) pairs should use LLM biocharge column
        # make sure these are not in no exist userids 
        self.llm_charge_pairs = set()
        if cfg.charge_col_llm is not None and cfg.llm_col_prob > 0:
            # Get unique (userid, date) combinations

            # make sure to exclude no_exist_userids
            unique_curves = self.index[['userid', 'date']].drop_duplicates()
            unique_curves = unique_curves[~unique_curves['userid'].astype(str).isin(self.no_exist_userids)]
            # Randomly sample llm_col_prob fraction of curves

            n_llm_curves = int(len(unique_curves) * cfg.llm_col_prob)
            if n_llm_curves > 0:
                sampled_curves = unique_curves.sample(n=n_llm_curves, random_state=42)
                # Store as set of (userid, date_string) tuples for fast lookup
                self.llm_charge_pairs = set(
                    (str(row['userid']), row['date'].strftime('%Y-%m-%d'))
                    for _, row in sampled_curves.iterrows()
                )
                print(f"Selected {len(self.llm_charge_pairs)} curves ({cfg.llm_col_prob*100:.1f}%) to use LLM biocharge column")

        # Normalization
        self.mu = None
        self.sd = None

        # Augmentation settings from config
        self.augment = cfg.enable_augmentation
        self.augment_prob = cfg.augment_prob
        
        self.use_windowed_zscore = cfg.use_windowed_zscore

        # Define indices for different time series types (will be set during first __getitem__)
        self.ts_feature_indices = None
        self.hr_indices = []
        self.hrr_indices = []
        self.stress_indices = []
        self.exercise_indices = []
        self.activity_indices = []
        self.binary_sequence_indices = []  # For exercise, nap, sleep markers, sleep_stage
        self.continuous_signal_indices = []  # For HR, stress, accelerometer
        self.jitter_indices = []  # For temporal jittering (excludes binary sequences)
        self.sleep_stage_indices = []  # For sleep stage specific handling
        self.z_data_norm = cfg.z_data_norm

    def get_indices(self):
        return self.sub_index
    
    def _get_hr_column_name(self, df_row=None):
        """Determine which HR column to use: prefer hr_filtered if available."""
        # Check if hr_filtered column exists in the dataframe
        if df_row is not None and hasattr(df_row, 'columns'):
            if 'timeseries.hrr_raw' in df_row.columns:
                return 'timeseries.hrr_raw'
        
        # Check if hr_filtered column exists in z-score file
        if self.zdf is not None:
            if 'global' in self.zdf and 'timeseries.hr_filtered' in self.zdf['global']:
                return 'timeseries.hr_filtered'
        
        # Default fallback to regular hr column
        return 'timeseries.hr'
    
    def filter_hr_data(self, user_id: str, date_str: str):
        """
        Filter HR data by applying Hampel-like filtering only when values are outside 
        physiologically reasonable range (30-220 bpm).
        
        Args:
            user_id: User identifier
            date_str: Date string in format "YYYY-MM-DD"
        """
        if user_id not in self.user_cache:
            return
            
        df = self.user_cache[user_id]
        df_row = df[df["date"] == date_str]
        
        if df_row.empty:
            return
            
        try:
            # Get HR data - use hr_filtered if available, otherwise timeseries.hr
            hr_column = self._get_hr_column_name(df_row)
            column_series_str = df_row[hr_column].values[0]
            column_values = self.safe_literal_eval(column_series_str)
            
            if not column_values or len(column_values) == 0:
                return
                
            # Convert to tensor
            hr_tensor = torch.tensor(column_values, dtype=torch.float32)
            
            # Create mask for values outside physiological range (30-220 bpm)
            outlier_mask = (hr_tensor < 30) | (hr_tensor > 220)
            
            if not outlier_mask.any():
                # No outliers, no filtering needed
                return
                
            # Apply Hampel filter to get replacement values for outliers
            filtered_values, _ = self.hampel_filter_torch(hr_tensor, window_size=7, n_sigmas=3.0)
            
            # Only replace values that are outside physiological range
            corrected_values = hr_tensor.clone()
            corrected_values[outlier_mask] = filtered_values[outlier_mask]
            
            # Update the cached dataframe
            df_row_index = df_row.index[0]
            hr_column = self._get_hr_column_name(df_row)
            self.user_cache[user_id].at[df_row_index, hr_column] = corrected_values.numpy().tolist()
            
        except Exception as e:
            print(f"Warning: Failed to filter HR data for user {user_id} on {date_str}: {e}")


    def hampel_filter_torch(self, x: torch.Tensor, window_size: int = 7, n_sigmas: float = 3.0):
        """
        Hampel filter for outlier removal in 1D torch tensors.
        Args:
            x: 1D torch.Tensor
            window_size: int, must be odd
            n_sigmas: threshold in number of standard deviations (MAD based)
        Returns:
            filtered_x: torch.Tensor (same shape)
            mask: torch.BoolTensor indicating where values were replaced
        """
        # Ensure input is float tensor
        if x.dtype != torch.float32:
            x = x.float()
            
        if x.ndim != 1:
            raise ValueError("Input must be a 1D tensor")

        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        
        # Handle empty or very small tensors
        if x.shape[0] == 0:
            return x, torch.zeros_like(x, dtype=torch.bool)
        
        if x.shape[0] < window_size:
            # Return original tensor if it's smaller than window size
            return x, torch.zeros_like(x, dtype=torch.bool)

        k = window_size // 2
        n = x.shape[0]

        # Create unfolding windows [N, window_size]
        # Pad at both ends to handle edges
        padded = torch.nn.functional.pad(x.unsqueeze(0).unsqueeze(0), (k, k), mode='reflect')
        windows = torch.nn.functional.unfold(
            padded,
            kernel_size=(1, window_size)
        ).squeeze(0).transpose(0, 1)  # shape: (N, window_size)

        # Median and MAD (Median Absolute Deviation)
        med = windows.median(dim=1).values
        abs_dev = (windows - med.unsqueeze(1)).abs()
        mad = abs_dev.median(dim=1).values

        # Hampel threshold
        threshold = n_sigmas * 1.4826 * mad
        diff = (x - med).abs()

        # Identify outliers
        mask = diff > threshold
        filtered_x = x.clone()
        filtered_x[mask] = med[mask]

        return filtered_x, mask

    def time_encoding_for_charge(self, minute_of_day):
        """
        Time encoding aligned with biocharge circadian model.
        minute_of_day: int or float in [0, 1440)
        """
        minute_of_day = minute_of_day % 1440  # safety wrap

        # Normalized fraction of day
        fraction = minute_of_day / 1440.0

        # Standard cyclic encoding
        sin_24h = np.sin(2 * np.pi * fraction)
        cos_24h = np.cos(2 * np.pi * fraction)

        # Circadian model (minute-based)
        circadian_output = (
            np.cos(2 * np.pi * (minute_of_day - 1080) / 1440) +
            0.5 * np.cos(2 * np.pi * (minute_of_day - 1260) / 720)
        )

        return np.array(
            [sin_24h, cos_24h, circadian_output],
            dtype=np.float32
        )

    def time_encoding_for_charge_torch(self, minute_of_day):
        """
        Torch version of time encoding aligned with biocharge circadian model.
        minute_of_day: int, float, or torch.Tensor in [0, 1440)
        Returns: torch.FloatTensor of shape (3,)
        """
        import torch
        pi = torch.tensor(np.pi) if not hasattr(torch, 'pi') else torch.pi
        if not torch.is_tensor(minute_of_day):
            minute_of_day = torch.tensor(minute_of_day, dtype=torch.float32)
        minute_of_day = minute_of_day % 1440
        fraction = minute_of_day / 1440.0
        sin_24h = torch.sin(2 * pi * fraction)
        cos_24h = torch.cos(2 * pi * fraction)
        circadian_output = (
            torch.cos(2 * pi * (minute_of_day - 1080) / 1440) +
            0.5 * torch.cos(2 * pi * (minute_of_day - 1260) / 720)
        )
        return torch.stack([sin_24h, cos_24h, circadian_output]).to(torch.float32)

    def time_of_day_encoding_continuous(self, minute_of_day):
        """
        Continuous time-of-day encoding using sine and cosine.

        Args:
            minute_of_day: int or float in [0, 1440)
        Returns:
            np.array: shape (2,) [sin(2π * t), cos(2π * t)]

            # continuous 
        """
        # Convert minutes to fraction of the day
        fraction = (minute_of_day % 1440) / 1440.0  # ensures wrap-around safety

        # Continuous cyclic encoding
        angle = 2 * np.pi * fraction
        encoding = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
        return encoding

    def time_of_day_encoding(self, minute_of_day, num_buckets=8):
        """
        Encode time of day into num_buckets (e.g., 8 buckets for 3-hour intervals).
        
        Args:
            minute_of_day: int in [0, 1440)
            num_buckets: number of intervals per day (8 for 3-hour buckets)
        
        Returns:
            np.array: encoding vector of shape (2,)  [sin, cos]
        """
        # Map minute of day to bucket
        minutes_per_bucket = 1440 // num_buckets
        bucket_idx = minute_of_day // minutes_per_bucket
        
        # Normalize bucket index to [0, 2π)
        angle = 2 * np.pi * (bucket_idx / num_buckets)
        
        # Cyclic encoding
        encoding = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
        return  encoding.astype(np.float32)

    def positional_encoding(self, time_idx, d_model=64):
        """
        Generate positional encoding for a specific time index.
        
        Args:
            time_idx: Time index in range [0, 1440] (minutes in a day)
            d_model: Dimension of the positional encoding vector
            
        Returns:
            np.array: Positional encoding vector of shape (d_model,)

        Note: This can be changed to the week of the day as well

        """
        # Normalize time_idx to [0, 1] range
        normalized_position = time_idx / 1440.0
        
        # Create positional encoding vector
        pe = np.zeros(d_model)
        
        # Generate sinusoidal encodings
        for i in range(0, d_model, 2):
            # Use different frequencies for different dimensions
            div_term = np.exp(i * -(np.log(10000.0) / d_model))
            
            # Apply sine to even indices
            pe[i] = np.sin(normalized_position * div_term)
            
            # Apply cosine to odd indices (if within bounds)
            if i + 1 < d_model:
                pe[i + 1] = np.cos(normalized_position * div_term)
        
        return pe.astype(np.float32)
    
    def _read_windowed_features(self, row: pd.Series, idx: int):
        """
        Read pre-calculated windowed features from the data file for a given row and time index.
        
        Key improvements:
        - Automatically selects correct columns based on window_size_minutes (15min vs 30min)
        - Supports include_current_hr flag to use current HR or previous HR (t-1)
        - Robust error handling with debugging information
        - Returns z-score normalized features from pre-calculated data
        
        Args:
            row: Current data row
            idx: Current time index
            
        Returns:
            dict: Dictionary containing windowed features read from file
                - hr_window: Single HR value (current or t-1 based on config)
                - acc_window: Single accelerometer value
                - hr_gradient: HR gradient (if include_gradients=True)
                - acc_gradient: Accelerometer gradient (if include_gradients=True)
        """
        features = {
            'hr_window': [],
            'acc_window': [],
            'hr_gradient': [],
            'acc_gradient': []
        }
        
        # Determine window size to use (15 or 30 minutes)
        window_size = self.cfg.window_size_minutes
        
        # Determine base data columns (needed for fallback)
        base_hr_col = self._get_hr_column_name(row)
        hr_data_column = base_hr_col
        acc_data_column = 'timeseries.acc_magnitude'
        hr_grad_data_column = 'hr_gradient_5min'
        acc_grad_data_column = 'acc_gradient_5min'

        # Choose appropriate column names based on window size
        if self.use_windowed_zscore:
            if window_size == 15:
                z_norm_hr_col = 'z_norm_hr_15'
                z_norm_acc_col = 'z_norm_acc_15'
                hr_grad_col = 'hr_grad_zscore_15min'
                acc_grad_col = 'acc_grad_zscore_15min'
            else:  # Default to 30 minutes or any other size
                z_norm_hr_col = 'z_norm_hr_30'
                z_norm_acc_col = 'z_norm_acc_30'
                hr_grad_col = 'hr_grad_zscore_30min'
                acc_grad_col = 'acc_grad_zscore_30min'
        else:
            # global normalized data columns - use dynamic HR column determination
            z_norm_hr_col = f'{base_hr_col}_zscore'
            z_norm_acc_col = 'timeseries.acc_zscore'
            hr_grad_col = 'hr_gradient_5min_zscore'
            acc_grad_col = 'acc_grad_zscore_5min'
            

        
        try:
            # Read z-score normalized HR window
            if z_norm_hr_col in row:
                hr_data = row.get(z_norm_hr_col, None)
                if hr_data is not None:
                    hr_values = self.safe_literal_eval(hr_data.values[0], default_value=[], column_name=z_norm_hr_col)
                    if isinstance(hr_values, list) and len(hr_values) > 0:
                        # Enhanced: Handle cases where idx >= len(hr_values)
                        effective_idx = min(idx, len(hr_values) - 1)
                        
                        if self.cfg.include_current_hr:
                            # Extract values from past window 15 or 30 min
                            start_idx = max(0, effective_idx - window_size)
                            end_idx = effective_idx
                            past_window = hr_values[start_idx:end_idx + 1]  # Include current
                            
                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with the first available value if we don't have enough history
                                padding_value = past_window[0] if past_window else 0.0
                                padding_needed = window_size - len(past_window)
                                past_window = [padding_value] * padding_needed + past_window
                            
                            features['hr_window'] = past_window[:window_size]  # Ensure exact size
                        else:
                            # Extract previous time point (t-1) to exclude current HR
                            prev_idx = max(0, effective_idx - 1)
                            start_idx = max(0, prev_idx - window_size)
                            past_window = hr_values[start_idx:prev_idx]
                            
                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with the first available value if we don't have enough history
                                padding_value = past_window[0] if past_window else 0.0
                                padding_needed = window_size - len(past_window)
                                past_window = [padding_value] * padding_needed + past_window
                                
                            features['hr_window'] = past_window[:window_size]  # Ensure exact size
                    else:
                        features['hr_window'] = [0.0] * window_size

            else:
                # Fallback: apply z_score normalization on the fly if pre-calculated column not found
                print(f"Warning: Pre-calculated column '{z_norm_hr_col}' not found, using on-the-fly normalization from '{hr_data_column}'")
                hr_mean, hr_std = 0.0, 1.0
                if self.zdf and 'global' in self.zdf and hr_data_column in self.zdf['global']:
                    hr_mean, hr_std = self.zdf['global'][hr_data_column]['mean'], self.zdf['global'][hr_data_column]['std']
                if hr_data_column in row:

                    hr_data = row.get(hr_data_column, None)
                    if hr_data is not None:
                        hr_values = self.safe_literal_eval(hr_data.values[0], default_value=[], column_name=hr_data_column)
                        if isinstance(hr_values, list) and len(hr_values) > 0:
                            # Enhanced: Handle cases where idx >= len(hr_values)
                            effective_idx = min(idx, len(hr_values) - 1)
                            
                            if self.cfg.include_current_hr:
                                # Extract values from past window 15 or 30 min
                                start_idx = max(0, effective_idx - window_size)
                                end_idx = effective_idx
                                past_window = hr_values[start_idx:end_idx + 1]  # Include current
                                
                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with the first available value if we don't have enough history
                                    padding_value = past_window[0] if past_window else 0.0
                                    padding_needed = window_size - len(past_window)
                                    past_window = [padding_value] * padding_needed + past_window

                                # Normalize with Z_score
                                features['hr_window'] = [(x - hr_mean) / hr_std for x in past_window[:window_size]]

                            else:
                                # Extract previous time point (t-1) to exclude current HR
                                prev_idx = max(0, effective_idx - 1)
                                start_idx = max(0, prev_idx - window_size)
                                past_window = hr_values[start_idx:prev_idx]
                                
                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with the first available value if we don't have enough history
                                    padding_value = past_window[0] if past_window else 0.0
                                    padding_needed = window_size - len(past_window)
                                    past_window = [padding_value] * padding_needed + past_window
                                    
                                features['hr_window'] = [(x - hr_mean) / hr_std for x in past_window[:window_size]]  # Ensure exact size

            
            # Read z-score normalized accelerometer window
            
            if z_norm_acc_col in row:
                acc_data = row.get(z_norm_acc_col, None)
                if acc_data is not None:
                    acc_values = self.safe_literal_eval(acc_data.values[0], default_value=[], column_name=z_norm_acc_col)
                    if isinstance(acc_values, list) and len(acc_values) > 0:
                        # Enhanced: Handle cases where idx >= len(acc_values)
                        effective_idx = min(idx, len(acc_values) - 1)

                        if self.cfg.include_current_hr:
                            # Extract values from past window including current
                            start_idx = max(0, effective_idx - window_size)
                            end_idx = effective_idx
                            past_window = acc_values[start_idx:end_idx + 1]  # Include current

                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with the first available value if we don't have enough history
                                padding_value = past_window[0] if past_window else 0.0
                                padding_needed = window_size - len(past_window)
                                past_window = [padding_value] * padding_needed + past_window

                            features['acc_window'] = past_window[:window_size]  # Ensure exact size
                        else:
                            # Extract previous time point (t-1) to exclude current ACC
                            prev_idx = max(0, effective_idx - 1)
                            start_idx = max(0, prev_idx - window_size)
                            past_window = acc_values[start_idx:prev_idx]

                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with the first available value if we don't have enough history
                                padding_value = past_window[0] if past_window else 0.0
                                padding_needed = window_size - len(past_window)
                                past_window = [padding_value] * padding_needed + past_window

                            features['acc_window'] = past_window[:window_size]  # Ensure exact size
                    else:
                        features['acc_window'] = [0.0] * window_size
            
            else:
                acc_mean, acc_std = 0.0, 1.0
                if self.zdf and 'global' in self.zdf and acc_data_column in self.zdf['global']:
                    acc_mean, acc_std = self.zdf['global'][acc_data_column]['mean'], self.zdf['global'][acc_data_column]['std']
                
                if acc_data_column in row:
                    acc_data = row.get(acc_data_column, None)
                    if acc_data is not None:
                        acc_values = self.safe_literal_eval(acc_data.values[0], default_value=[], column_name=acc_data_column)
                        if isinstance(acc_values, list) and len(acc_values) > 0:
                            # Enhanced: Handle cases where idx >= len(acc_values)
                            effective_idx = min(idx, len(acc_values) - 1)

                            if self.cfg.include_current_hr:
                                # Extract values from past window including current
                                start_idx = max(0, effective_idx - window_size)
                                end_idx = effective_idx
                                past_window = acc_values[start_idx:end_idx + 1]  # Include current

                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with the first available value if we don't have enough history
                                    padding_value = past_window[0] if past_window else 0.0
                                    padding_needed = window_size - len(past_window)
                                    past_window = [padding_value] * padding_needed + past_window

                                # Normalize with Z_score
                                features['acc_window'] = [(x - acc_mean) / acc_std for x in past_window[:window_size]]  # Ensure exact size
                            else:
                                # Extract previous time point (t-1) to exclude current ACC
                                prev_idx = max(0, effective_idx - 1)
                                start_idx = max(0, prev_idx - window_size)
                                past_window = acc_values[start_idx:prev_idx]

                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with the first available value if we don't have enough history
                                    padding_value = past_window[0] if past_window else 0.0
                                    padding_needed = window_size - len(past_window)
                                    past_window = [padding_value] * padding_needed + past_window

                                # Normalize with Z_score
                                features['acc_window'] = [(x - acc_mean) / acc_std for x in past_window[:window_size]]  # Ensure exact size
                        else:
                            features['acc_window'] = [0.0] * window_size
            # Read gradient features if enabled
            if self.cfg.include_gradients:
                # Read HR gradient
                if hr_grad_col in row:
                    hr_grad_data = row.get(hr_grad_col, None)
                    if hr_grad_data is not None:
                        hr_grad_values = self.safe_literal_eval(hr_grad_data.values[0], default_value=[], column_name=hr_grad_col)
                        if isinstance(hr_grad_values, list) and len(hr_grad_values) > 0:
                            # Enhanced: Handle cases where idx >= len(hr_grad_values)
                            effective_idx = min(idx, len(hr_grad_values) - 1)
                            start_idx = max(0, effective_idx - window_size)
                            past_window = hr_grad_values[start_idx:effective_idx]
                            
                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with zero for gradients (neutral gradient)
                                padding_needed = window_size - len(past_window)
                                past_window = [0.0] * padding_needed + past_window
                                
                            features['hr_gradient'] = past_window[:window_size]  # Ensure exact size
                        else:
                            features['hr_gradient'] = [0.0] * window_size
                else:
                    hr_grad_mean, hr_grad_std = 0.0, 1.0
                    if hr_grad_data_column in self.zdf['global']:
                        hr_grad_mean, hr_grad_std = self.zdf['global'][hr_grad_data_column]['mean'], self.zdf['global'][hr_grad_data_column]['std']
                    if hr_grad_data_column in row:
                        hr_grad_data = row.get(hr_grad_data_column, None)
                        if hr_grad_data is not None:
                            hr_grad_values = self.safe_literal_eval(hr_grad_data.values[0], default_value=[], column_name=hr_grad_data_column)
                            if isinstance(hr_grad_values, list) and len(hr_grad_values) > 0:
                                # Enhanced: Handle cases where idx >= len(hr_grad_values)
                                effective_idx = min(idx, len(hr_grad_values) - 1)
                                start_idx = max(0, effective_idx - window_size)
                                past_window = [(x-hr_grad_mean)/hr_grad_std for x in hr_grad_values[start_idx:effective_idx]]
                                
                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with zero for gradients (neutral gradient)
                                    padding_needed = window_size - len(past_window)
                                    past_window = [0.0] * padding_needed + past_window
                                    
                                features['hr_gradient'] = [(x-hr_grad_mean)/hr_grad_std for x in past_window[:window_size]]  # Ensure exact size
                            else:
                                features['hr_gradient'] = [0.0] * window_size
                if acc_grad_col in row:
                    acc_grad_data = row.get(acc_grad_col, None)
                    if acc_grad_data is not None:
                        acc_grad_values = self.safe_literal_eval(acc_grad_data.values[0], default_value=[], column_name=acc_grad_col)
                        if isinstance(acc_grad_values, list) and len(acc_grad_values) > 0:
                            # Enhanced: Handle cases where idx >= len(acc_grad_values)
                            effective_idx = min(idx, len(acc_grad_values) - 1)
                            start_idx = max(0, effective_idx - window_size)
                            past_window = acc_grad_values[start_idx:effective_idx]
                            
                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with zero for gradients (neutral gradient)
                                padding_needed = window_size - len(past_window)
                                past_window = [0.0] * padding_needed + past_window
                                
                            features['acc_gradient'] = past_window[:window_size]  # Ensure exact size
                        else:
                            features['acc_gradient'] = [0.0] * window_size
                else:
                    acc_grad_mean, acc_grad_std = 0.0, 1.0
                    if acc_grad_data_column in self.zdf['global']:
                        acc_grad_mean, acc_grad_std = self.zdf['global'][acc_grad_data_column]['mean'], self.zdf['global'][acc_grad_data_column]['std']
                    if acc_grad_data_column in row:
                        acc_grad_data = row.get(acc_grad_data_column, None)
                        if acc_grad_data is not None:
                            acc_grad_values = self.safe_literal_eval(acc_grad_data.values[0], default_value=[], column_name=acc_grad_data_column)
                            if isinstance(acc_grad_values, list) and len(acc_grad_values) > 0:
                                # Enhanced: Handle cases where idx >= len(acc_grad_values)
                                effective_idx = min(idx, len(acc_grad_values) - 1)
                                start_idx = max(0, effective_idx - window_size)
                                past_window = [(x - acc_grad_mean)/acc_grad_std for x in acc_grad_values[start_idx:effective_idx]]
                                
                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with zero for gradients (neutral gradient)
                                    padding_needed = window_size - len(past_window)
                                    past_window = [0.0] * padding_needed + past_window
                                    
                                features['acc_gradient'] = [(x - acc_grad_mean)/acc_grad_std for x in past_window[:window_size]]  # Ensure exact size
                            else:
                                features['acc_gradient'] = [0.0] * window_size
        except Exception as e:
            print(f"Warning: Error reading windowed features for window_size={window_size}min: {e}")
            print(f"Available columns: {list(row.index)}")
            # Return default values on error
            features = {
                'hr_window': [0.0],
                'acc_window': [0.0], 
                'hr_gradient': [0.0],
                'acc_gradient': [0.0]
            }
        
        return features 

    def _extract_windowed_features(self, row: pd.Series, idx: int, window_size: int = 30):
        """
        Extract HR and accelerometer features from the past window_size minutes.
        
        Args:
            row: Current data row
            idx: Current time index
            window_size: Window size in minutes (default 30)
            
        Returns:
            dict: Dictionary containing windowed features and gradients
                - hr_window: Array of HR values over the window
                - acc_window: Array of accelerometer values over the window
                - hr_gradient: Linear trend in HR (change per minute)
                - acc_gradient: Linear trend in accelerometer (change per minute)
                - hr_mean, hr_std: HR statistics over the window
                - acc_mean, acc_std: Accelerometer statistics over the window
        """
        features = {
            'hr_window': [],
            'acc_window': [],
            'hr_gradient': 0.0,
            'acc_gradient': 0.0,
            'hr_mean': 0.0,
            'hr_std': 0.0,
            'acc_mean': 0.0,
            'acc_std': 0.0
        }
        
        # Determine start index for the window
        start_idx = max(0, idx - window_size)
        
        # Find HR and accelerometer columns
        hr_cols = [col for col in self.cfg.ts_cols if 'hr' in col.lower() and 'hrr' not in col.lower()]
        acc_cols = [col for col in self.cfg.ts_cols if 'acc' in col.lower()]
        
        # Extract HR window (vectorized)
        hr_values = []
        if hr_cols:
            for hr_col in hr_cols:
                hr_data = row.get(hr_col, None)
                if hr_data is not None:
                    # Vectorized extraction using list comprehension
                    hr_values = [self._safe_extract_value(hr_data, i, 0.0) 
                               for i in range(start_idx, idx + 1)]
                    break  # Use first available HR column
        
        # Extract accelerometer window (vectorized)
        acc_values = []
        if acc_cols:
            for acc_col in acc_cols:
                acc_data = row.get(acc_col, None)
                if acc_data is not None:
                    # Vectorized extraction using list comprehension
                    acc_values = [self._safe_extract_value(acc_data, i, 0.0) 
                                for i in range(start_idx, idx + 1)]
                    break  # Use first available accelerometer column
        
        # Pad windows if necessary
        target_length = window_size + 1  # +1 to include current time point
        if len(hr_values) < target_length:
            hr_values = [0.0] * (target_length - len(hr_values)) + hr_values
        if len(acc_values) < target_length:
            acc_values = [0.0] * (target_length - len(acc_values)) + acc_values
        
        # Store windowed data
        features['hr_window'] = np.array(hr_values[-target_length:], dtype=np.float32)
        features['acc_window'] = np.array(acc_values[-target_length:], dtype=np.float32)
        
        # Calculate statistics
        if len(features['hr_window']) > 0:
            features['hr_mean'] = np.mean(features['hr_window'])
            features['hr_std'] = np.std(features['hr_window']) if len(features['hr_window']) > 1 else 0.0
        
        if len(features['acc_window']) > 0:
            features['acc_mean'] = np.mean(features['acc_window'])
            features['acc_std'] = np.std(features['acc_window']) if len(features['acc_window']) > 1 else 0.0
        
        # Calculate gradients (slope of linear regression) - per minute rate of change
        if self.cfg.include_gradients and len(features['hr_window']) > 1:
            # Time points represent minutes from start of window to current time
            # Each index corresponds to 1 minute, so gradient will be per minute
            time_points_minutes = np.arange(len(features['hr_window']))  # [0, 1, 2, ..., window_size] minutes
            
            # HR gradient (rolling window linear regression slope)
            try:
                # Use rolling window approach with convolution for efficient gradient calculation
                hr_tensor = torch.from_numpy(features['hr_window']).float()
                gradient_window = 5  # 5-minute rolling window for gradient calculation
                
                if len(hr_tensor) >= gradient_window:
                    # Precompute constants for linear regression
                    x = torch.arange(gradient_window, dtype=hr_tensor.dtype)
                    x_sum = x.sum()
                    x2_sum = (x ** 2).sum()
                    denominator = gradient_window * x2_sum - x_sum ** 2
                    
                    if denominator != 0:
                        # Reshape for conv1d (batch, channels, length)
                        hr_reshaped = hr_tensor.view(1, 1, -1)
                        
                        # Compute y_sum using conv1d with ones kernel
                        kernel_ones = torch.ones(1, 1, gradient_window, dtype=hr_tensor.dtype)
                        y_sum = torch.nn.functional.conv1d(hr_reshaped, kernel_ones).squeeze()
                        
                        # Compute xy_sum using conv1d with reversed x kernel
                        kernel_x = x.flip(0).view(1, 1, -1)
                        xy_sum = torch.nn.functional.conv1d(hr_reshaped, kernel_x).squeeze()
                        
                        # Ensure tensors are at least 1D for proper indexing
                        if y_sum.dim() == 0:
                            y_sum = y_sum.unsqueeze(0)
                        if xy_sum.dim() == 0:
                            xy_sum = xy_sum.unsqueeze(0)
                        
                        # Calculate gradients for all windows
                        gradients = (gradient_window * xy_sum - x_sum * y_sum) / denominator
                        
                        # Store all gradients as a list/array for feature extraction
                        features['hr_gradient'] = gradients.detach().cpu().numpy() if gradients.dim() > 0 else np.array([0.0])
                    else:
                        features['hr_gradient'] = np.array([0.0])
                else:
                    # Fallback for short windows
                    features['hr_gradient'] = np.array([0.0])
                    
            except (RuntimeError, ValueError, IndexError):
                features['hr_gradient'] = np.array([0.0])
            
            # Accelerometer gradient (rolling window linear regression slope)
            try:
                # Use rolling window approach with convolution for efficient gradient calculation
                acc_tensor = torch.from_numpy(features['acc_window']).float()
                gradient_window = 5  # 5-minute rolling window for gradient calculation
                
                if len(acc_tensor) >= gradient_window:
                    # Precompute constants for linear regression
                    x = torch.arange(gradient_window, dtype=acc_tensor.dtype)
                    x_sum = x.sum()
                    x2_sum = (x ** 2).sum()
                    denominator = gradient_window * x2_sum - x_sum ** 2
                    
                    if denominator != 0:
                        # Reshape for conv1d (batch, channels, length)
                        acc_reshaped = acc_tensor.view(1, 1, -1)
                        
                        # Compute y_sum using conv1d with ones kernel
                        kernel_ones = torch.ones(1, 1, gradient_window, dtype=acc_tensor.dtype)
                        y_sum = torch.nn.functional.conv1d(acc_reshaped, kernel_ones).squeeze()
                        
                        # Compute xy_sum using conv1d with reversed x kernel
                        kernel_x = x.flip(0).view(1, 1, -1)
                        xy_sum = torch.nn.functional.conv1d(acc_reshaped, kernel_x).squeeze()
                        
                        # Ensure tensors are at least 1D for proper indexing
                        if y_sum.dim() == 0:
                            y_sum = y_sum.unsqueeze(0)
                        if xy_sum.dim() == 0:
                            xy_sum = xy_sum.unsqueeze(0)
                        
                        # Calculate gradients for all windows
                        gradients = (gradient_window * xy_sum - x_sum * y_sum) / denominator
                        
                        # Store all gradients as a list/array for feature extraction
                        features['acc_gradient'] = gradients.detach().cpu().numpy() if gradients.dim() > 0 else np.array([0.0])
                    else:
                        features['acc_gradient'] = np.array([0.0])
                else:
                    # Fallback for short windows
                    features['acc_gradient'] = np.array([0.0])
                    
            except (RuntimeError, ValueError, IndexError):
                features['acc_gradient'] = np.array([0.0])
        
        return features
    

    def _apply_windowed_zscore(self, window_data: np.ndarray, epsilon: float = 1e-8):
        """
        Apply z-score normalization to windowed data using the window's own statistics.
        
        Args:
            window_data: Array of windowed values
            epsilon: Small value to prevent division by zero
            
        Returns:
            np.ndarray: Z-score normalized window data
        """
        if len(window_data) <= 1:
            return window_data
        
        window_mean = np.mean(window_data)
        window_std = np.std(window_data)
        
        if window_std < epsilon:
            return np.zeros_like(window_data)
        
        return (window_data - window_mean) / window_std
    
    def safe_literal_eval(self, data_string, default_value=None, column_name="unknown"):
        """
        Safely evaluate a string containing a Python literal expression.
        
        Args:
            data_string: String to evaluate
            default_value: Value to return if parsing fails (default: empty list)
            column_name: Name of the column being parsed (for error logging)
        
        Returns:
            Parsed data or default_value if parsing fails
        """
        if default_value is None:
            default_value = []
        
        try:
            # Handle NaN or None values
            if pd.isna(data_string) or data_string is None:
                # print(f"Warning: NaN or None value found in column '{column_name}', using default value")
                return default_value
            
            # Convert to string if not already
            if not isinstance(data_string, str):
                data_string = str(data_string)
            
            # Try to parse the string
            return ast.literal_eval(data_string)
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Warning: Failed to parse data in column '{column_name}': {e}. Using default value.")
            return default_value
    
    def get_feature_count(self):
        """
        Calculate the total number of features that will be generated.
        Useful for model initialization.

        IMPORTANT: Order must match _get_feature_vector exactly:
        1. Static + Sleep features
        2. Timeseries features
        3. Positional encoding (if enabled)
        4. Windowed features (if enabled)
        5. Past charge (if enabled)

        Returns:
            int: Total feature count
        """
        # 1. Static + Sleep + Timeseries
        # Account for BMI replacement: if both height and weight are present,
        # they are replaced by a single BMI feature in _get_feature_vector
        static_count = len(self.cfg.static_cols)
        if 'height' in self.cfg.static_cols and 'weight' in self.cfg.static_cols:
            static_count -= 1  # BMI replaces both height and weight (2 → 1)

        feature_count = (static_count +
                        len(self.cfg.sleep_cols) +
                        len(self.cfg.ts_cols))

        # 2. Positional encoding (comes BEFORE windowed features)
        if self.cfg.use_positional_encoding:
            if self.cfg.pos_encoding_type == "sinusoidal":
                feature_count += self.cfg.pos_encoding_dim
            elif self.cfg.pos_encoding_type == "biocharge_circadian":
                feature_count += 4  # 3 circadian features + 1 time_state feature
            else:  # "time_of_day"
                feature_count += 2  # time_of_day_encoding returns 2 elements [sin, cos]

        # 3. Windowed features (comes AFTER positional encoding)
        if self.cfg.use_windowed_features:
            # Windowed features add window_size_minutes features for HR and ACC each
            # The actual implementation returns slices of length window_size_minutes
            feature_count += self.cfg.window_size_minutes  # HR window slice
            feature_count += self.cfg.window_size_minutes  # ACC window slice

            if self.cfg.include_gradients:
                # Gradient features also add window_size_minutes features each
                feature_count += self.cfg.window_size_minutes  # HR gradient window slice
                feature_count += self.cfg.window_size_minutes  # ACC gradient window slice

        # 4. Past charge (comes LAST, after windowed features)
        if self.cfg.use_past_charge:
            feature_count += 1

        return feature_count


    def _initialize_feature_indices(self, total_features):
        """Initialize indices for different feature types for augmentation."""
        if self.ts_feature_indices is not None:
            return

        # Calculate feature positions based on your feature construction
        # Account for BMI replacement: if both height and weight are present,
        # they are replaced by a single BMI feature
        static_count = len(self.cfg.static_cols)
        if 'height' in self.cfg.static_cols and 'weight' in self.cfg.static_cols:
            static_count -= 1  # BMI replaces both height and weight (2 → 1)

        static_sleep_count  = static_count + len(self.cfg.sleep_cols)
        ts_start_idx        = static_sleep_count

        # Initialize category lists
        self.binary_sequence_indices   = []  # For exercise (modifiable binary features)
        self.continuous_signal_indices = []  # For HR, stress, accelerometer
        self.sleep_stage_indices       = []  # For sleep stage specific handling
        # PROTECTED: sleep_stage, nap_state, sleep_markers - values should NEVER be changed
        # (only feature dropout to -6 is acceptable)
        self.protected_state_indices   = []

        # Map time series columns to their feature indices and categories
        for i, col in enumerate(self.cfg.ts_cols):
            feature_idx = ts_start_idx + i

            # PROTECTED features: sleep_stage, nap_state, sleep_markers
            # These should NEVER have their values modified (no jitter, no flip, no noise)
            # Only feature dropout (setting to -6) is acceptable
            if any(protected_col in col.lower() for protected_col in ['nap_state', 'sleep_markers', 'sleep_stage']):
                self.protected_state_indices.append(feature_idx)

            # Categorize binary vs continuous signals
            # Only 'exercise' is modifiable among binary features
            if 'exercise' in col.lower() and 'exercise_event' not in col.lower():
                self.binary_sequence_indices.append(feature_idx)

            # Special handling for sleep_stage (for reference, but not for augmentation)
            if 'sleep_stage' in col.lower():
                self.sleep_stage_indices.append(feature_idx)
            
            if 'hrr' in col.lower() and 'hr' not in col.lower():  # HRR and not hr
                self.hrr_indices.append(feature_idx) 
                self.continuous_signal_indices.append(feature_idx)

            if 'hr_filtered' in col.lower() and 'hrr' not in col.lower():  # Heart rate but not HRR
                self.hr_indices.append(feature_idx) 
                self.continuous_signal_indices.append(feature_idx)

            elif 'stress' in col.lower():
                self.stress_indices.append(feature_idx)
                self.continuous_signal_indices.append(feature_idx)
            elif 'trimp' in col.lower():
                self.exercise_indices.append(feature_idx)
                if 'trimp' in col.lower():  # TRIMP is continuous
                    self.continuous_signal_indices.append(feature_idx)
            elif 'acc' in col.lower() or 'hrr' in col.lower():
                self.activity_indices.append(feature_idx)
                self.continuous_signal_indices.append(feature_idx)
        
        # ts_feature_indices should include ALL time series features, not exclude binary ones
        self.ts_feature_indices = list(range(ts_start_idx, ts_start_idx + len(self.cfg.ts_cols)))
        
        # Create separate indices for temporal jittering (excluding binary sequences)
        self.jitter_indices = []
        for i, col in enumerate(self.cfg.ts_cols):
            feature_idx = ts_start_idx + i
            # Skip binary sequence columns for temporal jittering
            if not any(binary_col in col.lower() for binary_col in ['exercise', 'nap_state', 'sleep_markers', 'sleep_stage']):
                self.jitter_indices.append(feature_idx)
        
        # Add indices for windowed features if enabled
        self.windowed_hr_indices = []
        self.windowed_acc_indices = []
        self.windowed_stats_indices = []
        self.windowed_gradient_indices = []
        
        if self.cfg.use_windowed_features:
            # CRITICAL: This order MUST match _get_feature_vector exactly!
            # Order: static+sleep -> timeseries -> positional_encoding -> windowed_features -> past_charge
            current_idx = ts_start_idx + len(self.cfg.ts_cols)

            # Account for positional encoding FIRST (it comes BEFORE windowed features in _get_feature_vector)
            if self.cfg.use_positional_encoding:
                if self.cfg.pos_encoding_type == "sinusoidal":
                    current_idx += self.cfg.pos_encoding_dim
                else:  # "time_of_day" or default
                    current_idx += 2

            # NOTE: past_charge is NOT added here because it comes AFTER windowed features!
            # In _get_feature_vector, the order is: positional_encoding -> windowed_features -> past_charge

            # HR window indices (window slice)
            self.windowed_hr_indices = list(range(current_idx, current_idx + self.cfg.window_size_minutes))
            current_idx += self.cfg.window_size_minutes

            # Accelerometer window indices (window slice)
            self.windowed_acc_indices = list(range(current_idx, current_idx + self.cfg.window_size_minutes))
            current_idx += self.cfg.window_size_minutes

            # No statistics indices as per user request
            self.windowed_stats_indices = []

            # Gradient indices if enabled (window slices)
            if self.cfg.include_gradients:
                # HR gradient window slice
                hr_grad_indices = list(range(current_idx, current_idx + self.cfg.window_size_minutes))
                current_idx += self.cfg.window_size_minutes

                # ACC gradient window slice
                acc_grad_indices = list(range(current_idx, current_idx + self.cfg.window_size_minutes))
                current_idx += self.cfg.window_size_minutes

                self.windowed_gradient_indices = hr_grad_indices + acc_grad_indices
            else:
                self.windowed_gradient_indices = []
            
            # Add windowed indices to continuous signal indices for augmentation
            self.continuous_signal_indices.extend(self.windowed_hr_indices)
            self.continuous_signal_indices.extend(self.windowed_acc_indices)
            # No stats indices to add
            if self.cfg.include_gradients:
                self.continuous_signal_indices.extend(self.windowed_gradient_indices)
        

    def flip_label(self, y, p=0.02):
        if np.random.rand() < p:
            return 1 - y
        return y
    
    def augment_sleep_stage_transitions(self, x_aug, sleep_stage_indices):
        """
        Apply simple signal corruption to sleep stage data.
        
        Args:
            x_aug: Feature vector to augment
            sleep_stage_indices: Indices of sleep stage features
        """
        if not sleep_stage_indices:
            return x_aug
            
        for idx in sleep_stage_indices:
            if idx < len(x_aug):
                current_stage = x_aug[idx]
                
                # Simple noise corruption with 10% probability
                if np.random.random() < 0.1:
                    # Add small amount of noise
                    noise = np.random.normal(0, 0.05)  # 5% noise
                    x_aug[idx] = np.clip(current_stage + noise, 0.0, 1.0)
        
        return x_aug
    
    def get_sleep_stage_info(self):
        """
        Get information about sleep_stage columns and their configuration.
        
        Returns:
            dict: Sleep stage configuration information
        """
        sleep_stage_cols = [col for col in self.cfg.ts_cols if 'sleep_stage' in col.lower()]
        
        return {
            "sleep_stage_columns": sleep_stage_cols,
            "sleep_stage_count": len(sleep_stage_cols),
            "sleep_stage_indices_initialized": hasattr(self, 'sleep_stage_indices') and bool(self.sleep_stage_indices),
            "augmentation_includes_sleep_stage": self.augment and len(sleep_stage_cols) > 0
        }


    def augment_sample(self, x, y):
        """
        Augment time series features with 30% probability.
        Applies physiologically realistic augmentations for different signal types.
        """
        if not self.augment or np.random.random() > self.augment_prob:
            return x, y

        # Initialize feature indices if not done
        self._initialize_feature_indices(len(x))

        # Skip augmentation if hrr_raw is -6 (missing value)
        if self.hr_indices:
            for idx in self.hr_indices:
                if idx < len(x) and x[idx] == -6:
                    return x, y

        # Create a copy to avoid modifying original
        x_aug = x.copy()

        # === BINARY SEQUENCE AUGMENTATION ===
        # Only for exercise features (NOT sleep_stage, nap_state, sleep_markers - these are protected)
        # Reasoning: sleep_stage, nap_state, and sleep_markers represent ground-truth physiological
        # states that should not be artificially modified. Modifying them would corrupt the
        # relationship between these states and the target variable (charge delta).
        if self.binary_sequence_indices and np.random.random() < 0.4:
            for idx in self.binary_sequence_indices:
                # Skip protected features (sleep_stage, nap_state, sleep_markers)
                if idx in self.protected_state_indices:
                    continue
                original_val = x_aug[idx]
                x_aug[idx] = self.flip_label(original_val, p = 0.2)

        
        # === HEART RATE AUGMENTATION ===
        # # For continuous HR signals - physiologically realistic variations
        if self.hr_indices and np.random.random() < 0.7:
            for idx in self.hr_indices:
                hr_val = x_aug[idx]
                
                # Strategy 1: Heart Rate Variability (HRV simulation)
                # Normal HR variation is ±2-5 bpm (roughly 2-3% of normalized signal)
                # if np.random.random() < 0.8:
                #     hrv_noise = np.random.normal(0, 0.025)  # 2.5% std
                #     x_aug[idx] = hr_val + hrv_noise
                
                # # Strategy 2: Physiological State Shifts
                # # Simulate different fitness/stress states affecting baseline HR
                # elif np.random.random() < 0.3:
                #     # Simulate fitness variations (±5% baseline shift)
                #     fitness_shift = np.random.uniform(0.95, 1.05)
                #     x_aug[idx] = hr_val * fitness_shift
                
                # Strategy 3: Measurement Artifact Simulation
                # Simulate sensor artifacts (brief spikes/drops)
                if np.random.random() < 0.1:
                    artifact_factor = np.random.choice([0.9, 1.1])  # 10% spike or drop
                    x_aug[idx] = hr_val * artifact_factor
        
        # === CONTINUOUS SIGNAL AUGMENTATION ===
        # For stress, accelerometer, TRIMP etc.
        if self.continuous_signal_indices and np.random.random() < 0.6:
            for idx in self.continuous_signal_indices:
                # if idx not in self.hr_indices:  # Skip HR (already handled above)
                signal_val = x_aug[idx]
                
                # Strategy 1: Sensor Noise
                if np.random.random() < 0.7:
                    noise_std = 0.03 if idx in self.stress_indices else 0.025
                    sensor_noise = np.random.normal(0, noise_std)
                    x_aug[idx] = signal_val + sensor_noise
                
                # Strategy 2: Calibration Drift
                elif np.random.random() < 0.2:
                    drift_factor = np.random.uniform(0.95, 1.05)
                    x_aug[idx] = signal_val * drift_factor
        
        # === TRIMP SPECIFIC AUGMENTATION ===
        if self.exercise_indices and np.random.random() < 0.4:
            for idx in self.exercise_indices:
                exercise_val = x_aug[idx]

                # Strategy 1: Training Load Variation (daily performance varies ±15%)
                if np.random.random() < 0.8:
                    performance_factor = np.random.uniform(0.85, 1.15)
                    x_aug[idx] = exercise_val * performance_factor

                # # Strategy 2: Exercise Type Variation
                # # Different exercise types have different intensity profiles
                # elif np.random.random() < 0.3:
                #     intensity_variation = np.random.uniform(0.9, 1.1)
                #     x_aug[idx] = exercise_val * intensity_variation

        # === MISSING VALUE AUGMENTATION (Non-windowed HR/ACC) ===
        # Simulate sensor dropouts for point-wise features (rare but realistic)

        # Non-windowed HR missing values (5% probability)
        if self.hr_indices and np.random.random() < 0.05:
            for idx in self.hr_indices:
                x_aug[idx] = -6

        # Non-windowed ACC missing values (5% probability, independent from HR)
        if self.activity_indices and np.random.random() < 0.05:
            for idx in self.activity_indices:
                x_aug[idx] = -6

        # === SLEEP STAGE SPECIFIC AUGMENTATION ===
        # DISABLED: sleep_stage is a protected feature - values should NEVER be modified
        # Reasoning: sleep_stage represents ground-truth sleep phases that directly influence
        # recovery rates. Artificially perturbing these values would corrupt the model's
        # ability to learn the true relationship between sleep stages and charge dynamics.
        # if self.sleep_stage_indices and np.random.random() < 0.3:
        #     x_aug = self.augment_sleep_stage_transitions(x_aug, self.sleep_stage_indices)

        # === TEMPORAL CONSISTENCY AUGMENTATION ===
        # Apply small temporal jittering to maintain physiological realism
        # Only apply to continuous signals, not binary sequences
        if np.random.random() < 0.3:
            jitter_factor = np.random.uniform(0.98, 1.02)  # ±2% temporal jitter
            for idx in self.jitter_indices:
                if idx < len(x_aug):  # Safety check to prevent out-of-bounds access
                    x_aug[idx] = x_aug[idx] * jitter_factor
        
        # === RANDOM FEATURE DROPOUT (Missingness Simulation) ===
        # Each feature independently has 10% probability of being dropped
        # This simulates realistic sensor failures and missing data patterns
        if np.random.random() < 0.05:
            # Determine which features can be dropped
            static_sleep_count = len(self.cfg.static_cols) + len(self.cfg.sleep_cols)
            static_count = len(self.cfg.static_cols)

            # Separate droppable indices by type
            static_indices = []
            sleep_indices = []
            other_droppable_indices = []
            pos_encoding_indices = []
            past_charge_idx = None

            # Static features (BMI)
            for i in range(static_count):
                static_indices.append(i)

            # Sleep features
            for i in range(static_count, static_sleep_count):
                sleep_indices.append(i)

            # Time series features can be dropped
            if self.ts_feature_indices:
                other_droppable_indices.extend(self.ts_feature_indices)

            # Windowed features can be dropped
            if self.cfg.use_windowed_features:
                other_droppable_indices.extend(self.windowed_hr_indices)
                other_droppable_indices.extend(self.windowed_acc_indices)
                if self.cfg.include_gradients:
                    other_droppable_indices.extend(self.windowed_gradient_indices)

            # Positional encoding features should NOT be dropped
            # Do NOT add positional encoding indices to droppable list
            # (pos_encoding_indices remains empty)

            # Past charge index (last feature, after positional encoding)
            if self.cfg.use_past_charge and len(x_aug) > 0:
                past_charge_idx = len(x_aug) - 1

            # Combine all droppable indices
            all_droppable = static_indices + sleep_indices + other_droppable_indices + pos_encoding_indices
            if past_charge_idx is not None:
                all_droppable.append(past_charge_idx)

            # Independent dropout: each feature has 10% probability of being dropped
            if all_droppable:
                for idx in all_droppable:
                    if np.random.random() < 0.1:  # 10% independent probability per feature
                        if idx < len(x_aug):
                            # Use -6 for all features to indicate missingness
                            x_aug[idx] = -6

        # === RANDOM NOISE AUGMENTATION (Excluding Positional Encoding) ===
        # Add small Gaussian noise to all features except positional encoding
        if np.random.random() < 0.5:
            # Calculate positional encoding indices to exclude
            pos_encoding_start = len(self.cfg.static_cols) + len(self.cfg.sleep_cols) + len(self.cfg.ts_cols)
            pos_encoding_end = pos_encoding_start

            if self.cfg.use_positional_encoding:
                if self.cfg.pos_encoding_type == "sinusoidal":
                    pos_encoding_end = pos_encoding_start + self.cfg.pos_encoding_dim
                elif self.cfg.pos_encoding_type == "biocharge_circadian":
                    pos_encoding_end = pos_encoding_start + 4  # 3 circadian + 1 time_state
                else:  # "time_of_day"
                    pos_encoding_end = pos_encoding_start + 2

            # Add noise to all features except positional encoding and past charge 
            # noise is added to past charge during training
            for idx in range(len(x_aug)-1):
                # Skip positional encoding indices
                if self.cfg.use_positional_encoding and pos_encoding_start <= idx < pos_encoding_end:
                    continue

                # Add small Gaussian noise (1-3% std)
                noise_std = np.random.uniform(0.01, 0.03)
                noise = np.random.normal(0, noise_std)
                x_aug[idx] = x_aug[idx] + noise

            # Also add noise to past_charge if enabled
            # Past charge is the last feature
            # if self.cfg.use_past_charge and len(x_aug) > 0:
            #     past_charge_idx = len(x_aug) - 1
            #     # Skip if this is in pos_encoding range (shouldn't be, but safety check)
            #     if not (self.cfg.use_positional_encoding and pos_encoding_start <= past_charge_idx < pos_encoding_end):
            #         noise_std = np.random.uniform(0.01, 0.03)  # Smaller noise for past_charge
            #         noise = np.random.normal(0, noise_std)
            #         x_aug[past_charge_idx] = x_aug[past_charge_idx] + noise

        # === BOUNDS CHECKING ===
        # Ensure binary sequences stay in [0,1] range and continuous signals stay reasonable
        #
        for idx in self.binary_sequence_indices:
            # CEHCK THIS 
            pass
        #     x_aug[idx] = np.clip(x_aug[idx], 0.0, 1.0)


        # add augmentation for windowed features if enabled
        if self.cfg.use_windowed_features:
            # HR window
            for idx in self.windowed_hr_indices:
                hr_window_val = x_aug[idx]
                if np.random.random() < 0.7:
                    hrv_noise = np.random.normal(0, 0.025)  # 2.5% std
                    x_aug[idx] = hr_window_val + hrv_noise
            
            # Accelerometer window
            for idx in self.windowed_acc_indices:
                acc_window_val = x_aug[idx]
                if np.random.random() < 0.6:
                    sensor_noise = np.random.normal(0, 0.025)
                    x_aug[idx] = acc_window_val + sensor_noise

            # === MISSING VALUE AUGMENTATION ===
            # Simulate sensor dropouts (rare but realistic)
            # Apply independently (asynchronously) to HR and ACC

            # HR missing values (5% probability)
            if self.windowed_hr_indices and np.random.random() < 0.05:
                # Random duration: 1-3 consecutive values
                missing_duration = np.random.randint(1, 4)
                # Random location within the window
                window_len = len(self.windowed_hr_indices)
                if window_len > missing_duration:
                    start_idx = np.random.randint(0, window_len - missing_duration + 1)
                    # Set to -6 (simulating missing/dropout)
                    for i in range(start_idx, start_idx + missing_duration):
                        x_aug[self.windowed_hr_indices[i]] = -6

            # ACC missing values (5% probability, independent from HR)
            if self.windowed_acc_indices and np.random.random() < 0.05:
                # Random duration: 1-3 consecutive values
                missing_duration = np.random.randint(1, 4)
                # Random location within the window
                window_len = len(self.windowed_acc_indices)
                if window_len > missing_duration:
                    start_idx = np.random.randint(0, window_len - missing_duration + 1)
                    # Set to -6 (simulating missing/dropout)
                    for i in range(start_idx, start_idx + missing_duration):
                        x_aug[self.windowed_acc_indices[i]] = -6

            # Statistics
            for idx in self.windowed_stats_indices:
                stat_val = x_aug[idx]
                if np.random.random() < 0.5:
                    noise = np.random.normal(0, 0.02)
                    x_aug[idx] = stat_val + noise
            
            # Gradients
            # for idx in self.windowed_gradient_indices:
            #     grad_val = x_aug[idx]
            #     if np.random.random() < 0.5:
            #         noise = np.random.normal(0, 0.01)
            #         x_aug[idx] = grad_val + noise

            for idx in self.windowed_gradient_indices:
                grad_val = x_aug[idx]
                if np.random.random() < 0.5:
                    # Slightly stronger, scale-aware noise (2–3%)
                    noise = np.random.normal(0, 0.025 * abs(grad_val) + 1e-3)
                    x_aug[idx] = grad_val + noise

            # Temporal jitter: Shift windowed sequences backward in time
            # This simulates slight temporal misalignment in data collection
            shift = np.random.randint(-2, 1)  # -2, -1, or 0 (shift backward only to maintain causality)
            if shift != 0:
                # shift < 0 means shift backward (e.g., -2 removes first 2 elements, pads with zeros at end)
                # For shift=-2: [a,b,c,d,e] -> [c,d,e,0,0] (removes early timepoints)

                # Apply same shift to HR and ACC windows
                for idx_group in [self.windowed_hr_indices, self.windowed_acc_indices]:
                    seq = x_aug[idx_group]
                    # Shift backward: remove first |shift| elements, pad with zeros at end
                    x_aug[idx_group] = np.concatenate([seq[-shift:], np.zeros(-shift)])

                # Apply same shift to gradient windows if they exist
                if self.cfg.include_gradients and len(self.windowed_gradient_indices) > 0:
                    # Split gradient indices into HR and ACC gradients
                    window_size = self.cfg.window_size_minutes
                    hr_grad_idx = self.windowed_gradient_indices[:window_size]
                    acc_grad_idx = self.windowed_gradient_indices[window_size:2*window_size]

                    for idx_group in [hr_grad_idx, acc_grad_idx]:
                        if len(idx_group) > 0:
                            seq = x_aug[idx_group]
                            x_aug[idx_group] = np.concatenate([seq[-shift:], np.zeros(-shift)])


        # === PAST CHARGE AUGMENTATION ===
        # Simulates train-inference mismatch (model uses own predictions at inference)
        if self.cfg.use_past_charge and np.random.random() < 0.25:
            past_charge_idx = len(x_aug) - 1
            if x_aug[past_charge_idx] != -6:  # Skip if missing
                noise = np.random.normal(0, 0.015)  # 1.5% std - very subtle
                x_aug[past_charge_idx] = x_aug[past_charge_idx] + noise

        # Clip continuous signals to reasonable bounds
        for idx in self.continuous_signal_indices:
            x_aug[idx] = np.clip(x_aug[idx], -4.0, 4.0)
        
        return x_aug, y 

    def __len__(self):
        return len(self.index)


    def find_sleep_start(self, series):
        """
        Given a binary sequence (0/1), find the index of the last 0->1 transition
        (searching from the end of the sequence).
        Returns None if no transition found or data is malformed.
        """
        try:
            if series is None or series.empty:
                return None
                
            series_val = series.iloc[0]
            arr = None
            
            if isinstance(series_val, str):
                try:
                    parsed = ast.literal_eval(series_val)
                    arr = np.array(parsed).astype(int)
                except (ValueError, SyntaxError, TypeError):
                    return None
            
            elif isinstance(series_val, (list, tuple)):
                try:
                    arr = np.array(series_val).astype(int)
                except (ValueError, TypeError):
                    return None
                    
            elif isinstance(series_val, np.ndarray):
                try:
                    arr = series_val.astype(int)
                except (ValueError, TypeError):
                    return None
            else:
                return None
            
            if arr is None or len(arr) <= 1:
                return None
            
            # Scan backwards to find 0->1 transition
            for i in range(len(arr) - 1, 0, -1):
                if arr[i] == 1 and arr[i-1] == 0:
                    return i
                    
            return None  # No transition found
            
        except Exception:
            return None


    def _load_user_file(self, user_id: str, date, idx) -> pd.DataFrame:
        """Load user file from folder (cached)."""

        date_str = date.strftime("%Y-%m-%d") 
        if user_id in self.user_cache:
            df = self.user_cache[user_id]
        else:
            # if userid in llm users then use self.cfg.llm_date_dir
            if self.cfg.data_dir_llm is not None and (user_id, date_str) in self.llm_charge_pairs:
                excel_path = os.path.join(self.cfg.data_dir_llm, f"{user_id}_processed.xlsx")
            else:
                excel_path = os.path.join(self.cfg.data_dir, f"{user_id}_processed.xlsx")

            # csv_path = os.path.join(self.cfg.data_dir, f"{user_id}.csv")
            # parquet_path = os.path.join(self.cfg.data_dir, f"{user_id}.parquet")

            if os.path.exists(excel_path):
                df = pd.read_excel(excel_path)
            else:
                raise FileNotFoundError(f"No file found for user {user_id}")
            
            self.user_cache[user_id] = df
            # Initialize processed dates cache for this user
            # self.processed_dates_cache[user_id] = set()


        # Try csv or parquet
        # read excel file

        df_row = df[df["date"] == date_str]
        df_yesterday_row = df[df["date"] == (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")]

        # # Check if this date has already been processed for this user
        # if user_id not in self.processed_dates_cache:
        #     self.processed_dates_cache[user_id] = set()
            
        # if date_str not in self.processed_dates_cache[user_id]:
        #     # Filter hr data only if not already processed - selective filtering for outliers only
        #     try:
        #         self.filter_hr_data(user_id, date_str)
        #         # Mark this date as processed
        #         self.processed_dates_cache[user_id].add(date_str)
                    
        #     except Exception as e:
        #         print(f"Warning: Failed to filter HR data for user {user_id} on {date_str}: {e}")
        #         # Mark as processed to avoid repeated attempts
        #         self.processed_dates_cache[user_id].add(date_str)
        
        # Get the updated row from cache
        df_row = self.user_cache[user_id][self.user_cache[user_id]["date"] == date_str]

        # Initialize caches for this user if needed
        if user_id not in self.processed_dates_cache:
            self.processed_dates_cache[user_id] = set()
        if user_id not in self.sleep_start_cache:
            self.sleep_start_cache[user_id] = {}
        
        # Check if sleep_start_idx is already cached for this date
        if date_str in self.sleep_start_cache[user_id]:
            sleep_start_idx = self.sleep_start_cache[user_id][date_str]
        else:
            # Calculate and cache sleep_start_idx
            sleep_start_idx = self.find_sleep_start(df_row['timeseries.sleep_markers'])
            self.sleep_start_cache[user_id][date_str] = sleep_start_idx

        # add to processed dates cache
        self.processed_dates_cache[user_id].add(date_str)


        # if sleep_start_idx is None:
        #     print(f"No sleep start found for user {user_id} on date {date_str}, using idx {idx}")
        # if df_yesterday_row.empty:
        #     prev_charge = 69
        # else:
        #     prev_charge = float(df_yesterday_row.iloc[0][self.cfg.charge_col])

        df["date"] = pd.to_datetime(df["date"])
        # self.user_cache[user_id] = df
        return df_row, df_yesterday_row, sleep_start_idx

    def _safe_extract_value(self, data, idx=None, default=0.0):
        """
        Safely extract value from various data types with robust error handling.
        Enhanced with better out-of-bounds protection and data length mismatch handling.
        """
        try:
            if data is None:
                return default
            
            # Handle all numpy scalar types
            if isinstance(data, (int, float, np.integer, np.floating, np.int64, np.int32, np.float64, np.float32)):
                val = float(data)
                return val if not (np.isnan(val) or np.isinf(val)) else default
            
            # Handle numpy boolean
            if isinstance(data, (bool, np.bool_)):
                return float(data)
            
            # Handle numpy arrays
            if isinstance(data, np.ndarray):
                if data.size == 0:
                    return default
                if idx is not None:
                    if 0 <= idx < len(data):
                        val = float(data[idx])
                        return val if not (np.isnan(val) or np.isinf(val)) else default
                    else:
                        # Enhanced: Handle out-of-bounds with intelligent fallback
                        if len(data) > 0:
                            # Use last available value for out-of-bounds access
                            val = float(data[-1])
                            return val if not (np.isnan(val) or np.isinf(val)) else default
                        return default
                else:
                    # Return first element if no index specified
                    val = float(data.flat[0])
                    return val if not (np.isnan(val) or np.isinf(val)) else default
            
            if isinstance(data, str):
                try:
                    # Try to parse as list/array
                    parsed = ast.literal_eval(data)
                    if isinstance(parsed, (list, tuple)) and idx is not None:
                        if 0 <= idx < len(parsed):
                            val = float(parsed[idx])
                            return val if not (np.isnan(val) or np.isinf(val)) else default
                        else:
                            # Enhanced: Handle out-of-bounds with intelligent fallback
                            if len(parsed) > 0:
                                # Use last available value for out-of-bounds access
                                val = float(parsed[-1])
                                return val if not (np.isnan(val) or np.isinf(val)) else default
                            return default
                    val = float(parsed)
                    return val if not (np.isnan(val) or np.isinf(val)) else default
                except (ValueError, SyntaxError, IndexError):
                    # If parsing fails, try direct float conversion
                    try:
                        val = float(data)
                        return val if not (np.isnan(val) or np.isinf(val)) else default
                    except (ValueError, TypeError):
                        return default
            
            if isinstance(data, (list, tuple)):
                if idx is not None:
                    if 0 <= idx < len(data):
                        val = data[idx]
                        # Recursively handle numpy types in lists
                        if isinstance(val, (np.integer, np.floating)):
                            val = float(val)
                        return val if not (np.isnan(val) or np.isinf(val)) else default
                    else:
                        # Enhanced: Handle out-of-bounds with intelligent fallback
                        if len(data) > 0:
                            # Use last available value for out-of-bounds access
                            val = data[-1]
                            if isinstance(val, (np.integer, np.floating)):
                                val = float(val)
                            return val if not (np.isnan(val) or np.isinf(val)) else default
                        return default
                elif idx is None and len(data) > 0:
                    val = data[0]
                    # Recursively handle numpy types in lists
                    if isinstance(val, (np.integer, np.floating)):
                        val = float(val)
                    return val if not (np.isnan(val) or np.isinf(val)) else default
                else:
                    return default
            
            # For pandas Series
            if hasattr(data, 'iloc'):
                try:
                    val = data.iloc[0]
                    return self._safe_extract_value(val, idx, default)
                except (IndexError, KeyError):
                    return default
            
            # For pandas scalar values
            if hasattr(data, 'item'):
                try:
                    val = float(data.item())
                    return val if not (np.isnan(val) or np.isinf(val)) else default
                except (ValueError, TypeError):
                    return default
            
            # Last resort: try direct float conversion
            try:
                val = float(data)
                return val if not (np.isnan(val) or np.isinf(val)) else default
            except (ValueError, TypeError):
                return default
            
        except Exception as e:
            # Enhanced: Add more specific error logging for debugging
            import traceback
            print(f"Debug: Exception in _safe_extract_value - data type: {type(data)}, idx: {idx}, error: {e}")
            print(f"Debug: Traceback: {traceback.format_exc()}")
            return default

    def _get_feature_vector(self, row: pd.Series, yesterday_row, idx: int, idx_sleep_start: int, charge_col: str = None) -> np.ndarray:
        feats = []

        
        # Use provided charge_col or fall back to config
        if charge_col is None:
            charge_col = self.cfg.charge_col

        # Handle malformed yesterday_row
        if yesterday_row is None or yesterday_row.empty:
            yesterday_row = row

        # Validate idx_sleep_start
        if idx_sleep_start is None:
            idx_sleep_start = 0
            
        # Static + sleep features with robust error handling
        # First, calculate BMI if we have height and weight in static_cols
        bmi_value = None
        if 'height' in self.cfg.static_cols and 'weight' in self.cfg.static_cols:
            try:
                height_val = self._safe_extract_value(row.get('height', None))  # in cm
                weight_val = self._safe_extract_value(row.get('weight', None))  # in kg
                if height_val > 0:
                    bmi_value = weight_val / ((height_val / 100) ** 2)  # BMI formula
                    bmi_value = bmi_value / 30.0  # Normalize BMI (typical range 15-40, so /30)
                else:
                    bmi_value = -6
            except:
                bmi_value = -6

        for col in self.cfg.static_cols + self.cfg.sleep_cols:

            # print(f"col: {col}")
            try:
                row_ = row.get(col, None)
                row_yesterday_ = yesterday_row.get(col, None)

                if row_ is None:
                    feats.append(-6.0)
                    continue

                # Handle BMI calculation (replace height and weight with BMI)
                if col == 'height' or col == 'weight':
                    # Skip individual height/weight, we'll use BMI instead
                    if col == 'weight' and bmi_value is not None:
                        # Add BMI when we encounter weight (skip height entirely)
                        feats.append(float(bmi_value))
                    # Skip height entirely
                    continue

                # normalize by constants with safe extraction
                elif 'age' in col.lower():
                    val = self._safe_extract_value(row_) / 80

                elif 'gender' in col:
                    val = self._safe_extract_value(row_)

                else:
                    # Determine which row to use based on sleep timing for ALL sleep metrics
                    # If before sleep start and yesterday's data exists, use yesterday's sleep data
                    # Otherwise use today's sleep data (including when row_yesterday_ is None)
                    if idx < idx_sleep_start and row_yesterday_ is not None:
                        source_row = row_yesterday_
                    else:
                        # # Use default values when today's sleep data isn't available yet
                        # if col not in ["z_rhr_7", "z_hrv_7", ]:
                        #     val = 0.0  # Default for z-scored values
                        # elif col in ["sleep_duration", "wakefulness_after_sleep_onset_duration"]:
                        #     val = 0.0  # Default for duration
                        # elif col in ["rem_sleep_ratio", "deep_sleep_ratio", "light_sleep_ratio"]:
                        #     val = 0.0  # Default for ratios
                        # else:
                        #     val = -6  # Missing indicator for other sleep features
                        # Don't use source_row, just use the default val set above
                        source_row = row_

                    # Only process from source_row if it's available
                    if source_row is not None:
                        # Handle sleep duration and waso columns - normalize by 660 minutes
                        if col in ["sleep_duration", "wakefulness_after_sleep_onset_duration"]:
                            val = self._safe_extract_value(source_row) / 660.0

                        # Handle sleep ratio columns (already in 0-1 range, just use directly)
                        elif col in ["rem_sleep_ratio", "deep_sleep_ratio", "light_sleep_ratio"]:
                            val = self._safe_extract_value(source_row)  # Already normalized to 0-1

                        # Handle z_rhr_7 and z_hrv_7 - already z-scored, use directly without normalization
                        elif col in ["z_rhr_7", "z_hrv_7"]:
                            val = self._safe_extract_value(source_row)  # Already z-scored, use as-is

                        # Legacy sleep duration columns (if still used)
                        elif col in ["deep_sleep_duration", "light_sleep_duration", "rem_sleep_duration"]:
                            val = self._safe_extract_value(source_row) / 720  # Normalize by max 12 hours

                        else:
                            val = self._safe_extract_value(source_row) / 100
                
                # Ensure the value is valid
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    val = -6.0
                
                # print(f"{col}: {val}")
                feats.append(float(val))
                
            except Exception as e:
                # Log malformed data for debugging (optional)
                print(f"Error processing column {col}: {e}")
                feats.append(0.0)

        # Timeseries columns with robust error handling
        for col in self.cfg.ts_cols:
            # print(f"col: {col}")
            val = 0.0

            try:
                row_ = row.get(col, None)
                if row_ is None:
                    feats.append(0.0)
                    continue
                
                #------------------------------------------------------------------------------
                #         Event Data (exercise, sleep_stage, sleep_markers, nap_state)
                # -----------------------------------------------------------------------------
                

                
                if 'exercise' in col.lower():
                    # Extract sleep stage value at specific index
                    raw_val = self._safe_extract_value(row_, idx, 0.0)
                    val = raw_val
                    feats.append(float(val) if not np.isnan(val) else 0.0)
                    continue
                
                if 'sleep_markers' in col.lower():
                    # Extract sleep stage value at specific index
                    raw_val = self._safe_extract_value(row_, idx, 0.0)
                    val = raw_val
                    feats.append(float(val) if not np.isnan(val) else 0.0)
                    continue
                
                if 'nap_state' in col.lower():
                    # Extract sleep stage value at specific index
                    raw_val = self._safe_extract_value(row_, idx, 0.0)
                    val = raw_val
                    feats.append(float(val) if not np.isnan(val) else 0.0)
                    continue

                # Special handling for sleep_stage column
                if 'sleep_stage' in col.lower():
                    # Extract sleep stage value at specific index
                    raw_val = self._safe_extract_value(row_, idx, 0.0)
                    
                    # Sleep stages are typically encoded as: -1=awake, 1=light, 2=deep, 3=REM
                    # Normalize to 0-1 range: 0=awake, 0.33=light, 0.66=deep, 1.0=REM
                    val =raw_val/3
                    feats.append(float(val) if not np.isnan(val) else 0.0)
                    continue
                
                # Safely extract time series value at specific index
                raw_val = self._safe_extract_value(row_, idx, 0.0)
                
                #---------------------------------------------------------
                #         Timeseries Data (hr)
                # --------------------------------------------------------
                # if 'hr' in col.lower() and 'hrr' not in col.lower():
                if 'hrr_raw' in col.lower():
                    col_string = self._get_hr_column_name(row)
                    raw_val = raw_val*0.01
                    feats.append(float(raw_val) if not np.isnan(raw_val) else 0.0)
                    continue

                else:
                    col_string = col
                
                # Apply z-score normalization if available
                if self.zdf is not None and self.cfg.use_zscores and (col_string in self.zdf['global']):
                    try:

                        if self.z_data_norm == 'population':
                            if 'global' in self.zdf and col_string in self.zdf['global']:
                                z_std = self.zdf['global'][col_string]['std']
                                z_mean = self.zdf['global'][col_string]['mean']
                            else:
                                # Fallback to raw normalization
                                val = raw_val / 100
                        else:
                            user_id_str = str(int(self._safe_extract_value(row.get('userid', ''))))
                            
                            # Try user-specific z-scores first
                            if (user_id_str in self.zdf and 
                                col_string in self.zdf[user_id_str]):
                                z_std = self.zdf[user_id_str][col_string]['std']
                                z_mean = self.zdf[user_id_str][col_string]['mean']
                            elif ('global' in self.zdf and 
                                col_string in self.zdf['global']):
                                z_std = self.zdf['global'][col_string]['std']
                                z_mean = self.zdf['global'][col_string]['mean']
                            else:
                                # No z-score available, use raw value
                                val = raw_val
                                feats.append(float(val) if not np.isnan(val) else 0.0)
                                continue
                        
                        # Apply z-score normalization
                        val = (raw_val - z_mean) / z_std if z_std > 0 else 0.0
                        
                    except (KeyError, TypeError, ValueError, IndexError):
                        val = raw_val
                else:
                    val = raw_val
                

                # RHR centered hr 
                if 'z_hr_filtered_7' in col.lower():
                    val = self._safe_extract_value('z_hr_filtered_7', idx, 0.0)  # mean_hr_7


                #---------------------------------------------------------
                #         Timeseries Data (acc)
                #--------------------------------------------------------
                if 'timeseries.acc' in col.lower():
                    
                    # Apply z-score normalization if available
                    if self.zdf is not None and self.cfg.use_zscores and (col_string in self.zdf['global']):
                        try:
                            if self.z_data_norm == 'population':
                                if 'global' in self.zdf and col_string in self.zdf['global']:
                                    z_std = self.zdf['global'][col_string]['std']
                                    z_mean = self.zdf['global'][col_string]['mean']
                                else:
                                    # Fallback to raw normalization
                                    val = raw_val / 100
                            else:
                                user_id_str = str(int(self._safe_extract_value(row.get('userid', ''))))
                                
                                # Try user-specific z-scores first
                                if (user_id_str in self.zdf and 
                                    col_string in self.zdf[user_id_str]):
                                    z_std = self.zdf[user_id_str][col_string]['std']
                                    z_mean = self.zdf[user_id_str][col_string]['mean']
                                elif ('global' in self.zdf and 
                                    col_string in self.zdf['global']):
                                    z_std = self.zdf['global'][col_string]['std']
                                    z_mean = self.zdf['global'][col_string]['mean']
                                else:
                                    # No z-score available, use raw value
                                    val = raw_val
                                    feats.append(float(val) if not np.isnan(val) else 0.0)
                                    continue
                            
                            # Apply z-score normalization
                            val = (raw_val - z_mean) / z_std if z_std > 0 else 0.0
                            
                        except (KeyError, TypeError, ValueError, IndexError):
                            val = raw_val

                # Ensure valid float value
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    val = -6            
                # print(f"timeseries{col}: {val}")
                feats.append(float(val))

            except Exception as e:
                # Log malformed time series data for debugging (optional)
                # print(f"Error processing time series column {col} at index {idx}: {e}")
                feats.append(0.0)

                
        # Add positional encoding if enabled
        if self.cfg.use_positional_encoding:
            try:
                if self.cfg.pos_encoding_type == "time_of_day":
                    # Use time_of_day_encoding which returns a 2-element cyclic encoding [sin, cos]
                    pos_encoding = self.time_of_day_encoding(idx % 1440)
                    feats.extend(pos_encoding.tolist())
                elif self.cfg.pos_encoding_type == "sinusoidal":
                    # Use sinusoidal positional encoding with configurable dimensions
                    
                    # pos_encoding = self.positional_encoding(idx % 1440, self.cfg.pos_encoding_dim)
                    pos_encoding = self.time_of_day_encoding_continuous(idx % 1440)
                    feats.extend(pos_encoding.tolist())
                else:
                    # Default to time_of_day if invalid type specified
                    pos_encoding = self.time_of_day_encoding(idx % 1440)
                    feats.extend(pos_encoding.tolist())
            except Exception:
                # Fallback to zero encoding if there's an error
                if self.cfg.pos_encoding_type == "sinusoidal":
                    # Use pos_encoding_dim zeros for sinusoidal
                    feats.extend([0.0] * self.cfg.pos_encoding_dim)
                else:
                    # Use 2 zeros for time_of_day (default)
                    feats.extend([0.0, 0.0])

        # Add windowed features if enabled
        if self.cfg.use_windowed_features:
            try:
                # Use the new method to read pre-calculated windowed features from file
                windowed_features = self._read_windowed_features(row, idx)
                
                # Add windowed HR values (already z-score normalized from file)
                if isinstance(windowed_features['hr_window'], list) and len(windowed_features['hr_window']) > 0:
                    feats.extend(windowed_features['hr_window'])
                else:
                    # Fallback: add zeros for entire window if data not available
                    feats.extend([0.0] * self.cfg.window_size_minutes)

                # Add windowed accelerometer values (already z-score normalized from file)
                if isinstance(windowed_features['acc_window'], list) and len(windowed_features['acc_window']) > 0:
                    feats.extend(windowed_features['acc_window'])
                else:
                    # Fallback: add zeros for entire window if data not available
                    feats.extend([0.0] * self.cfg.window_size_minutes)
                
                # Don't add summary
                
                # Add gradients if enabled (already z-score normalized from file)
                if self.cfg.include_gradients:
                    # Add HR gradient
                    if isinstance(windowed_features['hr_gradient'], list) and len(windowed_features['hr_gradient']) > 0:
                        feats.extend(windowed_features['hr_gradient'])
                    else:
                        feats.extend([0.0] * self.cfg.window_size_minutes)

                    # Add accelerometer gradient
                    if isinstance(windowed_features['acc_gradient'], list) and len(windowed_features['acc_gradient']) > 0:
                        feats.extend(windowed_features['acc_gradient'])
                    else:
                        feats.extend([0.0] * self.cfg.window_size_minutes)
                    
            except Exception as e:
                # Fallback to zero features if there's an error
                print(f"Error reading windowed features: {e}")
                
                # Add zero windowed values (window slices)
                feats.extend([0.0] * self.cfg.window_size_minutes)  # HR window slice
                feats.extend([0.0] * self.cfg.window_size_minutes)  # ACC window slice
                
                # Add zero gradients if enabled
                if self.cfg.include_gradients:
                    feats.extend([0.0] * self.cfg.window_size_minutes)  # HR gradient window slice
                    feats.extend([0.0] * self.cfg.window_size_minutes)  # ACC gradient window slice

        # DELTA MODE: Extract charge values for delta calculation
        # Get charge at time t and t-1
        try:
            charge_t_list = row.get(charge_col, None)
            if charge_t_list is not None:
                charge_t = self._safe_extract_value(charge_t_list, idx, 69.0)
                if idx == 0:
                    # last night's charge
                    # Megha : Monitor this value and add for each user. 
                    charge_t_1 = self._safe_extract_value(yesterday_row.get(charge_col, None), -1, 69.0) # get the last night charge's last value
                else:
                    charge_t_1 = self._safe_extract_value(charge_t_list, idx-1, 69.0)
            else:
                charge_t = 69.0  # Default charge value
                charge_t_1 = 69.0

            # Normalize to [0, 1] range
            charge_t = charge_t / 100.0
            charge_t_1 = charge_t_1 / 100.0
        except Exception:
            charge_t = 69.0 / 100.0
            charge_t_1 = 69.0 / 100.0

        # Append past charge value (ABSOLUTE, not delta) as feature
        # This is what the model will use to predict the delta
        if self.cfg.use_past_charge:
            feats.append(charge_t_1)

        # DELTA MODE: Calculate delta (change in charge)
        # Target is delta = charge_t - charge_{t-1}
        charge_delta = charge_t - charge_t_1

        # Convert to numpy array with robust NaN handling
        feats = np.asarray(feats, dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure charge_delta is valid
        if not isinstance(charge_delta, (int, float)) or np.isnan(charge_delta) or np.isinf(charge_delta):
            charge_delta = 0.0  # Default delta is 0 (no change)

        # charge_t_recon = None
        # if self.cfg.charge_reconstruction:
        charge_t_recon = charge_t if isinstance(charge_t, float) else 0.0

        # Return features with past_charge (absolute) and target delta
        return np.asarray(feats, dtype=np.float32), charge_delta, charge_t_recon

    def _validate_data_length(self, user_id: str, date: pd.Timestamp, idx: int, row: pd.Series) -> bool:
        """
        Validate that the requested index is within reasonable bounds for the data.
        Returns True if data seems valid, False if there are major inconsistencies.
        """
        try:
            # Check a few key time series columns to see their lengths
            sample_cols = [col for col in self.cfg.ts_cols[:3]]  # Check first 3 columns
            data_lengths = []
            
            for col in sample_cols:
                col_data = row.get(col, None)
                if col_data is not None:
                    try:
                        if isinstance(col_data, str):
                            parsed = ast.literal_eval(col_data)
                            if isinstance(parsed, (list, tuple)):
                                data_lengths.append(len(parsed))
                        elif isinstance(col_data, (list, tuple)):
                            data_lengths.append(len(col_data))
                        elif isinstance(col_data, np.ndarray):
                            data_lengths.append(len(col_data))
                    except:
                        continue
            
            if not data_lengths:
                return False  # No valid data found
            
            min_length = min(data_lengths)
            max_length = max(data_lengths)
            
            # Log data inconsistencies for debugging
            if idx >= min_length:
                print(f"Data length warning: User {user_id}, Date {date.strftime('%Y-%m-%d')}, "
                      f"requested idx={idx}, but min data length={min_length}, max={max_length}")
                
                # Still return True - we'll handle this in _safe_extract_value
                return True
            
            # Check for severely truncated data (less than 50% of expected day length)
            if max_length < 720:  # Less than 12 hours of data
                print(f"Severely truncated data: User {user_id}, Date {date.strftime('%Y-%m-%d')}, "
                      f"max data length={max_length} (expected ~1440)")
            
            return True
            
        except Exception as e:
            print(f"Error validating data length for user {user_id}, date {date}: {e}")
            return True  # Continue processing with enhanced error handling

    # write a tensor dataset based getitem 


    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        try:
            item = self.index.iloc[i]
            user_id = str(item["userid"])
            date = pd.to_datetime(item["date"])
            idx = int(item["index"])

            # Load user file, current row, yesterday row
            today_row, yesterday_row, idx_sleep_start = self._load_user_file(user_id, date, idx)

            # Enhanced: Validate data length consistency
            if not today_row.empty:
                self._validate_data_length(user_id, date, idx, today_row)

            # Handle case where sleep start is not found (malformed sleep data)
            if idx_sleep_start is None:
                idx_sleep_start = 0  # Default to beginning of day

            # check whether current is no wear idx 
            mask = 1.0
            non_wear = self._safe_extract_value(today_row.get('timeseries.min_status_list'), idx, 0.0)
            if non_wear == 3:
                mask = 0.0  # Exclude from loss calculation - no wear detected
                # print(f"Masking no-wear data: User {user_id}, Date {date.strftime('%Y-%m-%d')}, idx {idx}")

            if idx > 1800:
                mask = 0.0  # Exclude from loss calculation - out of bounds
                # print(f"Masking out-of-bounds data: User {user_id}, Date {date.strftime('%Y-%m-%d')}, idx {idx}")
            # Check if we should mask due to windowed features
            # if self.cfg.use_windowed_features and idx < self.cfg.window_size_minutes:
            #     mask = 0.0  # Exclude from loss calculation - insufficient historical data
            
            # Additional masking for severely truncated data
            # if not today_row.empty:
            #     # Sample a time series column to check data availability
            #     sample_col = self.cfg.ts_cols[0] if self.cfg.ts_cols else None
            #     if sample_col:
            #         sample_data = today_row.get(sample_col, None)
            #         if sample_data is not None:
            #             try:
            #                 if isinstance(sample_data, str):
            #                     parsed = ast.literal_eval(sample_data)
            #                     if isinstance(parsed, (list, tuple)) and len(parsed) <= idx:
            #                         mask = 0.0  # Insufficient data - exclude from loss
            #             except:
            #                 pass

            # Determine which charge column to use based on (userid, date)
            date_str = date.strftime('%Y-%m-%d')
            if (user_id, date_str) in self.llm_charge_pairs:
                charge_col_to_use = self.cfg.charge_col_llm
            else:
                charge_col_to_use = self.cfg.charge_col

            # Build feature vector with robust error handling
            # DELTA MODE: x contains past_charge (absolute), y is delta (charge_t - charge_{t-1})
            charge_recon = -1
            x, charge_delta, charge_recon = self._get_feature_vector(today_row, yesterday_row, idx, idx_sleep_start, charge_col=charge_col_to_use)

            # Apply augmentation if enabled (augments both features and delta)
            if self.augment:
                x, charge_delta = self.augment_sample(x, charge_delta) # augmentation in charge_delta

            # optionally add idx+1 feature vector for rollout calculation during training
            # First try idx+1 from today, if out of bounds use tomorrow's first idx
            try:
                x_add_1, charge_delta_add_1, charge_recon_add_1 = self._get_feature_vector(today_row, yesterday_row, idx+1, idx_sleep_start, charge_col=charge_col_to_use)
                y_add_1 = charge_delta_add_1

                # Check if we got sentinel/default values indicating idx+1 is out of bounds
                # If charge_recon_add_1 is 0.69 (default 69/100), idx+1 might be out of bounds
                if idx+1 >= 1800:  # likely beyond current day
                    raise IndexError("idx+1 likely out of bounds")
            except:
                # idx+1 is beyond current day, try tomorrow's data from cache
                try:
                    tomorrow_date = date + pd.Timedelta(days=1)
                    tomorrow_date_str = tomorrow_date.strftime("%Y-%m-%d")
                    df = self.user_cache[user_id]
                    df_tomorrow_row = df[df["date"] == tomorrow_date_str]

                    if not df_tomorrow_row.empty:
                        tomorrow_sleep_start = self.find_sleep_start(df_tomorrow_row['timeseries.sleep_markers'])
                        x_add_1, charge_delta_add_1, charge_recon_add_1 = self._get_feature_vector(df_tomorrow_row.iloc[0], today_row.iloc[0], 0, tomorrow_sleep_start, charge_col=charge_col_to_use)
                        y_add_1 = charge_delta_add_1
                    else:
                        # Next day doesn't exist
                        x_add_1 = np.full(self.get_feature_count(), -6.0, dtype=np.float32)
                        y_add_1 = -6.0
                except:
                    # Error loading next day
                    x_add_1 = np.full(self.get_feature_count(), -6.0, dtype=np.float32)
                    y_add_1 = -6.0

            return {
                "x": torch.from_numpy(x).float(),  # Ensure float32
                "y": torch.tensor([charge_delta*100], dtype=torch.float32),  # DELTA, not absolute charge SCALED
                "mask": torch.tensor([mask], dtype=torch.float32),
                'meta': {'user_id': user_id, 'date': date.strftime('%Y-%m-%d'), 'idx': idx},
                'charge_recon': charge_recon,
                "x_add_1": torch.from_numpy(x_add_1).float(),
                "y_add_1": torch.tensor([y_add_1*100], dtype=torch.float32)
            }
            
        except Exception as e:
            # Enhanced error handling with more context
            error_context = {
                'index': i,
                'user_id': user_id if 'user_id' in locals() else 'unknown',
                'date': date.strftime('%Y-%m-%d') if 'date' in locals() else 'unknown',
                'idx': idx if 'idx' in locals() else 'unknown'
            }
            
            print(f"Warning: Malformed data at index {error_context['index']}, "
                  f"user {error_context['user_id']} date {error_context['date']}, "
                  f"time_idx {error_context['idx']}: {e}")

            # Create default feature vector using get_feature_count() to ensure consistency
            # This automatically matches the order in _get_feature_vector
            default_feature_count = self.get_feature_count()

            x = np.zeros(default_feature_count, dtype=np.float32)
            charge_delta = 0.0  # DELTA MODE: Default delta is 0 (no change)

            return {
                "x": torch.from_numpy(x).float(),  # Ensure float32
                "y": torch.tensor([charge_delta], dtype=torch.float32),  # Return delta, not absolute
                "mask": torch.tensor([0.0], dtype=torch.float32)  # Exclude malformed data from loss
            }

    def fit_norm(self, loader: DataLoader, max_batches: int = 200):
        sums, sums2, count = None, None, 0
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                x = batch["x"].float()
                if sums is None:
                    sums, sums2 = x.sum(0), (x**2).sum(0)
                else:
                    sums += x.sum(0)
                    sums2 += (x**2).sum(0)
                count += x.shape[0]
                if bi >= max_batches:
                    break
        mu = (sums / count).numpy()
        var = (sums2 / count).numpy() - mu**2
        var = np.clip(var, 1e-8, None)
        sd = np.sqrt(var)
        self.mu = torch.from_numpy(mu).float()
        self.sd = torch.from_numpy(sd).float()

    def apply_norm(self, batch):
        if self.mu is None or self.sd is None:
            return batch
        x = batch["x"].float()
        x = (x - self.mu) / self.sd
        return {"x": x, "y": batch["y"]}
    
    def set_augmentation(self, enable: bool):
        """Enable or disable augmentation (useful for train/val switching)."""
        self.augment = enable
    
    def validate_sleep_stage_data(self, row: pd.Series, idx: int, col: str) -> bool:
        """
        Validate sleep stage data for consistency and reasonable values.
        
        Args:
            row: Current data row
            idx: Current time index
            col: Column name containing sleep stage data
            
        Returns:
            bool: True if data appears valid, False otherwise
        """
        try:
            sleep_stage_data = row.get(col, None)
            if sleep_stage_data is None:
                return False
                
            # Extract value at current index
            stage_value = self._safe_extract_value(sleep_stage_data, idx, -1)
            
            # Valid sleep stages are typically 0-3 (awake, light, deep, REM)
            if stage_value < 0 or stage_value > 3:
                return False
                
            return True
            
        except Exception:
            return False
        
    def get_augmentation_stats(self):
        """Return augmentation configuration for logging."""
        stats = {
            "augmentation_enabled": self.augment,
            "augmentation_probability": self.augment_prob,
           "ts_columns": self.cfg.ts_cols,
            "windowed_features_enabled": self.cfg.use_windowed_features,
            "has_sleep_stage": any('sleep_stage' in col.lower() for col in self.cfg.ts_cols),
        }
        
        if self.cfg.use_windowed_features:
            stats.update({
                "window_size_minutes": self.cfg.window_size_minutes,
                "include_gradients": self.cfg.include_gradients,
                "use_windowed_zscore": self.cfg.use_windowed_zscore,
                "include_current_hr": self.cfg.include_current_hr,
                "total_feature_count": self.get_feature_count()
            })
        
        # Add sleep stage information if available
        if hasattr(self, 'sleep_stage_indices') and self.sleep_stage_indices:
            stats.update({
                "sleep_stage_features": len(self.sleep_stage_indices),
                "sleep_stage_augmentation_enabled": True
            })
        
        return stats
    
    def debug_windowed_columns(self, user_id: str, date_str: str):
        """
        Debug utility to check what windowed feature columns are available in the data.
        
        Args:
            user_id: User identifier  
            date_str: Date string in format "YYYY-MM-DD"
            
        Returns:
            dict: Available columns and their presence
        """
        if user_id not in self.user_cache:
            try:
                excel_path = os.path.join(self.cfg.data_dir, f"{user_id}_processed.xlsx")
                if os.path.exists(excel_path):
                    df = pd.read_excel(excel_path)
                    self.user_cache[user_id] = df
                else:
                    return {"error": f"No file found for user {user_id}"}
            except Exception as e:
                return {"error": f"Failed to load file for user {user_id}: {e}"}
        
        df = self.user_cache[user_id]
        df_row = df[df["date"] == date_str]
        
        if df_row.empty:
            return {"error": f"No data found for date {date_str}"}
        
        # Check for expected windowed columns
        window_size = self.cfg.window_size_minutes
        expected_cols = {}
        
        
        
        if self.use_windowed_zscore:
            if window_size == 15:
                expected_cols = {
                    'z_norm_hr_15': 'z_norm_hr_15' in df_row.columns,
                    'z_norm_acc_15': 'z_norm_acc_15' in df_row.columns,
                    'hr_grad_zscore_15min': 'hr_grad_zscore_15min' in df_row.columns,
                    'acc_grad_zscore_15min': 'acc_grad_zscore_15min' in df_row.columns
                }
            else:
                expected_cols = {
                    'z_norm_hr_30': 'z_norm_hr_30' in df_row.columns,
                    'z_norm_acc_30': 'z_norm_acc_30' in df_row.columns,
                    'hr_grad_zscore_30min': 'hr_grad_zscore_30min' in df_row.columns,
                    'acc_grad_zscore_30min': 'acc_grad_zscore_30min' in df_row.columns}
        else:
            expected_cols = {
                'timeseries.hr_filtered_zscore': 'timeseries.hr_filtered_zscore' in df_row.columns,
                'timeseries.acc_zscore': 'timeseries.acc_zscore' in df_row.columns,
                'hr_gradient_5min_zscore': 'hr_gradient_5min_zscore' in df_row.columns,
                'acc_gradient_5min_zscore': 'acc_gradient_5min_zscore' in df_row.columns
                }
            
        
        # Get all columns containing windowed feature patterns
        windowed_cols = [col for col in df_row.columns if any(pattern in col for pattern in 
                        ['z_norm_hr', 'z_norm_acc', 'hr_grad_zscore', 'acc_grad_zscore'])]
        
        return {
            "window_size_minutes": window_size,
            "expected_columns": expected_cols,
            "all_windowed_columns": windowed_cols,
            "total_columns": len(df_row.columns)
        }



class TorchBiochargeDataset(Dataset):
    """
    Optimized dataset that loads pre-processed torch tensors.

    Applies same normalization and augmentation as BiochargeDataset.
    Uses cfg.static_cols, cfg.sleep_cols, cfg.ts_cols for column selection.
    Returns delta predictions (charge_t - charge_{t-1}).
    """

    def __init__(self, cfg: DatasetConfig, data_fraction: float = 1.0):
        super().__init__()
        self.cfg = cfg

        # Torch data directory - torch files are stored in a separate location
        self.data_dir = cfg.torch_data_dir if hasattr(cfg, 'torch_data_dir') else cfg.data_dir

        # Load index (contains [user_id, date, idx, ...])
        self.index = pd.read_csv(cfg.index_csv)
        # remove all indices start with 0, (off when plotting traejctory)
        # if not cfg.trajectory:
        #     self.index = self.index[self.index['index'] > 0]

        # Subsample data if requested (sample users, not indices, to keep user data together)
        if data_fraction < 1.0:
            # randomly sample a fraction of rows
            n_samples = int(len(self.index) * data_fraction)
            index_sub = self.index.sample(n=n_samples, random_state=42).reset_index(drop=True)
            self.index = index_sub

        self.sub_index = self.index.copy()
        # find all
        self.index["date"] = pd.to_datetime(self.index["date"])

        # Optional zscores
        self.zdf = None
        if cfg.zscores_file and cfg.use_zscores:
            with open(cfg.zscores_file, "r") as f:
                self.zdf = json.load(f)

        # Cache for user torch files (don't reload each time)
        self.user_cache: Dict[str, Dict] = {}

        # Cache for processed dates (to avoid re-filtering)
        self.processed_dates_cache: Dict[str, set] = {}


        self.add_hr_hrr = cfg.add_hr_hrr


        # Cache for sleep start indices
        self.sleep_start_cache: Dict[str, Dict[str, int]] = {}

        # Cache for data length validation to avoid repeated warnings
        self.data_length_cache: Dict[str, Dict[str, int]] = {}

        if cfg.llm_non_existent_userids_file is not None:
            with open(cfg.llm_non_existent_userids_file, "r") as f:
                self.no_exist_userids = set(json.load(f).keys())

        # Select which (userid, date) pairs should use LLM biocharge column
        # make sure these are not in no exist userids
        self.llm_charge_pairs = set()
        if cfg.charge_col_llm is not None and cfg.llm_col_prob > 0:
            # Get unique (userid, date) combinations

            # make sure to exclude no_exist_userids
            unique_curves = self.index[['userid', 'date']].drop_duplicates()
            unique_curves = unique_curves[~unique_curves['userid'].astype(str).isin(self.no_exist_userids)]
            # Randomly sample llm_col_prob fraction of curves

            n_llm_curves = int(len(unique_curves) * cfg.llm_col_prob)
            if n_llm_curves > 0:
                sampled_curves = unique_curves.sample(n=n_llm_curves, random_state=42)
                # Store as set of (userid, date_string) tuples for fast lookup
                self.llm_charge_pairs = set(
                    (str(row['userid']), row['date'].strftime('%Y-%m-%d'))
                    for _, row in sampled_curves.iterrows()
                )
                print(f"Selected {len(self.llm_charge_pairs)} curves ({cfg.llm_col_prob*100:.1f}%) to use LLM biocharge column")

        # Normalization
        self.mu = None
        self.sd = None

        # Augmentation settings from config
        self.augment = cfg.enable_augmentation
        self.augment_prob = cfg.augment_prob

        self.use_windowed_zscore = cfg.use_windowed_zscore

        # Define indices for different time series types (will be set during first __getitem__)
        self.ts_feature_indices = None
        self.hr_indices = []
        self.hrr_indices = []
        self.stress_indices = []
        self.exercise_indices = []
        self.activity_indices = []
        self.binary_sequence_indices = []  # For exercise, nap, sleep markers, sleep_stage
        self.continuous_signal_indices = []  # For HR, stress, accelerometer
        self.jitter_indices = []  # For temporal jittering (excludes binary sequences)
        self.sleep_stage_indices = []  # For sleep stage specific handling
        self.z_data_norm = cfg.z_data_norm

        self.sleep_state = 0 
        self.nap_state = 0

        self.sleep_stage = -1

        self.use_recovery_rate_feature = cfg.use_recovery_rate_feature

        self.weighted_sleep_score = cfg.weighted_sleep_score

        self.normalization_type = cfg.normalization_type

    def get_indices(self):
        return self.sub_index
    
    def _get_hr_column_name(self, df_row=None):
        """Determine which HR column to use: prefer hr_filtered if available."""
        # Check if hr_filtered column exists in the dataframe
        if df_row is not None and hasattr(df_row, 'columns'):
            if 'timeseries.hr_filtered' in df_row.columns:
                return 'timeseries.hr_filtered'
        
        # Check if hr_filtered column exists in z-score file
        if self.zdf is not None:
            if 'global' in self.zdf and 'timeseries.hr_filtered' in self.zdf['global']:
                return 'timeseries.hr_filtered'
        
        # Default fallback to regular hr column
        return 'timeseries.hr'
    
    def filter_hr_data(self, user_id: str, date_str: str):
        """
        Filter HR data by applying Hampel-like filtering only when values are outside 
        physiologically reasonable range (30-220 bpm).
        
        Args:
            user_id: User identifier
            date_str: Date string in format "YYYY-MM-DD"
        """
        if user_id not in self.user_cache:
            return
            
        df = self.user_cache[user_id]
        df_row = df[df["date"] == date_str]
        
        if df_row.empty:
            return
            
        try:
            # Get HR data - use hr_filtered if available, otherwise timeseries.hr
            hr_column = self._get_hr_column_name(df_row)
            column_series_str = df_row[hr_column].values[0]
            column_values = self.safe_literal_eval(column_series_str)
            
            if not column_values or len(column_values) == 0:
                return
                
            # Convert to tensor
            hr_tensor = torch.tensor(column_values, dtype=torch.float32)
            
            # Create mask for values outside physiological range (30-220 bpm)
            outlier_mask = (hr_tensor < 30) | (hr_tensor > 220)
            
            if not outlier_mask.any():
                # No outliers, no filtering needed
                return
                
            # Apply Hampel filter to get replacement values for outliers
            filtered_values, _ = self.hampel_filter_torch(hr_tensor, window_size=7, n_sigmas=3.0)
            
            # Only replace values that are outside physiological range
            corrected_values = hr_tensor.clone()
            corrected_values[outlier_mask] = filtered_values[outlier_mask]
            
            # Update the cached dataframe
            df_row_index = df_row.index[0]
            hr_column = self._get_hr_column_name(df_row)
            self.user_cache[user_id].at[df_row_index, hr_column] = corrected_values.numpy().tolist()
            
        except Exception as e:
            print(f"Warning: Failed to filter HR data for user {user_id} on {date_str}: {e}")


    def hampel_filter_torch(self, x: torch.Tensor, window_size: int = 7, n_sigmas: float = 3.0):
        """
        Hampel filter for outlier removal in 1D torch tensors.
        Args:
            x: 1D torch.Tensor
            window_size: int, must be odd
            n_sigmas: threshold in number of standard deviations (MAD based)
        Returns:
            filtered_x: torch.Tensor (same shape)
            mask: torch.BoolTensor indicating where values were replaced
        """
        # Ensure input is float tensor
        if x.dtype != torch.float32:
            x = x.float()
            
        if x.ndim != 1:
            raise ValueError("Input must be a 1D tensor")

        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        
        # Handle empty or very small tensors
        if x.shape[0] == 0:
            return x, torch.zeros_like(x, dtype=torch.bool)
        
        if x.shape[0] < window_size:
            # Return original tensor if it's smaller than window size
            return x, torch.zeros_like(x, dtype=torch.bool)

        k = window_size // 2
        n = x.shape[0]

        # Create unfolding windows [N, window_size]
        # Pad at both ends to handle edges
        padded = torch.nn.functional.pad(x.unsqueeze(0).unsqueeze(0), (k, k), mode='reflect')
        windows = torch.nn.functional.unfold(
            padded,
            kernel_size=(1, window_size)
        ).squeeze(0).transpose(0, 1)  # shape: (N, window_size)

        # Median and MAD (Median Absolute Deviation)
        med = windows.median(dim=1).values
        abs_dev = (windows - med.unsqueeze(1)).abs()
        mad = abs_dev.median(dim=1).values

        # Hampel threshold
        threshold = n_sigmas * 1.4826 * mad
        diff = (x - med).abs()

        # Identify outliers
        mask = diff > threshold
        filtered_x = x.clone()
        filtered_x[mask] = med[mask]

        return filtered_x, mask

    def time_of_day_encoding_continuous(self, minute_of_day):
        """
        Continuous time-of-day encoding using sine and cosine.

        Args:
            minute_of_day: int or float in [0, 1440)
        Returns:
            np.array: shape (2,) [sin(2π * t), cos(2π * t)]

            # continuous 
        """
        # Convert minutes to fraction of the day
        fraction = (minute_of_day % 1440) / 1440.0  # ensures wrap-around safety

        # Continuous cyclic encoding
        angle = 2 * np.pi * fraction
        encoding = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
        return encoding

    def time_of_day_encoding(self, minute_of_day, num_buckets=8):
        """
        Encode time of day into num_buckets (e.g., 8 buckets for 3-hour intervals).
        
        Args:
            minute_of_day: int in [0, 1440)
            num_buckets: number of intervals per day (8 for 3-hour buckets)
        
        Returns:
            np.array: encoding vector of shape (2,)  [sin, cos]
        """
        # Map minute of day to bucket
        minutes_per_bucket = 1440 // num_buckets
        bucket_idx = minute_of_day // minutes_per_bucket
        
        # Normalize bucket index to [0, 2π)
        angle = 2 * np.pi * (bucket_idx / num_buckets)
        
        # Cyclic encoding
        encoding = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
        return  encoding.astype(np.float32)

    def positional_encoding(self, time_idx, d_model=64):
        """
        Generate positional encoding for a specific time index.
        
        Args:
            time_idx: Time index in range [0, 1440] (minutes in a day)
            d_model: Dimension of the positional encoding vector
            
        Returns:
            np.array: Positional encoding vector of shape (d_model,)

        Note: This can be changed to the week of the day as well

        """
        # Normalize time_idx to [0, 1] range
        normalized_position = time_idx / 1440.0
        
        # Create positional encoding vector
        pe = np.zeros(d_model)
        
        # Generate sinusoidal encodings
        for i in range(0, d_model, 2):
            # Use different frequencies for different dimensions
            div_term = np.exp(i * -(np.log(10000.0) / d_model))
            
            # Apply sine to even indices
            pe[i] = np.sin(normalized_position * div_term)
            
            # Apply cosine to odd indices (if within bounds)
            if i + 1 < d_model:
                pe[i + 1] = np.cos(normalized_position * div_term)
        
        return pe.astype(np.float32)
    
    def time_encoding_for_charge(self, minute_of_day):
        """
        Time encoding aligned with biocharge circadian model.
        minute_of_day: int or float in [0, 1440)
        """
        minute_of_day = minute_of_day % 1440  # safety wrap

        # Normalized fraction of day
        fraction = minute_of_day / 1440.0

        # Standard cyclic encoding
        sin_24h = np.sin(2 * np.pi * fraction)
        cos_24h = np.cos(2 * np.pi * fraction)

        # Circadian model (minute-based)
        circadian_output = (
            np.cos(2 * np.pi * (minute_of_day - 1080) / 1440) +
            0.5 * np.cos(2 * np.pi * (minute_of_day - 1260) / 720)
        )

        return np.array(
            [sin_24h, cos_24h, circadian_output],
            dtype=np.float32
        )

    def time_encoding_for_charge_torch(self, minute_of_day):
        """
        Torch version of time encoding aligned with biocharge circadian model.
        minute_of_day: int, float, or torch.Tensor in [0, 1440)
        Returns: torch.FloatTensor of shape (3,)
        """
        # import torch
        pi = torch.tensor(np.pi) if not hasattr(torch, 'pi') else torch.pi
        if not torch.is_tensor(minute_of_day):
            minute_of_day = torch.tensor(minute_of_day, dtype=torch.float32)
        minute_of_day = minute_of_day % 1440
        fraction = minute_of_day / 1440.0
        sin_24h = torch.sin(2 * pi * fraction)
        cos_24h = torch.cos(2 * pi * fraction)
        circadian_output = (
            torch.cos(2 * pi * (minute_of_day - 1080) / 1440) +
            0.5 * torch.cos(2 * pi * (minute_of_day - 1260) / 720)
        )
        return torch.stack([sin_24h, cos_24h, circadian_output]).to(torch.float32)

    def time_in_state(self, minute, sleep_start, nap_start, wake_up_time=0, sleep_state=0, nap_state=0,
                    wake_up_scale=960.0, sleep_scale=480.0, nap_scale=90.0):
        '''Time since awake, sleep, or nap, normalized by typical duration. 
        sleep_state: 0=awake, 1=sleep; nap_state: 0=not nap, 1=nap'''
        if int(nap_state) == 1:
            t = torch.clamp(torch.tensor(minute - nap_start, dtype=torch.float32), min=0.0)
            scale = nap_scale
        elif int(sleep_state) == 0:
            t = torch.clamp(torch.tensor(minute - wake_up_time, dtype=torch.float32), min=0.0)
            scale = wake_up_scale
        else:
            t = torch.clamp(torch.tensor(minute - sleep_start, dtype=torch.float32), min=0.0)
            scale = sleep_scale
        norm = torch.log1p(t) / torch.log1p(torch.tensor(scale, dtype=torch.float32))
        return norm
    
    def get_stage_recovery_rate(self, sleep_stage, sleep_state):
        # Torch-friendly: support both scalar and tensor inputs
        # if torch.is_tensor(sleep_state):
        #     # Awake: recovery rate is 0.0 where sleep_state == 0
        #     stage_multiplier = (
        #         (sleep_stage == 0).to(torch.float32) * 3 +
        #         (sleep_stage == 1).to(torch.float32) * 9 +
        #         (sleep_stage == 2).to(torch.float32) * 12 +
        #         (~((sleep_stage == 0) | (sleep_stage == 1) | (sleep_stage == 2))).to(torch.float32) * 12
        #     )
        #     # Avoid division by zero
        #     stage_multiplier = torch.where(stage_multiplier == 0, torch.ones_like(stage_multiplier), stage_multiplier)
        #     rate = 1.0 / stage_multiplier
        #     # Set to 0.0 where sleep_state == 0 (awake)
        #     return torch.where(sleep_state == 0, torch.zeros_like(rate), rate)
        # else:
        if sleep_state == 0:  # Awake
            return 0.0

        # mental : {0: 3, 1: 9, 2: 12} # Deep , REM , Light 
        # physical : {0:1, 1:2, 2:1.5}: # Deep , REM , Light 

        # inverses: 
        # mental : (0.33, 0.111, 0.083)
        # physical: (1.0, 0.5, 0.665)

        # averages of inverses:
        # (0.33+1.0)/2, (0.111+0.5)/2, (0.083+0.665)/2 = 0.665, 0.305, 0.374

        # multiply 
        
        # in the anlytical model multipliers are different in mental and physical model
        # here we are combining hte two
        # stage_multiplier = {0: 3, 1: 9, 2: 12}.get(sleep_stage, 12): (0.33, 0.111, 0.083)
        # 1, 1/1.5, 1/2 = 1.0, 0.665, 0.5
        # (1+0.33)/2, (0.665+0.111)/2, (0.5+0.083)/2
        # 0.665, 0.388, 0.291
        # Mental: 
        # Deep, REM, Light
        

        # {0: 0.665, 1: 0.305, 2: 0.377}  # Deep, REM, Light
        # return 1.0 / stage_multiplier
        stage_multiplier = {0: 0.665, 1: 0.388, 2: 0.291} # old

        # stage_multiplier = {0: 0.665, 1: 0.305, 2: 0.374} # new
        # return stage_multiplier.get(sleep_stage, 0.305) # new
        return stage_multiplier.get(sleep_stage, 0.388) # old
    
    def _read_windowed_features(self, row: pd.Series, idx: int):
        """
        Read pre-calculated windowed features from the data file for a given row and time index.
        
        Key improvements:
        - Automatically selects correct columns based on window_size_minutes (15min vs 30min)
        - Supports include_current_hr flag to use current HR or previous HR (t-1)
        - Robust error handling with debugging information
        - Returns z-score normalized features from pre-calculated data
        
        Args:
            row: Current data row
            idx: Current time index
            
        Returns:
            dict: Dictionary containing windowed features read from file
                - hr_window: Single HR value (current or t-1 based on config)
                - acc_window: Single accelerometer value
                - hr_gradient: HR gradient (if include_gradients=True)
                - acc_gradient: Accelerometer gradient (if include_gradients=True)
        """
        features = {
            'hr_window': [],
            'acc_window': [],
            'hr_gradient': [],
            'acc_gradient': []
        }
        
        # Determine window size to use (15 or 30 minutes)
        window_size = self.cfg.window_size_minutes
        
        # Determine base data columns (needed for fallback)
        base_hr_col = self._get_hr_column_name(row)
        hr_data_column = base_hr_col
        acc_data_column = 'timeseries.acc_magnitude'
        hr_grad_data_column = 'hr_gradient_5min'
        acc_grad_data_column = 'acc_gradient_5min'

        # Choose appropriate column names based on window size
        if self.use_windowed_zscore:
            if window_size == 15:
                z_norm_hr_col = 'z_norm_hr_15'
                z_norm_acc_col = 'z_norm_acc_15'
                hr_grad_col = 'hr_grad_zscore_15min'
                acc_grad_col = 'acc_grad_zscore_15min'
            else:  # Default to 30 minutes or any other size
                z_norm_hr_col = 'z_norm_hr_30'
                z_norm_acc_col = 'z_norm_acc_30'
                hr_grad_col = 'hr_grad_zscore_30min'
                acc_grad_col = 'acc_grad_zscore_30min'
        else:
            # global normalized data columns - use dynamic HR column determination
            z_norm_hr_col = f'{base_hr_col}_zscore'
            z_norm_acc_col = 'timeseries.acc_zscore'
            hr_grad_col = 'hr_gradient_5min_zscore'
            acc_grad_col = 'acc_grad_zscore_5min'
            

        
        try:
            # Read z-score normalized HR window
            if z_norm_hr_col in row:
                hr_data = row.get(z_norm_hr_col, None)
                if hr_data is not None:
                    hr_values = self.safe_literal_eval(hr_data.values[0], default_value=[], column_name=z_norm_hr_col)
                    if isinstance(hr_values, list) and len(hr_values) > 0:
                        # Enhanced: Handle cases where idx >= len(hr_values)
                        effective_idx = min(idx, len(hr_values) - 1)
                        
                        if self.cfg.include_current_hr:
                            # Extract values from past window 15 or 30 min
                            start_idx = max(0, effective_idx - window_size)
                            end_idx = effective_idx
                            past_window = hr_values[start_idx:end_idx + 1]  # Include current
                            
                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with the first available value if we don't have enough history
                                padding_value = past_window[0] if past_window else 0.0
                                padding_needed = window_size - len(past_window)
                                past_window = [padding_value] * padding_needed + past_window
                            
                            features['hr_window'] = past_window[:window_size]  # Ensure exact size
                        else:
                            # Extract previous time point (t-1) to exclude current HR
                            prev_idx = max(0, effective_idx - 1)
                            start_idx = max(0, prev_idx - window_size)
                            past_window = hr_values[start_idx:prev_idx]
                            
                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with the first available value if we don't have enough history
                                padding_value = past_window[0] if past_window else 0.0
                                padding_needed = window_size - len(past_window)
                                past_window = [padding_value] * padding_needed + past_window
                                
                            features['hr_window'] = past_window[:window_size]  # Ensure exact size
                    else:
                        features['hr_window'] = [0.0] * window_size

            else:
                # Fallback: apply z_score normalization on the fly if pre-calculated column not found
                print(f"Warning: Pre-calculated column '{z_norm_hr_col}' not found, using on-the-fly normalization from '{hr_data_column}'")
                hr_mean, hr_std = 0.0, 1.0
                if self.zdf and 'global' in self.zdf and hr_data_column in self.zdf['global']:
                    hr_mean, hr_std = self.zdf['global'][hr_data_column]['mean'], self.zdf['global'][hr_data_column]['std']
                if hr_data_column in row:

                    hr_data = row.get(hr_data_column, None)
                    if hr_data is not None:
                        hr_values = self.safe_literal_eval(hr_data.values[0], default_value=[], column_name=hr_data_column)
                        if isinstance(hr_values, list) and len(hr_values) > 0:
                            # Enhanced: Handle cases where idx >= len(hr_values)
                            effective_idx = min(idx, len(hr_values) - 1)
                            
                            if self.cfg.include_current_hr:
                                # Extract values from past window 15 or 30 min
                                start_idx = max(0, effective_idx - window_size)
                                end_idx = effective_idx
                                past_window = hr_values[start_idx:end_idx + 1]  # Include current
                                
                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with the first available value if we don't have enough history
                                    padding_value = past_window[0] if past_window else 0.0
                                    padding_needed = window_size - len(past_window)
                                    past_window = [padding_value] * padding_needed + past_window

                                # Normalize with Z_score
                                features['hr_window'] = [(x - hr_mean) / hr_std for x in past_window[:window_size]]

                            else:
                                # Extract previous time point (t-1) to exclude current HR
                                prev_idx = max(0, effective_idx - 1)
                                start_idx = max(0, prev_idx - window_size)
                                past_window = hr_values[start_idx:prev_idx]
                                
                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with the first available value if we don't have enough history
                                    padding_value = past_window[0] if past_window else 0.0
                                    padding_needed = window_size - len(past_window)
                                    past_window = [padding_value] * padding_needed + past_window
                                    
                                features['hr_window'] = [(x - hr_mean) / hr_std for x in past_window[:window_size]]  # Ensure exact size

            
            # Read z-score normalized accelerometer window
            
            if z_norm_acc_col in row:
                acc_data = row.get(z_norm_acc_col, None)
                if acc_data is not None:
                    acc_values = self.safe_literal_eval(acc_data.values[0], default_value=[], column_name=z_norm_acc_col)
                    if isinstance(acc_values, list) and len(acc_values) > 0:
                        # Enhanced: Handle cases where idx >= len(acc_values)
                        effective_idx = min(idx, len(acc_values) - 1)

                        if self.cfg.include_current_hr:
                            # Extract values from past window including current
                            start_idx = max(0, effective_idx - window_size)
                            end_idx = effective_idx
                            past_window = acc_values[start_idx:end_idx + 1]  # Include current

                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with the first available value if we don't have enough history
                                padding_value = past_window[0] if past_window else 0.0
                                padding_needed = window_size - len(past_window)
                                past_window = [padding_value] * padding_needed + past_window

                            features['acc_window'] = past_window[:window_size]  # Ensure exact size
                        else:
                            # Extract previous time point (t-1) to exclude current ACC
                            prev_idx = max(0, effective_idx - 1)
                            start_idx = max(0, prev_idx - window_size)
                            past_window = acc_values[start_idx:prev_idx]

                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with the first available value if we don't have enough history
                                padding_value = past_window[0] if past_window else 0.0
                                padding_needed = window_size - len(past_window)
                                past_window = [padding_value] * padding_needed + past_window

                            features['acc_window'] = past_window[:window_size]  # Ensure exact size
                    else:
                        features['acc_window'] = [0.0] * window_size
            
            else:
                acc_mean, acc_std = 0.0, 1.0
                if self.zdf and 'global' in self.zdf and acc_data_column in self.zdf['global']:
                    acc_mean, acc_std = self.zdf['global'][acc_data_column]['mean'], self.zdf['global'][acc_data_column]['std']
                
                if acc_data_column in row:
                    acc_data = row.get(acc_data_column, None)
                    if acc_data is not None:
                        acc_values = self.safe_literal_eval(acc_data.values[0], default_value=[], column_name=acc_data_column)
                        if isinstance(acc_values, list) and len(acc_values) > 0:
                            # Enhanced: Handle cases where idx >= len(acc_values)
                            effective_idx = min(idx, len(acc_values) - 1)

                            if self.cfg.include_current_hr:
                                # Extract values from past window including current
                                start_idx = max(0, effective_idx - window_size)
                                end_idx = effective_idx
                                past_window = acc_values[start_idx:end_idx + 1]  # Include current

                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with the first available value if we don't have enough history
                                    padding_value = past_window[0] if past_window else 0.0
                                    padding_needed = window_size - len(past_window)
                                    past_window = [padding_value] * padding_needed + past_window

                                # Normalize with Z_score
                                features['acc_window'] = [(x - acc_mean) / acc_std for x in past_window[:window_size]]  # Ensure exact size
                            else:
                                # Extract previous time point (t-1) to exclude current ACC
                                prev_idx = max(0, effective_idx - 1)
                                start_idx = max(0, prev_idx - window_size)
                                past_window = acc_values[start_idx:prev_idx]

                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with the first available value if we don't have enough history
                                    padding_value = past_window[0] if past_window else 0.0
                                    padding_needed = window_size - len(past_window)
                                    past_window = [padding_value] * padding_needed + past_window

                                # Normalize with Z_score
                                features['acc_window'] = [(x - acc_mean) / acc_std for x in past_window[:window_size]]  # Ensure exact size
                        else:
                            features['acc_window'] = [0.0] * window_size
            # Read gradient features if enabled
            if self.cfg.include_gradients:
                # Read HR gradient
                if hr_grad_col in row:
                    hr_grad_data = row.get(hr_grad_col, None)
                    if hr_grad_data is not None:
                        hr_grad_values = self.safe_literal_eval(hr_grad_data.values[0], default_value=[], column_name=hr_grad_col)
                        if isinstance(hr_grad_values, list) and len(hr_grad_values) > 0:
                            # Enhanced: Handle cases where idx >= len(hr_grad_values)
                            effective_idx = min(idx, len(hr_grad_values) - 1)
                            start_idx = max(0, effective_idx - window_size)
                            past_window = hr_grad_values[start_idx:effective_idx]
                            
                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with zero for gradients (neutral gradient)
                                padding_needed = window_size - len(past_window)
                                past_window = [0.0] * padding_needed + past_window
                                
                            features['hr_gradient'] = past_window[:window_size]  # Ensure exact size
                        else:
                            features['hr_gradient'] = [0.0] * window_size
                else:
                    hr_grad_mean, hr_grad_std = 0.0, 1.0
                    if hr_grad_data_column in self.zdf['global']:
                        hr_grad_mean, hr_grad_std = self.zdf['global'][hr_grad_data_column]['mean'], self.zdf['global'][hr_grad_data_column]['std']
                    if hr_grad_data_column in row:
                        hr_grad_data = row.get(hr_grad_data_column, None)
                        if hr_grad_data is not None:
                            hr_grad_values = self.safe_literal_eval(hr_grad_data.values[0], default_value=[], column_name=hr_grad_data_column)
                            if isinstance(hr_grad_values, list) and len(hr_grad_values) > 0:
                                # Enhanced: Handle cases where idx >= len(hr_grad_values)
                                effective_idx = min(idx, len(hr_grad_values) - 1)
                                start_idx = max(0, effective_idx - window_size)
                                past_window = [(x-hr_grad_mean)/hr_grad_std for x in hr_grad_values[start_idx:effective_idx]]
                                
                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with zero for gradients (neutral gradient)
                                    padding_needed = window_size - len(past_window)
                                    past_window = [0.0] * padding_needed + past_window
                                    
                                features['hr_gradient'] = [(x-hr_grad_mean)/hr_grad_std for x in past_window[:window_size]]  # Ensure exact size
                            else:
                                features['hr_gradient'] = [0.0] * window_size
                if acc_grad_col in row:
                    acc_grad_data = row.get(acc_grad_col, None)
                    if acc_grad_data is not None:
                        acc_grad_values = self.safe_literal_eval(acc_grad_data.values[0], default_value=[], column_name=acc_grad_col)
                        if isinstance(acc_grad_values, list) and len(acc_grad_values) > 0:
                            # Enhanced: Handle cases where idx >= len(acc_grad_values)
                            effective_idx = min(idx, len(acc_grad_values) - 1)
                            start_idx = max(0, effective_idx - window_size)
                            past_window = acc_grad_values[start_idx:effective_idx]
                            
                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with zero for gradients (neutral gradient)
                                padding_needed = window_size - len(past_window)
                                past_window = [0.0] * padding_needed + past_window
                                
                            features['acc_gradient'] = past_window[:window_size]  # Ensure exact size
                        else:
                            features['acc_gradient'] = [0.0] * window_size
                else:
                    acc_grad_mean, acc_grad_std = 0.0, 1.0
                    if acc_grad_data_column in self.zdf['global']:
                        acc_grad_mean, acc_grad_std = self.zdf['global'][acc_grad_data_column]['mean'], self.zdf['global'][acc_grad_data_column]['std']
                    if acc_grad_data_column in row:
                        acc_grad_data = row.get(acc_grad_data_column, None)
                        if acc_grad_data is not None:
                            acc_grad_values = self.safe_literal_eval(acc_grad_data.values[0], default_value=[], column_name=acc_grad_data_column)
                            if isinstance(acc_grad_values, list) and len(acc_grad_values) > 0:
                                # Enhanced: Handle cases where idx >= len(acc_grad_values)
                                effective_idx = min(idx, len(acc_grad_values) - 1)
                                start_idx = max(0, effective_idx - window_size)
                                past_window = [(x - acc_grad_mean)/acc_grad_std for x in acc_grad_values[start_idx:effective_idx]]
                                
                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with zero for gradients (neutral gradient)
                                    padding_needed = window_size - len(past_window)
                                    past_window = [0.0] * padding_needed + past_window
                                    
                                features['acc_gradient'] = [(x - acc_grad_mean)/acc_grad_std for x in past_window[:window_size]]  # Ensure exact size
                            else:
                                features['acc_gradient'] = [0.0] * window_size
        except Exception as e:
            print(f"Warning: Error reading windowed features for window_size={window_size}min: {e}")
            print(f"Available columns: {list(row.index)}")
            # Return default values on error
            features = {
                'hr_window': [0.0],
                'acc_window': [0.0], 
                'hr_gradient': [0.0],
                'acc_gradient': [0.0]
            }
        
        return features 

    def _extract_windowed_features(self, row: pd.Series, idx: int, window_size: int = 30):
        """
        Extract HR and accelerometer features from the past window_size minutes.
        
        Args:
            row: Current data row
            idx: Current time index
            window_size: Window size in minutes (default 30)
            
        Returns:
            dict: Dictionary containing windowed features and gradients
                - hr_window: Array of HR values over the window
                - acc_window: Array of accelerometer values over the window
                - hr_gradient: Linear trend in HR (change per minute)
                - acc_gradient: Linear trend in accelerometer (change per minute)
                - hr_mean, hr_std: HR statistics over the window
                - acc_mean, acc_std: Accelerometer statistics over the window
        """
        features = {
            'hr_window': [],
            'acc_window': [],
            'hr_gradient': 0.0,
            'acc_gradient': 0.0,
            'hr_mean': 0.0,
            'hr_std': 0.0,
            'acc_mean': 0.0,
            'acc_std': 0.0
        }
        
        # Determine start index for the window
        start_idx = max(0, idx - window_size)
        
        # Find HR and accelerometer columns
        hr_cols = [col for col in self.cfg.ts_cols if 'hr' in col.lower() and 'hrr' not in col.lower()]
        acc_cols = [col for col in self.cfg.ts_cols if 'acc' in col.lower()]
        
        # Extract HR window (vectorized)
        hr_values = []
        if hr_cols:
            for hr_col in hr_cols:
                hr_data = row.get(hr_col, None)
                if hr_data is not None:
                    # Vectorized extraction using list comprehension
                    hr_values = [self._safe_extract_value(hr_data, i, 0.0) 
                               for i in range(start_idx, idx + 1)]
                    break  # Use first available HR column
        
        # Extract accelerometer window (vectorized)
        acc_values = []
        if acc_cols:
            for acc_col in acc_cols:
                acc_data = row.get(acc_col, None)
                if acc_data is not None:
                    # Vectorized extraction using list comprehension
                    acc_values = [self._safe_extract_value(acc_data, i, 0.0) 
                                for i in range(start_idx, idx + 1)]
                    break  # Use first available accelerometer column
        
        # Pad windows if necessary
        target_length = window_size + 1  # +1 to include current time point
        if len(hr_values) < target_length:
            hr_values = [0.0] * (target_length - len(hr_values)) + hr_values
        if len(acc_values) < target_length:
            acc_values = [0.0] * (target_length - len(acc_values)) + acc_values
        
        # Store windowed data
        features['hr_window'] = np.array(hr_values[-target_length:], dtype=np.float32)
        features['acc_window'] = np.array(acc_values[-target_length:], dtype=np.float32)
        
        # Calculate statistics
        if len(features['hr_window']) > 0:
            features['hr_mean'] = np.mean(features['hr_window'])
            features['hr_std'] = np.std(features['hr_window']) if len(features['hr_window']) > 1 else 0.0
        
        if len(features['acc_window']) > 0:
            features['acc_mean'] = np.mean(features['acc_window'])
            features['acc_std'] = np.std(features['acc_window']) if len(features['acc_window']) > 1 else 0.0
        
        # Calculate gradients (slope of linear regression) - per minute rate of change
        if self.cfg.include_gradients and len(features['hr_window']) > 1:
            # Time points represent minutes from start of window to current time
            # Each index corresponds to 1 minute, so gradient will be per minute
            time_points_minutes = np.arange(len(features['hr_window']))  # [0, 1, 2, ..., window_size] minutes
            
            # HR gradient (rolling window linear regression slope)
            try:
                # Use rolling window approach with convolution for efficient gradient calculation
                hr_tensor = torch.from_numpy(features['hr_window']).float()
                gradient_window = 5  # 5-minute rolling window for gradient calculation
                
                if len(hr_tensor) >= gradient_window:
                    # Precompute constants for linear regression
                    x = torch.arange(gradient_window, dtype=hr_tensor.dtype)
                    x_sum = x.sum()
                    x2_sum = (x ** 2).sum()
                    denominator = gradient_window * x2_sum - x_sum ** 2
                    
                    if denominator != 0:
                        # Reshape for conv1d (batch, channels, length)
                        hr_reshaped = hr_tensor.view(1, 1, -1)
                        
                        # Compute y_sum using conv1d with ones kernel
                        kernel_ones = torch.ones(1, 1, gradient_window, dtype=hr_tensor.dtype)
                        y_sum = torch.nn.functional.conv1d(hr_reshaped, kernel_ones).squeeze()
                        
                        # Compute xy_sum using conv1d with reversed x kernel
                        kernel_x = x.flip(0).view(1, 1, -1)
                        xy_sum = torch.nn.functional.conv1d(hr_reshaped, kernel_x).squeeze()
                        
                        # Ensure tensors are at least 1D for proper indexing
                        if y_sum.dim() == 0:
                            y_sum = y_sum.unsqueeze(0)
                        if xy_sum.dim() == 0:
                            xy_sum = xy_sum.unsqueeze(0)
                        
                        # Calculate gradients for all windows
                        gradients = (gradient_window * xy_sum - x_sum * y_sum) / denominator
                        
                        # Store all gradients as a list/array for feature extraction
                        features['hr_gradient'] = gradients.detach().cpu().numpy() if gradients.dim() > 0 else np.array([0.0])
                    else:
                        features['hr_gradient'] = np.array([0.0])
                else:
                    # Fallback for short windows
                    features['hr_gradient'] = np.array([0.0])
                    
            except (RuntimeError, ValueError, IndexError):
                features['hr_gradient'] = np.array([0.0])
            
            # Accelerometer gradient (rolling window linear regression slope)
            try:
                # Use rolling window approach with convolution for efficient gradient calculation
                acc_tensor = torch.from_numpy(features['acc_window']).float()
                gradient_window = 5  # 5-minute rolling window for gradient calculation
                
                if len(acc_tensor) >= gradient_window:
                    # Precompute constants for linear regression
                    x = torch.arange(gradient_window, dtype=acc_tensor.dtype)
                    x_sum = x.sum()
                    x2_sum = (x ** 2).sum()
                    denominator = gradient_window * x2_sum - x_sum ** 2
                    
                    if denominator != 0:
                        # Reshape for conv1d (batch, channels, length)
                        acc_reshaped = acc_tensor.view(1, 1, -1)
                        
                        # Compute y_sum using conv1d with ones kernel
                        kernel_ones = torch.ones(1, 1, gradient_window, dtype=acc_tensor.dtype)
                        y_sum = torch.nn.functional.conv1d(acc_reshaped, kernel_ones).squeeze()
                        
                        # Compute xy_sum using conv1d with reversed x kernel
                        kernel_x = x.flip(0).view(1, 1, -1)
                        xy_sum = torch.nn.functional.conv1d(acc_reshaped, kernel_x).squeeze()
                        
                        # Ensure tensors are at least 1D for proper indexing
                        if y_sum.dim() == 0:
                            y_sum = y_sum.unsqueeze(0)
                        if xy_sum.dim() == 0:
                            xy_sum = xy_sum.unsqueeze(0)
                        
                        # Calculate gradients for all windows
                        gradients = (gradient_window * xy_sum - x_sum * y_sum) / denominator
                        
                        # Store all gradients as a list/array for feature extraction
                        features['acc_gradient'] = gradients.detach().cpu().numpy() if gradients.dim() > 0 else np.array([0.0])
                    else:
                        features['acc_gradient'] = np.array([0.0])
                else:
                    # Fallback for short windows
                    features['acc_gradient'] = np.array([0.0])
                    
            except (RuntimeError, ValueError, IndexError):
                features['acc_gradient'] = np.array([0.0])
        
        return features
    

    def _apply_windowed_zscore(self, window_data: np.ndarray, epsilon: float = 1e-8):
        """
        Apply z-score normalization to windowed data using the window's own statistics.
        
        Args:
            window_data: Array of windowed values
            epsilon: Small value to prevent division by zero
            
        Returns:
            np.ndarray: Z-score normalized window data
        """
        if len(window_data) <= 1:
            return window_data
        
        window_mean = np.mean(window_data)
        window_std = np.std(window_data)
        
        if window_std < epsilon:
            return np.zeros_like(window_data)
        
        return (window_data - window_mean) / window_std
    
    def safe_literal_eval(self, data_string, default_value=None, column_name="unknown"):
        """
        Safely evaluate a string containing a Python literal expression.
        
        Args:
            data_string: String to evaluate
            default_value: Value to return if parsing fails (default: empty list)
            column_name: Name of the column being parsed (for error logging)
        
        Returns:
            Parsed data or default_value if parsing fails
        """
        if default_value is None:
            default_value = []
        
        try:
            # Handle NaN or None values
            if pd.isna(data_string) or data_string is None:
                # print(f"Warning: NaN or None value found in column '{column_name}', using default value")
                return default_value
            
            # Convert to string if not already
            if not isinstance(data_string, str):
                data_string = str(data_string)
            
            # Try to parse the string
            return ast.literal_eval(data_string)
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Warning: Failed to parse data in column '{column_name}': {e}. Using default value.")
            return default_value
    
    def get_feature_count(self):
        """
        Calculate the total number of features that will be generated.
        Useful for model initialization.

        IMPORTANT: Order must match _get_feature_vector exactly:
        1. Static + Sleep features
        2. Timeseries features
        3. Positional encoding (if enabled)
        4. Windowed features (if enabled)
        5. Past charge (if enabled)

        Returns:
            int: Total feature count
        """
        # 1. Static + Sleep + Timeseries
        # Account for BMI replacement: if both height and weight are present,
        # they are replaced by a single BMI feature in _get_feature_vector
        static_count = len(self.cfg.static_cols)
        if 'height' in self.cfg.static_cols and 'weight' in self.cfg.static_cols:
            static_count -= 1  # BMI replaces both height and weight (2 → 1)

        feature_count = (static_count +
                        len(self.cfg.sleep_cols) +
                        len(self.cfg.ts_cols))

        # 2. Positional encoding (comes BEFORE windowed features)
        if self.cfg.use_positional_encoding:
            if self.cfg.pos_encoding_type == "sinusoidal":
                feature_count += self.cfg.pos_encoding_dim
            elif self.cfg.pos_encoding_type == "biocharge_circadian":
                feature_count += 4  # 3 circadian features + 1 time_state feature
            else:  # "time_of_day"
                feature_count += 2  # time_of_day_encoding returns 2 elements [sin, cos]

        # 3. Windowed features (comes AFTER positional encoding)
        if self.cfg.use_windowed_features:
            # Windowed features add window_size_minutes features for HR and ACC each
            # The actual implementation returns slices of length window_size_minutes
            feature_count += self.cfg.window_size_minutes  # HR window slice
            feature_count += self.cfg.window_size_minutes  # ACC window slice

            if self.cfg.include_gradients:
                # Gradient features also add window_size_minutes features each
                feature_count += self.cfg.window_size_minutes  # HR gradient window slice
                feature_count += self.cfg.window_size_minutes  # ACC gradient window slice

        # 4. Past charge (comes LAST, after windowed features)
        if self.cfg.use_past_charge:
            feature_count += 1

        return feature_count

    def flip_label(self, y, p=0.02):
        if np.random.rand() < p:
            return 1 - y
        return y
    
    def augment_sleep_stage_transitions(self, x_aug, sleep_stage_indices):
        """
        Apply simple signal corruption to sleep stage data.
        
        Args:
            x_aug: Feature vector to augment
            sleep_stage_indices: Indices of sleep stage features
        """
        if not sleep_stage_indices:
            return x_aug
            
        for idx in sleep_stage_indices:
            if idx < len(x_aug):
                current_stage = x_aug[idx]
                
                # Simple noise corruption with 10% probability
                if np.random.random() < 0.1:
                    # Add small amount of noise
                    noise = np.random.normal(0, 0.05)  # 5% noise
                    x_aug[idx] = np.clip(current_stage + noise, 0.0, 1.0)
        
        return x_aug
    
    def get_sleep_stage_info(self):
        """
        Get information about sleep_stage columns and their configuration.
        
        Returns:
            dict: Sleep stage configuration information
        """
        sleep_stage_cols = [col for col in self.cfg.ts_cols if 'sleep_stage' in col.lower()]
        
        return {
            "sleep_stage_columns": sleep_stage_cols,
            "sleep_stage_count": len(sleep_stage_cols),
            "sleep_stage_indices_initialized": hasattr(self, 'sleep_stage_indices') and bool(self.sleep_stage_indices),
            "augmentation_includes_sleep_stage": self.augment and len(sleep_stage_cols) > 0
        }


    def augment_sample(self, x, y):
        """
        Augment time series features with 30% probability.
        Applies physiologically realistic augmentations for different signal types.
        """
        if not self.augment or np.random.random() > self.augment_prob:
            return x, y

        # Initialize feature indices if not done
        self._initialize_feature_indices(len(x))

        # Skip augmentation if hrr_raw is -6 (missing value)
        if self.hr_indices:
            for idx in self.hr_indices:
                if idx < len(x) and x[idx] == -6:
                    return x, y

        # Create a copy to avoid modifying original
        x_aug = x.copy()

        # === STATIC FEATURE AUGMENTATION ===
        # Age: ±3 years (normalized by 80, so ±0.0375)
        # BMI (weight proxy): ±4kg ≈ ±1.4 BMI points (normalized by 30, so ±0.047)
        if np.random.random() < 0.3:
            static_idx = 0
            for col in self.cfg.static_cols:
                if col == 'height':
                    continue  # Height is skipped, BMI added at weight
                if 'age' in col.lower():
                    if x_aug[static_idx] != -6:  # Skip if missing
                        age_noise = np.random.uniform(-0.0375, 0.0375)  # ±3 years
                        x_aug[static_idx] = x_aug[static_idx] + age_noise
                elif col == 'weight':
                    # BMI is stored here (replaces height+weight)
                    if x_aug[static_idx] != -6:  # Skip if missing
                        bmi_noise = np.random.uniform(-0.047, 0.047)  # ±4kg equivalent
                        x_aug[static_idx] = x_aug[static_idx] + bmi_noise
                static_idx += 1


        # === BINARY SEQUENCE AUGMENTATION ===
        # Only for exercise features (NOT sleep_stage, nap_state, sleep_markers - these are protected)
        # Reasoning: sleep_stage, nap_state, and sleep_markers represent ground-truth physiological
        # states that should not be artificially modified. Modifying them would corrupt the
        # relationship between these states and the target variable (charge delta).
        if self.binary_sequence_indices and np.random.random() < 0.4:
            for idx in self.binary_sequence_indices:
                # Skip protected features (sleep_stage, nap_state, sleep_markers)
                if idx in self.protected_state_indices:
                    continue
                original_val = x_aug[idx]
                x_aug[idx] = self.flip_label(original_val, p = 0.2)

        # === HEART RATE AUGMENTATION ===
        # # For continuous HR signals - physiologically realistic variations
        if self.hr_indices and np.random.random() < 0.7:
            for idx in self.hr_indices:
                hr_val = x_aug[idx]

                # # Strategy 1: Heart Rate Variability (HRV simulation)
                # # Normal HR variation is ±2-5 bpm (roughly 2-3% of normalized signal)
                # if np.random.random() < 0.8:
                #     hrv_noise = np.random.normal(0, 0.02)  # 2.5% std
                #     x_aug[idx] = hr_val + hrv_noise
            
                
                # Strategy 3: Measurement Artifact Simulation
                # Simulate sensor artifacts (brief spikes/drops)
                if np.random.random() < 0.1:
                    artifact_factor = np.random.choice([0.9, 1.1])  # 10% spike or drop
                    x_aug[idx] = hr_val * artifact_factor
        
        # === CONTINUOUS SIGNAL AUGMENTATION ===
        # For stress, accelerometer, TRIMP etc.
        if self.continuous_signal_indices and np.random.random() < 0.6:
            for idx in self.continuous_signal_indices:
                # if idx not in self.hr_indices:  # Skip HR (already handled above)
                signal_val = x_aug[idx]
                
                # Strategy 1: Sensor Noise
                if np.random.random() < 0.7:
                    noise_std = 0.03 if idx in self.stress_indices else 0.025
                    sensor_noise = np.random.normal(0, noise_std)
                    x_aug[idx] = signal_val + sensor_noise
                
                # Strategy 2: Calibration Drift
                elif np.random.random() < 0.2:
                    drift_factor = np.random.uniform(0.95, 1.05)
                    x_aug[idx] = signal_val * drift_factor
        
        # === TRIMP SPECIFIC AUGMENTATION ===
        if self.exercise_indices and np.random.random() < 0.4:
            for idx in self.exercise_indices:
                exercise_val = x_aug[idx]

                # Strategy 1: Training Load Variation (daily performance varies ±15%)
                if np.random.random() < 0.8:
                    performance_factor = np.random.uniform(0.85, 1.15)
                    x_aug[idx] = exercise_val * performance_factor



        # === MISSING VALUE AUGMENTATION (Non-windowed HR/ACC) ===
        # Simulate sensor dropouts for point-wise features (rare but realistic)

        # Non-windowed HR missing values (5% probability)
        if self.hr_indices and np.random.random() < 0.05:
            for idx in self.hr_indices:
                x_aug[idx] = -6

        # Non-windowed ACC missing values (5% probability, independent from HR)
        if self.activity_indices and np.random.random() < 0.05:
            for idx in self.activity_indices:
                x_aug[idx] = -6

        # === SLEEP STAGE SPECIFIC AUGMENTATION ===
        # DISABLED: sleep_stage is a protected feature - values should NEVER be modified
        # Reasoning: sleep_stage represents ground-truth sleep phases that directly influence
        # recovery rates. Artificially perturbing these values would corrupt the model's
        # ability to learn the true relationship between sleep stages and charge dynamics.
        # if self.sleep_stage_indices and np.random.random() < 0.3:
        #     x_aug = self.augment_sleep_stage_transitions(x_aug, self.sleep_stage_indices)

        # === TEMPORAL CONSISTENCY AUGMENTATION ===
        # Apply small temporal jittering to maintain physiological realism
        # Only apply to continuous signals, not binary sequences
        if np.random.random() < 0.3:
            jitter_factor = np.random.uniform(0.98, 1.02)  # ±2% temporal jitter
            for idx in self.jitter_indices:
                if idx < len(x_aug) and x_aug[idx] != -6:  # Skip missing values
                    x_aug[idx] = x_aug[idx] * jitter_factor

        # === RANDOM FEATURE DROPOUT (Missingness Simulation) ===
        # Each feature independently has 10% probability of being dropped
        # This simulates realistic sensor failures and missing data patterns
        if np.random.random() < 0.05:
            # Determine which features can be dropped
            # Account for BMI replacement: height+weight become 1 BMI feature
            static_count = len(self.cfg.static_cols)
            if 'height' in self.cfg.static_cols and 'weight' in self.cfg.static_cols:
                static_count -= 1
            static_sleep_count = static_count + len(self.cfg.sleep_cols)

            # Separate droppable indices by type
            static_indices = []
            sleep_indices = []
            other_droppable_indices = []
            pos_encoding_indices = []
            past_charge_idx = None

            # Static features (BMI)
            for i in range(static_count):
                static_indices.append(i)

            # Sleep features
            for i in range(static_count, static_sleep_count):
                sleep_indices.append(i)

            # Time series features can be dropped
            if self.ts_feature_indices:
                other_droppable_indices.extend(self.ts_feature_indices)

            # Windowed features can be dropped
            if self.cfg.use_windowed_features:
                other_droppable_indices.extend(self.windowed_hr_indices)
                other_droppable_indices.extend(self.windowed_acc_indices)
                if self.cfg.include_gradients:
                    other_droppable_indices.extend(self.windowed_gradient_indices)

            # Positional encoding features can be dropped
            # Note: Feature order is static+sleep -> ts_cols -> pos_encoding -> windowed -> past_charge
            # Positional encoding comes BEFORE windowed features, not after
            if self.cfg.use_positional_encoding:
                pos_encoding_start = static_sleep_count + len(self.cfg.ts_cols)
                # No windowed features offset needed - pos_encoding is BEFORE windowed features

                if self.cfg.pos_encoding_type == "sinusoidal":
                    pos_encoding_end = pos_encoding_start + self.cfg.pos_encoding_dim
                else:  # "time_of_day"
                    pos_encoding_end = pos_encoding_start + 2

                for i in range(pos_encoding_start, pos_encoding_end):
                    pos_encoding_indices.append(i)

            # Past charge index (last feature, after positional encoding)
            if self.cfg.use_past_charge and len(x_aug) > 0:
                past_charge_idx = len(x_aug) - 1

            # Combine all droppable indices
            all_droppable = static_indices + sleep_indices + other_droppable_indices + pos_encoding_indices
            if past_charge_idx is not None:
                all_droppable.append(past_charge_idx)

            # Independent dropout: each feature has 10% probability of being dropped
            if all_droppable:
                for idx in all_droppable:
                    if np.random.random() < 0.1:  # 10% independent probability per feature
                        if idx < len(x_aug):
                            # Use -6 for all features to indicate missingness
                            x_aug[idx] = -6

        # === RANDOM NOISE AUGMENTATION (Excluding Positional Encoding) ===
        # Add small Gaussian noise to all features except positional encoding
        if np.random.random() < 0.5:
            # Calculate positional encoding indices to exclude
            # Account for BMI replacement: height+weight become 1 BMI feature
            static_count_for_noise = len(self.cfg.static_cols)
            if 'height' in self.cfg.static_cols and 'weight' in self.cfg.static_cols:
                static_count_for_noise -= 1
            pos_encoding_start = static_count_for_noise + len(self.cfg.sleep_cols) + len(self.cfg.ts_cols)
            pos_encoding_end = pos_encoding_start

            if self.cfg.use_positional_encoding:
                if self.cfg.pos_encoding_type == "sinusoidal":
                    pos_encoding_end = pos_encoding_start + self.cfg.pos_encoding_dim
                else:  # "time_of_day"
                    pos_encoding_end = pos_encoding_start + 2

            # Add noise to all features except positional encoding and protected features
            for idx in range(len(x_aug)):
                # Skip positional encoding indices
                if self.cfg.use_positional_encoding and pos_encoding_start <= idx < pos_encoding_end:
                    continue

                # Skip protected features (sleep_stage, nap_state, sleep_markers)
                # Reasoning: These represent ground-truth physiological states that should not be modified
                if idx in self.protected_state_indices:
                    continue

                # Skip missing values (-6)
                if x_aug[idx] == -6:
                    continue

                # Add small Gaussian noise (1-3% std)
                noise_std = np.random.uniform(0.01, 0.03)
                noise = np.random.normal(0, noise_std)
                x_aug[idx] = x_aug[idx] + noise

        # add augmentation for windowed features if enabled
        if self.cfg.use_windowed_features:
            # HR window
            for idx in self.windowed_hr_indices:
                hr_window_val = x_aug[idx]
                if np.random.random() < 0.7:
                    hrv_noise = np.random.normal(0, 0.025)  # 2.5% std
                    x_aug[idx] = hr_window_val + hrv_noise
            
            # Accelerometer window
            for idx in self.windowed_acc_indices:
                acc_window_val = x_aug[idx]
                if np.random.random() < 0.6:
                    sensor_noise = np.random.normal(0, 0.025)
                    x_aug[idx] = acc_window_val + sensor_noise

            # === MISSING VALUE AUGMENTATION ===
            # Simulate sensor dropouts (rare but realistic)
            # Apply independently (asynchronously) to HR and ACC

            # HR missing values (5% probability)
            if self.windowed_hr_indices and np.random.random() < 0.05:
                # Random duration: 1-3 consecutive values
                missing_duration = np.random.randint(1, 4)
                # Random location within the window
                window_len = len(self.windowed_hr_indices)
                if window_len > missing_duration:
                    start_idx = np.random.randint(0, window_len - missing_duration + 1)
                    # Set to -6 (simulating missing/dropout)
                    for i in range(start_idx, start_idx + missing_duration):
                        x_aug[self.windowed_hr_indices[i]] = -6

            # ACC missing values (5% probability, independent from HR)
            if self.windowed_acc_indices and np.random.random() < 0.05:
                # Random duration: 1-3 consecutive values
                missing_duration = np.random.randint(1, 4)
                # Random location within the window
                window_len = len(self.windowed_acc_indices)
                if window_len > missing_duration:
                    start_idx = np.random.randint(0, window_len - missing_duration + 1)
                    # Set to -6 (simulating missing/dropout)
                    for i in range(start_idx, start_idx + missing_duration):
                        x_aug[self.windowed_acc_indices[i]] = -6

            # Statistics
            for idx in self.windowed_stats_indices:
                stat_val = x_aug[idx]
                if np.random.random() < 0.5:
                    noise = np.random.normal(0, 0.02)
                    x_aug[idx] = stat_val + noise
            
            # Gradients
            # for idx in self.windowed_gradient_indices:
            #     grad_val = x_aug[idx]
            #     if np.random.random() < 0.5:
            #         noise = np.random.normal(0, 0.01)
            #         x_aug[idx] = grad_val + noise

            for idx in self.windowed_gradient_indices:
                grad_val = x_aug[idx]
                if np.random.random() < 0.5:
                    # Slightly stronger, scale-aware noise (2–3%)
                    noise = np.random.normal(0, 0.025 * abs(grad_val) + 1e-3)
                    x_aug[idx] = grad_val + noise

            # Temporal jitter: Shift windowed sequences backward in time
            # This simulates slight temporal misalignment in data collection
            shift = np.random.randint(-2, 1)  # -2, -1, or 0 (shift backward only to maintain causality)
            if shift != 0:
                # shift < 0 means shift backward (e.g., -2 removes first 2 elements, pads with zeros at end)
                # For shift=-2: [a,b,c,d,e] -> [c,d,e,0,0] (removes early timepoints)

                # Apply same shift to HR and ACC windows
                for idx_group in [self.windowed_hr_indices, self.windowed_acc_indices]:
                    seq = x_aug[idx_group]
                    # Shift backward: remove first |shift| elements, pad with zeros at end
                    x_aug[idx_group] = np.concatenate([seq[-shift:], np.zeros(-shift)])

                # Apply same shift to gradient windows if they exist
                if self.cfg.include_gradients and len(self.windowed_gradient_indices) > 0:
                    # Split gradient indices into HR and ACC gradients
                    window_size = self.cfg.window_size_minutes
                    hr_grad_idx = self.windowed_gradient_indices[:window_size]
                    acc_grad_idx = self.windowed_gradient_indices[window_size:2*window_size]

                    for idx_group in [hr_grad_idx, acc_grad_idx]:
                        if len(idx_group) > 0:
                            seq = x_aug[idx_group]
                            x_aug[idx_group] = np.concatenate([seq[-shift:], np.zeros(-shift)])

        # Clip continuous signals to reasonable bounds
        for idx in self.continuous_signal_indices:
            x_aug[idx] = np.clip(x_aug[idx], -4.0, 4.0)
        
        # === PAST CHARGE AUGMENTATION ===
        # Simulates train-inference mismatch (model uses own predictions at inference)
        if self.cfg.use_past_charge and np.random.random() < 0.25:
            past_charge_idx = len(x_aug) - 1
            if x_aug[past_charge_idx] != -6:  # Skip if missing
                noise = np.random.normal(0, 0.015)  # 1.5% std - very subtle
                x_aug[past_charge_idx] = x_aug[past_charge_idx] + noise

        return x_aug, y 

    def __len__(self):
        return len(self.index)


    def find_sleep_start(self, series):
        """
        Given a binary sequence (0/1), find the index of the last 0->1 transition
        (searching from the end of the sequence).
        Returns None if no transition found or data is malformed.
        """
        try:
            if series is None or series.empty:
                return None
                
            series_val = series.iloc[0]
            arr = None
            
            if isinstance(series_val, str):
                try:
                    parsed = ast.literal_eval(series_val)
                    arr = np.array(parsed).astype(int)
                except (ValueError, SyntaxError, TypeError):
                    return None
            
            elif isinstance(series_val, (list, tuple)):
                try:
                    arr = np.array(series_val).astype(int)
                except (ValueError, TypeError):
                    return None
                    
            elif isinstance(series_val, np.ndarray):
                try:
                    arr = series_val.astype(int)
                except (ValueError, TypeError):
                    return None
            else:
                return None
            
            if arr is None or len(arr) <= 1:
                return None
            
            # Scan backwards to find 0->1 transition
            for i in range(len(arr) - 1, 0, -1):
                if arr[i] == 1 and arr[i-1] == 0:
                    return i
                    
            return None  # No transition found
            
        except Exception:
            return None


    def _load_user_file(self, user_id: str, date, idx) -> pd.DataFrame:
        """Load user file from folder (cached)."""

        date_str = date.strftime("%Y-%m-%d") 
        if user_id in self.user_cache:
            df = self.user_cache[user_id]
        else:
            # if userid in llm users then use self.cfg.llm_date_dir
            if self.cfg.data_dir_llm is not None and (user_id, date_str) in self.llm_charge_pairs:
                excel_path = os.path.join(self.cfg.data_dir_llm, f"{user_id}_processed.xlsx")
            else:
                excel_path = os.path.join(self.cfg.data_dir, f"{user_id}_processed.xlsx")

            # csv_path = os.path.join(self.cfg.data_dir, f"{user_id}.csv")
            # parquet_path = os.path.join(self.cfg.data_dir, f"{user_id}.parquet")

            if os.path.exists(excel_path):
                df = pd.read_excel(excel_path)
            else:
                raise FileNotFoundError(f"No file found for user {user_id}")
            
            self.user_cache[user_id] = df
            # Initialize processed dates cache for this user
            # self.processed_dates_cache[user_id] = set()


        # Try csv or parquet
        # read excel file

        df_row = df[df["date"] == date_str]
        df_yesterday_row = df[df["date"] == (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")]

        # # Check if this date has already been processed for this user
        # if user_id not in self.processed_dates_cache:
        #     self.processed_dates_cache[user_id] = set()
            
        # if date_str not in self.processed_dates_cache[user_id]:
        #     # Filter hr data only if not already processed - selective filtering for outliers only
        #     try:
        #         self.filter_hr_data(user_id, date_str)
        #         # Mark this date as processed
        #         self.processed_dates_cache[user_id].add(date_str)
                    
        #     except Exception as e:
        #         print(f"Warning: Failed to filter HR data for user {user_id} on {date_str}: {e}")
        #         # Mark as processed to avoid repeated attempts
        #         self.processed_dates_cache[user_id].add(date_str)
        
        # Get the updated row from cache
        df_row = self.user_cache[user_id][self.user_cache[user_id]["date"] == date_str]

        # Initialize caches for this user if needed
        if user_id not in self.processed_dates_cache:
            self.processed_dates_cache[user_id] = set()
        if user_id not in self.sleep_start_cache:
            self.sleep_start_cache[user_id] = {}
        
        # Check if sleep_start_idx is already cached for this date
        if date_str in self.sleep_start_cache[user_id]:
            sleep_start_idx = self.sleep_start_cache[user_id][date_str]
        else:
            # Calculate and cache sleep_start_idx
            sleep_start_idx = self.find_sleep_start(df_row['timeseries.sleep_markers'])
            self.sleep_start_cache[user_id][date_str] = sleep_start_idx

        # add to processed dates cache
        self.processed_dates_cache[user_id].add(date_str)


        # if sleep_start_idx is None:
        #     print(f"No sleep start found for user {user_id} on date {date_str}, using idx {idx}")
        # if df_yesterday_row.empty:
        #     prev_charge = 69
        # else:
        #     prev_charge = float(df_yesterday_row.iloc[0][self.cfg.charge_col])

        df["date"] = pd.to_datetime(df["date"])
        # self.user_cache[user_id] = df
        return df_row, df_yesterday_row, sleep_start_idx

    def _safe_extract_value(self, data, idx=None, default=0.0):
        """
        Safely extract value from various data types with robust error handling.
        Enhanced with better out-of-bounds protection and data length mismatch handling.
        """
        try:
            if data is None:
                return default
            
            # Handle all numpy scalar types
            if isinstance(data, (int, float, np.integer, np.floating, np.int64, np.int32, np.float64, np.float32)):
                val = float(data)
                return val if not (np.isnan(val) or np.isinf(val)) else default
            
            # Handle numpy boolean
            if isinstance(data, (bool, np.bool_)):
                return float(data)
            
            # Handle numpy arrays
            if isinstance(data, np.ndarray):
                if data.size == 0:
                    return default
                if idx is not None:
                    if 0 <= idx < len(data):
                        val = float(data[idx])
                        return val if not (np.isnan(val) or np.isinf(val)) else default
                    else:
                        # Enhanced: Handle out-of-bounds with intelligent fallback
                        if len(data) > 0:
                            # Use last available value for out-of-bounds access
                            val = float(data[-1])
                            return val if not (np.isnan(val) or np.isinf(val)) else default
                        return default
                else:
                    # Return first element if no index specified
                    val = float(data.flat[0])
                    return val if not (np.isnan(val) or np.isinf(val)) else default
            
            if isinstance(data, str):
                try:
                    # Try to parse as list/array
                    parsed = ast.literal_eval(data)
                    if isinstance(parsed, (list, tuple)) and idx is not None:
                        if 0 <= idx < len(parsed):
                            val = float(parsed[idx])
                            return val if not (np.isnan(val) or np.isinf(val)) else default
                        else:
                            # Enhanced: Handle out-of-bounds with intelligent fallback
                            if len(parsed) > 0:
                                # Use last available value for out-of-bounds access
                                val = float(parsed[-1])
                                return val if not (np.isnan(val) or np.isinf(val)) else default
                            return default
                    val = float(parsed)
                    return val if not (np.isnan(val) or np.isinf(val)) else default
                except (ValueError, SyntaxError, IndexError):
                    # If parsing fails, try direct float conversion
                    try:
                        val = float(data)
                        return val if not (np.isnan(val) or np.isinf(val)) else default
                    except (ValueError, TypeError):
                        return default
            
            if isinstance(data, (list, tuple)):
                if idx is not None:
                    if 0 <= idx < len(data):
                        val = data[idx]
                        # Recursively handle numpy types in lists
                        if isinstance(val, (np.integer, np.floating)):
                            val = float(val)
                        return val if not (np.isnan(val) or np.isinf(val)) else default
                    else:
                        # Enhanced: Handle out-of-bounds with intelligent fallback
                        if len(data) > 0:
                            # Use last available value for out-of-bounds access
                            val = data[-1]
                            if isinstance(val, (np.integer, np.floating)):
                                val = float(val)
                            return val if not (np.isnan(val) or np.isinf(val)) else default
                        return default
                elif idx is None and len(data) > 0:
                    val = data[0]
                    # Recursively handle numpy types in lists
                    if isinstance(val, (np.integer, np.floating)):
                        val = float(val)
                    return val if not (np.isnan(val) or np.isinf(val)) else default
                else:
                    return default
            
            # For pandas Series
            if hasattr(data, 'iloc'):
                try:
                    val = data.iloc[0]
                    return self._safe_extract_value(val, idx, default)
                except (IndexError, KeyError):
                    return default
            
            # For pandas scalar values
            if hasattr(data, 'item'):
                try:
                    val = float(data.item())
                    return val if not (np.isnan(val) or np.isinf(val)) else default
                except (ValueError, TypeError):
                    return default
            
            # Last resort: try direct float conversion
            try:
                val = float(data)
                return val if not (np.isnan(val) or np.isinf(val)) else default
            except (ValueError, TypeError):
                return default
            
        except Exception as e:
            # Enhanced: Add more specific error logging for debugging
            import traceback
            print(f"Debug: Exception in _safe_extract_value - data type: {type(data)}, idx: {idx}, error: {e}")
            print(f"Debug: Traceback: {traceback.format_exc()}")
            return default

    def _get_feature_vector(self, row: pd.Series, yesterday_row, idx: int, idx_sleep_start: int, charge_col: str = None) -> np.ndarray:
        feats = []

        
        # Use provided charge_col or fall back to config
        if charge_col is None:
            charge_col = self.cfg.charge_col

        # Handle malformed yesterday_row
        if yesterday_row is None or yesterday_row.empty:
            yesterday_row = row

        # Validate idx_sleep_start
        if idx_sleep_start is None:
            idx_sleep_start = 0
            
        # Static + sleep features with robust error handling
        # First, calculate BMI if we have height and weight in static_cols
        bmi_value = None
        if 'height' in self.cfg.static_cols and 'weight' in self.cfg.static_cols:
            try:
                height_val = self._safe_extract_value(row.get('height', None))  # in cm
                weight_val = self._safe_extract_value(row.get('weight', None))  # in kg
                if height_val > 0:
                    bmi_value = weight_val / ((height_val / 100) ** 2)  # BMI formula
                    bmi_value = bmi_value / 30.0  # Normalize BMI (typical range 15-40, so /30)
                else:
                    bmi_value = -6
            except:
                bmi_value = -6

        for col in self.cfg.static_cols + self.cfg.sleep_cols:

            # print(f"col: {col}")
            try:
                row_ = row.get(col, None)
                row_yesterday_ = yesterday_row.get(col, None)

                if row_ is None:
                    feats.append(-6.0)
                    continue

                # Handle BMI calculation (replace height and weight with BMI)
                if col == 'height' or col == 'weight':
                    # Skip individual height/weight, we'll use BMI instead
                    if col == 'weight' and bmi_value is not None:
                        # Add BMI when we encounter weight (skip height entirely)
                        feats.append(float(bmi_value))
                    # Skip height entirely
                    continue

                # normalize by constants with safe extraction
                elif 'age' in col.lower():
                    val = self._safe_extract_value(row_) / 80

                elif 'gender' in col:
                    val = self._safe_extract_value(row_)

                else:
                    # Determine which row to use based on sleep timing for ALL sleep metrics
                    # If before sleep start and yesterday's data exists, use yesterday's sleep data
                    # Otherwise use today's sleep data (including when row_yesterday_ is None)
                    if idx < idx_sleep_start and row_yesterday_ is not None:
                        source_row = row_yesterday_
                    else:
                        # # Use default values when today's sleep data isn't available yet
                        # if col not in ["z_rhr_7", "z_hrv_7", ]:
                        #     val = 0.0  # Default for z-scored values
                        # elif col in ["sleep_duration", "wakefulness_after_sleep_onset_duration"]:
                        #     val = 0.0  # Default for duration
                        # elif col in ["rem_sleep_ratio", "deep_sleep_ratio", "light_sleep_ratio"]:
                        #     val = 0.0  # Default for ratios
                        # else:
                        #     val = -6  # Missing indicator for other sleep features
                        # Don't use source_row, just use the default val set above
                        source_row = row_

                    # Only process from source_row if it's available
                    if source_row is not None:
                        # Handle sleep duration and waso columns - normalize by 660 minutes
                        if col in ["sleep_duration", "wakefulness_after_sleep_onset_duration"]:
                            val = self._safe_extract_value(source_row) / 660.0

                        # Handle sleep ratio columns (already in 0-1 range, just use directly)
                        elif col in ["rem_sleep_ratio", "deep_sleep_ratio", "light_sleep_ratio"]:
                            val = self._safe_extract_value(source_row)  # Already normalized to 0-1

                        # Handle z_rhr_7 and z_hrv_7 - already z-scored, use directly without normalization
                        elif col in ["z_rhr_7", "z_hrv_7"]:
                            val = self._safe_extract_value(source_row)  # Already z-scored, use as-is

                        # Legacy sleep duration columns (if still used)
                        elif col in ["deep_sleep_duration", "light_sleep_duration", "rem_sleep_duration"]:
                            val = self._safe_extract_value(source_row) / 720  # Normalize by max 12 hours

                        else:
                            val = self._safe_extract_value(source_row) / 100
                
                # Ensure the value is valid
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    val = -6.0
                
                # print(f"{col}: {val}")
                feats.append(float(val))
                
            except Exception as e:
                # Log malformed data for debugging (optional)
                print(f"Error processing column {col}: {e}")
                feats.append(0.0)

        # Timeseries columns with robust error handling
        for col in self.cfg.ts_cols:
            # print(f"col: {col}")

            try:
                row_ = row.get(col, None)
                if row_ is None:
                    feats.append(0.0)
                    continue
                
                #------------------------------------------------------------------------------
                #         Event Data (exercise, sleep_stage, sleep_markers, nap_state)
                # -----------------------------------------------------------------------------
                if 'exercise' in col.lower():
                    # Extract sleep stage value at specific index
                    raw_val = self._safe_extract_value(row_, idx, 0.0)
                    
                    feats.append(float(val) if not np.isnan(val) else 0.0)
                    continue
                
                if 'sleep_markers' in col.lower():
                    # Extract sleep stage value at specific index
                    raw_val = self._safe_extract_value(row_, idx, 0.0)
                    
                    feats.append(float(val) if not np.isnan(val) else 0.0)
                    continue
                
                if 'nap_state' in col.lower():
                    # Extract sleep stage value at specific index
                    raw_val = self._safe_extract_value(row_, idx, 0.0)
                    
                    feats.append(float(val) if not np.isnan(val) else 0.0)
                    continue

                # Special handling for sleep_stage column
                if 'sleep_stage' in col.lower():
                    # Extract sleep stage value at specific index
                    raw_val = self._safe_extract_value(row_, idx, 0.0)
                    
                    # Sleep stages are typically encoded as: -1=awake, 1=light, 2=deep, 3=REM
                    # Normalize to 0-1 range: 0=awake, 0.33=light, 0.66=deep, 1.0=REM
                    val =raw_val/3
                    feats.append(float(val) if not np.isnan(val) else 0.0)
                    continue
                
                # Safely extract time series value at specific index
                raw_val = self._safe_extract_value(row_, idx, 0.0)
                
                #---------------------------------------------------------
                #         Timeseries Data (hr)
                # --------------------------------------------------------
                if 'hr' in col.lower() and 'hrr' not in col.lower():
                    col_string = self._get_hr_column_name()
                else:
                    col_string = col
                
                # Apply z-score normalization if available
                if self.zdf is not None and self.cfg.use_zscores and (col_string in self.zdf['global']):
                    try:

                        if self.z_data_norm == 'population':
                            if 'global' in self.zdf and col_string in self.zdf['global']:
                                z_std = self.zdf['global'][col_string]['std']
                                z_mean = self.zdf['global'][col_string]['mean']
                            else:
                                # Fallback to raw normalization
                                val = raw_val / 100
                        else:
                            user_id_str = str(int(self._safe_extract_value(row.get('userid', ''))))
                            
                            # Try user-specific z-scores first
                            if (user_id_str in self.zdf and 
                                col_string in self.zdf[user_id_str]):
                                z_std = self.zdf[user_id_str][col_string]['std']
                                z_mean = self.zdf[user_id_str][col_string]['mean']
                            elif ('global' in self.zdf and 
                                col_string in self.zdf['global']):
                                z_std = self.zdf['global'][col_string]['std']
                                z_mean = self.zdf['global'][col_string]['mean']
                            else:
                                # No z-score available, use raw value
                                val = raw_val
                                feats.append(float(val) if not np.isnan(val) else 0.0)
                                continue
                        
                        # Apply z-score normalization
                        val = (raw_val - z_mean) / z_std if z_std > 0 else 0.0
                        
                    except (KeyError, TypeError, ValueError, IndexError):
                        val = raw_val
                else:
                    val = raw_val
                

                # RHR centered hr 
                if 'z_hr_filtered_7' in col.lower():
                    val = self._safe_extract_value('z_hr_filtered_7', idx, 0.0)  # mean_hr_7


                #---------------------------------------------------------
                #         Timeseries Data (acc)
                #--------------------------------------------------------
                if 'timeseries.acc' in col.lower():
                    
                    # Apply z-score normalization if available
                    if self.zdf is not None and self.cfg.use_zscores and (col_string in self.zdf['global']):
                        try:
                            if self.z_data_norm == 'population':
                                if 'global' in self.zdf and col_string in self.zdf['global']:
                                    z_std = self.zdf['global'][col_string]['std']
                                    z_mean = self.zdf['global'][col_string]['mean']
                                else:
                                    # Fallback to raw normalization
                                    val = raw_val / 100
                            else:
                                user_id_str = str(int(self._safe_extract_value(row.get('userid', ''))))
                                
                                # Try user-specific z-scores first
                                if (user_id_str in self.zdf and 
                                    col_string in self.zdf[user_id_str]):
                                    z_std = self.zdf[user_id_str][col_string]['std']
                                    z_mean = self.zdf[user_id_str][col_string]['mean']
                                elif ('global' in self.zdf and 
                                    col_string in self.zdf['global']):
                                    z_std = self.zdf['global'][col_string]['std']
                                    z_mean = self.zdf['global'][col_string]['mean']
                                else:
                                    # No z-score available, use raw value
                                    val = raw_val
                                    feats.append(float(val) if not np.isnan(val) else 0.0)
                                    continue
                            
                            # Apply z-score normalization
                            val = (raw_val - z_mean) / z_std if z_std > 0 else 0.0
                            
                        except (KeyError, TypeError, ValueError, IndexError):
                            val = raw_val

                # Ensure valid float value
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    val = -6            
                # print(f"timeseries{col}: {val}")
                feats.append(float(val))

            except Exception as e:
                # Log malformed time series data for debugging (optional)
                # print(f"Error processing time series column {col} at index {idx}: {e}")
                feats.append(0.0)

                
        # Add positional encoding if enabled
        if self.cfg.use_positional_encoding:
            try:
                if self.cfg.pos_encoding_type == "time_of_day":
                    # Use time_of_day_encoding which returns a 2-element cyclic encoding [sin, cos]
                    pos_encoding = self.time_of_day_encoding(idx % 1440)
                    feats.extend(pos_encoding.tolist())
                elif self.cfg.pos_encoding_type == "sinusoidal":
                    # Use sinusoidal positional encoding with configurable dimensions
                    
                    # pos_encoding = self.positional_encoding(idx % 1440, self.cfg.pos_encoding_dim)
                    pos_encoding = self.time_of_day_encoding_continuous(idx % 1440)
                    feats.extend(pos_encoding.tolist())
                else:
                    # Default to time_of_day if invalid type specified
                    pos_encoding = self.time_of_day_encoding(idx % 1440)
                    feats.extend(pos_encoding.tolist())
            except Exception:
                # Fallback to zero encoding if there's an error
                if self.cfg.pos_encoding_type == "sinusoidal":
                    # Use pos_encoding_dim zeros for sinusoidal
                    feats.extend([0.0] * self.cfg.pos_encoding_dim)
                else:
                    # Use 2 zeros for time_of_day (default)
                    feats.extend([0.0, 0.0])

        # Add windowed features if enabled
        if self.cfg.use_windowed_features:
            try:
                # Use the new method to read pre-calculated windowed features from file
                windowed_features = self._read_windowed_features(row, idx)
                
                # Add windowed HR values (already z-score normalized from file)
                if isinstance(windowed_features['hr_window'], list) and len(windowed_features['hr_window']) > 0:
                    feats.extend(windowed_features['hr_window'])
                else:
                    # Fallback: add zeros for entire window if data not available
                    feats.extend([0.0] * self.cfg.window_size_minutes)

                # Add windowed accelerometer values (already z-score normalized from file)
                if isinstance(windowed_features['acc_window'], list) and len(windowed_features['acc_window']) > 0:
                    feats.extend(windowed_features['acc_window'])
                else:
                    # Fallback: add zeros for entire window if data not available
                    feats.extend([0.0] * self.cfg.window_size_minutes)
                
                # Don't add summary
                
                # Add gradients if enabled (already z-score normalized from file)
                if self.cfg.include_gradients:
                    # Add HR gradient
                    if isinstance(windowed_features['hr_gradient'], list) and len(windowed_features['hr_gradient']) > 0:
                        feats.extend(windowed_features['hr_gradient'])
                    else:
                        feats.extend([0.0] * self.cfg.window_size_minutes)

                    # Add accelerometer gradient
                    if isinstance(windowed_features['acc_gradient'], list) and len(windowed_features['acc_gradient']) > 0:
                        feats.extend(windowed_features['acc_gradient'])
                    else:
                        feats.extend([0.0] * self.cfg.window_size_minutes)
                    
            except Exception as e:
                # Fallback to zero features if there's an error
                print(f"Error reading windowed features: {e}")
                
                # Add zero windowed values (window slices)
                feats.extend([0.0] * self.cfg.window_size_minutes)  # HR window slice
                feats.extend([0.0] * self.cfg.window_size_minutes)  # ACC window slice
                
                # Add zero gradients if enabled
                if self.cfg.include_gradients:
                    feats.extend([0.0] * self.cfg.window_size_minutes)  # HR gradient window slice
                    feats.extend([0.0] * self.cfg.window_size_minutes)  # ACC gradient window slice

        # DELTA MODE: Extract charge values for delta calculation
        # Get charge at time t and t-1
        try:
            charge_t_list = row.get(charge_col, None)
            if charge_t_list is not None:
                charge_t = self._safe_extract_value(charge_t_list, idx, 69.0)
                if idx == 0:
                    # last night's charge
                    # Megha : Monitor this value and add for each user. 
                    charge_t_1 = self._safe_extract_value(yesterday_row.get(charge_col, None), -1, 69.0) # get the last night charge's last value
                else:
                    charge_t_1 = self._safe_extract_value(charge_t_list, idx-1, 69.0)
            else:
                charge_t = 69.0  # Default charge value
                charge_t_1 = 69.0

            # Normalize to [0, 1] range
            charge_t = charge_t / 100.0
            charge_t_1 = charge_t_1 / 100.0
        except Exception:
            charge_t = 69.0 / 100.0
            charge_t_1 = 69.0 / 100.0

        # Append past charge value (ABSOLUTE, not delta) as feature
        # This is what the model will use to predict the delta
        if self.cfg.use_past_charge:
            feats.append(charge_t_1)

        # DELTA MODE: Calculate delta (change in charge)
        # Target is delta = charge_t - charge_{t-1}
        charge_delta = charge_t - charge_t_1

        # Convert to numpy array with robust NaN handling
        feats = np.asarray(feats, dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure charge_delta is valid
        if not isinstance(charge_delta, (int, float)) or np.isnan(charge_delta) or np.isinf(charge_delta):
            charge_delta = 0.0  # Default delta is 0 (no change)

        # charge_t_recon = None
        # if self.cfg.charge_reconstruction:
        charge_t_recon = charge_t if isinstance(charge_t, float) else 0.0

        # Return features with past_charge (absolute) and target delta
        return np.asarray(feats, dtype=np.float32), charge_delta, charge_t_recon

    def _validate_data_length(self, user_id: str, date: pd.Timestamp, idx: int, row: pd.Series) -> bool:
        """
        Validate that the requested index is within reasonable bounds for the data.
        Returns True if data seems valid, False if there are major inconsistencies.
        """
        try:
            # Check a few key time series columns to see their lengths
            sample_cols = [col for col in self.cfg.ts_cols[:3]]  # Check first 3 columns
            data_lengths = []
            
            for col in sample_cols:
                col_data = row.get(col, None)
                if col_data is not None:
                    try:
                        if isinstance(col_data, str):
                            parsed = ast.literal_eval(col_data)
                            if isinstance(parsed, (list, tuple)):
                                data_lengths.append(len(parsed))
                        elif isinstance(col_data, (list, tuple)):
                            data_lengths.append(len(col_data))
                        elif isinstance(col_data, np.ndarray):
                            data_lengths.append(len(col_data))
                    except:
                        continue
            
            if not data_lengths:
                return False  # No valid data found
            
            min_length = min(data_lengths)
            max_length = max(data_lengths)
            
            # Log data inconsistencies for debugging
            if idx >= min_length:
                print(f"Data length warning: User {user_id}, Date {date.strftime('%Y-%m-%d')}, "
                      f"requested idx={idx}, but min data length={min_length}, max={max_length}")
                
                # Still return True - we'll handle this in _safe_extract_value
                return True
            
            # Check for severely truncated data (less than 50% of expected day length)
            if max_length < 720:  # Less than 12 hours of data
                print(f"Severely truncated data: User {user_id}, Date {date.strftime('%Y-%m-%d')}, "
                      f"max data length={max_length} (expected ~1440)")
            
            return True
            
        except Exception as e:
            print(f"Error validating data length for user {user_id}, date {date}: {e}")
            return True  # Continue processing with enhanced error handling

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        """
        Load data from torch tensors and apply same normalization/augmentation as BiochargeDataset.
        Returns delta predictions (charge_t - charge_{t-1}) in DELTA MODE.
        """
        try:
            item        = self.index.iloc[i]
            user_id     = str(item['userid'])
            date        = pd.to_datetime(item['date'])
            idx         = int(item['index'])

            # Load user torch data (with caching and memory mapping)
            if user_id not in self.user_cache:
                torch_path = os.path.join(self.data_dir, f"{user_id}.pt")
                self.user_cache[user_id] = torch.load(torch_path, mmap=True, weights_only=False)

            data = self.user_cache[user_id]
            date_str = date.strftime('%Y-%m-%d')
            query_date = np.datetime64(date_str, "D").astype("int64")

            # Find the date index in torch data
            date_matches = (data["dates"] == query_date).nonzero(as_tuple=True)[0]
            if len(date_matches) == 0:
                # Date not found - return default
                raise ValueError(f"Date {date_str} not found for user {user_id}")

            date_idx = date_matches[0].item()

            # Extract raw values from torch tensor and apply normalization
            x, charge_delta, charge_recon = self._extract_and_normalize_torch(
                data, date_idx, idx, user_id, date_str
            )

            # Apply augmentation if enabled
            if self.augment:
                x, charge_delta = self.augment_sample(x, charge_delta)

            # Add x_add_1 logic for rollout calculation during training
            try:
                x_add_1, charge_delta_add_1, charge_recon_add_1 = self._extract_and_normalize_torch(
                    data, date_idx, idx+1, user_id, date_str
                )
                y_add_1 = charge_delta_add_1
                
               
            except:
                # idx+1 is beyond current day, try tomorrow's data
                try:
                    tomorrow_date = date + pd.Timedelta(days=1)
                    tomorrow_date_str = tomorrow_date.strftime('%Y-%m-%d')
                    tomorrow_query = np.datetime64(tomorrow_date_str, "D").astype("int64")
                    tomorrow_matches = (data["dates"] == tomorrow_query).nonzero(as_tuple=True)[0]
                    
                    if len(tomorrow_matches) > 0:
                        tomorrow_date_idx = tomorrow_matches[0].item()
                        x_add_1, charge_delta_add_1, charge_recon_add_1 = self._extract_and_normalize_torch(
                            data, tomorrow_date_idx, 0, user_id, tomorrow_date_str
                        )
                        y_add_1 = charge_delta_add_1
                    else:
                        # Next day doesn't exist
                        x_add_1 = np.full(self.get_feature_count(), -6.0, dtype=np.float32)
                        y_add_1 = -6.0
                except:
                    # Error loading next day
                    x_add_1 = np.full(self.get_feature_count(), -6.0, dtype=np.float32)
                    y_add_1 = -6.0

            # Masking logic
            mask = 1.0
            # Check for non-wear from torch data
            if "features_data" in data and "column_to_idx" in data:
                if "timeseries.min_status_list" in data["column_to_idx"]:
                    min_status_idx = data["column_to_idx"]["timeseries.min_status_list"]
                    min_status_data = data["features_data"][min_status_idx][date_idx]
                    if isinstance(min_status_data, (list, tuple)) and len(min_status_data) > idx:
                        if min_status_data[idx] == 3:
                            mask = 0.0

            # if idx > 1500:
            #     mask = 0.0

            # Delta scaling depends on normalization type
            if self.normalization_type == "max":
                # Normalize delta to [-1, 1] range using min=-0.4, max=0.7
                y = 2 * ((torch.tensor([charge_delta * 100], dtype=torch.float32) - (-0.4)) / (0.7 - (-0.4))) - 1
                y_add_1_t = 2 * ((torch.tensor([y_add_1 * 100], dtype=torch.float32) - (-0.4)) / (0.7 - (-0.4))) - 1
            else:
                y = torch.tensor([charge_delta * 100], dtype=torch.float32)
                y_add_1_t = torch.tensor([y_add_1 * 100], dtype=torch.float32)

            return {
                "x": torch.from_numpy(x).float(),
                "y": y,  # DELTA scaled
                "mask": torch.tensor([mask], dtype=torch.float32),
                'meta': {'user_id': user_id, 'date': date_str, 'idx': idx},
                'charge_recon': charge_recon,
                "x_add_1": torch.from_numpy(x_add_1).float(),
                "y_add_1": y_add_1_t
            }

        except Exception as e:
            print(f"Error in TorchBiochargeDataset[{i}]: {e}")
            # Return default values
            default_dim = len(self.cfg.static_cols) + len(self.cfg.sleep_cols) + len(self.cfg.ts_cols)
            x = np.zeros(default_dim, dtype=np.float32)
            x_add_1 = np.full(self.get_feature_count(), -6.0, dtype=np.float32)
            return {
                "x": torch.from_numpy(x).float(),
                "y": torch.tensor([0.0], dtype=torch.float32),
                "mask": torch.tensor([0.0], dtype=torch.float32),
                "x_add_1": torch.from_numpy(x_add_1).float(),
                "y_add_1": torch.tensor([-6.0 * 100], dtype=torch.float32)
            }

    def get_feature_list(self) -> List:
        """
        Returns a dictionary mapping feature names to their indices in the feature vector.
        The order matches the feature vector produced by _extract_and_normalize_torch.
        """
        feature_names = []
        idx = 0

        # BMI replacement logic for static features
        static_cols = self.cfg.static_cols.copy()
        if 'height' in static_cols and 'weight' in static_cols:
            # Only add BMI at 'weight', skip 'height'
            static_cols_no_height = []
            for col in static_cols:
                if col == 'height':
                    continue
                elif col == 'weight':
                    static_cols_no_height.append('BMI')
                else:
                    static_cols_no_height.append(col)
            static_cols = static_cols_no_height

        # Static + sleep features
        for col in static_cols + self.cfg.sleep_cols:
            feature_names.append(col)
            idx += 1

        # Timeseries features
        for col in self.cfg.ts_cols:
            feature_names.append(col)
            idx += 1

        # Positional encoding (biocharge_circadian: 4 features, else: pos_encoding_dim or 2)
        if self.cfg.use_positional_encoding:
            if self.cfg.pos_encoding_type == "biocharge_circadian":
                feature_names.extend([
                    "circadian_sin", "circadian_cos", "circadian_model", "time_state"
                ])
                idx += 4
            elif self.cfg.pos_encoding_type == "sinusoidal":
                for i in range(self.cfg.pos_encoding_dim):
                    feature_names.append(f"sinusoidal_pos_{i}")
                idx += self.cfg.pos_encoding_dim
            else:
                feature_names.extend(["time_of_day_sin", "time_of_day_cos"])
                idx += 2

        # Recovery rate feature
        if getattr(self, "use_recovery_rate_feature", False):
            feature_names.append("recovery_rate")
            idx += 1

        # Weighted sleep score
        if getattr(self, "weighted_sleep_score", False):
            feature_names.append("weighted_sleep_score")
            idx += 1

        # Past charge (always last if enabled)
        if self.cfg.use_past_charge:
            feature_names.append("past_charge")
            idx += 1

        return feature_names

    def _extract_and_normalize_torch(self, data, date_idx, idx, user_id, date_str):
        """
        Extract raw values from torch tensor and apply BiochargeDataset-style normalization.
        This matches the normalization logic in BiochargeDataset._get_feature_vector.
        """
        feats = []
        features_data = data["features_data"]
        column_to_idx = data["column_to_idx"]

        # Get yesterday's data for sleep features (if idx < sleep_start)
        yesterday_date = (pd.to_datetime(date_str) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        yesterday_query = np.datetime64(yesterday_date, "D").astype("int64")
        yesterday_matches = (data["dates"] == yesterday_query).nonzero(as_tuple=True)[0]
        yesterday_date_idx = yesterday_matches[0].item() if len(yesterday_matches) > 0 else None

        # --- Cache sleep and nap start indices per user/date ---
        if not hasattr(self, 'sleep_start_cache_torch'):
            self.sleep_start_cache_torch = {}
        if not hasattr(self, 'nap_start_cache_torch'):
            self.nap_start_cache_torch = {}

        if user_id not in self.sleep_start_cache_torch:
            self.sleep_start_cache_torch[user_id] = {}
        if user_id not in self.nap_start_cache_torch:
            self.nap_start_cache_torch[user_id] = {}

        # Sleep start index
        if date_str in self.sleep_start_cache_torch[user_id]:
            idx_sleep_start = self.sleep_start_cache_torch[user_id][date_str]
        else:
            idx_sleep_start = 0
            if "timeseries.sleep_markers" in column_to_idx:
                sleep_markers_idx = column_to_idx["timeseries.sleep_markers"]
                sleep_markers = features_data[sleep_markers_idx][date_idx]
                if isinstance(sleep_markers, (list, tuple)):
                    for i, val in enumerate(sleep_markers):
                        if val == 1:
                            idx_sleep_start = i
                            break
            self.sleep_start_cache_torch[user_id][date_str] = idx_sleep_start

        # Nap start index
        if date_str in self.nap_start_cache_torch[user_id]:
            idx_nap_start = self.nap_start_cache_torch[user_id][date_str]
        else:
            idx_nap_start = 0
            if "timeseries.nap_state" in column_to_idx:
                nap_state_idx = column_to_idx["timeseries.nap_state"]
                nap_state = features_data[nap_state_idx][date_idx]
                if isinstance(nap_state, (list, tuple)):
                    for i, val in enumerate(nap_state):
                        if val == 1:
                            idx_nap_start = i
                            break
            self.nap_start_cache_torch[user_id][date_str] = idx_nap_start

        # Calculate BMI if height and weight are both present (matching BiochargeDataset)
        bmi_value = None
        if 'height' in self.cfg.static_cols and 'weight' in self.cfg.static_cols:
            if 'height' in column_to_idx and 'weight' in column_to_idx:
                height_val = float(features_data[column_to_idx['height']][date_idx])
                weight_val = float(features_data[column_to_idx['weight']][date_idx])
                if height_val > 0:
                    bmi_value = weight_val / ((height_val / 100) ** 2)
                    bmi_value = bmi_value / 30.0  # Normalize BMI
                else:
                    bmi_value = -6.0

        # STATIC + SLEEP features with normalization (matching BiochargeDataset logic)
        for col in self.cfg.static_cols + self.cfg.sleep_cols:
            if col not in column_to_idx:
                feats.append(-6.0)
                continue

            col_idx = column_to_idx[col]

            # Determine which date to use for sleep features
            if col in self.cfg.sleep_cols and idx < idx_sleep_start and yesterday_date_idx is not None:
                col_data = features_data[col_idx][yesterday_date_idx]
            else:
                col_data = features_data[col_idx][date_idx]

            # Handle BMI replacement (matching BiochargeDataset behavior)
            if col == 'height' or col == 'weight':
                if col == 'weight' and bmi_value is not None:
                    # Add BMI when we encounter weight (skip height entirely)
                    feats.append(float(bmi_value))
                # Skip height entirely
                continue
            
            # # Apply same normalization as BiochargeDataset
            # if 'age' in col.lower():
            #     val = float(col_data) / 80.0
            # elif 'gender' in col:
            #     val = float(col_data)
            # elif col in ["sleep_duration", "wakefulness_after_sleep_onset_duration"]:
            #     val = float(col_data) / 660.0
            # elif col in ["rem_sleep_ratio", "deep_sleep_ratio", "light_sleep_ratio"]:
            #     val = float(col_data)  # Already in 0-1 range
            # elif col in ["z_rhr_7", "z_hrv_7"]:
            #     val = float(col_data)  # Already z-scored
            # else:
            #     val = float(col_data) / 100.0

            # Apply same normalization as BiochargeDataset
            if 'age' in col.lower():
                val = float(col_data) / 80.0
            elif 'gender' in col:
                val = float(col_data)
            elif col in ["sleep.waso_score", "sleep.start_time_score", "sleep.duration_score", "sleep.deep_sleep_score"]:
                
                val = float(col_data) / 100
                if col == "sleep.duration_score":
                    self.sleep_duration_score = val
                elif col == "sleep.start_time_score":
                    self.sleep_start_time_score = val
                elif col == "sleep.waso_score":
                    self.waso_score = val

            # elif col in ['stress.fitness_fatigue_difference', 'hrv_factor', 'rhr_factor']: # this feature is very sparse # this should be the final version, New
            elif col in ['stress.fitness_fatigue_difference']: # this feature is very sparse
                # use yesterday row if current idx is less than sleep start
                if idx < idx_sleep_start and yesterday_date_idx is not None:
                    col_data = features_data[col_idx][yesterday_date_idx]
                else:
                    col_data = features_data[col_idx][date_idx]
                
                
                val = float(col_data) # already in range

            # elif col in ["sleep.duration_score", "sleep.deep_sleep_score", "light_sleep_ratio"]:
            #     val = float(col_data)  # Already in 0-1 range
            elif col in ["rhrScore", "hrvScore"]:

                # depending on sleep index # NEW
                if idx < idx_sleep_start and yesterday_date_idx is not None:
                    col_data = features_data[col_idx][yesterday_date_idx]
                else:
                    col_data = features_data[col_idx][date_idx]

                val = float(col_data)/100
            elif col in ["z_rhr_7", "z_hrv_7"]:
                val = float(col_data)  # Already z-scored
            else:
                val = float(col_data) / 100.0

            feats.append(val if not np.isnan(val) and not np.isinf(val) else -6.0)

        # TIMESERIES features with z-score normalization
        for col in self.cfg.ts_cols:
            # Fallback: if config says hr_filtered but data only has hr, use hr
            lookup_col = col
            if col not in column_to_idx:
                if 'hr_filtered' in col.lower() and 'timeseries.hr' in column_to_idx:
                    lookup_col = 'timeseries.hr'
                else:
                    feats.append(0.0)
                    continue

            col_idx = column_to_idx[lookup_col]
            col_data = features_data[col_idx][date_idx]

            # Extract value at idx
            if isinstance(col_data, (list, tuple)) and len(col_data) > idx:
                raw_val = float(col_data[idx])
            else:
                raw_val = 0.0

            # stress
            if 'timeseries.full_stress_list' in col.lower():
                val = raw_val
                feats.append(val if not np.isnan(val) and not np.isinf(val) else -6.0)
                continue

            # know the current sleep state
            if "timeseries.sleep_markers" in col.lower():
                self.sleep_state = raw_val # either 0 / 1

            if "timeseries.nap_state" in col.lower():
                self.nap_state = raw_val # either 0 / 1

            # Special handling for sleep_stage (biocharge logic: {4:2, 8:1, 5:0} → deep=0, REM=1, light=2)
            if 'sleep_stage' in col.lower():
                val = raw_val / 3.0  # Normalize to 0-1
                feats.append(val)

                self.sleep_stage = raw_val  # store for recovery rate feature
                continue

            # Apply z-score normalization if available
            col_string = 'timeseries.hr_filtered' if ('hr' in col.lower() and 'hrr' not in col.lower()) else col

            if 'hrr_raw' in col.lower():
                col_string = 'timeseries.hrr_raw'
                raw_val = raw_val / 100.0  # Normalize HRR raw
                feats.append(raw_val)
                continue

            if self.normalization_type == "max":
                if 'acc' in col.lower():
                    if raw_val > 150:
                        raw_val = -6.0  # missing
                    else:
                        raw_val = raw_val / 150  # Normalize ACC with population max

                if 'hr_filtered' in col.lower() or (lookup_col == 'timeseries.hr' and 'hr_filtered' in col.lower()):
                    if raw_val > 240:
                        raw_val = -6.0  # missing
                    else:
                        raw_val = raw_val / 240  # Normalize HR filtered

                val = raw_val
                
            else:

                if self.zdf and self.cfg.use_zscores and col_string in self.zdf.get('global', {}):
                    if self.z_data_norm == 'population':
                        z_std = self.zdf['global'][col_string]['std']
                        z_mean = self.zdf['global'][col_string]['mean']
                    else:
                        user_id_str = str(user_id)
                        if user_id_str in self.zdf and col_string in self.zdf[user_id_str]:
                            z_std = self.zdf[user_id_str][col_string]['std']
                            z_mean = self.zdf[user_id_str][col_string]['mean']
                        elif col_string in self.zdf['global']:
                            z_std = self.zdf['global'][col_string]['std']
                            z_mean = self.zdf['global'][col_string]['mean']
                        else:
                            feats.append(raw_val)
                            continue

                    val = (raw_val - z_mean) / z_std if z_std > 0 else 0.0

            feats.append(val if not np.isnan(val) and not np.isinf(val) else -6.0)

        # positional encoding
        if self.cfg.use_positional_encoding:
            try:
                if self.cfg.pos_encoding_type == "biocharge_circadian":
                    # total 4 more additional features, right before past charge
                    circardian_feature = self.time_encoding_for_charge_torch(idx % 1440) # 3
                    time_state = self.time_in_state(idx, nap_state=self.nap_state, nap_start=idx_nap_start, sleep_state=self.sleep_state, sleep_start=idx_sleep_start, wake_up_time= 0) # 1

                    feats.extend(circardian_feature.tolist())
                    feats.append(time_state.item())  # time_state is scalar tensor, use .item()
            except Exception:
                print(f"Error in positional encoding for user {user_id} on date {date_str}")
                if self.cfg.pos_encoding_type == "biocharge_circadian":
                    feats.extend([0.0] * 4)  # biocharge_circadian always has 4 features
                else:
                    feats.extend([0.0, 0.0])
        
        if self.use_recovery_rate_feature:
            try:
                recovery_rate = self.get_stage_recovery_rate(sleep_stage=self.sleep_stage, sleep_state=self.sleep_state)
                feats.append(recovery_rate)
            except Exception:
                feats.append(-6.0)
        
        if self.weighted_sleep_score: 
            # explicit sleep score feature
            try:
                weighted_sleep_score = (self.sleep_duration_score * 0.5) + (self.sleep_start_time_score * 0.25) + (self.waso_score * 0.25)
                feats.append(weighted_sleep_score)
            except Exception:
                feats.append(-6.0)

        # DELTA MODE: Calculate charge delta
        charge_col = self.cfg.charge_col_llm if (user_id, date_str) in self.llm_charge_pairs else self.cfg.charge_col

        if charge_col in column_to_idx:
            charge_idx = column_to_idx[charge_col]
            charge_data = features_data[charge_idx][date_idx]

            if isinstance(charge_data, (list, tuple)):
                charge_t = float(charge_data[idx]) / 100.0 if len(charge_data) > idx else 0.69
                if idx == 0 and yesterday_date_idx is not None:
                    yesterday_charge = features_data[charge_idx][yesterday_date_idx]
                    charge_t_1 = float(yesterday_charge[-1]) / 100.0 if isinstance(yesterday_charge, (list, tuple)) else 0.69
                else:
                    charge_t_1 = float(charge_data[idx - 1]) / 100.0 if idx > 0 and len(charge_data) > idx - 1 else 0.69
            else:
                charge_t, charge_t_1 = 0.69, 0.69
        else:
            charge_t, charge_t_1 = 0.69, 0.69

        charge_delta = charge_t - charge_t_1

        # Add past_charge if enabled
        if self.cfg.use_past_charge:
            feats.append(charge_t_1)

        feats_array = np.asarray(feats, dtype=np.float32)
        feats_array = np.nan_to_num(feats_array, nan=0.0, posinf=0.0, neginf=0.0)

        return feats_array, charge_delta, charge_t

    def fit_norm(self, loader: DataLoader, max_batches: int = 200):
        sums, sums2, count = None, None, 0
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                x = batch["x"].float()
                if sums is None:
                    sums, sums2 = x.sum(0), (x**2).sum(0)
                else:
                    sums += x.sum(0)
                    sums2 += (x**2).sum(0)
                count += x.shape[0]
                if bi >= max_batches:
                    break
        mu = (sums / count).numpy()
        var = (sums2 / count).numpy() - mu**2
        var = np.clip(var, 1e-8, None)
        sd = np.sqrt(var)
        self.mu = torch.from_numpy(mu).float()
        self.sd = torch.from_numpy(sd).float()

    def apply_norm(self, batch):
        if self.mu is None or self.sd is None:
            return batch
        x = batch["x"].float()
        x = (x - self.mu) / self.sd
        return {"x": x, "y": batch["y"]}

    def set_augmentation(self, enable: bool):
        """Enable or disable augmentation (useful for train/val switching)."""
        self.augment = enable

    def _initialize_feature_indices(self, total_features):
        """Initialize indices for different feature types for augmentation (torch version)."""
        if self.ts_feature_indices is not None:
            return
        
        current_idx = 0
        # Account for BMI replacement: if both height and weight are present,
        # they are replaced by a single BMI feature
        static_count = len(self.cfg.static_cols)
        if 'height' in self.cfg.static_cols and 'weight' in self.cfg.static_cols:
            static_count -= 1  # BMI replaces both height and weight (2 → 1)

        static_sleep_count = static_count + len(self.cfg.sleep_cols)
        ts_start_idx = static_sleep_count

        self.binary_sequence_indices = []  # Only for exercise (modifiable binary features)
        self.continuous_signal_indices = []
        self.sleep_stage_indices = []
        # PROTECTED: sleep_stage, nap_state, sleep_markers - values should NEVER be changed
        # (only feature dropout to -6 is acceptable)
        self.protected_state_indices = []

        for i, col in enumerate(self.cfg.ts_cols):
            feature_idx = ts_start_idx + i

            # PROTECTED features: sleep_stage, nap_state, sleep_markers
            # These should NEVER have their values modified (no jitter, no flip, no noise)
            # Only feature dropout (setting to -6) is acceptable
            if any(protected_col in col.lower() for protected_col in ['nap_state', 'sleep_markers', 'sleep_stage']):
                self.protected_state_indices.append(feature_idx)

            # Categorize binary vs continuous signals
            # Only 'exercise' is modifiable among binary features
            if 'exercise' in col.lower() and 'exercise_event' not in col.lower():
                self.binary_sequence_indices.append(feature_idx)

            if 'sleep_stage' in col.lower():
                self.sleep_stage_indices.append(feature_idx)

            # if 'hrr' in col.lower():
            #     self.hrr_indices.append(feature_idx)

            if 'hr_filtered' in col.lower() and 'hrr' not in col.lower():
                self.hr_indices.append(feature_idx)
                self.continuous_signal_indices.append(feature_idx)
            
            if 'hrr_raw' in col.lower():
                self.hr_indices.append(feature_idx)
                self.continuous_signal_indices.append(feature_idx)
            elif 'stress' in col.lower():
                self.stress_indices.append(feature_idx)
                self.continuous_signal_indices.append(feature_idx)
            elif 'trimp' in col.lower():
                self.exercise_indices.append(feature_idx)
                self.continuous_signal_indices.append(feature_idx)
            elif 'acc' in col.lower():
                self.activity_indices.append(feature_idx)
                self.continuous_signal_indices.append(feature_idx)

        self.ts_feature_indices = list(range(ts_start_idx, ts_start_idx + len(self.cfg.ts_cols)))

        self.jitter_indices = []
        for i, col in enumerate(self.cfg.ts_cols):
            feature_idx = ts_start_idx + i
            if not any(binary_col in col.lower() for binary_col in ['exercise', 'nap_state', 'sleep_markers', 'sleep_stage']):
                self.jitter_indices.append(feature_idx)

        if self.cfg.use_positional_encoding:
            if self.cfg.pos_encoding_type == "biocharge_circadian":
                pos_enc_start = ts_start_idx + len(self.cfg.ts_cols)
                pos_enc_dim = 4
                current_idx+=4
                
            else:
                pos_enc_start = ts_start_idx + len(self.cfg.ts_cols)
                pos_enc_dim = self.cfg.pos_encoding_dim if self.cfg.pos_encoding_type == "sinusoidal" else 2
                current_idx += pos_enc_dim
                
        # Add indices for windowed features if enabled
        self.windowed_hr_indices = []
        self.windowed_acc_indices = []
        self.windowed_stats_indices = []
        self.windowed_gradient_indices = []

        if self.cfg.use_windowed_features:
            current_idx = ts_start_idx + len(self.cfg.ts_cols)

            # Account for positional encoding FIRST (it comes BEFORE windowed features)
            if self.cfg.use_positional_encoding:
                if self.cfg.pos_encoding_type == "sinusoidal":
                    current_idx += self.cfg.pos_encoding_dim
                elif self.cfg.pos_encoding_type == "biocharge_circadian":
                    current_idx += 4  # 3 circadian + 1 time_state
                else:  # time_of_day
                    current_idx += 2

            # HR window indices
            self.windowed_hr_indices = list(range(current_idx, current_idx + self.cfg.window_size_minutes))
            current_idx += self.cfg.window_size_minutes

            # ACC window indices
            self.windowed_acc_indices = list(range(current_idx, current_idx + self.cfg.window_size_minutes))
            current_idx += self.cfg.window_size_minutes

            # Gradient indices if enabled
            if self.cfg.include_gradients:
                hr_grad_indices = list(range(current_idx, current_idx + self.cfg.window_size_minutes))
                current_idx += self.cfg.window_size_minutes
                acc_grad_indices = list(range(current_idx, current_idx + self.cfg.window_size_minutes))
                current_idx += self.cfg.window_size_minutes
                self.windowed_gradient_indices = hr_grad_indices + acc_grad_indices

            # Add windowed indices to continuous signal indices
            self.continuous_signal_indices.extend(self.windowed_hr_indices)
            self.continuous_signal_indices.extend(self.windowed_acc_indices)
            if self.cfg.include_gradients:
                self.continuous_signal_indices.extend(self.windowed_gradient_indices)

    def validate_sleep_stage_data(self, row: pd.Series, idx: int, col: str) -> bool:
        """
        Validate sleep stage data for consistency and reasonable values.
        
        Args:
            row: Current data row
            idx: Current time index
            col: Column name containing sleep stage data
            
        Returns:
            bool: True if data appears valid, False otherwise
        """
        try:
            sleep_stage_data = row.get(col, None)
            if sleep_stage_data is None:
                return False
                
            # Extract value at current index
            stage_value = self._safe_extract_value(sleep_stage_data, idx, -1)
            
            # Valid sleep stages are typically 0-3 (awake, light, deep, REM)
            if stage_value < 0 or stage_value > 3:
                return False
                
            return True
            
        except Exception:
            return False
        
    def get_augmentation_stats(self):
        """Return augmentation configuration for logging."""
        stats = {
            "augmentation_enabled": self.augment,
            "augmentation_probability": self.augment_prob,
           "ts_columns": self.cfg.ts_cols,
            "windowed_features_enabled": self.cfg.use_windowed_features,
            "has_sleep_stage": any('sleep_stage' in col.lower() for col in self.cfg.ts_cols),
        }
        
        if self.cfg.use_windowed_features:
            stats.update({
                "window_size_minutes": self.cfg.window_size_minutes,
                "include_gradients": self.cfg.include_gradients,
                "use_windowed_zscore": self.cfg.use_windowed_zscore,
                "include_current_hr": self.cfg.include_current_hr,
                "total_feature_count": self.get_feature_count()
            })
        
        # Add sleep stage information if available
        if hasattr(self, 'sleep_stage_indices') and self.sleep_stage_indices:
            stats.update({
                "sleep_stage_features": len(self.sleep_stage_indices),
                "sleep_stage_augmentation_enabled": True
            })
        
        return stats
    
    def debug_windowed_columns(self, user_id: str, date_str: str):
        """
        Debug utility to check what windowed feature columns are available in the data.
        
        Args:
            user_id: User identifier  
            date_str: Date string in format "YYYY-MM-DD"
            
        Returns:
            dict: Available columns and their presence
        """
        if user_id not in self.user_cache:
            try:
                excel_path = os.path.join(self.cfg.data_dir, f"{user_id}_processed.xlsx")
                if os.path.exists(excel_path):
                    df = pd.read_excel(excel_path)
                    self.user_cache[user_id] = df
                else:
                    return {"error": f"No file found for user {user_id}"}
            except Exception as e:
                return {"error": f"Failed to load file for user {user_id}: {e}"}
        
        df = self.user_cache[user_id]
        df_row = df[df["date"] == date_str]
        
        if df_row.empty:
            return {"error": f"No data found for date {date_str}"}
        
        # Check for expected windowed columns
        window_size = self.cfg.window_size_minutes
        expected_cols = {}
        
        
        
        if self.use_windowed_zscore:
            if window_size == 15:
                expected_cols = {
                    'z_norm_hr_15': 'z_norm_hr_15' in df_row.columns,
                    'z_norm_acc_15': 'z_norm_acc_15' in df_row.columns,
                    'hr_grad_zscore_15min': 'hr_grad_zscore_15min' in df_row.columns,
                    'acc_grad_zscore_15min': 'acc_grad_zscore_15min' in df_row.columns
                }
            else:
                expected_cols = {
                    'z_norm_hr_30': 'z_norm_hr_30' in df_row.columns,
                    'z_norm_acc_30': 'z_norm_acc_30' in df_row.columns,
                    'hr_grad_zscore_30min': 'hr_grad_zscore_30min' in df_row.columns,
                    'acc_grad_zscore_30min': 'acc_grad_zscore_30min' in df_row.columns}
        else:
            expected_cols = {
                'timeseries.hr_filtered_zscore': 'timeseries.hr_filtered_zscore' in df_row.columns,
                'timeseries.acc_zscore': 'timeseries.acc_zscore' in df_row.columns,
                'hr_gradient_5min_zscore': 'hr_gradient_5min_zscore' in df_row.columns,
                'acc_gradient_5min_zscore': 'acc_gradient_5min_zscore' in df_row.columns
                }
            
        
        # Get all columns containing windowed feature patterns
        windowed_cols = [col for col in df_row.columns if any(pattern in col for pattern in 
                        ['z_norm_hr', 'z_norm_acc', 'hr_grad_zscore', 'acc_grad_zscore'])]
        
        return {
            "window_size_minutes": window_size,
            "expected_columns": expected_cols,
            "all_windowed_columns": windowed_cols,
            "total_columns": len(df_row.columns)
        }



#-------------------------------------------
# Start: Data loader for selective Sampling 
#--------------------------------------------

class SampleTorchBiochargeDataset(Dataset):
    """
    Optimized dataset that loads pre-processed torch tensors.

    Applies same normalization and augmentation as BiochargeDataset.
    Uses cfg.static_cols, cfg.sleep_cols, cfg.ts_cols for column selection.
    Returns delta predictions (charge_t - charge_{t-1}).
    """

    def __init__(self, cfg: DatasetConfig, data_fraction: float = 1.0):
        super().__init__()
        self.cfg = cfg

        # Torch data directory - torch files are stored in a separate location
        self.data_dir = cfg.torch_data_dir if hasattr(cfg, 'torch_data_dir') else cfg.data_dir

        # Load index (contains [user_id, date, idx, ...])
        self.index = pd.read_csv(cfg.index_csv)
        # remove all indices start with 0, (off when plotting traejctory)
        # if not cfg.trajectory:
        #     self.index = self.index[self.index['index'] > 0]

        # Subsample data if requested (sample users, not indices, to keep user data together)
        if data_fraction < 1.0:
            # randomly sample a fraction of rows
            n_samples = int(len(self.index) * data_fraction)
            index_sub = self.index.sample(n=n_samples, random_state=42).reset_index(drop=True)
            self.index = index_sub

        self.sub_index = self.index.copy()
        # find all
        self.index["date"] = pd.to_datetime(self.index["date"])

        # Optional zscores
        self.zdf = None
        if cfg.zscores_file and cfg.use_zscores:
            with open(cfg.zscores_file, "r") as f:
                self.zdf = json.load(f)

        # Cache for user torch files (don't reload each time)
        self.user_cache: Dict[str, Dict] = {}

        # Cache for processed dates (to avoid re-filtering)
        self.processed_dates_cache: Dict[str, set] = {}


        self.generate_trajectory = cfg.generate_trajectory

        # Cache for sleep start indices
        self.sleep_start_cache: Dict[str, Dict[str, int]] = {}

        # Cache for data length validation to avoid repeated warnings
        self.data_length_cache: Dict[str, Dict[str, int]] = {}

        if cfg.llm_non_existent_userids_file is not None:
            with open(cfg.llm_non_existent_userids_file, "r") as f:
                self.no_exist_userids = set(json.load(f).keys())

        # Select which (userid, date) pairs should use LLM biocharge column
        # make sure these are not in no exist userids
        self.llm_charge_pairs = set()
        if cfg.charge_col_llm is not None and cfg.llm_col_prob > 0:
            # Get unique (userid, date) combinations

            # make sure to exclude no_exist_userids
            unique_curves = self.index[['userid', 'date']].drop_duplicates()
            unique_curves = unique_curves[~unique_curves['userid'].astype(str).isin(self.no_exist_userids)]
            # Randomly sample llm_col_prob fraction of curves

            n_llm_curves = int(len(unique_curves) * cfg.llm_col_prob)
            if n_llm_curves > 0:
                sampled_curves = unique_curves.sample(n=n_llm_curves, random_state=42)
                # Store as set of (userid, date_string) tuples for fast lookup
                self.llm_charge_pairs = set(
                    (str(row['userid']), row['date'].strftime('%Y-%m-%d'))
                    for _, row in sampled_curves.iterrows()
                )
                print(f"Selected {len(self.llm_charge_pairs)} curves ({cfg.llm_col_prob*100:.1f}%) to use LLM biocharge column")

        # Normalization
        self.mu = None
        self.sd = None

        # Augmentation settings from config
        self.augment = cfg.enable_augmentation
        self.augment_prob = cfg.augment_prob

        self.use_windowed_zscore = cfg.use_windowed_zscore

        # Define indices for different time series types (will be set during first __getitem__)
        self.ts_feature_indices = None
        self.hr_indices = []
        self.hrr_indices = []
        self.stress_indices = []
        self.exercise_indices = []
        self.activity_indices = []
        self.binary_sequence_indices = []  # For exercise, nap, sleep markers, sleep_stage
        self.continuous_signal_indices = []  # For HR, stress, accelerometer
        self.jitter_indices = []  # For temporal jittering (excludes binary sequences)
        self.sleep_stage_indices = []  # For sleep stage specific handling
        self.z_data_norm = cfg.z_data_norm

    def get_indices(self):
        return self.sub_index
    
    def _get_hr_column_name(self, df_row=None):
        """Determine which HR column to use: prefer hr_filtered if available."""
        # Check if hr_filtered column exists in the dataframe
        if df_row is not None and hasattr(df_row, 'columns'):
            if 'timeseries.hr_filtered' in df_row.columns:
                return 'timeseries.hr_filtered'
        
        # Check if hr_filtered column exists in z-score file
        if self.zdf is not None:
            if 'global' in self.zdf and 'timeseries.hr_filtered' in self.zdf['global']:
                return 'timeseries.hr_filtered'
        
        # Default fallback to regular hr column
        return 'timeseries.hr'
    
    def filter_hr_data(self, user_id: str, date_str: str):
        """
        Filter HR data by applying Hampel-like filtering only when values are outside 
        physiologically reasonable range (30-220 bpm).
        
        Args:
            user_id: User identifier
            date_str: Date string in format "YYYY-MM-DD"
        """
        if user_id not in self.user_cache:
            return
            
        df = self.user_cache[user_id]
        df_row = df[df["date"] == date_str]
        
        if df_row.empty:
            return
            
        try:
            # Get HR data - use hr_filtered if available, otherwise timeseries.hr
            hr_column = self._get_hr_column_name(df_row)
            column_series_str = df_row[hr_column].values[0]
            column_values = self.safe_literal_eval(column_series_str)
            
            if not column_values or len(column_values) == 0:
                return
                
            # Convert to tensor
            hr_tensor = torch.tensor(column_values, dtype=torch.float32)
            
            # Create mask for values outside physiological range (30-220 bpm)
            outlier_mask = (hr_tensor < 30) | (hr_tensor > 220)
            
            if not outlier_mask.any():
                # No outliers, no filtering needed
                return
                
            # Apply Hampel filter to get replacement values for outliers
            filtered_values, _ = self.hampel_filter_torch(hr_tensor, window_size=7, n_sigmas=3.0)
            
            # Only replace values that are outside physiological range
            corrected_values = hr_tensor.clone()
            corrected_values[outlier_mask] = filtered_values[outlier_mask]
            
            # Update the cached dataframe
            df_row_index = df_row.index[0]
            hr_column = self._get_hr_column_name(df_row)
            self.user_cache[user_id].at[df_row_index, hr_column] = corrected_values.numpy().tolist()
            
        except Exception as e:
            print(f"Warning: Failed to filter HR data for user {user_id} on {date_str}: {e}")


    def hampel_filter_torch(self, x: torch.Tensor, window_size: int = 7, n_sigmas: float = 3.0):
        """
        Hampel filter for outlier removal in 1D torch tensors.
        Args:
            x: 1D torch.Tensor
            window_size: int, must be odd
            n_sigmas: threshold in number of standard deviations (MAD based)
        Returns:
            filtered_x: torch.Tensor (same shape)
            mask: torch.BoolTensor indicating where values were replaced
        """
        # Ensure input is float tensor
        if x.dtype != torch.float32:
            x = x.float()
            
        if x.ndim != 1:
            raise ValueError("Input must be a 1D tensor")

        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        
        # Handle empty or very small tensors
        if x.shape[0] == 0:
            return x, torch.zeros_like(x, dtype=torch.bool)
        
        if x.shape[0] < window_size:
            # Return original tensor if it's smaller than window size
            return x, torch.zeros_like(x, dtype=torch.bool)

        k = window_size // 2
        n = x.shape[0]

        # Create unfolding windows [N, window_size]
        # Pad at both ends to handle edges
        padded = torch.nn.functional.pad(x.unsqueeze(0).unsqueeze(0), (k, k), mode='reflect')
        windows = torch.nn.functional.unfold(
            padded,
            kernel_size=(1, window_size)
        ).squeeze(0).transpose(0, 1)  # shape: (N, window_size)

        # Median and MAD (Median Absolute Deviation)
        med = windows.median(dim=1).values
        abs_dev = (windows - med.unsqueeze(1)).abs()
        mad = abs_dev.median(dim=1).values

        # Hampel threshold
        threshold = n_sigmas * 1.4826 * mad
        diff = (x - med).abs()

        # Identify outliers
        mask = diff > threshold
        filtered_x = x.clone()
        filtered_x[mask] = med[mask]

        return filtered_x, mask

    def time_of_day_encoding_continuous(self, minute_of_day):
        """
        Continuous time-of-day encoding using sine and cosine.

        Args:
            minute_of_day: int or float in [0, 1440)
        Returns:
            np.array: shape (2,) [sin(2π * t), cos(2π * t)]

            # continuous 
        """
        # Convert minutes to fraction of the day
        fraction = (minute_of_day % 1440) / 1440.0  # ensures wrap-around safety

        # Continuous cyclic encoding
        angle = 2 * np.pi * fraction
        encoding = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
        return encoding

    def time_of_day_encoding(self, minute_of_day, num_buckets=8):
        """
        Encode time of day into num_buckets (e.g., 8 buckets for 3-hour intervals).
        
        Args:
            minute_of_day: int in [0, 1440)
            num_buckets: number of intervals per day (8 for 3-hour buckets)
        
        Returns:
            np.array: encoding vector of shape (2,)  [sin, cos]
        """
        # Map minute of day to bucket
        minutes_per_bucket = 1440 // num_buckets
        bucket_idx = minute_of_day // minutes_per_bucket
        
        # Normalize bucket index to [0, 2π)
        angle = 2 * np.pi * (bucket_idx / num_buckets)
        
        # Cyclic encoding
        encoding = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
        return  encoding.astype(np.float32)

    def positional_encoding(self, time_idx, d_model=64):
        """
        Generate positional encoding for a specific time index.
        
        Args:
            time_idx: Time index in range [0, 1440] (minutes in a day)
            d_model: Dimension of the positional encoding vector
            
        Returns:
            np.array: Positional encoding vector of shape (d_model,)

        Note: This can be changed to the week of the day as well

        """
        # Normalize time_idx to [0, 1] range
        normalized_position = time_idx / 1440.0
        
        # Create positional encoding vector
        pe = np.zeros(d_model)
        
        # Generate sinusoidal encodings
        for i in range(0, d_model, 2):
            # Use different frequencies for different dimensions
            div_term = np.exp(i * -(np.log(10000.0) / d_model))
            
            # Apply sine to even indices
            pe[i] = np.sin(normalized_position * div_term)
            
            # Apply cosine to odd indices (if within bounds)
            if i + 1 < d_model:
                pe[i + 1] = np.cos(normalized_position * div_term)
        
        return pe.astype(np.float32)
    
    def _read_windowed_features(self, row: pd.Series, idx: int):
        """
        Read pre-calculated windowed features from the data file for a given row and time index.
        
        Key improvements:
        - Automatically selects correct columns based on window_size_minutes (15min vs 30min)
        - Supports include_current_hr flag to use current HR or previous HR (t-1)
        - Robust error handling with debugging information
        - Returns z-score normalized features from pre-calculated data
        
        Args:
            row: Current data row
            idx: Current time index
            
        Returns:
            dict: Dictionary containing windowed features read from file
                - hr_window: Single HR value (current or t-1 based on config)
                - acc_window: Single accelerometer value
                - hr_gradient: HR gradient (if include_gradients=True)
                - acc_gradient: Accelerometer gradient (if include_gradients=True)
        """
        features = {
            'hr_window': [],
            'acc_window': [],
            'hr_gradient': [],
            'acc_gradient': []
        }
        
        # Determine window size to use (15 or 30 minutes)
        window_size = self.cfg.window_size_minutes
        
        # Determine base data columns (needed for fallback)
        base_hr_col = self._get_hr_column_name(row)
        hr_data_column = base_hr_col
        acc_data_column = 'timeseries.acc_magnitude'
        hr_grad_data_column = 'hr_gradient_5min'
        acc_grad_data_column = 'acc_gradient_5min'

        # Choose appropriate column names based on window size
        if self.use_windowed_zscore:
            if window_size == 15:
                z_norm_hr_col = 'z_norm_hr_15'
                z_norm_acc_col = 'z_norm_acc_15'
                hr_grad_col = 'hr_grad_zscore_15min'
                acc_grad_col = 'acc_grad_zscore_15min'
            else:  # Default to 30 minutes or any other size
                z_norm_hr_col = 'z_norm_hr_30'
                z_norm_acc_col = 'z_norm_acc_30'
                hr_grad_col = 'hr_grad_zscore_30min'
                acc_grad_col = 'acc_grad_zscore_30min'
        else:
            # global normalized data columns - use dynamic HR column determination
            z_norm_hr_col = f'{base_hr_col}_zscore'
            z_norm_acc_col = 'timeseries.acc_zscore'
            hr_grad_col = 'hr_gradient_5min_zscore'
            acc_grad_col = 'acc_grad_zscore_5min'
            

        
        try:
            # Read z-score normalized HR window
            if z_norm_hr_col in row:
                hr_data = row.get(z_norm_hr_col, None)
                if hr_data is not None:
                    hr_values = self.safe_literal_eval(hr_data.values[0], default_value=[], column_name=z_norm_hr_col)
                    if isinstance(hr_values, list) and len(hr_values) > 0:
                        # Enhanced: Handle cases where idx >= len(hr_values)
                        effective_idx = min(idx, len(hr_values) - 1)
                        
                        if self.cfg.include_current_hr:
                            # Extract values from past window 15 or 30 min
                            start_idx = max(0, effective_idx - window_size)
                            end_idx = effective_idx
                            past_window = hr_values[start_idx:end_idx + 1]  # Include current
                            
                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with the first available value if we don't have enough history
                                padding_value = past_window[0] if past_window else 0.0
                                padding_needed = window_size - len(past_window)
                                past_window = [padding_value] * padding_needed + past_window
                            
                            features['hr_window'] = past_window[:window_size]  # Ensure exact size
                        else:
                            # Extract previous time point (t-1) to exclude current HR
                            prev_idx = max(0, effective_idx - 1)
                            start_idx = max(0, prev_idx - window_size)
                            past_window = hr_values[start_idx:prev_idx]
                            
                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with the first available value if we don't have enough history
                                padding_value = past_window[0] if past_window else 0.0
                                padding_needed = window_size - len(past_window)
                                past_window = [padding_value] * padding_needed + past_window
                                
                            features['hr_window'] = past_window[:window_size]  # Ensure exact size
                    else:
                        features['hr_window'] = [0.0] * window_size

            else:
                # Fallback: apply z_score normalization on the fly if pre-calculated column not found
                print(f"Warning: Pre-calculated column '{z_norm_hr_col}' not found, using on-the-fly normalization from '{hr_data_column}'")
                hr_mean, hr_std = 0.0, 1.0
                if self.zdf and 'global' in self.zdf and hr_data_column in self.zdf['global']:
                    hr_mean, hr_std = self.zdf['global'][hr_data_column]['mean'], self.zdf['global'][hr_data_column]['std']
                if hr_data_column in row:

                    hr_data = row.get(hr_data_column, None)
                    if hr_data is not None:
                        hr_values = self.safe_literal_eval(hr_data.values[0], default_value=[], column_name=hr_data_column)
                        if isinstance(hr_values, list) and len(hr_values) > 0:
                            # Enhanced: Handle cases where idx >= len(hr_values)
                            effective_idx = min(idx, len(hr_values) - 1)
                            
                            if self.cfg.include_current_hr:
                                # Extract values from past window 15 or 30 min
                                start_idx = max(0, effective_idx - window_size)
                                end_idx = effective_idx
                                past_window = hr_values[start_idx:end_idx + 1]  # Include current
                                
                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with the first available value if we don't have enough history
                                    padding_value = past_window[0] if past_window else 0.0
                                    padding_needed = window_size - len(past_window)
                                    past_window = [padding_value] * padding_needed + past_window

                                # Normalize with Z_score
                                features['hr_window'] = [(x - hr_mean) / hr_std for x in past_window[:window_size]]

                            else:
                                # Extract previous time point (t-1) to exclude current HR
                                prev_idx = max(0, effective_idx - 1)
                                start_idx = max(0, prev_idx - window_size)
                                past_window = hr_values[start_idx:prev_idx]
                                
                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with the first available value if we don't have enough history
                                    padding_value = past_window[0] if past_window else 0.0
                                    padding_needed = window_size - len(past_window)
                                    past_window = [padding_value] * padding_needed + past_window
                                    
                                features['hr_window'] = [(x - hr_mean) / hr_std for x in past_window[:window_size]]  # Ensure exact size

            
            # Read z-score normalized accelerometer window
            
            if z_norm_acc_col in row:
                acc_data = row.get(z_norm_acc_col, None)
                if acc_data is not None:
                    acc_values = self.safe_literal_eval(acc_data.values[0], default_value=[], column_name=z_norm_acc_col)
                    if isinstance(acc_values, list) and len(acc_values) > 0:
                        # Enhanced: Handle cases where idx >= len(acc_values)
                        effective_idx = min(idx, len(acc_values) - 1)

                        if self.cfg.include_current_hr:
                            # Extract values from past window including current
                            start_idx = max(0, effective_idx - window_size)
                            end_idx = effective_idx
                            past_window = acc_values[start_idx:end_idx + 1]  # Include current

                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with the first available value if we don't have enough history
                                padding_value = past_window[0] if past_window else 0.0
                                padding_needed = window_size - len(past_window)
                                past_window = [padding_value] * padding_needed + past_window

                            features['acc_window'] = past_window[:window_size]  # Ensure exact size
                        else:
                            # Extract previous time point (t-1) to exclude current ACC
                            prev_idx = max(0, effective_idx - 1)
                            start_idx = max(0, prev_idx - window_size)
                            past_window = acc_values[start_idx:prev_idx]

                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with the first available value if we don't have enough history
                                padding_value = past_window[0] if past_window else 0.0
                                padding_needed = window_size - len(past_window)
                                past_window = [padding_value] * padding_needed + past_window

                            features['acc_window'] = past_window[:window_size]  # Ensure exact size
                    else:
                        features['acc_window'] = [0.0] * window_size
            
            else:
                acc_mean, acc_std = 0.0, 1.0
                if self.zdf and 'global' in self.zdf and acc_data_column in self.zdf['global']:
                    acc_mean, acc_std = self.zdf['global'][acc_data_column]['mean'], self.zdf['global'][acc_data_column]['std']
                
                if acc_data_column in row:
                    acc_data = row.get(acc_data_column, None)
                    if acc_data is not None:
                        acc_values = self.safe_literal_eval(acc_data.values[0], default_value=[], column_name=acc_data_column)
                        if isinstance(acc_values, list) and len(acc_values) > 0:
                            # Enhanced: Handle cases where idx >= len(acc_values)
                            effective_idx = min(idx, len(acc_values) - 1)

                            if self.cfg.include_current_hr:
                                # Extract values from past window including current
                                start_idx = max(0, effective_idx - window_size)
                                end_idx = effective_idx
                                past_window = acc_values[start_idx:end_idx + 1]  # Include current

                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with the first available value if we don't have enough history
                                    padding_value = past_window[0] if past_window else 0.0
                                    padding_needed = window_size - len(past_window)
                                    past_window = [padding_value] * padding_needed + past_window

                                # Normalize with Z_score
                                features['acc_window'] = [(x - acc_mean) / acc_std for x in past_window[:window_size]]  # Ensure exact size
                            else:
                                # Extract previous time point (t-1) to exclude current ACC
                                prev_idx = max(0, effective_idx - 1)
                                start_idx = max(0, prev_idx - window_size)
                                past_window = acc_values[start_idx:prev_idx]

                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with the first available value if we don't have enough history
                                    padding_value = past_window[0] if past_window else 0.0
                                    padding_needed = window_size - len(past_window)
                                    past_window = [padding_value] * padding_needed + past_window

                                # Normalize with Z_score
                                features['acc_window'] = [(x - acc_mean) / acc_std for x in past_window[:window_size]]  # Ensure exact size
                        else:
                            features['acc_window'] = [0.0] * window_size
            # Read gradient features if enabled
            if self.cfg.include_gradients:
                # Read HR gradient
                if hr_grad_col in row:
                    hr_grad_data = row.get(hr_grad_col, None)
                    if hr_grad_data is not None:
                        hr_grad_values = self.safe_literal_eval(hr_grad_data.values[0], default_value=[], column_name=hr_grad_col)
                        if isinstance(hr_grad_values, list) and len(hr_grad_values) > 0:
                            # Enhanced: Handle cases where idx >= len(hr_grad_values)
                            effective_idx = min(idx, len(hr_grad_values) - 1)
                            start_idx = max(0, effective_idx - window_size)
                            past_window = hr_grad_values[start_idx:effective_idx]
                            
                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with zero for gradients (neutral gradient)
                                padding_needed = window_size - len(past_window)
                                past_window = [0.0] * padding_needed + past_window
                                
                            features['hr_gradient'] = past_window[:window_size]  # Ensure exact size
                        else:
                            features['hr_gradient'] = [0.0] * window_size
                else:
                    hr_grad_mean, hr_grad_std = 0.0, 1.0
                    if hr_grad_data_column in self.zdf['global']:
                        hr_grad_mean, hr_grad_std = self.zdf['global'][hr_grad_data_column]['mean'], self.zdf['global'][hr_grad_data_column]['std']
                    if hr_grad_data_column in row:
                        hr_grad_data = row.get(hr_grad_data_column, None)
                        if hr_grad_data is not None:
                            hr_grad_values = self.safe_literal_eval(hr_grad_data.values[0], default_value=[], column_name=hr_grad_data_column)
                            if isinstance(hr_grad_values, list) and len(hr_grad_values) > 0:
                                # Enhanced: Handle cases where idx >= len(hr_grad_values)
                                effective_idx = min(idx, len(hr_grad_values) - 1)
                                start_idx = max(0, effective_idx - window_size)
                                past_window = [(x-hr_grad_mean)/hr_grad_std for x in hr_grad_values[start_idx:effective_idx]]
                                
                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with zero for gradients (neutral gradient)
                                    padding_needed = window_size - len(past_window)
                                    past_window = [0.0] * padding_needed + past_window
                                    
                                features['hr_gradient'] = [(x-hr_grad_mean)/hr_grad_std for x in past_window[:window_size]]  # Ensure exact size
                            else:
                                features['hr_gradient'] = [0.0] * window_size
                if acc_grad_col in row:
                    acc_grad_data = row.get(acc_grad_col, None)
                    if acc_grad_data is not None:
                        acc_grad_values = self.safe_literal_eval(acc_grad_data.values[0], default_value=[], column_name=acc_grad_col)
                        if isinstance(acc_grad_values, list) and len(acc_grad_values) > 0:
                            # Enhanced: Handle cases where idx >= len(acc_grad_values)
                            effective_idx = min(idx, len(acc_grad_values) - 1)
                            start_idx = max(0, effective_idx - window_size)
                            past_window = acc_grad_values[start_idx:effective_idx]
                            
                            # Ensure we have enough data points
                            if len(past_window) < window_size:
                                # Pad with zero for gradients (neutral gradient)
                                padding_needed = window_size - len(past_window)
                                past_window = [0.0] * padding_needed + past_window
                                
                            features['acc_gradient'] = past_window[:window_size]  # Ensure exact size
                        else:
                            features['acc_gradient'] = [0.0] * window_size
                else:
                    acc_grad_mean, acc_grad_std = 0.0, 1.0
                    if acc_grad_data_column in self.zdf['global']:
                        acc_grad_mean, acc_grad_std = self.zdf['global'][acc_grad_data_column]['mean'], self.zdf['global'][acc_grad_data_column]['std']
                    if acc_grad_data_column in row:
                        acc_grad_data = row.get(acc_grad_data_column, None)
                        if acc_grad_data is not None:
                            acc_grad_values = self.safe_literal_eval(acc_grad_data.values[0], default_value=[], column_name=acc_grad_data_column)
                            if isinstance(acc_grad_values, list) and len(acc_grad_values) > 0:
                                # Enhanced: Handle cases where idx >= len(acc_grad_values)
                                effective_idx = min(idx, len(acc_grad_values) - 1)
                                start_idx = max(0, effective_idx - window_size)
                                past_window = [(x - acc_grad_mean)/acc_grad_std for x in acc_grad_values[start_idx:effective_idx]]
                                
                                # Ensure we have enough data points
                                if len(past_window) < window_size:
                                    # Pad with zero for gradients (neutral gradient)
                                    padding_needed = window_size - len(past_window)
                                    past_window = [0.0] * padding_needed + past_window
                                    
                                features['acc_gradient'] = [(x - acc_grad_mean)/acc_grad_std for x in past_window[:window_size]]  # Ensure exact size
                            else:
                                features['acc_gradient'] = [0.0] * window_size
        except Exception as e:
            print(f"Warning: Error reading windowed features for window_size={window_size}min: {e}")
            print(f"Available columns: {list(row.index)}")
            # Return default values on error
            features = {
                'hr_window': [0.0],
                'acc_window': [0.0], 
                'hr_gradient': [0.0],
                'acc_gradient': [0.0]
            }
        
        return features 

    def _extract_windowed_features(self, row: pd.Series, idx: int, window_size: int = 30):
        """
        Extract HR and accelerometer features from the past window_size minutes.
        
        Args:
            row: Current data row
            idx: Current time index
            window_size: Window size in minutes (default 30)
            
        Returns:
            dict: Dictionary containing windowed features and gradients
                - hr_window: Array of HR values over the window
                - acc_window: Array of accelerometer values over the window
                - hr_gradient: Linear trend in HR (change per minute)
                - acc_gradient: Linear trend in accelerometer (change per minute)
                - hr_mean, hr_std: HR statistics over the window
                - acc_mean, acc_std: Accelerometer statistics over the window
        """
        features = {
            'hr_window': [],
            'acc_window': [],
            'hr_gradient': 0.0,
            'acc_gradient': 0.0,
            'hr_mean': 0.0,
            'hr_std': 0.0,
            'acc_mean': 0.0,
            'acc_std': 0.0
        }
        
        # Determine start index for the window
        start_idx = max(0, idx - window_size)
        
        # Find HR and accelerometer columns
        hr_cols = [col for col in self.cfg.ts_cols if 'hr' in col.lower() and 'hrr' not in col.lower()]
        acc_cols = [col for col in self.cfg.ts_cols if 'acc' in col.lower()]
        
        # Extract HR window (vectorized)
        hr_values = []
        if hr_cols:
            for hr_col in hr_cols:
                hr_data = row.get(hr_col, None)
                if hr_data is not None:
                    # Vectorized extraction using list comprehension
                    hr_values = [self._safe_extract_value(hr_data, i, 0.0) 
                               for i in range(start_idx, idx + 1)]
                    break  # Use first available HR column
        
        # Extract accelerometer window (vectorized)
        acc_values = []
        if acc_cols:
            for acc_col in acc_cols:
                acc_data = row.get(acc_col, None)
                if acc_data is not None:
                    # Vectorized extraction using list comprehension
                    acc_values = [self._safe_extract_value(acc_data, i, 0.0) 
                                for i in range(start_idx, idx + 1)]
                    break  # Use first available accelerometer column
        
        # Pad windows if necessary
        target_length = window_size + 1  # +1 to include current time point
        if len(hr_values) < target_length:
            hr_values = [0.0] * (target_length - len(hr_values)) + hr_values
        if len(acc_values) < target_length:
            acc_values = [0.0] * (target_length - len(acc_values)) + acc_values
        
        # Store windowed data
        features['hr_window'] = np.array(hr_values[-target_length:], dtype=np.float32)
        features['acc_window'] = np.array(acc_values[-target_length:], dtype=np.float32)
        
        # Calculate statistics
        if len(features['hr_window']) > 0:
            features['hr_mean'] = np.mean(features['hr_window'])
            features['hr_std'] = np.std(features['hr_window']) if len(features['hr_window']) > 1 else 0.0
        
        if len(features['acc_window']) > 0:
            features['acc_mean'] = np.mean(features['acc_window'])
            features['acc_std'] = np.std(features['acc_window']) if len(features['acc_window']) > 1 else 0.0
        
        # Calculate gradients (slope of linear regression) - per minute rate of change
        if self.cfg.include_gradients and len(features['hr_window']) > 1:
            # Time points represent minutes from start of window to current time
            # Each index corresponds to 1 minute, so gradient will be per minute
            time_points_minutes = np.arange(len(features['hr_window']))  # [0, 1, 2, ..., window_size] minutes
            
            # HR gradient (rolling window linear regression slope)
            try:
                # Use rolling window approach with convolution for efficient gradient calculation
                hr_tensor = torch.from_numpy(features['hr_window']).float()
                gradient_window = 5  # 5-minute rolling window for gradient calculation
                
                if len(hr_tensor) >= gradient_window:
                    # Precompute constants for linear regression
                    x = torch.arange(gradient_window, dtype=hr_tensor.dtype)
                    x_sum = x.sum()
                    x2_sum = (x ** 2).sum()
                    denominator = gradient_window * x2_sum - x_sum ** 2
                    
                    if denominator != 0:
                        # Reshape for conv1d (batch, channels, length)
                        hr_reshaped = hr_tensor.view(1, 1, -1)
                        
                        # Compute y_sum using conv1d with ones kernel
                        kernel_ones = torch.ones(1, 1, gradient_window, dtype=hr_tensor.dtype)
                        y_sum = torch.nn.functional.conv1d(hr_reshaped, kernel_ones).squeeze()
                        
                        # Compute xy_sum using conv1d with reversed x kernel
                        kernel_x = x.flip(0).view(1, 1, -1)
                        xy_sum = torch.nn.functional.conv1d(hr_reshaped, kernel_x).squeeze()
                        
                        # Ensure tensors are at least 1D for proper indexing
                        if y_sum.dim() == 0:
                            y_sum = y_sum.unsqueeze(0)
                        if xy_sum.dim() == 0:
                            xy_sum = xy_sum.unsqueeze(0)
                        
                        # Calculate gradients for all windows
                        gradients = (gradient_window * xy_sum - x_sum * y_sum) / denominator
                        
                        # Store all gradients as a list/array for feature extraction
                        features['hr_gradient'] = gradients.detach().cpu().numpy() if gradients.dim() > 0 else np.array([0.0])
                    else:
                        features['hr_gradient'] = np.array([0.0])
                else:
                    # Fallback for short windows
                    features['hr_gradient'] = np.array([0.0])
                    
            except (RuntimeError, ValueError, IndexError):
                features['hr_gradient'] = np.array([0.0])
            
            # Accelerometer gradient (rolling window linear regression slope)
            try:
                # Use rolling window approach with convolution for efficient gradient calculation
                acc_tensor = torch.from_numpy(features['acc_window']).float()
                gradient_window = 5  # 5-minute rolling window for gradient calculation
                
                if len(acc_tensor) >= gradient_window:
                    # Precompute constants for linear regression
                    x = torch.arange(gradient_window, dtype=acc_tensor.dtype)
                    x_sum = x.sum()
                    x2_sum = (x ** 2).sum()
                    denominator = gradient_window * x2_sum - x_sum ** 2
                    
                    if denominator != 0:
                        # Reshape for conv1d (batch, channels, length)
                        acc_reshaped = acc_tensor.view(1, 1, -1)
                        
                        # Compute y_sum using conv1d with ones kernel
                        kernel_ones = torch.ones(1, 1, gradient_window, dtype=acc_tensor.dtype)
                        y_sum = torch.nn.functional.conv1d(acc_reshaped, kernel_ones).squeeze()
                        
                        # Compute xy_sum using conv1d with reversed x kernel
                        kernel_x = x.flip(0).view(1, 1, -1)
                        xy_sum = torch.nn.functional.conv1d(acc_reshaped, kernel_x).squeeze()
                        
                        # Ensure tensors are at least 1D for proper indexing
                        if y_sum.dim() == 0:
                            y_sum = y_sum.unsqueeze(0)
                        if xy_sum.dim() == 0:
                            xy_sum = xy_sum.unsqueeze(0)
                        
                        # Calculate gradients for all windows
                        gradients = (gradient_window * xy_sum - x_sum * y_sum) / denominator
                        
                        # Store all gradients as a list/array for feature extraction
                        features['acc_gradient'] = gradients.detach().cpu().numpy() if gradients.dim() > 0 else np.array([0.0])
                    else:
                        features['acc_gradient'] = np.array([0.0])
                else:
                    # Fallback for short windows
                    features['acc_gradient'] = np.array([0.0])
                    
            except (RuntimeError, ValueError, IndexError):
                features['acc_gradient'] = np.array([0.0])
        
        return features
    

    def _apply_windowed_zscore(self, window_data: np.ndarray, epsilon: float = 1e-8):
        """
        Apply z-score normalization to windowed data using the window's own statistics.
        
        Args:
            window_data: Array of windowed values
            epsilon: Small value to prevent division by zero
            
        Returns:
            np.ndarray: Z-score normalized window data
        """
        if len(window_data) <= 1:
            return window_data
        
        window_mean = np.mean(window_data)
        window_std = np.std(window_data)
        
        if window_std < epsilon:
            return np.zeros_like(window_data)
        
        return (window_data - window_mean) / window_std
    
    def safe_literal_eval(self, data_string, default_value=None, column_name="unknown"):
        """
        Safely evaluate a string containing a Python literal expression.
        
        Args:
            data_string: String to evaluate
            default_value: Value to return if parsing fails (default: empty list)
            column_name: Name of the column being parsed (for error logging)
        
        Returns:
            Parsed data or default_value if parsing fails
        """
        if default_value is None:
            default_value = []
        
        try:
            # Handle NaN or None values
            if pd.isna(data_string) or data_string is None:
                # print(f"Warning: NaN or None value found in column '{column_name}', using default value")
                return default_value
            
            # Convert to string if not already
            if not isinstance(data_string, str):
                data_string = str(data_string)
            
            # Try to parse the string
            return ast.literal_eval(data_string)
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Warning: Failed to parse data in column '{column_name}': {e}. Using default value.")
            return default_value
    
    def get_feature_count(self):
        """
        Calculate the total number of features that will be generated.
        Useful for model initialization.

        IMPORTANT: Order must match _get_feature_vector exactly:
        1. Static + Sleep features
        2. Timeseries features
        3. Positional encoding (if enabled)
        4. Windowed features (if enabled)
        5. Past charge (if enabled)

        Returns:
            int: Total feature count
        """
        # 1. Static + Sleep + Timeseries
        # Account for BMI replacement: if both height and weight are present,
        # they are replaced by a single BMI feature in _get_feature_vector
        static_count = len(self.cfg.static_cols)
        if 'height' in self.cfg.static_cols and 'weight' in self.cfg.static_cols:
            static_count -= 1  # BMI replaces both height and weight (2 → 1)

        feature_count = (static_count +
                        len(self.cfg.sleep_cols) +
                        len(self.cfg.ts_cols))

        # 2. Positional encoding (comes BEFORE windowed features)
        if self.cfg.use_positional_encoding:
            if self.cfg.pos_encoding_type == "sinusoidal":
                feature_count += self.cfg.pos_encoding_dim
            elif self.cfg.pos_encoding_type == "biocharge_circadian":
                feature_count += 4  # 3 circadian features + 1 time_state feature
            else:  # "time_of_day"
                feature_count += 2  # time_of_day_encoding returns 2 elements [sin, cos]

        # 3. Windowed features (comes AFTER positional encoding)
        if self.cfg.use_windowed_features:
            # Windowed features add window_size_minutes features for HR and ACC each
            # The actual implementation returns slices of length window_size_minutes
            feature_count += self.cfg.window_size_minutes  # HR window slice
            feature_count += self.cfg.window_size_minutes  # ACC window slice

            if self.cfg.include_gradients:
                # Gradient features also add window_size_minutes features each
                feature_count += self.cfg.window_size_minutes  # HR gradient window slice
                feature_count += self.cfg.window_size_minutes  # ACC gradient window slice

        # 4. Past charge (comes LAST, after windowed features)
        if self.cfg.use_past_charge:
            feature_count += 1

        return feature_count

    def flip_label(self, y, p=0.02):
        if np.random.rand() < p:
            return 1 - y
        return y
    
    def augment_sleep_stage_transitions(self, x_aug, sleep_stage_indices):
        """
        Apply simple signal corruption to sleep stage data.
        
        Args:
            x_aug: Feature vector to augment
            sleep_stage_indices: Indices of sleep stage features
        """
        if not sleep_stage_indices:
            return x_aug
            
        for idx in sleep_stage_indices:
            if idx < len(x_aug):
                current_stage = x_aug[idx]
                
                # Simple noise corruption with 10% probability
                if np.random.random() < 0.1:
                    # Add small amount of noise
                    noise = np.random.normal(0, 0.05)  # 5% noise
                    x_aug[idx] = np.clip(current_stage + noise, 0.0, 1.0)
        
        return x_aug
    
    def get_sleep_stage_info(self):
        """
        Get information about sleep_stage columns and their configuration.
        
        Returns:
            dict: Sleep stage configuration information
        """
        sleep_stage_cols = [col for col in self.cfg.ts_cols if 'sleep_stage' in col.lower()]
        
        return {
            "sleep_stage_columns": sleep_stage_cols,
            "sleep_stage_count": len(sleep_stage_cols),
            "sleep_stage_indices_initialized": hasattr(self, 'sleep_stage_indices') and bool(self.sleep_stage_indices),
            "augmentation_includes_sleep_stage": self.augment and len(sleep_stage_cols) > 0
        }

    def __len__(self):
        return len(self.index)

    def time_encoding_for_charge_torch(self, minute_of_day):
        """
        Torch version of time encoding aligned with biocharge circadian model.
        minute_of_day: int, float, or torch.Tensor in [0, 1440)
        Returns: torch.FloatTensor of shape (3,)
        """
        import torch
        pi = torch.tensor(np.pi) if not hasattr(torch, 'pi') else torch.pi
        if not torch.is_tensor(minute_of_day):
            minute_of_day = torch.tensor(minute_of_day, dtype=torch.float32)
        minute_of_day = minute_of_day % 1440
        fraction = minute_of_day / 1440.0
        sin_24h = torch.sin(2 * pi * fraction)
        cos_24h = torch.cos(2 * pi * fraction)
        circadian_output = (
            torch.cos(2 * pi * (minute_of_day - 1080) / 1440) +
            0.5 * torch.cos(2 * pi * (minute_of_day - 1260) / 720)
        )
        return torch.stack([sin_24h, cos_24h, circadian_output]).to(torch.float32)

    def time_in_state(self, minute, sleep_start, nap_start, wake_up_time=0, sleep_state=0, nap_state=0,
                    wake_up_scale=960.0, sleep_scale=480.0, nap_scale=90.0):
        '''Time since awake, sleep, or nap, normalized by typical duration.
        sleep_state: 0=awake, 1=sleep; nap_state: 0=not nap, 1=nap'''
        import torch
        minute = torch.tensor(minute, dtype=torch.float32) if not torch.is_tensor(minute) else minute
        if nap_state == 1:
            t = torch.clamp(minute - nap_start, min=0)
            scale = nap_scale
        elif sleep_state == 0:
            t = torch.clamp(minute - wake_up_time, min=0)
            scale = wake_up_scale
        else:
            t = torch.clamp(minute - sleep_start, min=0)
            scale = sleep_scale
        norm = torch.log1p(t) / torch.log1p(torch.tensor(scale))
        return norm

    def find_sleep_start(self, series):
        """
        Given a binary sequence (0/1), find the index of the last 0->1 transition
        (searching from the end of the sequence).
        Returns None if no transition found or data is malformed.
        """
        try:
            if series is None or series.empty:
                return None
                
            series_val = series.iloc[0]
            arr = None
            
            if isinstance(series_val, str):
                try:
                    parsed = ast.literal_eval(series_val)
                    arr = np.array(parsed).astype(int)
                except (ValueError, SyntaxError, TypeError):
                    return None
            
            elif isinstance(series_val, (list, tuple)):
                try:
                    arr = np.array(series_val).astype(int)
                except (ValueError, TypeError):
                    return None
                    
            elif isinstance(series_val, np.ndarray):
                try:
                    arr = series_val.astype(int)
                except (ValueError, TypeError):
                    return None
            else:
                return None
            
            if arr is None or len(arr) <= 1:
                return None
            
            # Scan backwards to find 0->1 transition
            for i in range(len(arr) - 1, 0, -1):
                if arr[i] == 1 and arr[i-1] == 0:
                    return i
                    
            return None  # No transition found
            
        except Exception:
            return None


    def _load_user_file(self, user_id: str, date, idx) -> pd.DataFrame:
        """Load user file from folder (cached)."""

        date_str = date.strftime("%Y-%m-%d") 
        if user_id in self.user_cache:
            df = self.user_cache[user_id]
        else:
            # if userid in llm users then use self.cfg.llm_date_dir
            if self.cfg.data_dir_llm is not None and (user_id, date_str) in self.llm_charge_pairs:
                excel_path = os.path.join(self.cfg.data_dir_llm, f"{user_id}_processed.xlsx")
            else:
                excel_path = os.path.join(self.cfg.data_dir, f"{user_id}_processed.xlsx")

            # csv_path = os.path.join(self.cfg.data_dir, f"{user_id}.csv")
            # parquet_path = os.path.join(self.cfg.data_dir, f"{user_id}.parquet")

            if os.path.exists(excel_path):
                df = pd.read_excel(excel_path)
            else:
                raise FileNotFoundError(f"No file found for user {user_id}")
            
            self.user_cache[user_id] = df
            # Initialize processed dates cache for this user
            # self.processed_dates_cache[user_id] = set()


        # Try csv or parquet
        # read excel file

        df_row = df[df["date"] == date_str]
        df_yesterday_row = df[df["date"] == (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")]

        # # Check if this date has already been processed for this user
        # if user_id not in self.processed_dates_cache:
        #     self.processed_dates_cache[user_id] = set()
            
        # if date_str not in self.processed_dates_cache[user_id]:
        #     # Filter hr data only if not already processed - selective filtering for outliers only
        #     try:
        #         self.filter_hr_data(user_id, date_str)
        #         # Mark this date as processed
        #         self.processed_dates_cache[user_id].add(date_str)
                    
        #     except Exception as e:
        #         print(f"Warning: Failed to filter HR data for user {user_id} on {date_str}: {e}")
        #         # Mark as processed to avoid repeated attempts
        #         self.processed_dates_cache[user_id].add(date_str)
        
        # Get the updated row from cache
        df_row = self.user_cache[user_id][self.user_cache[user_id]["date"] == date_str]

        # Initialize caches for this user if needed
        if user_id not in self.processed_dates_cache:
            self.processed_dates_cache[user_id] = set()
        if user_id not in self.sleep_start_cache:
            self.sleep_start_cache[user_id] = {}
        
        # Check if sleep_start_idx is already cached for this date
        if date_str in self.sleep_start_cache[user_id]:
            sleep_start_idx = self.sleep_start_cache[user_id][date_str]
        else:
            # Calculate and cache sleep_start_idx
            sleep_start_idx = self.find_sleep_start(df_row['timeseries.sleep_markers'])
            self.sleep_start_cache[user_id][date_str] = sleep_start_idx

        # add to processed dates cache
        self.processed_dates_cache[user_id].add(date_str)


        # if sleep_start_idx is None:
        #     print(f"No sleep start found for user {user_id} on date {date_str}, using idx {idx}")
        # if df_yesterday_row.empty:
        #     prev_charge = 69
        # else:
        #     prev_charge = float(df_yesterday_row.iloc[0][self.cfg.charge_col])

        df["date"] = pd.to_datetime(df["date"])
        # self.user_cache[user_id] = df
        return df_row, df_yesterday_row, sleep_start_idx

    def _safe_extract_value(self, data, idx=None, default=0.0):
        """
        Safely extract value from various data types with robust error handling.
        Enhanced with better out-of-bounds protection and data length mismatch handling.
        """
        try:
            if data is None:
                return default
            
            # Handle all numpy scalar types
            if isinstance(data, (int, float, np.integer, np.floating, np.int64, np.int32, np.float64, np.float32)):
                val = float(data)
                return val if not (np.isnan(val) or np.isinf(val)) else default
            
            # Handle numpy boolean
            if isinstance(data, (bool, np.bool_)):
                return float(data)
            
            # Handle numpy arrays
            if isinstance(data, np.ndarray):
                if data.size == 0:
                    return default
                if idx is not None:
                    if 0 <= idx < len(data):
                        val = float(data[idx])
                        return val if not (np.isnan(val) or np.isinf(val)) else default
                    else:
                        # Enhanced: Handle out-of-bounds with intelligent fallback
                        if len(data) > 0:
                            # Use last available value for out-of-bounds access
                            val = float(data[-1])
                            return val if not (np.isnan(val) or np.isinf(val)) else default
                        return default
                else:
                    # Return first element if no index specified
                    val = float(data.flat[0])
                    return val if not (np.isnan(val) or np.isinf(val)) else default
            
            if isinstance(data, str):
                try:
                    # Try to parse as list/array
                    parsed = ast.literal_eval(data)
                    if isinstance(parsed, (list, tuple)) and idx is not None:
                        if 0 <= idx < len(parsed):
                            val = float(parsed[idx])
                            return val if not (np.isnan(val) or np.isinf(val)) else default
                        else:
                            # Enhanced: Handle out-of-bounds with intelligent fallback
                            if len(parsed) > 0:
                                # Use last available value for out-of-bounds access
                                val = float(parsed[-1])
                                return val if not (np.isnan(val) or np.isinf(val)) else default
                            return default
                    val = float(parsed)
                    return val if not (np.isnan(val) or np.isinf(val)) else default
                except (ValueError, SyntaxError, IndexError):
                    # If parsing fails, try direct float conversion
                    try:
                        val = float(data)
                        return val if not (np.isnan(val) or np.isinf(val)) else default
                    except (ValueError, TypeError):
                        return default
            
            if isinstance(data, (list, tuple)):
                if idx is not None:
                    if 0 <= idx < len(data):
                        val = data[idx]
                        # Recursively handle numpy types in lists
                        if isinstance(val, (np.integer, np.floating)):
                            val = float(val)
                        return val if not (np.isnan(val) or np.isinf(val)) else default
                    else:
                        # Enhanced: Handle out-of-bounds with intelligent fallback
                        if len(data) > 0:
                            # Use last available value for out-of-bounds access
                            val = data[-1]
                            if isinstance(val, (np.integer, np.floating)):
                                val = float(val)
                            return val if not (np.isnan(val) or np.isinf(val)) else default
                        return default
                elif idx is None and len(data) > 0:
                    val = data[0]
                    # Recursively handle numpy types in lists
                    if isinstance(val, (np.integer, np.floating)):
                        val = float(val)
                    return val if not (np.isnan(val) or np.isinf(val)) else default
                else:
                    return default
            
            # For pandas Series
            if hasattr(data, 'iloc'):
                try:
                    val = data.iloc[0]
                    return self._safe_extract_value(val, idx, default)
                except (IndexError, KeyError):
                    return default
            
            # For pandas scalar values
            if hasattr(data, 'item'):
                try:
                    val = float(data.item())
                    return val if not (np.isnan(val) or np.isinf(val)) else default
                except (ValueError, TypeError):
                    return default
            
            # Last resort: try direct float conversion
            try:
                val = float(data)
                return val if not (np.isnan(val) or np.isinf(val)) else default
            except (ValueError, TypeError):
                return default
            
        except Exception as e:
            # Enhanced: Add more specific error logging for debugging
            import traceback
            print(f"Debug: Exception in _safe_extract_value - data type: {type(data)}, idx: {idx}, error: {e}")
            print(f"Debug: Traceback: {traceback.format_exc()}")
            return default

    def _get_feature_vector(self, row: pd.Series, yesterday_row, idx: int, idx_sleep_start: int, charge_col: str = None) -> np.ndarray:
        feats = []

        
        # Use provided charge_col or fall back to config
        if charge_col is None:
            charge_col = self.cfg.charge_col

        # Handle malformed yesterday_row
        if yesterday_row is None or yesterday_row.empty:
            yesterday_row = row

        # Validate idx_sleep_start
        if idx_sleep_start is None:
            idx_sleep_start = 0
            
        # Static + sleep features with robust error handling
        # First, calculate BMI if we have height and weight in static_cols
        bmi_value = None
        if 'height' in self.cfg.static_cols and 'weight' in self.cfg.static_cols:
            try:
                height_val = self._safe_extract_value(row.get('height', None))  # in cm
                weight_val = self._safe_extract_value(row.get('weight', None))  # in kg
                if height_val > 0:
                    bmi_value = weight_val / ((height_val / 100) ** 2)  # BMI formula
                    bmi_value = bmi_value / 30.0  # Normalize BMI (typical range 15-40, so /30)
                else:
                    bmi_value = -6
            except:
                bmi_value = -6

        for col in self.cfg.static_cols + self.cfg.sleep_cols:

            # print(f"col: {col}")
            try:
                row_ = row.get(col, None)
                row_yesterday_ = yesterday_row.get(col, None)

                if row_ is None:
                    feats.append(-6.0)
                    continue

                # Handle BMI calculation (replace height and weight with BMI)
                if col == 'height' or col == 'weight':
                    # Skip individual height/weight, we'll use BMI instead
                    if col == 'weight' and bmi_value is not None:
                        # Add BMI when we encounter weight (skip height entirely)
                        feats.append(float(bmi_value))
                    # Skip height entirely
                    continue

                # normalize by constants with safe extraction
                elif 'age' in col.lower():
                    val = self._safe_extract_value(row_) / 80

                elif 'gender' in col:
                    val = self._safe_extract_value(row_)

                else:
                    # Determine which row to use based on sleep timing for ALL sleep metrics
                    # If before sleep start and yesterday's data exists, use yesterday's sleep data
                    # Otherwise use today's sleep data (including when row_yesterday_ is None)
                    if idx < idx_sleep_start and row_yesterday_ is not None:
                        source_row = row_yesterday_
                    else:
                        # # Use default values when today's sleep data isn't available yet
                        # if col not in ["z_rhr_7", "z_hrv_7", ]:
                        #     val = 0.0  # Default for z-scored values
                        # elif col in ["sleep_duration", "wakefulness_after_sleep_onset_duration"]:
                        #     val = 0.0  # Default for duration
                        # elif col in ["rem_sleep_ratio", "deep_sleep_ratio", "light_sleep_ratio"]:
                        #     val = 0.0  # Default for ratios
                        # else:
                        #     val = -6  # Missing indicator for other sleep features
                        # Don't use source_row, just use the default val set above
                        source_row = row_

                    # Only process from source_row if it's available
                    if source_row is not None:
                        # Handle sleep duration and waso columns - normalize by 660 minutes
                        if col in ["sleep_duration", "wakefulness_after_sleep_onset_duration"]:
                            val = self._safe_extract_value(source_row) / 660.0

                        # Handle sleep ratio columns (already in 0-1 range, just use directly)
                        elif col in ["rem_sleep_ratio", "deep_sleep_ratio", "light_sleep_ratio"]:
                            val = self._safe_extract_value(source_row)  # Already normalized to 0-1

                        # Handle z_rhr_7 and z_hrv_7 - already z-scored, use directly without normalization
                        elif col in ["z_rhr_7", "z_hrv_7"]:
                            val = self._safe_extract_value(source_row)  # Already z-scored, use as-is

                        # Legacy sleep duration columns (if still used)
                        elif col in ["deep_sleep_duration", "light_sleep_duration", "rem_sleep_duration"]:
                            val = self._safe_extract_value(source_row) / 720  # Normalize by max 12 hours

                        else:
                            val = self._safe_extract_value(source_row) / 100
                
                # Ensure the value is valid
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    val = -6.0
                
                # print(f"{col}: {val}")
                feats.append(float(val))
                
            except Exception as e:
                # Log malformed data for debugging (optional)
                print(f"Error processing column {col}: {e}")
                feats.append(0.0)

        # Timeseries columns with robust error handling
        for col in self.cfg.ts_cols:
            # print(f"col: {col}")

            try:
                row_ = row.get(col, None)
                if row_ is None:
                    feats.append(0.0)
                    continue
                
                #------------------------------------------------------------------------------
                #         Event Data (exercise, sleep_stage, sleep_markers, nap_state)
                # -----------------------------------------------------------------------------
                if 'exercise' in col.lower():
                    # Extract sleep stage value at specific index
                    raw_val = self._safe_extract_value(row_, idx, 0.0)
                    
                    feats.append(float(val) if not np.isnan(val) else 0.0)
                    continue
                
                if 'sleep_markers' in col.lower():
                    # Extract sleep stage value at specific index
                    raw_val = self._safe_extract_value(row_, idx, 0.0)
                    
                    feats.append(float(val) if not np.isnan(val) else 0.0)
                    continue
                
                if 'nap_state' in col.lower():
                    # Extract sleep stage value at specific index
                    raw_val = self._safe_extract_value(row_, idx, 0.0)
                    
                    feats.append(float(val) if not np.isnan(val) else 0.0)
                    continue

                # Special handling for sleep_stage column
                if 'sleep_stage' in col.lower():
                    # Extract sleep stage value at specific index
                    raw_val = self._safe_extract_value(row_, idx, 0.0)
                    
                    # Sleep stages are typically encoded as: -1=awake, 1=light, 2=deep, 3=REM
                    # Normalize to 0-1 range: 0=awake, 0.33=light, 0.66=deep, 1.0=REM
                    val =raw_val/3
                    feats.append(float(val) if not np.isnan(val) else 0.0)
                    continue
                
                # Safely extract time series value at specific index
                raw_val = self._safe_extract_value(row_, idx, 0.0)
                
                #---------------------------------------------------------
                #         Timeseries Data (hr)
                # --------------------------------------------------------
                if 'hr' in col.lower() and 'hrr' not in col.lower():
                    col_string = self._get_hr_column_name()
                else:
                    col_string = col
                
                # Apply z-score normalization if available
                if self.zdf is not None and self.cfg.use_zscores and (col_string in self.zdf['global']):
                    try:

                        if self.z_data_norm == 'population':
                            if 'global' in self.zdf and col_string in self.zdf['global']:
                                z_std = self.zdf['global'][col_string]['std']
                                z_mean = self.zdf['global'][col_string]['mean']
                            else:
                                # Fallback to raw normalization
                                val = raw_val / 100
                        else:
                            user_id_str = str(int(self._safe_extract_value(row.get('userid', ''))))
                            
                            # Try user-specific z-scores first
                            if (user_id_str in self.zdf and 
                                col_string in self.zdf[user_id_str]):
                                z_std = self.zdf[user_id_str][col_string]['std']
                                z_mean = self.zdf[user_id_str][col_string]['mean']
                            elif ('global' in self.zdf and 
                                col_string in self.zdf['global']):
                                z_std = self.zdf['global'][col_string]['std']
                                z_mean = self.zdf['global'][col_string]['mean']
                            else:
                                # No z-score available, use raw value
                                val = raw_val
                                feats.append(float(val) if not np.isnan(val) else 0.0)
                                continue
                        
                        # Apply z-score normalization
                        val = (raw_val - z_mean) / z_std if z_std > 0 else 0.0
                        
                    except (KeyError, TypeError, ValueError, IndexError):
                        val = raw_val
                else:
                    val = raw_val
                

                # RHR centered hr 
                if 'z_hr_filtered_7' in col.lower():
                    val = self._safe_extract_value('z_hr_filtered_7', idx, 0.0)  # mean_hr_7


                #---------------------------------------------------------
                #         Timeseries Data (acc)
                #--------------------------------------------------------
                if 'timeseries.acc' in col.lower():
                    
                    # Apply z-score normalization if available
                    if self.zdf is not None and self.cfg.use_zscores and (col_string in self.zdf['global']):
                        try:
                            if self.z_data_norm == 'population':
                                if 'global' in self.zdf and col_string in self.zdf['global']:
                                    z_std = self.zdf['global'][col_string]['std']
                                    z_mean = self.zdf['global'][col_string]['mean']
                                else:
                                    # Fallback to raw normalization
                                    val = raw_val / 100
                            else:
                                user_id_str = str(int(self._safe_extract_value(row.get('userid', ''))))
                                
                                # Try user-specific z-scores first
                                if (user_id_str in self.zdf and 
                                    col_string in self.zdf[user_id_str]):
                                    z_std = self.zdf[user_id_str][col_string]['std']
                                    z_mean = self.zdf[user_id_str][col_string]['mean']
                                elif ('global' in self.zdf and 
                                    col_string in self.zdf['global']):
                                    z_std = self.zdf['global'][col_string]['std']
                                    z_mean = self.zdf['global'][col_string]['mean']
                                else:
                                    # No z-score available, use raw value
                                    val = raw_val
                                    feats.append(float(val) if not np.isnan(val) else 0.0)
                                    continue
                            
                            # Apply z-score normalization
                            val = (raw_val - z_mean) / z_std if z_std > 0 else 0.0
                            
                        except (KeyError, TypeError, ValueError, IndexError):
                            val = raw_val

                # Ensure valid float value
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    val = -6            
                # print(f"timeseries{col}: {val}")
                feats.append(float(val))

            except Exception as e:
                # Log malformed time series data for debugging (optional)
                # print(f"Error processing time series column {col} at index {idx}: {e}")
                feats.append(0.0)

                
        # Add positional encoding if enabled
        if self.cfg.use_positional_encoding:
            try:
                if self.cfg.pos_encoding_type == "time_of_day":
                    # Use time_of_day_encoding which returns a 2-element cyclic encoding [sin, cos]
                    pos_encoding = self.time_of_day_encoding(idx % 1440)
                    feats.extend(pos_encoding.tolist())
                elif self.cfg.pos_encoding_type == "sinusoidal":
                    # Use sinusoidal positional encoding with configurable dimensions
                    
                    # pos_encoding = self.positional_encoding(idx % 1440, self.cfg.pos_encoding_dim)
                    pos_encoding = self.time_of_day_encoding_continuous(idx % 1440)
                    feats.extend(pos_encoding.tolist())
                else:
                    # Default to time_of_day if invalid type specified
                    pos_encoding = self.time_of_day_encoding(idx % 1440)
                    feats.extend(pos_encoding.tolist())
            except Exception:
                # Fallback to zero encoding if there's an error
                if self.cfg.pos_encoding_type == "sinusoidal":
                    # Use pos_encoding_dim zeros for sinusoidal
                    feats.extend([0.0] * self.cfg.pos_encoding_dim)
                else:
                    # Use 2 zeros for time_of_day (default)
                    feats.extend([0.0, 0.0])

        # Add windowed features if enabled
        if self.cfg.use_windowed_features:
            try:
                # Use the new method to read pre-calculated windowed features from file
                windowed_features = self._read_windowed_features(row, idx)
                
                # Add windowed HR values (already z-score normalized from file)
                if isinstance(windowed_features['hr_window'], list) and len(windowed_features['hr_window']) > 0:
                    feats.extend(windowed_features['hr_window'])
                else:
                    # Fallback: add zeros for entire window if data not available
                    feats.extend([0.0] * self.cfg.window_size_minutes)

                # Add windowed accelerometer values (already z-score normalized from file)
                if isinstance(windowed_features['acc_window'], list) and len(windowed_features['acc_window']) > 0:
                    feats.extend(windowed_features['acc_window'])
                else:
                    # Fallback: add zeros for entire window if data not available
                    feats.extend([0.0] * self.cfg.window_size_minutes)
                
                # Don't add summary
                
                # Add gradients if enabled (already z-score normalized from file)
                if self.cfg.include_gradients:
                    # Add HR gradient
                    if isinstance(windowed_features['hr_gradient'], list) and len(windowed_features['hr_gradient']) > 0:
                        feats.extend(windowed_features['hr_gradient'])
                    else:
                        feats.extend([0.0] * self.cfg.window_size_minutes)

                    # Add accelerometer gradient
                    if isinstance(windowed_features['acc_gradient'], list) and len(windowed_features['acc_gradient']) > 0:
                        feats.extend(windowed_features['acc_gradient'])
                    else:
                        feats.extend([0.0] * self.cfg.window_size_minutes)
                    
            except Exception as e:
                # Fallback to zero features if there's an error
                print(f"Error reading windowed features: {e}")
                
                # Add zero windowed values (window slices)
                feats.extend([0.0] * self.cfg.window_size_minutes)  # HR window slice
                feats.extend([0.0] * self.cfg.window_size_minutes)  # ACC window slice
                
                # Add zero gradients if enabled
                if self.cfg.include_gradients:
                    feats.extend([0.0] * self.cfg.window_size_minutes)  # HR gradient window slice
                    feats.extend([0.0] * self.cfg.window_size_minutes)  # ACC gradient window slice

        # DELTA MODE: Extract charge values for delta calculation
        # Get charge at time t and t-1
        try:
            charge_t_list = row.get(charge_col, None)
            if charge_t_list is not None:
                charge_t = self._safe_extract_value(charge_t_list, idx, 69.0)
                if idx == 0:
                    # last night's charge
                    # Megha : Monitor this value and add for each user. 
                    charge_t_1 = self._safe_extract_value(yesterday_row.get(charge_col, None), -1, 69.0) # get the last night charge's last value
                else:
                    charge_t_1 = self._safe_extract_value(charge_t_list, idx-1, 69.0)
            else:
                charge_t = 69.0  # Default charge value
                charge_t_1 = 69.0

            # Normalize to [0, 1] range
            charge_t = charge_t / 100.0
            charge_t_1 = charge_t_1 / 100.0
        except Exception:
            charge_t = 69.0 / 100.0
            charge_t_1 = 69.0 / 100.0

        # Append past charge value (ABSOLUTE, not delta) as feature
        # This is what the model will use to predict the delta
        if self.cfg.use_past_charge:
            feats.append(charge_t_1)

        # DELTA MODE: Calculate delta (change in charge)
        # Target is delta = charge_t - charge_{t-1}
        charge_delta = charge_t - charge_t_1

        # Convert to numpy array with robust NaN handling
        feats = np.asarray(feats, dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure charge_delta is valid
        if not isinstance(charge_delta, (int, float)) or np.isnan(charge_delta) or np.isinf(charge_delta):
            charge_delta = 0.0  # Default delta is 0 (no change)

        # charge_t_recon = None
        # if self.cfg.charge_reconstruction:
        charge_t_recon = charge_t if isinstance(charge_t, float) else 0.0

        # Return features with past_charge (absolute) and target delta
        return np.asarray(feats, dtype=np.float32), charge_delta, charge_t_recon

    def _validate_data_length(self, user_id: str, date: pd.Timestamp, idx: int, row: pd.Series) -> bool:
        """
        Validate that the requested index is within reasonable bounds for the data.
        Returns True if data seems valid, False if there are major inconsistencies.
        """
        try:
            # Check a few key time series columns to see their lengths
            sample_cols = [col for col in self.cfg.ts_cols[:3]]  # Check first 3 columns
            data_lengths = []
            
            for col in sample_cols:
                col_data = row.get(col, None)
                if col_data is not None:
                    try:
                        if isinstance(col_data, str):
                            parsed = ast.literal_eval(col_data)
                            if isinstance(parsed, (list, tuple)):
                                data_lengths.append(len(parsed))
                        elif isinstance(col_data, (list, tuple)):
                            data_lengths.append(len(col_data))
                        elif isinstance(col_data, np.ndarray):
                            data_lengths.append(len(col_data))
                    except:
                        continue
            
            if not data_lengths:
                return False  # No valid data found
            
            min_length = min(data_lengths)
            max_length = max(data_lengths)
            
            # Log data inconsistencies for debugging
            if idx >= min_length:
                print(f"Data length warning: User {user_id}, Date {date.strftime('%Y-%m-%d')}, "
                      f"requested idx={idx}, but min data length={min_length}, max={max_length}")
                
                # Still return True - we'll handle this in _safe_extract_value
                return True
            
            # Check for severely truncated data (less than 50% of expected day length)
            if max_length < 720:  # Less than 12 hours of data
                print(f"Severely truncated data: User {user_id}, Date {date.strftime('%Y-%m-%d')}, "
                      f"max data length={max_length} (expected ~1440)")
            
            return True
            
        except Exception as e:
            print(f"Error validating data length for user {user_id}, date {date}: {e}")
            return True  # Continue processing with enhanced error handling

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        """
        Load data from torch tensors and apply same normalization/augmentation as BiochargeDataset.
        Returns delta predictions (charge_t - charge_{t-1}) in DELTA MODE.
        """
        try:
            item        = self.index.iloc[i]
            user_id     = str(item['userid'])
            date        = pd.to_datetime(item['date'])
            idx         = int(item['index'])

            # Load user torch data (with caching and memory mapping)
            if user_id not in self.user_cache:
                torch_path = os.path.join(self.data_dir, f"{user_id}.pt")
                self.user_cache[user_id] = torch.load(torch_path, mmap=True, weights_only=False)

            data = self.user_cache[user_id]
            date_str = date.strftime('%Y-%m-%d')
            query_date = np.datetime64(date_str, "D").astype("int64")

            # Find the date index in torch data
            date_matches = (data["dates"] == query_date).nonzero(as_tuple=True)[0]
            if len(date_matches) == 0:
                # Date not found - return default
                raise ValueError(f"Date {date_str} not found for user {user_id}")

            date_idx = date_matches[0].item()

            # Extract raw values from torch tensor and apply normalization
            x, charge_delta, charge_recon = self._extract_and_normalize_torch(
                data, date_idx, idx, user_id, date_str
            )

            # Apply augmentation if enabled
            if self.augment:
                x, charge_delta = self.augment_sample(x, charge_delta)

            # Add x_add_1 logic for rollout calculation during training
            try:
                x_add_1, charge_delta_add_1, charge_recon_add_1 = self._extract_and_normalize_torch(
                    data, date_idx, idx+1, user_id, date_str
                )
                y_add_1 = charge_delta_add_1
                
               
            except:
                # idx+1 is beyond current day, try tomorrow's data
                try:
                    tomorrow_date = date + pd.Timedelta(days=1)
                    tomorrow_date_str = tomorrow_date.strftime('%Y-%m-%d')
                    tomorrow_query = np.datetime64(tomorrow_date_str, "D").astype("int64")
                    tomorrow_matches = (data["dates"] == tomorrow_query).nonzero(as_tuple=True)[0]
                    
                    if len(tomorrow_matches) > 0:
                        tomorrow_date_idx = tomorrow_matches[0].item()
                        x_add_1, charge_delta_add_1, charge_recon_add_1 = self._extract_and_normalize_torch(
                            data, tomorrow_date_idx, 0, user_id, tomorrow_date_str
                        )
                        y_add_1 = charge_delta_add_1
                    else:
                        # Next day doesn't exist
                        x_add_1 = np.full(self.get_feature_count(), -6.0, dtype=np.float32)
                        y_add_1 = -6.0
                except:
                    # Error loading next day
                    x_add_1 = np.full(self.get_feature_count(), -6.0, dtype=np.float32)
                    y_add_1 = -6.0

            # Masking logic
            mask = 1.0
            # Check for non-wear from torch data
            if "features_data" in data and "column_to_idx" in data:
                if "timeseries.min_status_list" in data["column_to_idx"]:
                    min_status_idx = data["column_to_idx"]["timeseries.min_status_list"]
                    min_status_data = data["features_data"][min_status_idx][date_idx]
                    if isinstance(min_status_data, (list, tuple)) and len(min_status_data) > idx:
                        if min_status_data[idx] == 3:
                            mask = 0.0

            # if idx > 1500:
            #     mask = 0.0

            return {
                "x": torch.from_numpy(x).float(),
                "y": torch.tensor([charge_delta * 100], dtype=torch.float32),  # DELTA scaled
                "mask": torch.tensor([mask], dtype=torch.float32),
                'meta': {'user_id': user_id, 'date': date_str, 'idx': idx},
                'charge_recon': charge_recon, 
                "x_add_1": torch.from_numpy(x_add_1).float(),
                "y_add_1": torch.tensor([y_add_1*100], dtype=torch.float32)
            }

        except Exception as e:
            print(f"Error in TorchBiochargeDataset[{i}]: {e}")
            # Return default values
            default_dim = len(self.cfg.static_cols) + len(self.cfg.sleep_cols) + len(self.cfg.ts_cols)
            x = np.zeros(default_dim, dtype=np.float32)
            x_add_1 = np.full(self.get_feature_count(), -6.0, dtype=np.float32)
            return {
                "x": torch.from_numpy(x).float(),
                "y": torch.tensor([0.0], dtype=torch.float32),
                "mask": torch.tensor([0.0], dtype=torch.float32),
                "x_add_1": torch.from_numpy(x_add_1).float(),
                "y_add_1": torch.tensor([-6.0*100], dtype=torch.float32)
            }

    def _extract_and_normalize_torch(self, data, date_idx, idx, user_id, date_str):
        """
        Extract raw values from torch tensor and apply BiochargeDataset-style normalization.
        This matches the normalization logic in BiochargeDataset._get_feature_vector.
        """
        feats = []
        features_data = data["features_data"]
        column_to_idx = data["column_to_idx"]

        # Get yesterday's data for sleep features (if idx < sleep_start)
        yesterday_date = (pd.to_datetime(date_str) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        yesterday_query = np.datetime64(yesterday_date, "D").astype("int64")
        yesterday_matches = (data["dates"] == yesterday_query).nonzero(as_tuple=True)[0]
        yesterday_date_idx = yesterday_matches[0].item() if len(yesterday_matches) > 0 else None

        # simulate real-world HR availability
        # hr_available = (np.random.rand() < 0.25)

        # Find sleep start index from sleep_markers
        idx_sleep_start = 0
        if "timeseries.sleep_markers" in column_to_idx:
            sleep_markers_idx = column_to_idx["timeseries.sleep_markers"]
            sleep_markers = features_data[sleep_markers_idx][date_idx]
            if isinstance(sleep_markers, (list, tuple)):
                for i, val in enumerate(sleep_markers):
                    if val == 1:
                        idx_sleep_start = i
                        break

        # Find nap start index from nap_state
        idx_nap_start = 0
        if "timeseries.nap_state" in column_to_idx:
            nap_state_idx = column_to_idx["timeseries.nap_state"]
            nap_state_data = features_data[nap_state_idx][date_idx]
            if isinstance(nap_state_data, (list, tuple)):
                for i, val in enumerate(nap_state_data):
                    if val == 1:
                        idx_nap_start = i
                        break

        # Initialize sleep/nap state tracking for positional encoding
        current_sleep_state = 0
        current_nap_state = 0

        # Calculate BMI if height and weight are both present (matching BiochargeDataset)
        bmi_value = None
        if 'height' in self.cfg.static_cols and 'weight' in self.cfg.static_cols:
            if 'height' in column_to_idx and 'weight' in column_to_idx:
                height_val = float(features_data[column_to_idx['height']][date_idx])
                weight_val = float(features_data[column_to_idx['weight']][date_idx])
                if height_val > 0:
                    bmi_value = weight_val / ((height_val / 100) ** 2)
                    bmi_value = bmi_value / 30.0  # Normalize BMI
                else:
                    bmi_value = -6.0

        # STATIC + SLEEP features with normalization (matching BiochargeDataset logic)
        for col in self.cfg.static_cols + self.cfg.sleep_cols:
            if col not in column_to_idx:
                feats.append(-6.0)
                continue

            col_idx = column_to_idx[col]

            # Determine which date to use for sleep features
            if col in self.cfg.sleep_cols and idx < idx_sleep_start and yesterday_date_idx is not None:
                col_data = features_data[col_idx][yesterday_date_idx]
            else:
                col_data = features_data[col_idx][date_idx]

            # Handle BMI replacement (matching BiochargeDataset behavior)
            if col == 'height' or col == 'weight':
                if col == 'weight' and bmi_value is not None:
                    # Add BMI when we encounter weight (skip height entirely)
                    feats.append(float(bmi_value))
                # Skip height entirely
                continue

            # Apply same normalization as BiochargeDataset
            if 'age' in col.lower():
                val = float(col_data) / 80.0
            elif 'gender' in col:
                val = float(col_data)
            elif col in ["sleep_duration", "wakefulness_after_sleep_onset_duration"]:
                val = float(col_data) / 660.0
            elif col in ["rem_sleep_ratio", "deep_sleep_ratio", "light_sleep_ratio"]:
                val = float(col_data)  # Already in 0-1 range
            elif col in ["z_rhr_7", "z_hrv_7"]:
                val = float(col_data)  # Already z-scored
            else:
                val = float(col_data) / 100.0

            feats.append(val if not np.isnan(val) and not np.isinf(val) else -6.0)

        # TIMESERIES features with z-score normalization
        for col in self.cfg.ts_cols:
            if col not in column_to_idx:
                feats.append(0.0)
                continue

            col_idx = column_to_idx[col]
            col_data = features_data[col_idx][date_idx]

            # Extract value at idx
            if isinstance(col_data, (list, tuple)) and len(col_data) > idx:
                raw_val = float(col_data[idx])
            else:
                raw_val = 0.0

            # Special handling for sleep_stage
            if 'sleep_stage' in col.lower():
                val = raw_val / 3.0  # Normalize to 0-1
                feats.append(val)
                continue

            # Apply z-score normalization if available
            col_string = 'timeseries.hr_filtered' if ('hr' in col.lower() and 'hrr' not in col.lower()) else col

            
            if 'hrr_raw' in col.lower():
                col_string = 'timeseries.hrr_raw'
                raw_val = raw_val / 100.0  # Normalize HRR raw
                
                # Check if HR should be dropped entirely
                if hasattr(self.cfg, 'drop_hr') and self.cfg.drop_hr:
                    feats.append(-6.0)
                    continue

                if not self.generate_trajectory: # during training
                    # Get exercise, sleep_markers, and nap_state at current idx
                    exercise_val = 0
                    sleep_markers_val = 0
                    nap_state_val = 0

                    if "timeseries.exercise" in column_to_idx:
                        exercise_data = features_data[column_to_idx["timeseries.exercise"]][date_idx]
                        if isinstance(exercise_data, (list, tuple)) and len(exercise_data) > idx:
                            exercise_val = float(exercise_data[idx])

                    if "timeseries.sleep_markers" in column_to_idx:
                        sleep_markers_data = features_data[column_to_idx["timeseries.sleep_markers"]][date_idx]
                        if isinstance(sleep_markers_data, (list, tuple)) and len(sleep_markers_data) > idx:
                            sleep_markers_val = float(sleep_markers_data[idx])

                    if "timeseries.nap_state" in column_to_idx:
                        nap_state_data = features_data[column_to_idx["timeseries.nap_state"]][date_idx]
                        if isinstance(nap_state_data, (list, tuple)) and len(nap_state_data) > idx:
                            nap_state_val = float(nap_state_data[idx])

                    # Apply conditional logic
                    if exercise_val == 1:
                        # Always use raw value when exercising
                        final_hrr = raw_val
                    elif sleep_markers_val == 1 or nap_state_val == 1:
                        # Use value 1/5th of the time during sleep/nap, otherwise set to -6
                        if np.random.random() < 0.25:
                            final_hrr = raw_val
                        else:
                            final_hrr = -6.0
                    else:
                        # When awake, use value 1/10th of the time, otherwise set to -6
                        if np.random.random() < 0.2:
                            final_hrr = raw_val
                        else:
                            final_hrr = -6.0

                    feats.append(final_hrr)
                    continue
                else:
                    # during trajectory generation
                    # get 'hr_available' column from torch data
                    if "timeseries.hr_available" in column_to_idx:
                        hr_available_idx = column_to_idx["timeseries.hr_available"]
                        hr_available_data = features_data[hr_available_idx][date_idx]

                        # Check per-minute availability if it's a list
                        if isinstance(hr_available_data, (list, tuple)) and len(hr_available_data) > idx:
                            hr_avail_at_idx = hr_available_data[idx]
                        else:
                            hr_avail_at_idx = hr_available_data  # single value for whole day

                        if hr_avail_at_idx == 1:
                            feats.append(raw_val)  # raw_val is already normalized
                            continue
                        else:
                            feats.append(-6.0)
                            continue
                    else:
                        # Fallback: no hr_available column, use -6 (missing)
                        feats.append(-6.0)
                        continue


            if self.zdf and self.cfg.use_zscores and col_string in self.zdf.get('global', {}):
                if self.z_data_norm == 'population':
                    z_std = self.zdf['global'][col_string]['std']
                    z_mean = self.zdf['global'][col_string]['mean']
                else:
                    user_id_str = str(user_id)
                    if user_id_str in self.zdf and col_string in self.zdf[user_id_str]:
                        z_std = self.zdf[user_id_str][col_string]['std']
                        z_mean = self.zdf[user_id_str][col_string]['mean']
                    elif col_string in self.zdf['global']:
                        z_std = self.zdf['global'][col_string]['std']
                        z_mean = self.zdf['global'][col_string]['mean']
                    else:
                        feats.append(raw_val)
                        continue

                val = (raw_val - z_mean) / z_std if z_std > 0 else 0.0
            else:
                val = raw_val

            feats.append(val if not np.isnan(val) and not np.isinf(val) else -6.0)

            # Track sleep/nap state for positional encoding
            if "timeseries.sleep_markers" in col.lower():
                current_sleep_state = raw_val
            if "timeseries.nap_state" in col.lower():
                current_nap_state = raw_val

        # Positional encoding (must come BEFORE past_charge, which is always last)
        if self.cfg.use_positional_encoding:
            try:
                if self.cfg.pos_encoding_type == "biocharge_circadian":
                    # total 4 more additional features, right before past charge
                    circardian_feature = self.time_encoding_for_charge_torch(idx % 1440)  # 3 features
                    time_state = self.time_in_state(idx, nap_state=current_nap_state, nap_start=idx_nap_start,
                                                   sleep_state=current_sleep_state, sleep_start=idx_sleep_start,
                                                   wake_up_time=0)  # 1 feature
                    feats.extend(circardian_feature.tolist())
                    feats.append(time_state.item())  # time_state is scalar tensor, use .item()
            except Exception:
                if self.cfg.pos_encoding_type == "biocharge_circadian":
                    feats.extend([0.0] * 4)  # biocharge_circadian always has 4 features
                else:
                    feats.extend([0.0, 0.0])

        # DELTA MODE: Calculate charge delta
        charge_col = self.cfg.charge_col_llm if (user_id, date_str) in self.llm_charge_pairs else self.cfg.charge_col

        if charge_col in column_to_idx:
            charge_idx = column_to_idx[charge_col]
            charge_data = features_data[charge_idx][date_idx]

            if isinstance(charge_data, (list, tuple)):
                charge_t = float(charge_data[idx]) / 100.0 if len(charge_data) > idx else 0.69
                if idx == 0 and yesterday_date_idx is not None:
                    yesterday_charge = features_data[charge_idx][yesterday_date_idx]
                    charge_t_1 = float(yesterday_charge[-1]) / 100.0 if isinstance(yesterday_charge, (list, tuple)) else 0.69
                else:
                    charge_t_1 = float(charge_data[idx - 1]) / 100.0 if idx > 0 and len(charge_data) > idx - 1 else 0.69
            else:
                charge_t, charge_t_1 = 0.69, 0.69
        else:
            charge_t, charge_t_1 = 0.69, 0.69

        charge_delta = charge_t - charge_t_1

        # Add past_charge if enabled
        if self.cfg.use_past_charge:
            feats.append(charge_t_1)

        feats_array = np.asarray(feats, dtype=np.float32)
        feats_array = np.nan_to_num(feats_array, nan=0.0, posinf=0.0, neginf=0.0)

        return feats_array, charge_delta, charge_t

    def fit_norm(self, loader: DataLoader, max_batches: int = 200):
        sums, sums2, count = None, None, 0
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                x = batch["x"].float()
                if sums is None:
                    sums, sums2 = x.sum(0), (x**2).sum(0)
                else:
                    sums += x.sum(0)
                    sums2 += (x**2).sum(0)
                count += x.shape[0]
                if bi >= max_batches:
                    break
        mu = (sums / count).numpy()
        var = (sums2 / count).numpy() - mu**2
        var = np.clip(var, 1e-8, None)
        sd = np.sqrt(var)
        self.mu = torch.from_numpy(mu).float()
        self.sd = torch.from_numpy(sd).float()

    def apply_norm(self, batch):
        if self.mu is None or self.sd is None:
            return batch
        x = batch["x"].float()
        x = (x - self.mu) / self.sd
        return {"x": x, "y": batch["y"]}

    def set_augmentation(self, enable: bool):
        """Enable or disable augmentation (useful for train/val switching)."""
        self.augment = enable

    def _initialize_feature_indices(self, total_features):
        """Initialize indices for different feature types for augmentation (torch version)."""
        if self.ts_feature_indices is not None:
            return

        # Account for BMI replacement: if both height and weight are present,
        # they are replaced by a single BMI feature
        static_count = len(self.cfg.static_cols)
        if 'height' in self.cfg.static_cols and 'weight' in self.cfg.static_cols:
            static_count -= 1  # BMI replaces both height and weight (2 → 1)

        static_sleep_count = static_count + len(self.cfg.sleep_cols)
        ts_start_idx = static_sleep_count

        self.binary_sequence_indices = []  # Only for exercise (modifiable binary features)
        self.continuous_signal_indices = []
        self.sleep_stage_indices = []
        # PROTECTED: sleep_stage, nap_state, sleep_markers - values should NEVER be changed
        # (only feature dropout to -6 is acceptable)
        self.protected_state_indices = []

        for i, col in enumerate(self.cfg.ts_cols):
            feature_idx = ts_start_idx + i

            # PROTECTED features: sleep_stage, nap_state, sleep_markers
            # These should NEVER have their values modified (no jitter, no flip, no noise)
            # Only feature dropout (setting to -6) is acceptable
            if any(protected_col in col.lower() for protected_col in ['nap_state', 'sleep_markers', 'sleep_stage']):
                self.protected_state_indices.append(feature_idx)

            # Categorize binary vs continuous signals
            # Only 'exercise' is modifiable among binary features
            if 'exercise' in col.lower() and 'exercise_event' not in col.lower():
                self.binary_sequence_indices.append(feature_idx)

            if 'sleep_stage' in col.lower():
                self.sleep_stage_indices.append(feature_idx)

            # Use hrr_raw for hr_indices (matches TorchBiochargeDataset)
            if 'hrr_raw' in col.lower():
                self.hr_indices.append(feature_idx)
                self.continuous_signal_indices.append(feature_idx)
            elif 'stress' in col.lower():
                self.stress_indices.append(feature_idx)
                self.continuous_signal_indices.append(feature_idx)
            elif 'trimp' in col.lower():
                self.exercise_indices.append(feature_idx)
                self.continuous_signal_indices.append(feature_idx)
            elif 'acc' in col.lower():
                self.activity_indices.append(feature_idx)
                self.continuous_signal_indices.append(feature_idx)

        self.ts_feature_indices = list(range(ts_start_idx, ts_start_idx + len(self.cfg.ts_cols)))

        self.jitter_indices = []
        for i, col in enumerate(self.cfg.ts_cols):
            feature_idx = ts_start_idx + i
            if not any(binary_col in col.lower() for binary_col in ['exercise', 'nap_state', 'sleep_markers', 'sleep_stage']):
                self.jitter_indices.append(feature_idx)

        # Add indices for windowed features if enabled
        self.windowed_hr_indices = []
        self.windowed_acc_indices = []
        self.windowed_stats_indices = []
        self.windowed_gradient_indices = []

        if self.cfg.use_windowed_features:
            current_idx = ts_start_idx + len(self.cfg.ts_cols)

            # Account for positional encoding FIRST (it comes BEFORE windowed features)
            if self.cfg.use_positional_encoding:
                if self.cfg.pos_encoding_type == "sinusoidal":
                    current_idx += self.cfg.pos_encoding_dim
                elif self.cfg.pos_encoding_type == "biocharge_circadian":
                    current_idx += 4  # 3 circadian + 1 time_state
                else:  # time_of_day
                    current_idx += 2

            # HR window indices
            self.windowed_hr_indices = list(range(current_idx, current_idx + self.cfg.window_size_minutes))
            current_idx += self.cfg.window_size_minutes

            # ACC window indices
            self.windowed_acc_indices = list(range(current_idx, current_idx + self.cfg.window_size_minutes))
            current_idx += self.cfg.window_size_minutes

            # Gradient indices if enabled
            if self.cfg.include_gradients:
                hr_grad_indices = list(range(current_idx, current_idx + self.cfg.window_size_minutes))
                current_idx += self.cfg.window_size_minutes
                acc_grad_indices = list(range(current_idx, current_idx + self.cfg.window_size_minutes))
                current_idx += self.cfg.window_size_minutes
                self.windowed_gradient_indices = hr_grad_indices + acc_grad_indices

            # Add windowed indices to continuous signal indices
            self.continuous_signal_indices.extend(self.windowed_hr_indices)
            self.continuous_signal_indices.extend(self.windowed_acc_indices)
            if self.cfg.include_gradients:
                self.continuous_signal_indices.extend(self.windowed_gradient_indices)

    def augment_sample(self, x, y):
        """Augment features with physiologically realistic transformations (torch version)."""
        if not self.augment or np.random.random() > self.augment_prob:
            return x, y

        self._initialize_feature_indices(len(x))

        # Skip augmentation if hrr_raw is -6 (missing value)
        if self.hr_indices:
            for idx in self.hr_indices:
                if idx < len(x) and x[idx] == -6:
                    return x, y

        x_aug = x.copy()

        # Binary sequence augmentation (ONLY exercise - NOT sleep_stage, nap_state, sleep_markers)
        # Reasoning: sleep_stage, nap_state, and sleep_markers represent ground-truth physiological
        # states that should not be artificially modified.
        if self.binary_sequence_indices and np.random.random() < 0.4:
            for idx in self.binary_sequence_indices:
                # Skip protected features (sleep_stage, nap_state, sleep_markers)
                if idx in self.protected_state_indices:
                    continue
                if np.random.random() < 0.2:  # 20% flip probability
                    x_aug[idx] = 1.0 - x_aug[idx] if x_aug[idx] in [0.0, 1.0] else x_aug[idx]

        # Heart rate augmentation
        if self.hr_indices and np.random.random() < 0.7:
            for idx in self.hr_indices:
                hr_val = x_aug[idx]
                # Only apply augmentation when hrr_raw is not -6 (not masked)
                if hr_val != -6.0:
                    if np.random.random() < 0.8:
                        hrv_noise = np.random.normal(0, 0.025)
                        x_aug[idx] = hr_val + hrv_noise
                    elif np.random.random() < 0.1:
                        artifact_factor = np.random.choice([0.9, 1.1])
                        x_aug[idx] = hr_val * artifact_factor

        # Continuous signal augmentation (stress, acc, etc)
        if self.continuous_signal_indices and np.random.random() < 0.6:
            for idx in self.continuous_signal_indices:
                signal_val = x_aug[idx]
                if signal_val != -6 and np.random.random() < 0.7:  # Skip if missing
                    noise_std = 0.03 if idx in self.stress_indices else 0.025
                    sensor_noise = np.random.normal(0, noise_std)
                    x_aug[idx] = signal_val + sensor_noise

        # Exercise/TRIMP augmentation
        if self.exercise_indices and np.random.random() < 0.4:
            for idx in self.exercise_indices:
                if np.random.random() < 0.8:
                    performance_factor = np.random.uniform(0.85, 1.15)
                    x_aug[idx] = x_aug[idx] * performance_factor

        # Temporal jittering
        if np.random.random() < 0.3:
            jitter_factor = np.random.uniform(0.98, 1.02)
            for idx in self.jitter_indices:
                if idx < len(x_aug):
                    x_aug[idx] = x_aug[idx] * jitter_factor

        # === STATIC FEATURES NOISE AUGMENTATION (AGE, HEIGHT, WEIGHT/BMI) ===
        # Add very small noise to static features (age, height, weight/BMI)
        # if np.random.random() < 0.3:
        #     # Calculate static feature indices
        #     static_count = len(self.cfg.static_cols)
        #     if 'height' in self.cfg.static_cols and 'weight' in self.cfg.static_cols:
        #         static_count -= 1  # BMI replaces both height and weight (2 → 1)
            
        #     for idx in range(static_count):
        #         if idx < len(x_aug):
        #             # Very small noise for static features (0.5-1.5% std)
        #             noise_std = np.random.uniform(0.005, 0.015)
        #             noise = np.random.normal(0, noise_std)
        #             x_aug[idx] = x_aug[idx] + noise
                    # *** HIGHLIGHT: Adding small noise to static features (age, height, weight/BMI) ***



        # === WINDOWED FEATURE AUGMENTATION ===
        if self.cfg.use_windowed_features:
            # HR window augmentation
            for idx in self.windowed_hr_indices:
                if idx < len(x_aug):
                    hr_window_val = x_aug[idx]
                    if hr_window_val != -6 and np.random.random() < 0.7:  # Skip if missing
                        hrv_noise = np.random.normal(0, 0.025)
                        x_aug[idx] = hr_window_val + hrv_noise

            # Accelerometer window augmentation
            for idx in self.windowed_acc_indices:
                if idx < len(x_aug):
                    acc_window_val = x_aug[idx]
                    if acc_window_val != -6 and np.random.random() < 0.6:  # Skip if missing
                        sensor_noise = np.random.normal(0, 0.025)
                        x_aug[idx] = acc_window_val + sensor_noise

        # === PAST CHARGE NOISE AUGMENTATION ===
        if self.cfg.use_past_charge and np.random.random() < 0.25:
            past_charge_idx = len(x_aug) - 1
            if x_aug[past_charge_idx] != -6:  # Skip if missing
                noise = np.random.normal(0, 0.015)  # 1.5% std - very subtle
                x_aug[past_charge_idx] = x_aug[past_charge_idx] + noise

        return x_aug, y
    
    def validate_sleep_stage_data(self, row: pd.Series, idx: int, col: str) -> bool:
        """
        Validate sleep stage data for consistency and reasonable values.
        
        Args:
            row: Current data row
            idx: Current time index
            col: Column name containing sleep stage data
            
        Returns:
            bool: True if data appears valid, False otherwise
        """
        try:
            sleep_stage_data = row.get(col, None)
            if sleep_stage_data is None:
                return False
                
            # Extract value at current index
            stage_value = self._safe_extract_value(sleep_stage_data, idx, -1)
            
            # Valid sleep stages are typically 0-3 (awake, light, deep, REM)
            if stage_value < 0 or stage_value > 3:
                return False
                
            return True
            
        except Exception:
            return False
        
    def get_augmentation_stats(self):
        """Return augmentation configuration for logging."""
        stats = {
            "augmentation_enabled": self.augment,
            "augmentation_probability": self.augment_prob,
           "ts_columns": self.cfg.ts_cols,
            "windowed_features_enabled": self.cfg.use_windowed_features,
            "has_sleep_stage": any('sleep_stage' in col.lower() for col in self.cfg.ts_cols),
        }
        
        if self.cfg.use_windowed_features:
            stats.update({
                "window_size_minutes": self.cfg.window_size_minutes,
                "include_gradients": self.cfg.include_gradients,
                "use_windowed_zscore": self.cfg.use_windowed_zscore,
                "include_current_hr": self.cfg.include_current_hr,
                "total_feature_count": self.get_feature_count()
            })
        
        # Add sleep stage information if available
        if hasattr(self, 'sleep_stage_indices') and self.sleep_stage_indices:
            stats.update({
                "sleep_stage_features": len(self.sleep_stage_indices),
                "sleep_stage_augmentation_enabled": True
            })
        
        return stats
    
    def debug_windowed_columns(self, user_id: str, date_str: str):
        """
        Debug utility to check what windowed feature columns are available in the data.
        
        Args:
            user_id: User identifier  
            date_str: Date string in format "YYYY-MM-DD"
            
        Returns:
            dict: Available columns and their presence
        """
        if user_id not in self.user_cache:
            try:
                excel_path = os.path.join(self.cfg.data_dir, f"{user_id}_processed.xlsx")
                if os.path.exists(excel_path):
                    df = pd.read_excel(excel_path)
                    self.user_cache[user_id] = df
                else:
                    return {"error": f"No file found for user {user_id}"}
            except Exception as e:
                return {"error": f"Failed to load file for user {user_id}: {e}"}
        
        df = self.user_cache[user_id]
        df_row = df[df["date"] == date_str]
        
        if df_row.empty:
            return {"error": f"No data found for date {date_str}"}
        
        # Check for expected windowed columns
        window_size = self.cfg.window_size_minutes
        expected_cols = {}
        
        
        
        if self.use_windowed_zscore:
            if window_size == 15:
                expected_cols = {
                    'z_norm_hr_15': 'z_norm_hr_15' in df_row.columns,
                    'z_norm_acc_15': 'z_norm_acc_15' in df_row.columns,
                    'hr_grad_zscore_15min': 'hr_grad_zscore_15min' in df_row.columns,
                    'acc_grad_zscore_15min': 'acc_grad_zscore_15min' in df_row.columns
                }
            else:
                expected_cols = {
                    'z_norm_hr_30': 'z_norm_hr_30' in df_row.columns,
                    'z_norm_acc_30': 'z_norm_acc_30' in df_row.columns,
                    'hr_grad_zscore_30min': 'hr_grad_zscore_30min' in df_row.columns,
                    'acc_grad_zscore_30min': 'acc_grad_zscore_30min' in df_row.columns}
        else:
            expected_cols = {
                'timeseries.hr_filtered_zscore': 'timeseries.hr_filtered_zscore' in df_row.columns,
                'timeseries.acc_zscore': 'timeseries.acc_zscore' in df_row.columns,
                'hr_gradient_5min_zscore': 'hr_gradient_5min_zscore' in df_row.columns,
                'acc_gradient_5min_zscore': 'acc_gradient_5min_zscore' in df_row.columns
                }
            
        
        # Get all columns containing windowed feature patterns
        windowed_cols = [col for col in df_row.columns if any(pattern in col for pattern in 
                        ['z_norm_hr', 'z_norm_acc', 'hr_grad_zscore', 'acc_grad_zscore'])]
        
        return {
            "window_size_minutes": window_size,
            "expected_columns": expected_cols,
            "all_windowed_columns": windowed_cols,
            "total_columns": len(df_row.columns)
        }

#-------------------------------------------
# End: Data loader for selective Sampling 
#--------------------------------------------

class AdvancedAugmentedBiochargeDataset(BiochargeDataset):
    """
    Advanced augmentation dataset with sophisticated time series augmentation techniques.
    Use this for more aggressive augmentation during training.
    """
    def __init__(self, cfg, augment_prob=0.3, enable_advanced_aug=True, data_fraction=1.0):
        super().__init__(cfg, data_fraction=data_fraction)
        self.augment_prob = augment_prob
        self.enable_advanced_aug = enable_advanced_aug
        
    def augment_sample(self, x, y):
        """Enhanced augmentation with multiple strategies."""
        if not self.augment or np.random.random() > self.augment_prob:
            return x, y
            
        # Call parent augmentation first
        x_aug, y = super().augment_sample(x, y)
        
        if not self.enable_advanced_aug:
            return x_aug, y
        
        # Advanced augmentation techniques (applied on top of basic augmentation)
        
        # === ADVANCED BINARY SEQUENCE TECHNIQUES ===
        if self.binary_sequence_indices and np.random.random() < 0.25:
            # 1. State Pattern Disruption (simulate irregular behavior patterns)
            for idx in self.binary_sequence_indices:
                if np.random.random() < 0.3:
                    # Add "micro-interruptions" - brief state changes
                    interruption_noise = np.random.normal(0, 0.1)
                    x_aug[idx] = x_aug[idx] + interruption_noise
                    
        # 2. Binary Sequence Smoothing (simulate gradual state transitions)
        if self.binary_sequence_indices and np.random.random() < 0.2:
            for idx in self.binary_sequence_indices:
                # Smooth sharp binary transitions
                if 0.1 < x_aug[idx] < 0.9:  # Near binary transitions
                    smoothing_factor = np.random.uniform(0.9, 1.1)
                    x_aug[idx] = x_aug[idx] * smoothing_factor
        
        # === ADVANCED HEART RATE TECHNIQUES ===
        if self.hr_indices and np.random.random() < 0.3:
            # 3. Circadian Rhythm Simulation
            for idx in self.hr_indices:
                # Simulate time-of-day effects on HR
                circadian_factor = np.random.uniform(0.95, 1.05)
                x_aug[idx] = x_aug[idx] * circadian_factor
        
        # 4. Exercise Response Simulation
        if self.hr_indices and self.exercise_indices and np.random.random() < 0.2:
            # Correlate HR changes with exercise state
            exercise_intensity = np.mean([x_aug[idx] for idx in self.exercise_indices])
            if exercise_intensity > 0.5:  # During exercise
                hr_boost = np.random.uniform(1.1, 1.3)  # 10-30% HR increase
                for idx in self.hr_indices:
                    x_aug[idx] = x_aug[idx] * hr_boost
        
        # === PHYSIOLOGICAL STATE COMPLEXES ===
        # 5. Sleep State Complex (affects multiple signals together)
        if self.binary_sequence_indices and self.hr_indices and np.random.random() < 0.15:
            sleep_indices = [idx for idx in self.binary_sequence_indices 
                           if any('sleep' in col for col in self.cfg.ts_cols)]
            if sleep_indices:
                sleep_state = np.mean([x_aug[idx] for idx in sleep_indices])
                if sleep_state > 0.5:  # During sleep
                    # Reduce HR during sleep
                    sleep_hr_factor = np.random.uniform(0.8, 0.9)
                    for idx in self.hr_indices:
                        x_aug[idx] = x_aug[idx] * sleep_hr_factor
        
        # 6. Stress State Complex
        if self.stress_indices and self.hr_indices and np.random.random() < 0.2:
            # Simulate acute stress response
            stress_multiplier = np.random.uniform(1.1, 1.25)
            for idx in self.hr_indices + self.stress_indices:
                x_aug[idx] = x_aug[idx] * stress_multiplier
        
        # 7. Feature Masking (simulate sensor dropouts)
        if np.random.random() < 0.1 and self.ts_feature_indices:
            # Randomly mask 1-2 time series features (sensor failure simulation)
            n_mask = np.random.randint(1, min(3, len(self.ts_feature_indices)))
            mask_indices = np.random.choice(self.ts_feature_indices, n_mask, replace=False)
            for idx in mask_indices:
                if idx not in self.binary_sequence_indices:  # Don't mask binary sequences completely
                    x_aug[idx] = x_aug[idx] * 0.1  # Reduce to 10% (partial sensor failure)
        
        # 8. Magnitude Warping (different for binary vs continuous)
        if np.random.random() < 0.2:
            # Binary sequences: gentle warping
            for idx in self.binary_sequence_indices:
                warp_factor = np.random.uniform(0.95, 1.05)
                x_aug[idx] = x_aug[idx] * warp_factor
            
            # Continuous signals: more aggressive warping
            for idx in self.continuous_signal_indices:
                if idx not in self.binary_sequence_indices:
                    warp_factor = np.random.uniform(0.85, 1.15)
                    x_aug[idx] = x_aug[idx] * warp_factor
        
        # === BOUNDS CHECKING FOR ADVANCED TECHNIQUES ===
        # Ensure binary sequences stay in [0,1] range
        for idx in self.binary_sequence_indices:
            x_aug[idx] = np.clip(x_aug[idx], 0.0, 1.0)
        
        # Ensure continuous signals stay within reasonable bounds
        for idx in self.continuous_signal_indices:
            x_aug[idx] = np.clip(x_aug[idx], -5.0, 5.0)
        
        return x_aug, y
    
    
# def collate_skip_none(batch):
#     # Drop Nones that came from __getitem__
#     batch = [b for b in batch if b is not None]
#     if not batch:  # edge case: batch had only None
#         return None
#     return default_collate(batch)

class SequenceBiochargeDataset(BiochargeDataset):
    """
    Dataset that returns sequences of K consecutive timesteps for rollout training.
    Each sample is a sequence of (x_0, y_0), (x_1, y_1), ..., (x_{K-1}, y_{K-1})
    from the same user on the same day.
    """

    def __init__(self, cfg: DatasetConfig, sequence_length: int = 30, data_fraction: float = 1.0):
        super().__init__(cfg, data_fraction=data_fraction)
        self.sequence_length = sequence_length

        # Filter index to only include samples where we can get K consecutive timesteps
        # Remove samples that are too close to the end of their day
        valid_indices = []
        for idx in range(len(self.index)):
            row = self.index.iloc[idx]
            day_idx = row["index"]

            # Check if we have enough samples ahead
            if day_idx + sequence_length <= 1440:  # 1440 minutes in a day
                valid_indices.append(idx)

        # Update index to only include valid starting points
        self.index = self.index.iloc[valid_indices].reset_index(drop=True)
        print(f"SequenceDataset: {len(self.index)} valid sequence starting points (sequence_length={sequence_length})")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Returns a sequence of K consecutive samples for rollout training.

        DELTA MODE: Each y_k is a delta (change in charge), not absolute charge.
        The model will predict deltas and accumulate them during rollout.

        Returns:
            dict: {
                "x_sequence": [x_0, x_1, ..., x_{K-1}],  # List of feature tensors (past_charge is absolute)
                "y_sequence": [delta_0, delta_1, ..., delta_{K-1}],  # List of DELTA targets
                "user_id": user_id,
                "start_idx": starting day index
            }
        """
        row = self.index.iloc[idx]
        user_id = str(row["userid"])
        date = pd.to_datetime(row["date"])
        start_day_idx = row["index"]

        # Load user file once for the entire sequence
        today_row, yesterday_row, idx_sleep_start = self._load_user_file(user_id, date, start_day_idx)

        # Determine which charge column to use
        date_str = date.strftime('%Y-%m-%d')
        if (user_id, date_str) in self.llm_charge_pairs:
            charge_col_to_use = self.cfg.charge_col_llm
        else:
            charge_col_to_use = self.cfg.charge_col

        x_sequence = []
        y_sequence = []

        # Get K consecutive samples
        # DELTA MODE: Each y_k will be a delta (charge_{k} - charge_{k-1})
        for k in range(self.sequence_length):
            current_idx = start_day_idx + k

            # Get the single sample using parent class logic
            # _get_feature_vector returns (features, delta) in delta mode
            x_k, delta_k = self._get_feature_vector(today_row, yesterday_row, current_idx, idx_sleep_start, charge_col=charge_col_to_use)

            x_sequence.append(torch.tensor(x_k, dtype=torch.float32))
            y_sequence.append(torch.tensor([delta_k], dtype=torch.float32))  # DELTA target

        return {
            "x_sequence": x_sequence,  # List of K tensors (features with past_charge)
            "y_sequence": y_sequence,  # List of K DELTA tensors
            "user_id": user_id,
            "start_idx": start_day_idx
        }


def sequence_collate_fn(batch):
    """
    Custom collate function for sequence batches.

    Args:
        batch: List of dicts from SequenceBiochargeDataset

    Returns:
        dict: {
            "x_sequence": Tensor [batch_size, sequence_length, feature_dim],
            "y_sequence": Tensor [batch_size, sequence_length, 1],
            "user_ids": List of user_ids,
            "start_indices": List of start indices
        }
    """
    batch_size = len(batch)
    sequence_length = len(batch[0]["x_sequence"])
    feature_dim = batch[0]["x_sequence"][0].shape[0]

    # Pre-allocate tensors
    x_sequences = torch.zeros(batch_size, sequence_length, feature_dim)
    y_sequences = torch.zeros(batch_size, sequence_length, 1)
    user_ids = []
    start_indices = []

    for i, sample in enumerate(batch):
        # Stack the sequence lists into tensors
        x_sequences[i] = torch.stack(sample["x_sequence"])  # [seq_len, feat_dim]
        y_sequences[i] = torch.stack(sample["y_sequence"])  # [seq_len, 1]
        user_ids.append(sample["user_id"])
        start_indices.append(sample["start_idx"])

    return {
        "x_sequence": x_sequences,
        "y_sequence": y_sequences,
        "user_ids": user_ids,
        "start_indices": start_indices
    }

def build_test_dataloaders_torch(cfg_test: DatasetConfig, batch_size: int = 128, data_fraction: float = 1.0):
    """
    Build test dataloaders using TorchBiochargeDataset instead of BiochargeDataset.
    """
    from dataset_delta_v2 import TorchBiochargeDataset
    
    test_ds = TorchBiochargeDataset(cfg_test, data_fraction=data_fraction)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return test_loader, test_ds

def build_test_dataloaders(cfg_test: DatasetConfig,
                      batch_size: int = 128, data_fraction: float = 1.0):

    test_ds = BiochargeDataset(cfg_test, data_fraction=data_fraction)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return test_loader, test_ds

def build_test_dataloaders_sample(cfg_test: DatasetConfig,
                      batch_size: int = 128, data_fraction: float = 1.0):

    test_ds = SampleTorchBiochargeDataset(cfg_test, data_fraction=data_fraction)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return test_loader, test_ds

def build_dataloaders_sample(cfg_train: DatasetConfig, cfg_val: DatasetConfig,
                      batch_size: int = 512, use_advanced_augmentation: bool = False, num_workers: int = 4,
                      data_fraction: float = 1.0, torch_data_loader: bool = True):

    # Choose dataset class based on augmentation preference
    if use_advanced_augmentation:
        train_ds = AdvancedAugmentedBiochargeDataset(cfg_train, data_fraction=data_fraction)
    else:
        if torch_data_loader:
            train_ds = SampleTorchBiochargeDataset(cfg_train, data_fraction=data_fraction)
        else:
            train_ds = BiochargeDataset(cfg_train, data_fraction=data_fraction)
        
    # Validation should not have augmentation
    cfg_val.enable_augmentation = False
    if torch_data_loader:
        val_ds = SampleTorchBiochargeDataset(cfg_val, data_fraction=data_fraction)
    else:
        val_ds = BiochargeDataset(cfg_val, data_fraction=data_fraction)

    # build loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, train_ds, val_ds

def build_dataloaders(cfg_train: DatasetConfig, cfg_val: DatasetConfig,
                      batch_size: int = 512, use_advanced_augmentation: bool = False, num_workers: int = 4,
                      data_fraction: float = 1.0, torch_data_loader: bool = True):

    # Choose dataset class based on augmentation preference
    if use_advanced_augmentation:
        train_ds = AdvancedAugmentedBiochargeDataset(cfg_train, data_fraction=data_fraction)
    else:
        if torch_data_loader:
            train_ds = TorchBiochargeDataset(cfg_train, data_fraction=data_fraction)
        else:
            train_ds = BiochargeDataset(cfg_train, data_fraction=data_fraction)
        
    # Validation should not have augmentation
    cfg_val.enable_augmentation = False
    if torch_data_loader:
        val_ds = TorchBiochargeDataset(cfg_val, data_fraction=data_fraction)
    else:
        val_ds = BiochargeDataset(cfg_val, data_fraction=data_fraction)

    # build loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, train_ds, val_ds


def build_rollout_dataloaders(cfg_train: DatasetConfig, cfg_val: DatasetConfig,
                               batch_size: int = 32, sequence_length: int = 30,
                               num_workers: int = 4, data_fraction: float = 1.0):
    """
    Build dataloaders for K-step rollout training.

    Args:
        cfg_train: Training dataset config
        cfg_val: Validation dataset config
        batch_size: Batch size (typically smaller than standard training due to sequences)
        sequence_length: Number of consecutive timesteps (K)
        num_workers: Number of dataloader workers
        data_fraction: Fraction of data to use

    Returns:
        train_loader, val_loader, train_ds, val_ds
    """
    # Disable augmentation for rollout (to ensure clean sequences)
    cfg_train.enable_augmentation = False
    cfg_val.enable_augmentation = False

    # Create sequence datasets
    train_ds = SequenceBiochargeDataset(cfg_train, sequence_length=sequence_length, data_fraction=data_fraction)
    val_ds = SequenceBiochargeDataset(cfg_val, sequence_length=sequence_length, data_fraction=data_fraction)

    # Build loaders with custom collate function
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=sequence_collate_fn,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sequence_collate_fn,
        pin_memory=False
    )

    return train_loader, val_loader, train_ds, val_ds


def create_windowed_config_example():
    """
    Example function showing how to configure the dataset with windowed features.
    
    Returns:
        DatasetConfig: Configured dataset with windowed features enabled
    """
    
    cfg = DatasetConfig(
        data_dir="path/to/your/data",
        index_csv="path/to/your/index.csv",
        zscores_file="path/to/your/zscores.json",
        charge_col="charge_column_name",
        static_cols=["weight", "height", "age", "gender"],
        sleep_cols=["sleep_duration", "deep_sleep_duration", "light_sleep_duration", "rem_sleep_duration"],
        ts_cols=["timeseries.hr", "timeseries.acc_magnitude", "timeseries.stress", "timeseries.exercise"],
        
        # Standard options
        use_zscores=True,
        enable_augmentation=True,
        use_past_charge=True,
        use_positional_encoding=True,
        pos_encoding_dim=64,
        
        # New windowed features options
        use_windowed_features=True,        # Enable windowed features
        window_size_minutes=30,            # Use 30-minute windows
        include_gradients=True,            # Include HR and ACC gradients
        use_windowed_zscore=True,          # Apply windowed z-score normalization
        include_current_hr=True            # Include current HR data (set False to use t-1)
    )
    
    return cfg


def example_usage():
    """
    Example of how to use the windowed features dataset.
    """
    
    # Create configuration with windowed features
    cfg = create_windowed_config_example()
    
    # Create dataset
    dataset = BiochargeDataset(cfg)
    
    # Print feature information
    print(f"Total feature count: {dataset.get_feature_count()}")
    print(f"Configuration stats: {dataset.get_augmentation_stats()}")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Example of accessing a batch
    for batch in dataloader:
        x = batch["x"]  # Shape: [batch_size, total_features]
        y = batch["y"]  # Shape: [batch_size, 1]
        
        print(f"Input shape: {x.shape}")
        print(f"Target shape: {y.shape}")
        
        # If windowed features are enabled, the feature vector contains:
        # - Static features (weight, height, age, gender)
        # - Sleep features (from previous night)
        # - Time series features (current time point)
        # - Past charge (if enabled)
        # - Positional encoding (if enabled)
        # - HR window (single z-score normalized value, current or t-1 based on include_current_hr)
        # - ACC window (single z-score normalized value)
        # - Gradients (hr_gradient, acc_gradient if include_gradients=True)
        # 
        # IMPORTANT: Samples with idx < window_size_minutes have mask=0.0 and should be
        # excluded from loss calculations during training.
        
        break
    
    return dataset, dataloader


if __name__ == "__main__":
    # Run example usage
    dataset, dataloader = example_usage()
