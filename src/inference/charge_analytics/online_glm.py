import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import time

# --- Hyperparameters ---
LEARNING_RATE = 0.0001
ALPHA = 0.01  # Decay factor for EWMA for features
DECOMPOSE_WINDOW = 15
SEASONAL_PERIOD = 240 # Fluctuations expected in a 4-hour window (240 mins)
SEASONAL_ALPHA = 0.8 # Smoothing factor for updating seasonal components
MINUTE_EWMA_ALPHA = 0.1 # Smoothing factor for the minute-of-day baseline

class MinuteOfDayAverage:
    """
    Calculates and updates an online Exponentially Weighted Moving Average (EWMA)
    and variance for a value at each minute of the day to define a confidence interval.
    """
    def __init__(self, alpha=MINUTE_EWMA_ALPHA):
        """
        Initializes the tracker with 1440 minutes (24 hours * 60 minutes).

        Args:
            alpha (float): The smoothing factor for the EWMA.
        """
        self.alpha = alpha
        self.daily_ewma = np.zeros(1440)
        self.daily_ewma_sq = np.zeros(1440) # To calculate variance

    def update(self, minute_of_day, value):
        """
        Updates the running EWMA and EWMA of squares for a specific minute.

        Args:
            minute_of_day (int): The minute of the day (0-1439).
            value (float): The new value to include in the average.
        """
        minute_idx = int(minute_of_day)
        if 0 <= minute_idx < 1440:
            # Update EWMA of the value
            previous_ewma = self.daily_ewma[minute_idx]
            self.daily_ewma[minute_idx] = (self.alpha * value) + (1 - self.alpha) * previous_ewma
            
            # Update EWMA of the squared value
            previous_ewma_sq = self.daily_ewma_sq[minute_idx]
            self.daily_ewma_sq[minute_idx] = (self.alpha * (value**2)) + (1 - self.alpha) * previous_ewma_sq

    def get_average(self, minute_of_day):
        """
        Gets the current EWMA for a specific minute of the day.

        Args:
            minute_of_day (int): The minute of the day (0-1439).

        Returns:
            float: The current EWMA for that minute.
        """
        minute_idx = int(minute_of_day)
        if 0 <= minute_idx < 1440:
            return self.daily_ewma[minute_idx]
        return 0.0

    def get_confidence_interval(self, minute_of_day):
        """
        Calculates the 80% confidence interval for a specific minute of the day.

        Args:
            minute_of_day (int): The minute of the day (0-1439).

        Returns:
            tuple: A tuple containing the lower and upper bounds of the confidence interval.
                   Returns (nan, nan) if variance is not positive.
        """
        minute_idx = int(minute_of_day)
        if 0 <= minute_idx < 1440:
            ewma = self.daily_ewma[minute_idx]
            ewma_sq = self.daily_ewma_sq[minute_idx]
            
            # Variance = E[X^2] - (E[X])^2
            variance = ewma_sq - (ewma**2)
            
            if variance > 0:
                std_dev = np.sqrt(variance)
                z_score_80 = 1.282  # Z-score for 80% confidence
                margin_of_error = z_score_80 * std_dev
                return (ewma - margin_of_error, ewma + margin_of_error)
        
        return (np.nan, np.nan) # Return NaN if variance is not positive

class OnlineDecomposer:
    """
    Handles incremental decomposition of a time series into trend and seasonality.
    --- OPTIMIZATION: Uses a NumPy array as a circular buffer for faster trend calculation. ---
    """
    def __init__(self, window=DECOMPOSE_WINDOW, period=SEASONAL_PERIOD, alpha_seasonal=SEASONAL_ALPHA):
        self.window = window
        self.period = period
        self.alpha_seasonal = alpha_seasonal
        self.history = np.zeros(window) # Use a NumPy array for history
        self.history_idx = 0
        self.is_full = False
        self.seasonal_factors = np.zeros(period)

    def update(self, value, time_step):
        # Update circular buffer for trend calculation
        self.history[self.history_idx] = value
        self.history_idx = (self.history_idx + 1) % self.window
        if not self.is_full and self.history_idx == 0:
            self.is_full = True

        # Calculate current trend using the NumPy array (much faster than list)
        if self.is_full:
            current_trend = np.mean(self.history)
        else:
            current_trend = np.mean(self.history[:self.history_idx]) if self.history_idx > 0 else value

        # Update seasonality
        seasonal_idx = time_step % self.period
        current_seasonal_comp = self.seasonal_factors[seasonal_idx]
        detrended_value = value - current_trend
        self.seasonal_factors[seasonal_idx] = (
            (1 - self.alpha_seasonal) * self.seasonal_factors[seasonal_idx] +
            self.alpha_seasonal * detrended_value
        )
        return current_trend, current_seasonal_comp

class OnlineScaler:
    """
    A highly efficient online scaler using Welford's algorithm.
    """
    def __init__(self, num_features):
        self.n = 0
        self.mean = np.zeros(num_features)
        self.M2 = np.zeros(num_features) 

    def partial_fit(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def transform(self, x):
        if self.n < 2:
            return x
        variance = self.M2 / (self.n - 1)
        std = np.sqrt(variance)
        std[std == 0] = 1.0 
        return (x - self.mean) / std

    def fit_transform(self, x):
        self.partial_fit(x)
        return self.transform(x)

class OnlineGLM:
    """
    A class to handle the state and updates for an online GLM with SGD and EWMA.
    """
    def __init__(self, learning_rate=LEARNING_RATE, alpha=ALPHA):
        self.feature_names = ['const', 'hr', 'activity', 'minute_of_day']
        self.num_features = len(self.feature_names)
        self.params = np.zeros(self.num_features)
        hr_index = self.feature_names.index('hr')
        activity_index = self.feature_names.index('activity')
        self.params[hr_index] = 0.5
        self.params[activity_index] = 0.2
        self.ewma_features = np.zeros(self.num_features)
        self.learning_rate = learning_rate
        self.alpha = alpha

    def update(self, features, target):
        self.ewma_features = self.alpha * features + (1 - self.alpha) * self.ewma_features
        prediction = np.dot(self.ewma_features, self.params)
        error = prediction - target
        gradient = error * self.ewma_features
        if np.isnan(gradient).any() or np.isinf(gradient).any():
            return prediction, error
        np.clip(gradient, -1.0, 1.0, out=gradient)
        self.params = self.params - self.learning_rate * gradient
        return prediction, error

    def predict(self, features):
        self.ewma_features = self.alpha * features + (1 - self.alpha) * self.ewma_features
        return np.dot(self.ewma_features, self.params)

    def get_feature_importance(self):
        importance = pd.Series(self.params, index=self.feature_names)
        return importance.reindex(importance.abs().sort_values(ascending=False).index)


def run_online_training(model, decomposer, minute_of_day_tracker, features_df, stress_signal, is_missing_mask, indices_to_process):
    """
    Runs the main online training loop with optimizations for speed.
    """
    y = stress_signal
    model_features = [f for f in model.feature_names if f != 'const']
    
    # --- OPTIMIZATION: Convert pandas objects to NumPy arrays before the loop ---
    y_np = y.to_numpy()
    model_features_np = features_df[model_features].to_numpy()
    hr_np = features_df['hr'].to_numpy()
    minute_of_day_np = features_df['minute_of_day'].to_numpy()
    is_missing_mask_np = np.array(is_missing_mask)

    if indices_to_process is None:
        indices_to_process = range(len(y))

    final_predictions = np.full(len(y), np.nan)
    actual_targets = np.full(len(y), np.nan)
    feature_invalid_mask = np.full(len(y), False, dtype=bool)
    
    last_valid_prediction = 50.0 
    last_valid_stress_avg = 0.0
    scaler = OnlineScaler(num_features=len(model_features))

    for i in indices_to_process:
        # --- OPTIMIZATION: Use fast NumPy array indexing instead of .iloc ---
        current_raw_target = y_np[i]
        current_minute = minute_of_day_np[i]
        current_features_unscaled = model_features_np[i]
        
        features_scaled = scaler.fit_transform(current_features_unscaled)
        # --- OPTIMIZATION: Use faster np.insert instead of sm.add_constant ---
        current_scaled_features = np.insert(features_scaled, 0, 1.0)

        target_is_missing = is_missing_mask_np[i]
        feature_is_invalid = hr_np[i] >= 250
        
        if target_is_missing:
            if feature_is_invalid:
                feature_invalid_mask[i] = True
                final_prediction = last_valid_prediction
            else:
                stress_avg_baseline = minute_of_day_tracker.get_average(current_minute)
                is_cold_start = (stress_avg_baseline == 0.0)
                
                if is_cold_start:
                    final_prediction = 40.0 + np.random.normal(loc=0, scale=5.0)
                else:
                    previous_residual = last_valid_prediction - last_valid_stress_avg
                    trend, seasonality = decomposer.update(previous_residual, time_step=i)
                    prediction_glm_residual = model.predict(current_scaled_features)
                    final_prediction = stress_avg_baseline + prediction_glm_residual + trend + seasonality
                    
                    lower_bound, upper_bound = minute_of_day_tracker.get_confidence_interval(current_minute)
                    if not np.isnan(lower_bound) and (final_prediction < lower_bound or final_prediction > upper_bound):
                        final_prediction = stress_avg_baseline
        else:
            final_prediction = current_raw_target
            if not feature_is_invalid:
                stress_avg_baseline = minute_of_day_tracker.get_average(current_minute)
                residual_target = current_raw_target - stress_avg_baseline
                trend, seasonality = decomposer.update(residual_target, time_step=i)
                glm_target = residual_target - trend - seasonality
                model.update(current_scaled_features, glm_target)
                minute_of_day_tracker.update(current_minute, current_raw_target)
            else:
                feature_invalid_mask[i] = True
        
        final_prediction = np.clip(final_prediction, 0, 100)
        final_predictions[i] = final_prediction
        actual_targets[i] = current_raw_target
        
        if not np.isnan(final_prediction):
            last_valid_prediction = final_prediction
            if not feature_is_invalid:
                last_valid_stress_avg = minute_of_day_tracker.get_average(current_minute)
            
    return final_predictions

# --- Example Main Execution ---
if __name__ == '__main__':
    model = OnlineGLM()
    decomposer = OnlineDecomposer()
    minute_tracker = MinuteOfDayAverage(alpha=MINUTE_EWMA_ALPHA)

    feature_names = ['hr', 'activity', 'minute_of_day']
    data_len = 200000 # Increased length to better demonstrate performance
    timeseries_sheet = pd.DataFrame({
        'hr': np.random.randint(60, 100, size=data_len),
        'activity': np.random.rand(data_len) * 5,
        'minute_of_day': np.arange(data_len) % 1440,
        'stress': np.random.randint(20, 80, size=data_len)
    })
    features = timeseries_sheet[feature_names]
    
    is_missing_mask = [False] * data_len
    for i in range(50):
        is_missing_mask[i] = True
        
    indices_to_process = range(data_len)

    start_time = time.time()
    
    imputed_stress = run_online_training(
        model, 
        decomposer, 
        minute_tracker,
        features_df=features, 
        stress_signal=timeseries_sheet['stress'], 
        is_missing_mask=is_missing_mask, 
        indices_to_process=indices_to_process
    )
    
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    print("\nFinal Feature Importances:")
    print(model.get_feature_importance())
