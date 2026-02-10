import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

class SignalBPredictor:
    """
    Predicts the next value of a signal based on a short history of its
    recent valid readings using linear regression.
    """
    def __init__(self, history_size=5):
        """
        Initializes the predictor.

        Args:
            history_size (int): The number of recent data points to use for prediction.
        """
        self.history = deque(maxlen=history_size)
        self.last_valid_value = np.nan

    def update_history(self, value):
        """Adds a new valid value to the history."""
        self.history.append(value)
        self.last_valid_value = value

    def predict(self):
        """
        Predicts the next value using linear regression on the history.
        Falls back to last known value if history is too short.
        """
        # Fallback if we don't have enough data for regression
        if len(self.history) < 2:
            return self.last_valid_value

        # Perform linear regression: y = mx + c
        y = np.array(self.history)
        x = np.arange(len(y))
        
        # Fit a line (degree 1 polynomial)
        slope, intercept = np.polyfit(x, y, 1)
        
        # Predict the next point in the sequence
        next_x = len(y)
        prediction = slope * next_x + intercept
        
        return prediction

class StreamingDecomposer:
    """
    Performs dynamic, streaming decomposition of a signal into trend and seasonal
    components for each new data point.
    """
    def __init__(self, period=15, alpha=0.1):
        """
        Initializes the streaming decomposer.

        Args:
            period (int): The period of the seasonality.
            alpha (float): The base smoothing factor for the EWMA trend calculation.
        """
        self.period = period
        self.base_alpha = alpha
        
        # State variables
        self.trend = 0.0
        self.seasonal_means = np.zeros(period)
        self.seasonal_counts = np.zeros(period)
        self.t = 0 # Time step counter

    def update(self, value, alpha_scaler=1.0):
        """
        Updates the decomposer with a new data point and returns the current
        trend and seasonal components. Handles NaN values by holding state.

        Args:
            value (float): The new data point from the signal.
            alpha_scaler (float): A multiplier to dynamically adjust the alpha based on volatility.

        Returns:
            tuple: A tuple containing (current_trend, current_seasonal_component).
        """
        seasonal_index = self.t % self.period
        
        if np.isnan(value):
            current_seasonal = self.seasonal_means[seasonal_index]
            return self.trend, current_seasonal

        if self.t == 0:
            self.trend = value

        # Dynamically adjust alpha based on the scaler
        current_alpha = np.clip(self.base_alpha * alpha_scaler, 0.01, 0.99)
        self.trend = current_alpha * value + (1 - current_alpha) * self.trend
        
        detrended_value = value - self.trend
        
        self.seasonal_counts[seasonal_index] += 1
        prev_mean = self.seasonal_means[seasonal_index]
        count = self.seasonal_counts[seasonal_index]
        self.seasonal_means[seasonal_index] = prev_mean + (detrended_value - prev_mean) / count
        
        current_seasonal = self.seasonal_means[seasonal_index]
        
        self.t += 1
        return self.trend, current_seasonal

class RunningStats:
    """
    Calculates running mean and standard deviation using Welford's algorithm.
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, value):
        if np.isnan(value):
            return
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def std(self):
        if self.n < 2:
            return 0.0
        variance = self.M2 / self.n
        return np.sqrt(variance)

    @property
    def sem(self):
        if self.n == 0:
            return 0.0
        effective_n = min(self.n, 240)
        if effective_n == 0:
            return 0.0
        return self.std / np.sqrt(effective_n)

def soft_clip_and_scale(value, low_bound=0, high_bound=100, margin=10, scale=0.05):
    """
    Compresses values outside the primary range [low_bound, high_bound] into margins.
    """
    if value < low_bound:
        return low_bound + margin * np.exp(scale * (value - low_bound))
    elif value > high_bound:
        return high_bound - margin * np.exp(-scale * (value - high_bound))
    else:
        range_width = high_bound - low_bound
        margin_width = range_width - 2 * margin
        return (value - low_bound) * (margin_width / range_width) + low_bound + margin

class DynamicSignalProcessor:
    """
    Orchestrates the full dynamic adjustment pipeline with adaptive decomposition.
    """
    def __init__(self, period=15, alpha=0.1):
        self.alpha = alpha
        self.decomposer_c = StreamingDecomposer(period, alpha)
        self.decomposer_adj_b = StreamingDecomposer(period, alpha)
        
        self.stats_a = RunningStats()
        self.stats_final_a = RunningStats()
        self.predictor_b = SignalBPredictor()

        # Stats for adaptive variance stabilization
        self.stats_c = RunningStats()
        self.stats_adj_b = RunningStats()

        self.stats_reconstructed_a = RunningStats()
        self.ewma_reconstructed_a = 0.0
        self.is_ewma_initialized = False

    def process_point(self, a, b, c, state):
        """
        Processes a single new data point from each signal.
        """
        if b >= 250:
            b = self.predictor_b.predict()
        else:
            self.predictor_b.update_history(b)

        if np.isnan(a) or np.isnan(b) or np.isnan(c):
            return np.nan

        # --- Step 1: Decompose based on C with adaptive alpha for variance stabilization ---
        self.stats_c.update(c)
        coeff_var_c = self.stats_c.std / (abs(self.stats_c.mean) + 1e-8)
        alpha_scaler_c = 1 + np.tanh(coeff_var_c) # Scale alpha between 1x and 2x
        
        trend_c, seasonal_c = self.decomposer_c.update(c, alpha_scaler=alpha_scaler_c)
        adjusted_a_c = a - trend_c - seasonal_c
        adjusted_b_c = b - trend_c - seasonal_c
        
        # --- Step 2: Decompose based on adjusted B with adaptive alpha ---
        self.stats_adj_b.update(adjusted_b_c)
        coeff_var_b = self.stats_adj_b.std / (abs(self.stats_adj_b.mean) + 1e-8)
        alpha_scaler_b = 1 + np.tanh(coeff_var_b)

        trend_adj_b, seasonal_adj_b = self.decomposer_adj_b.update(adjusted_b_c, alpha_scaler=alpha_scaler_b)
        final_adjusted_a = adjusted_a_c - trend_adj_b - seasonal_adj_b
        
        # --- Step 3: Update stats for reconstruction ---
        self.stats_a.update(a)
        self.stats_final_a.update(final_adjusted_a)
        
        # --- Step 4: Z-score reconstruction ---
        mean_a = self.stats_a.mean
        sem_a = self.stats_a.sem
        
        mean_final = self.stats_final_a.mean
        sem_final = self.stats_final_a.sem

        z_score = (final_adjusted_a - mean_final) / (sem_final + 1e-8)
        reconstructed_a = z_score * sem_a + mean_a

        # --- Step 5: Apply domain knowledge modifier ---
        modifier_map = {
            -1: 1,
            0: 0.80,
            1: 0.50,
            2: 0.40,
        }
        
        self.stats_reconstructed_a.update(reconstructed_a)
        if not self.is_ewma_initialized:
            self.ewma_reconstructed_a = reconstructed_a
            self.is_ewma_initialized = True
        else:
            self.ewma_reconstructed_a = self.alpha * reconstructed_a + (1 - self.alpha) * self.ewma_reconstructed_a
        
        modified_a = reconstructed_a
        modifier = modifier_map.get(state)
        if modifier:
            modified_a = reconstructed_a * modifier

        final_value = soft_clip_and_scale(modified_a)
        
        return final_value

    def run_online_training(self, y, indices_to_process=None):
        if indices_to_process is None:
            indices_to_process = range(len(y))

        results = []
        indices_to_process = set(indices_to_process)
        for i, (a_val, b_val, c_val, state_val) in enumerate(y):
            if i in indices_to_process:
                processed_val = self.process_point(a_val, b_val, c_val, state_val)
                results.append(processed_val)
            else:
                results.append(np.nan)
        return results

# --- Example Usage ---
if __name__ == '__main__':
    # 1. Generate Sample Data
    data_points = 500
    period = 24
    time = np.arange(data_points)
    signal_c = pd.Series(0.5 * time + 15 * np.sin(2 * np.pi * time / period) + np.random.normal(0, 2, data_points))
    signal_a = pd.Series(0.4 * time + 30 * np.sin(2 * np.pi * time / (period/2)) + 50 + np.random.normal(0, 10, data_points)) - 20
    signal_b = pd.Series(10 * np.cos(2 * np.pi * time / (period*2)) + 100 + np.random.normal(0, 4, data_points))
    state_signal = pd.Series((time // 50) % 4 - 1) 

    signal_a.iloc[50:60] = np.nan
    signal_c.iloc[150:155] = np.nan
    signal_b.iloc[220] = np.nan
    
    signal_b_spiked = signal_b.copy()
    signal_b_spiked.iloc[100:110] = 255
    signal_b_spiked.iloc[300:320] = 254

    # 2. Simulate online processing with run_online_training
    print("Simulating online signal processing...")
    processor = DynamicSignalProcessor(period=period, alpha=0.05)
    y_data = list(zip(signal_a, signal_b_spiked, signal_c, state_signal))
    indices_to_process = set(range(100)) | set(range(175, 250))
    reconstructed_signal = processor.run_online_training(y_data, indices_to_process)
    reconstructed_signal = pd.Series(reconstructed_signal)
    print("Processing complete.")

    # 3. Visualize the results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 7))
    
    ax1.plot(signal_a, label='Original Signal A', color='blue', alpha=0.4, linewidth=2)
    ax1.plot(reconstructed_signal, label='Processed Signal A', color='purple', linestyle=':', marker='o', markersize=2)
    ax1.set_title('Dynamic Signal Processing (via run_online_training)', fontsize=16)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Signal Value')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax1.axhline(100, color='k', linestyle=':', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.step(time, state_signal, label='State', color='green', alpha=0.6, where='post')
    ax2.set_ylabel('State', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_yticks(np.unique(state_signal))
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
