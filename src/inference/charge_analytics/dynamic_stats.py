import math

class DynamicStats:
    """
    Calculates mean and standard deviation in a single pass using Welford's
    online algorithm for numerical stability.
    """
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from the mean

    def push(self, x: float):
        """Adds a new value to the running calculation."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        # The new delta is (x - new_mean)
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self) -> float:
        """Returns the current mean."""
        return self.mean

    def get_variance(self, sample: bool = True) -> float:
        """
        Returns the current variance.
        
        Args:
            sample (bool): If True, returns sample variance (N-1 denominator).
                           If False, returns population variance (N denominator).
        """
        if self.count < 2:
            return float('nan') # Variance is undefined for fewer than 2 samples
        
        return self.M2 / (self.count - 1) if sample else self.M2 / self.count

    def get_std(self, sample: bool = True) -> float:
        """
        Returns the current standard deviation.
        
        Args:
            sample (bool): If True, returns sample standard deviation.
                           If False, returns population standard deviation.
        """
        return math.sqrt(self.get_variance(sample))

if __name__ == '__main__':
    # --- Example Usage ---
    # 1. Initialize the calculator
    stats_tracker = RunningStats()

    # 2. Add some initial data points
    initial_data = [10, 12, 15, 11, 13]
    for value in initial_data:
        stats_tracker.push(value)

    print(f"After initial data: {initial_data}")
    print(f"Mean: {stats_tracker.get_mean():.4f}")
    print(f"Sample Std Dev: {stats_tracker.get_std():.4f}\n")

    # 3. Add a single new value and see the updated stats
    new_value = 25
    print(f"--- Adding new value: {new_value} ---\n")
    stats_tracker.push(new_value)

    print(f"After adding {new_value}:")
    print(f"New Mean: {stats_tracker.get_mean():.4f}")
    print(f"New Sample Std Dev: {stats_tracker.get_std():.4f}")