# t_digest_wrapper.py

from tdigest import TDigest
import numpy as np
import pickle
from typing import Union

class TDigestWrapper:
    """
    A wrapper class for the t-digest algorithm to provide a simple,
    consistent interface for creating, updating, merging, and querying
    quantile summaries.
    """

    def __init__(self):
        """Initializes an empty t-digest object."""
        self.digest = TDigest()

    def update(self, value: float):
        """
        Updates the digest with a single data point.

        Args:
            value (float): The data point to add.
        """
        self.digest.update(value)

    def batch_update(self, data: Union[list[float], np.ndarray]):
        """
        Updates the digest with a batch of data points.

        Args:
            data (list[float] | np.ndarray): A list or NumPy array of data points.
        """
        self.digest.batch_update(data)

    def merge(self, other_wrapper: 'TDigestWrapper'):
        """
        Merges another TDigestWrapper's digest into this one.
        The current digest is updated in-place using the '+' operator.

        Args:
            other_wrapper (TDigestWrapper): Another instance of the wrapper
                                            whose digest will be merged.
        """
        self.digest += other_wrapper.digest

    def get_percentile(self, percentile: float) -> Union[float, None]:
        """
        Calculates the value at a given percentile.

        Args:
            percentile (float): The percentile to calculate (e.g., 50 for median).
                                Must be between 0 and 100.

        Returns:
            float | None: The estimated value at the percentile, or None if the
                          digest is empty.
        """
        # Corrected line: Check the length of the digest to see if it's empty.
        if len(self.digest) == 0:
            return None
        return self.digest.percentile(percentile)

    def save(self, filepath: str):
        """
        Saves the internal digest object to a file using pickle.

        Args:
            filepath (str): The path to the file where the digest will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.digest, f)

    @classmethod
    def load(cls, filepath: str) -> 'TDigestWrapper':
        """
        Loads a digest object from a file and returns a new TDigestWrapper instance.

        Args:
            filepath (str): The path to the file containing the saved digest.

        Returns:
            TDigestWrapper: A new instance containing the loaded digest.
        """
        with open(filepath, 'rb') as f:
            loaded_digest = pickle.load(f)
        
        wrapper = cls()
        wrapper.digest = loaded_digest
        return wrapper

# --- Example Usage: Dynamic Timeseries of Unknown Length (Accumulation Method) ---
if __name__ == '__main__':
    print("--- T-Digest Wrapper: Dynamic Timeseries of Unknown Length (Accumulation Method) ---")

    # 1. Simulate a data source.
    print("Generating 10,250 data points for the simulation (to ensure a final partial batch)...")
    np.random.seed(42) # Use a seed for reproducibility
    full_data_stream = np.random.normal(loc=100, scale=25, size=10250)
    batch_size = 500

    # 2. This is the main digest that will accumulate all batches over time.
    main_digest_wrapper = TDigestWrapper()

    print(f"\nProcessing data by accumulating points into batches of {batch_size}...")
    print("-" * 70)

    total_points_processed = 0
    batch_num = 1
    current_batch_data = []

    # 3. Process the stream by iterating through each point and accumulating a batch.
    for data_point in full_data_stream:
        current_batch_data.append(data_point)

        # When the batch is full, process it.
        if len(current_batch_data) == batch_size:
            total_points_processed += len(current_batch_data)

            # Create a temporary digest for ONLY the current batch
            batch_wrapper = TDigestWrapper()
            batch_wrapper.batch_update(current_batch_data)

            # Merge the new batch's digest into the main (cumulative) digest
            main_digest_wrapper.merge(batch_wrapper)

            # Get the cumulative 90th percentile after processing the current batch
            cumulative_90th = main_digest_wrapper.get_percentile(90)
            
            # For demonstration, calculate the true percentile on all data *seen so far*.
            data_seen_so_far = full_data_stream[:total_points_processed]
            true_90th = np.percentile(data_seen_so_far, 90)

            print(f"Batch {batch_num:2d} | Points: {total_points_processed:5d} | Cumulative 90th Pct Estimate: {cumulative_90th:8.4f} | True Value: {true_90th:8.4f}")
            
            # Reset the batch buffer and increment the batch number
            current_batch_data = []
            batch_num += 1
    
    # 4. After the loop, process any leftover data in the final partial batch.
    if current_batch_data:
        print("\nProcessing final partial batch...")
        total_points_processed += len(current_batch_data)

        batch_wrapper = TDigestWrapper()
        batch_wrapper.batch_update(current_batch_data)
        main_digest_wrapper.merge(batch_wrapper)

        cumulative_90th = main_digest_wrapper.get_percentile(90)
        true_90th = np.percentile(full_data_stream, 90) # True value is now for the whole stream

        print(f"Batch {batch_num:2d} | Points: {total_points_processed:5d} | Cumulative 90th Pct Estimate: {cumulative_90th:8.4f} | True Value: {true_90th:8.4f}")


    # 5. Final verification against the entire dataset
    print("-" * 70)
    final_estimate = main_digest_wrapper.get_percentile(90)
    final_true_value = np.percentile(full_data_stream, 90)
    
    print("\n--- Final Verification ---")
    print(f"Final T-Digest Estimate: {final_estimate:.4f}")
    print(f"Final NumPy True Value:  {final_true_value:.4f}")
    print(f"Final Error: {abs(final_estimate - final_true_value):.4f}")
