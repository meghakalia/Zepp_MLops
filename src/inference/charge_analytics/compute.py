"""
Wrapper for computing biocharge analytical ground truth for a single user.

Uses fixed parameters from charge_main.py's main() function (NOT finetune).
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# Static parameters from charge_main.py's main() function
MENTAL_PARAMETERS_TUNED = [1.2, 1.4, 2.2921, 1.3, -2.0, -2.4, 3, 9, 12, 0.001]
PHYSICAL_PARAMETERS_TUNED = [0.002, 22000.0, 40.0, 200.0, 25.6, 1, 1.5, 2, 0.001]

# Data files directory (same directory as this file)
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_biocharge_ground_truth(
    user_id: str,
    data_folder: str,
    result_path: str,
) -> str:
    """
    Compute biocharge analytical values (ground truth) for a single user.

    Calls calculate_one_user() with the fixed (non-finetune) parameters
    from the main() function. Writes {user_id}_processed.xlsx to result_path.

    Args:
        user_id: User ID (string or int)
        data_folder: Directory containing user_score_data/ and user_sleep_data/
                     subdirectories (i.e., what puller.py produces under data/raw/)
        result_path: Directory where {user_id}_processed.xlsx will be written

    Returns:
        Path to the generated {user_id}_processed.xlsx file

    Raises:
        FileNotFoundError: If required data files are missing
        RuntimeError: If output file was not created
    """
    os.makedirs(result_path, exist_ok=True)

    # Validate input data exists
    score_file = os.path.join(data_folder, "user_score_data", f"{user_id}.csv")
    sleep_dir = os.path.join(data_folder, "user_sleep_data", str(user_id))
    if not os.path.exists(score_file):
        raise FileNotFoundError(f"Score data not found: {score_file}")
    if not os.path.isdir(sleep_dir):
        raise FileNotFoundError(f"Sleep data directory not found: {sleep_dir}")

    # Load guidance DataFrame from package data
    guidance_path = os.path.join(_PACKAGE_DIR, "guidance_online.xlsx")
    if not os.path.exists(guidance_path):
        raise FileNotFoundError(f"guidance_online.xlsx not found at {guidance_path}")
    df_guidance = pd.read_excel(guidance_path)

    # current_dir is used by calculate_one_day to find data_mbat_new.csv and data_pbat_new.csv
    current_dir = _PACKAGE_DIR

    logger.info(
        "Computing biocharge ground truth for user %s (data_folder=%s, result_path=%s)",
        user_id, data_folder, result_path,
    )

    from .charge_main import calculate_one_user

    calculate_one_user(
        user_ID=user_id,
        physical_parameters_tuned=PHYSICAL_PARAMETERS_TUNED,
        mental_parameters_tuned=MENTAL_PARAMETERS_TUNED,
        data_folder=data_folder,
        current_dir=current_dir,
        result_path=result_path,
        df_guidance=df_guidance,
    )

    output_file = os.path.join(result_path, f"{user_id}_processed.xlsx")
    if not os.path.exists(output_file):
        raise RuntimeError(f"Expected output file was not created: {output_file}")

    logger.info("Biocharge ground truth saved: %s", output_file)
    return output_file
