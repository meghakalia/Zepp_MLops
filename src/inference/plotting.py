"""
Trajectory curve plotting for biocharge inference.

Generates multi-panel plots showing predicted vs actual biocharge curves
with physiological event annotations (exercise, sleep, nap, non-wear),
day boundaries, and HR/ACC subplots.
"""

from __future__ import annotations

import ast
import io
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_parse(x):
    """Parse data that may be string, numeric, or NaN into a list."""
    if pd.isna(x):
        return []
    if isinstance(x, (int, float)):
        return [x]
    try:
        return ast.literal_eval(x)
    except Exception:
        try:
            return json.loads(x)
        except Exception:
            return []


def generate_trajectory_plot(
    preds: list[float],
    targets: list[float],
    user_id: str,
    dates: list[str],
    data_dir: str,
    error_dict: Optional[dict] = None,
    output_path: Optional[str] = None,
) -> bytes:
    """
    Generate a trajectory plot showing autoregressive predictions vs ground truth
    with physiological event annotations and HR/ACC subplots.

    Args:
        preds: Predicted biocharge values (concatenated across days)
        targets: Ground truth biocharge values (concatenated across days)
        user_id: User identifier
        dates: List of date strings (YYYY-MM-DD)
        data_dir: Directory containing {user_id}_processed.xlsx
        error_dict: Optional region-based error metrics to display
        output_path: Optional path to save the plot PNG

    Returns:
        PNG bytes of the plot
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    user_file = os.path.join(data_dir, f"{user_id}_processed.xlsx")
    has_user_data = os.path.exists(user_file)

    # Load timeseries data for annotations
    all_exercise = []
    all_sleep = []
    all_nap = []
    all_non_wear = []
    all_hr = []
    all_acc = []
    day_boundaries = [0]
    current_length = 0

    rendered_dates = []

    if has_user_data:
        df = pd.read_excel(user_file)
        # Normalise date column to "YYYY-MM-DD" strings so the lookup always works
        # regardless of whether Excel stores dates as datetime objects or strings.
        try:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        except Exception:
            df['date'] = df['date'].astype(str).str.strip()

        for i, date_str in enumerate(dates):
            df_row = df[df['date'] == date_str]
            if df_row.empty:
                continue

            row = df_row.iloc[0]
            charge = _safe_parse(row.get('charge.timeseries', '[]'))
            exercise = _safe_parse(row.get('timeseries.exercise', '[]'))
            sleep = _safe_parse(row.get('timeseries.sleep_markers', '[]'))
            nap = _safe_parse(row.get('timeseries.nap_state', '[]'))
            non_wear = _safe_parse(row.get('timeseries.min_status_list', '[]'))
            hr = _safe_parse(row.get('timeseries.hrr_raw', '[]'))
            acc = _safe_parse(row.get('timeseries.acc', '[]'))

            # Use charge length as reference; fall back to hr/acc so those subplots
            # appear even when charge.timeseries is absent for a row.
            day_len = len(charge) or len(hr) or len(acc) or 0
            if day_len == 0:
                continue

            rendered_dates.append(date_str)

            # Event flags padded with 0 (not active); continuous signals with NaN
            # so gaps render as breaks rather than a false zero baseline.
            exercise = (exercise[:day_len] + [0] * day_len)[:day_len]
            sleep = (sleep[:day_len] + [0] * day_len)[:day_len]
            nap = (nap[:day_len] + [0] * day_len)[:day_len]
            non_wear = (non_wear[:day_len] + [0] * day_len)[:day_len]
            hr = (hr[:day_len] + [float('nan')] * day_len)[:day_len]
            acc = (acc[:day_len] + [float('nan')] * day_len)[:day_len]

            all_exercise.extend(exercise)
            all_sleep.extend(sleep)
            all_nap.extend(nap)
            all_non_wear.extend(non_wear)
            all_hr.extend(hr)
            all_acc.extend(acc)

            current_length += day_len
            if i < len(dates) - 1:
                day_boundaries.append(current_length)

    pred_len = min(len(preds), len(targets))
    x_axis = np.arange(pred_len)

    # Determine subplot layout — only add a subplot when there is real data
    def _has_real_data(series: list) -> bool:
        if not series:
            return False
        arr = np.array(series[:pred_len], dtype=float)
        return bool(np.any(np.isfinite(arr)))

    has_hr_data = _has_real_data(all_hr)
    has_acc_data = _has_real_data(all_acc)

    n_subplots = 1  # main charge plot
    if has_hr_data:
        n_subplots += 1
    if has_acc_data:
        n_subplots += 1

    height_ratios = [3] + [1] * (n_subplots - 1)
    fig, axes = plt.subplots(
        n_subplots, 1, figsize=(16, 4 * n_subplots),
        sharex=True, gridspec_kw={'height_ratios': height_ratios}
    )
    if n_subplots == 1:
        axes = [axes]

    ax_main = axes[0]

    # --- Main charge plot ---
    ax_main.plot(x_axis, targets[:pred_len], label="Ground Truth", color="blue", alpha=0.8, linewidth=1.2)
    ax_main.plot(x_axis, preds[:pred_len], label="Predicted (Autoregressive)", color="red", alpha=0.8, linewidth=1.2)

    # Event annotations as colored background bands
    if all_exercise:
        _add_event_bands(ax_main, all_exercise[:pred_len], {1}, color='green', alpha=0.15, label='Exercise')
    if all_sleep:
        _add_event_bands(ax_main, all_sleep[:pred_len], {1}, color='purple', alpha=0.1, label='Sleep')
    if all_nap:
        _add_event_bands(ax_main, all_nap[:pred_len], {1}, color='cyan', alpha=0.1, label='Nap')
    if all_non_wear:
        _add_event_bands(ax_main, all_non_wear[:pred_len], {3}, color='gray', alpha=0.15, label='Non-wear')

    # Day boundaries
    for boundary in day_boundaries[1:]:
        if boundary < pred_len:
            ax_main.axvline(x=boundary, color='black', linestyle='--', alpha=0.4, linewidth=0.8)

    # Error metrics in title
    overall_mae = error_dict.get('overall_traj_mae', float('nan')) if error_dict else float('nan')
    overall_mse = error_dict.get('overall_traj_mse', float('nan')) if error_dict else float('nan')

    title_parts = [f"Biocharge Trajectory — User {user_id}"]
    plot_dates = rendered_dates if rendered_dates else dates
    if len(plot_dates) == 1:
        title_parts.append(f"Date: {plot_dates[0]}")
    else:
        title_parts.append(f"Dates: {', '.join(plot_dates)}")
    if not np.isnan(overall_mae):
        title_parts.append(f"Overall MAE={overall_mae:.2f}, MSE={overall_mse:.2f}")

    ax_main.set_title("\n".join(title_parts), fontsize=11)
    ax_main.set_ylabel("Charge (%)")
    ax_main.set_ylim(0, 105)
    ax_main.grid(True, alpha=0.3)

    # Legend with event patches
    handles, labels = ax_main.get_legend_handles_labels()
    event_patches = []
    if all_exercise:
        event_patches.append(Patch(facecolor='green', alpha=0.3, label='Exercise'))
    if all_sleep:
        event_patches.append(Patch(facecolor='purple', alpha=0.2, label='Sleep'))
    if all_nap:
        event_patches.append(Patch(facecolor='cyan', alpha=0.2, label='Nap'))
    if all_non_wear:
        event_patches.append(Patch(facecolor='gray', alpha=0.3, label='Non-wear'))
    ax_main.legend(handles=handles + event_patches, loc='upper right', fontsize=8)

    # --- Region error annotation box ---
    if error_dict:
        error_lines = []
        for event_type in ['exercise', 'sleep', 'nap', 'non_wear']:
            mae_key = f"{event_type}_mae"
            if mae_key in error_dict and not np.isnan(error_dict[mae_key]):
                error_lines.append(f"{event_type}: MAE={error_dict[mae_key]:.2f}")
        for key in ['day_start_mae', 'day_end_mae']:
            if key in error_dict and not np.isnan(error_dict[key]):
                label = key.replace('_', ' ').title()
                error_lines.append(f"{label}: {error_dict[key]:.2f}")

        if error_lines:
            error_text = "\n".join(error_lines)
            ax_main.text(
                0.01, 0.02, error_text, transform=ax_main.transAxes,
                fontsize=7, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7)
            )

    # --- HRR subplot ---
    subplot_idx = 1
    if has_hr_data and subplot_idx < n_subplots:
        ax_hr = axes[subplot_idx]
        hr_data = np.array(all_hr[:pred_len], dtype=float)
        ax_hr.plot(x_axis, hr_data, color='orangered', alpha=0.7, linewidth=0.6, label='HRR raw')
        ax_hr.set_ylabel("HRR (raw)", fontsize=9)
        ax_hr.grid(True, alpha=0.3)
        ax_hr.legend(loc='upper right', fontsize=7)
        for boundary in day_boundaries[1:]:
            if boundary < pred_len:
                ax_hr.axvline(x=boundary, color='black', linestyle='--', alpha=0.4, linewidth=0.8)
        subplot_idx += 1

    # --- ACC subplot ---
    if has_acc_data and subplot_idx < n_subplots:
        ax_acc = axes[subplot_idx]
        acc_data = np.array(all_acc[:pred_len], dtype=float)
        ax_acc.plot(x_axis, acc_data, color='teal', alpha=0.7, linewidth=0.6, label='ACC')
        ax_acc.set_ylabel("ACC", fontsize=9)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend(loc='upper right', fontsize=7)
        for boundary in day_boundaries[1:]:
            if boundary < pred_len:
                ax_acc.axvline(x=boundary, color='black', linestyle='--', alpha=0.4, linewidth=0.8)

    # Set x label on bottom subplot
    axes[-1].set_xlabel("Timestep (minutes)")

    fig.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        logger.info("Trajectory plot saved to: %s", output_path)

    return png_bytes


def _add_event_bands(ax, series, active_values, color='green', alpha=0.15, label=None):
    """Add colored background bands for active event periods."""
    in_event = False
    start = 0
    added_label = False

    for i, val in enumerate(series):
        is_active = val in active_values
        if is_active and not in_event:
            start = i
            in_event = True
        elif not is_active and in_event:
            ax.axvspan(start, i, color=color, alpha=alpha,
                       label=label if not added_label else None)
            added_label = True
            in_event = False

    if in_event:
        ax.axvspan(start, len(series), color=color, alpha=alpha,
                   label=label if not added_label else None)
