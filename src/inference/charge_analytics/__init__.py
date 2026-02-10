"""
Biocharge analytical ground truth computation package.

Standalone copy of the charge calculation engine from charge_old/charge_logic/.
Uses the main() path with fixed (non-finetune) parameters.
"""

from .compute import compute_biocharge_ground_truth

__all__ = ["compute_biocharge_ground_truth"]
