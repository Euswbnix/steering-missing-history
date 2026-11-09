"""Checkpoint-selection metric for steering angles in degrees."""

import numpy as np


def macro_mae(prediction, target):
    """Average MAE across nonempty absolute-angle bins [0,5), [5,15), [15,inf)."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or target.ndim != 1:
        raise ValueError("prediction and target must be matching one-dimensional arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction and target must be finite")
    angle = np.abs(target)
    values = []
    for lo, hi in ((0, 5), (5, 15), (15, np.inf)):
        selected = (angle >= lo) & (angle < hi)
        if selected.any():
            values.append(np.abs(prediction[selected] - target[selected]).mean())
    return float(np.mean(values)) if values else float("nan")
