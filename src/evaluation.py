"""Regression metrics, including the PHM08 asymmetric score.

The PHM score is the domain metric: over-predicting RUL (saying an engine
has 50 cycles left when it has 10) is operationally worse than
under-predicting (replacing too early wastes a part; replacing too late
risks an in-flight failure). The score penalizes late predictions
exponentially harder.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def phm_score(y_true, y_pred):
    """PHM08 asymmetric scoring function. Lower is better.

    d = y_pred - y_true. d > 0 means we predicted MORE life than remains
    (late / dangerous): exp(d/10) - 1. d < 0 (early / conservative):
    exp(-d/13) - 1.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluate(y_true, y_pred):
    """Full metric bundle for one prediction set."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "phm_score": phm_score(y_true, y_pred),
        "n": int(len(np.asarray(y_true))),
    }
