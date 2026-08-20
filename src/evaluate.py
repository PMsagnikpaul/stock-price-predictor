"""
evaluate.py
-----------
Shared evaluation utilities for regression-based stock price predictions:
RMSE, MAE, MAPE, and directional accuracy.
"""

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error, expressed as a percentage."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true, y_pred, prev_actual) -> float:
    """
    Percentage of times the model correctly predicts the direction of movement
    (up or down) relative to the previous actual value.

    Parameters
    ----------
    y_true : array-like — actual values at time t
    y_pred : array-like — predicted values at time t
    prev_actual : array-like — actual values at time t-1 (the reference point
        both the true and predicted direction are measured against)
    """
    y_true, y_pred, prev_actual = np.asarray(y_true), np.asarray(y_pred), np.asarray(prev_actual)
    true_direction = np.sign(y_true - prev_actual)
    pred_direction = np.sign(y_pred - prev_actual)
    correct = (true_direction == pred_direction)
    return float(np.mean(correct) * 100)


def evaluate_predictions(y_true, y_pred, prev_actual=None) -> dict:
    """Compute the full metrics suite and return as a dict."""
    metrics = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
    }
    if prev_actual is not None:
        metrics["Directional_Accuracy_%"] = directional_accuracy(y_true, y_pred, prev_actual)
    return metrics


def print_metrics(name: str, metrics: dict):
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k:28s}: {v:.4f}")


def metrics_to_dataframe(results: dict) -> pd.DataFrame:
    """
    Combine multiple models' metrics dicts into a single comparison table.

    Parameters
    ----------
    results : dict
        {"Model Name": {metric_dict}, ...}
    """
    return pd.DataFrame(results).T
