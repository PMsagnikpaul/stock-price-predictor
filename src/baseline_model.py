"""
baseline_model.py
------------------
Baseline next-day closing price predictor using Linear Regression on
engineered features (technical indicators + lag features).

Uses a strict chronological train/test split (no shuffling) to avoid
look-ahead bias, which is the single most common mistake in stock
prediction projects.

Usage (CLI):
    python -m src.baseline_model --ticker AAPL --test-size 0.2

Usage (as a library):
    from src.baseline_model import train_baseline
    result = train_baseline(feature_df)
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import load_data
from src.features import build_features, get_feature_columns
from src.evaluate import evaluate_predictions, print_metrics

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def chronological_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split a time-indexed dataframe into train/test WITHOUT shuffling."""
    split_idx = int(len(df) * (1 - test_size))
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    return train, test


def train_baseline(feature_df: pd.DataFrame, test_size: float = 0.2, save_path: str = None) -> dict:
    """
    Train a Linear Regression baseline on engineered features.

    Returns a dict with the fitted model, scaler, predictions, actuals,
    dates, and evaluation metrics.
    """
    feature_cols = get_feature_columns(feature_df)
    train_df, test_df = chronological_split(feature_df, test_size)

    X_train, y_train = train_df[feature_cols], train_df["Target"]
    X_test, y_test = test_df[feature_cols], test_df["Target"]

    # Fit scaler on TRAIN ONLY to avoid leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)

    # "previous actual" for directional accuracy = actual Close at t
    # (since Target is Close at t+horizon)
    prev_actual = test_df["Close"].values

    metrics = evaluate_predictions(y_test.values, preds, prev_actual=prev_actual)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols}, save_path)

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "y_test": y_test.values,
        "preds": preds,
        "dates": test_df.index,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train baseline Linear Regression model.")
    parser.add_argument("--ticker", required=True, help="Ticker previously cached via data_pipeline")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-lags", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()

    raw_df = load_data(args.ticker)
    feat_df = build_features(raw_df, n_lags=args.n_lags, horizon=args.horizon)

    save_path = os.path.join(MODELS_DIR, f"{args.ticker.replace('.', '_')}_linear.joblib")
    result = train_baseline(feat_df, test_size=args.test_size, save_path=save_path)

    print_metrics(f"Linear Regression — {args.ticker}", result["metrics"])
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
