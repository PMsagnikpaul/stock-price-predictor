"""
lstm_model.py
-------------
Advanced next-day closing price predictor using a stacked LSTM network
over a sliding window of past trading days.

Key design decision — predicting RETURNS, not raw price:
Price-level features (SMA, EMA, MACD, Bollinger Bands, lagged closes) and
the prediction target are all naturally on the same scale as the stock's
absolute price. For a stock that trends significantly over the training
window (very common — e.g. a stock that roughly doubles over 5 years), the
most recent (test) portion of the data sits at price levels the model's
scaler never saw during training. Neural nets, unlike linear models, do not
extrapolate gracefully beyond their training input range, so this silently
produces much worse LSTM performance than the Linear Regression baseline
even though the LSTM is the more "advanced" model.

The fix: express price-level features as RATIOS relative to that row's own
Close price (e.g. SMA_20 -> SMA_20/Close - 1), and predict the next-day
RETURN (percentage change) rather than the absolute price. Returns and
price-relative ratios are approximately stationary regardless of the
stock's absolute price level, so the scaler's train-fit range stays valid
on the test period. Predicted price is then reconstructed as
Close_today * (1 + predicted_return).

Sequence framing: given the last `window_size` days of (relativized)
features, predict the return `horizon` days ahead. Data is split
chronologically (train on the earliest portion, test on the most recent)
— never shuffled.

Usage (CLI):
    python -m src.lstm_model --ticker AAPL --window-size 60 --epochs 50

Usage (as a library):
    from src.lstm_model import train_lstm
    result = train_lstm(feature_df)
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from src.data_pipeline import load_data
from src.features import build_features, get_feature_columns
from src.evaluate import evaluate_predictions, print_metrics

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Feature columns that are on an absolute price scale and need to be
# expressed relative to that row's Close before feeding the LSTM.
PRICE_LEVEL_COLS = [
    "Open", "High", "Low",
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "MACD", "MACD_signal", "MACD_diff",
    "BB_high", "BB_low",
    "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag4", "Close_lag5",
]


def relativize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Convert absolute price-level feature columns into ratios relative to that
    row's own Close price, so the resulting features are approximately
    stationary regardless of the stock's absolute price trend over time.
    """
    out = df.copy()
    close = out["Close"]
    for c in feature_cols:
        if c in PRICE_LEVEL_COLS and c in out.columns:
            out[c] = out[c] / close - 1.0
    if "BB_width" in feature_cols and "BB_width" in out.columns:
        out["BB_width"] = out["BB_width"] / close
    return out


def make_sequences(X: np.ndarray, y: np.ndarray, window_size: int):
    """Convert flat feature/target arrays into sliding-window sequences."""
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i - window_size:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def build_lstm_model(input_shape, units=64, dropout=0.2):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_lstm(feature_df: pd.DataFrame, window_size: int = 60, test_size: float = 0.2,
                epochs: int = 50, batch_size: int = 32, save_path: str = None,
                verbose: int = 0) -> dict:
    """
    Train an LSTM model on engineered features using a sliding window,
    predicting next-`horizon` RETURN rather than absolute price (see module
    docstring for why).

    Returns a dict with the fitted model, scalers, predictions, actuals
    (both reconstructed to original PRICE scale), dates, and evaluation
    metrics — same interface as before, so callers (API, Streamlit,
    scripts/run_full_report.py) don't need to change.
    """
    all_feature_cols = get_feature_columns(feature_df)
    # Drop raw "Close" from LSTM inputs: after relativizing everything else
    # relative to Close, Close/Close-1 would just be a constant zero and
    # carries no information as a direct input.
    lstm_feature_cols = [c for c in all_feature_cols if c != "Close"]

    rel_df = relativize_features(feature_df, lstm_feature_cols)
    X_raw = rel_df[lstm_feature_cols].values

    close_raw = feature_df["Close"].values.astype(float)
    target_raw = feature_df["Target"].values.astype(float)
    # Next-`horizon` return relative to TODAY's close (row-aligned, matches
    # the reference point later used for directional accuracy).
    return_raw = (target_raw - close_raw) / close_raw

    dates = feature_df.index
    split_idx = int(len(feature_df) * (1 - test_size))

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(X_raw[:split_idx])
    y_scaler.fit(return_raw[:split_idx].reshape(-1, 1))

    X_scaled = x_scaler.transform(X_raw)
    y_scaled = y_scaler.transform(return_raw.reshape(-1, 1)).flatten()

    X_seq, y_seq = make_sequences(X_scaled, y_scaled, window_size)
    seq_dates = dates[window_size:]
    seq_close = close_raw[window_size:]  # Close[i] aligned with y_seq's row i — the reference price both for reconstructing predicted price and for directional accuracy

    seq_split_idx = split_idx - window_size
    if seq_split_idx <= 0:
        raise ValueError(
            f"window_size ({window_size}) is too large relative to the training "
            f"portion of the data. Reduce window_size or provide more history."
        )

    X_train, X_test = X_seq[:seq_split_idx], X_seq[seq_split_idx:]
    y_train, y_test = y_seq[:seq_split_idx], y_seq[seq_split_idx:]
    test_dates = seq_dates[seq_split_idx:]
    test_ref_close = seq_close[seq_split_idx:]

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
    )

    pred_return_scaled = model.predict(X_test, verbose=0)
    pred_return = y_scaler.inverse_transform(pred_return_scaled).flatten()

    # Reconstruct absolute price from the predicted return and that row's
    # known reference close price.
    preds = test_ref_close * (1 + pred_return)
    y_test_actual = test_ref_close * (1 + y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten())

    metrics = evaluate_predictions(y_test_actual, preds, prev_actual=test_ref_close)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        joblib.dump(
            {"x_scaler": x_scaler, "y_scaler": y_scaler, "feature_cols": lstm_feature_cols,
             "window_size": window_size},
            save_path.replace(".keras", "_meta.joblib"),
        )

    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_cols": lstm_feature_cols,
        "window_size": window_size,
        "y_test": y_test_actual,
        "preds": preds,
        "dates": test_dates,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train LSTM model for stock price prediction.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-lags", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    raw_df = load_data(args.ticker)
    feat_df = build_features(raw_df, n_lags=args.n_lags, horizon=args.horizon)

    save_path = os.path.join(MODELS_DIR, f"{args.ticker.replace('.', '_')}_lstm.keras")
    result = train_lstm(
        feat_df,
        window_size=args.window_size,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=save_path,
        verbose=args.verbose,
    )

    print_metrics(f"LSTM — {args.ticker}", result["metrics"])
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
