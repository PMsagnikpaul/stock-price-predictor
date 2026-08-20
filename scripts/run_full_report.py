"""
run_full_report.py
-------------------
End-to-end driver: fetch data -> engineer features -> train Linear Regression
baseline + LSTM -> print comparison table -> save an actual-vs-predicted plot.

Usage:
    python scripts/run_full_report.py --ticker AAPL --period 5y --epochs 30
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import fetch_stock_data, save_data
from src.features import build_features
from src.baseline_model import train_baseline
from src.evaluate import metrics_to_dataframe, print_metrics
from src.lstm_model import train_lstm

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def main():
    parser = argparse.ArgumentParser(description="Run full baseline + LSTM comparison report.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--offline-csv", default=None,
                         help="Optional: path to a locally cached CSV (skips live fetch, "
                              "for offline/demo use, e.g. data/DEMO.csv)")
    args = parser.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)

    if args.offline_csv:
        import pandas as pd
        raw_df = pd.read_csv(args.offline_csv, index_col="Date", parse_dates=True)
    else:
        raw_df = fetch_stock_data(args.ticker, period=args.period)
        save_data(raw_df, args.ticker)

    feat_df = build_features(raw_df)
    print(f"Feature-engineered dataset: {feat_df.shape[0]} rows, {feat_df.shape[1]} columns")

    lin_path = os.path.join(MODELS_DIR, f"{args.ticker.replace('.', '_')}_linear.joblib")
    lin_result = train_baseline(feat_df, save_path=lin_path)
    print_metrics(f"Linear Regression — {args.ticker}", lin_result["metrics"])

    lstm_path = os.path.join(MODELS_DIR, f"{args.ticker.replace('.', '_')}_lstm.keras")
    lstm_result = train_lstm(feat_df, window_size=args.window_size, epochs=args.epochs,
                              save_path=lstm_path, verbose=0)
    print_metrics(f"LSTM — {args.ticker}", lstm_result["metrics"])

    comparison = metrics_to_dataframe({
        "Linear Regression": lin_result["metrics"],
        "LSTM": lstm_result["metrics"],
    })
    print("\n=== Model Comparison ===")
    print(comparison.round(4).to_string())

    csv_path = os.path.join(REPORTS_DIR, f"{args.ticker.replace('.', '_')}_comparison.csv")
    comparison.to_csv(csv_path)
    print(f"\nComparison table saved to {csv_path}")

    # Plot: actual vs predicted for both models
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lin_result["dates"], lin_result["y_test"], label="Actual", color="black", linewidth=1.5)
    ax.plot(lin_result["dates"], lin_result["preds"], label="Linear Regression", alpha=0.8)
    ax.plot(lstm_result["dates"], lstm_result["preds"], label="LSTM", alpha=0.8)
    ax.set_title(f"{args.ticker} — Actual vs Predicted Close Price (Test Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    plot_path = os.path.join(REPORTS_DIR, f"{args.ticker.replace('.', '_')}_comparison_plot.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
