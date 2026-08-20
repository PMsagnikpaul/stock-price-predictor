"""
generate_sample_data.py
------------------------
Generates realistic SYNTHETIC OHLCV data (via geometric Brownian motion with
mild mean-reverting drift) and saves it in the same format data_pipeline.py
produces from yfinance. This exists purely so the rest of the pipeline can
be developed and tested WITHOUT live internet access to Yahoo Finance.

On your local machine, with internet access, you don't need this file —
just run:
    python -m src.data_pipeline --ticker AAPL --period 5y

This script is only a stand-in for that during offline development/testing.

Usage:
    python scripts/generate_sample_data.py --ticker DEMO --days 1500
"""

import argparse
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def generate_synthetic_ohlcv(days: int = 1500, start_price: float = 150.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    n = len(dates)

    # Geometric Brownian motion with slight drift + a slow-moving regime
    # component so technical indicators have something real to pick up on.
    mu, sigma = 0.0003, 0.018
    regime = np.sin(np.linspace(0, 6 * np.pi, n)) * 0.0004
    daily_returns = rng.normal(mu, sigma, n) + regime

    close = start_price * np.cumprod(1 + daily_returns)

    # Derive Open/High/Low around Close with plausible intraday noise
    intraday_noise = rng.normal(0, 0.006, n)
    open_ = close * (1 + intraday_noise)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(1_000_000, 8_000_000, n)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)
    df.index.name = "Date"
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OHLCV data for offline testing.")
    parser.add_argument("--ticker", default="DEMO")
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--start-price", type=float, default=150.0)
    args = parser.parse_args()

    df = generate_synthetic_ohlcv(days=args.days, start_price=args.start_price)
    df["Ticker"] = args.ticker

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{args.ticker}.csv")
    df.to_csv(path)
    print(f"Generated {len(df)} synthetic rows for '{args.ticker}' -> {path}")
    print(df.tail())


if __name__ == "__main__":
    main()
