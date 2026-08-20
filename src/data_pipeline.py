"""
data_pipeline.py
-----------------
Fetches historical OHLCV stock data using yfinance, cleans it, and
saves it to disk as CSV for downstream feature engineering and modeling.

Usage (CLI):
    python -m src.data_pipeline --ticker RELIANCE.NS --period 5y
    python -m src.data_pipeline --ticker AAPL --start 2015-01-01 --end 2026-08-01

Usage (as a library):
    from src.data_pipeline import fetch_stock_data, save_data
    df = fetch_stock_data("AAPL", period="5y")
    save_data(df, "AAPL")
"""

import argparse
import os
import sys
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def fetch_stock_data(ticker: str, period: str = "5y", start: str = None, end: str = None,
                      interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker.

    Parameters
    ----------
    ticker : str
        Stock symbol. Use suffix '.NS' for NSE (India), e.g. 'RELIANCE.NS', 'TCS.NS'.
        US tickers need no suffix, e.g. 'AAPL', 'MSFT'.
    period : str
        Data period if start/end not given. e.g. '1y','2y','5y','10y','max'.
    start, end : str
        Explicit date range in 'YYYY-MM-DD' format. Overrides `period` if provided.
    interval : str
        Data granularity: '1d' (daily), '1wk' (weekly), '1mo' (monthly).

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV dataframe indexed by Date, columns:
        Open, High, Low, Close, Volume (Close is already split/dividend adjusted
        by yfinance's `auto_adjust=True` default).
    """
    if start and end:
        raw = yf.download(ticker, start=start, end=end, interval=interval,
                           auto_adjust=True, progress=False)
    else:
        raw = yf.download(ticker, period=period, interval=interval,
                           auto_adjust=True, progress=False)

    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. Check the symbol is correct "
            f"(NSE tickers need a '.NS' suffix, e.g. 'TCS.NS')."
        )

    # yfinance sometimes returns MultiIndex columns (ticker, field) when
    # multiple tickers are involved internally — flatten to single index.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"

    # Forward-fill any gaps (e.g. holidays that slipped through), then drop
    # any remaining leading NaNs.
    df = df.ffill().dropna()

    # Sanity checks
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Data contains non-positive prices — source data looks corrupted.")

    df = df.sort_index()
    df["Ticker"] = ticker
    return df


def save_data(df: pd.DataFrame, ticker: str, data_dir: str = DATA_DIR) -> str:
    """Save dataframe to CSV under data/<ticker>.csv, returns the file path."""
    os.makedirs(data_dir, exist_ok=True)
    safe_name = ticker.replace(".", "_")
    path = os.path.join(data_dir, f"{safe_name}.csv")
    df.to_csv(path)
    return path


def load_data(ticker: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load previously saved CSV for a ticker."""
    safe_name = ticker.replace(".", "_")
    path = os.path.join(data_dir, f"{safe_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached data found for '{ticker}' at {path}. "
            f"Run fetch_stock_data() and save_data() first."
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch and cache historical stock data.")
    parser.add_argument("--ticker", required=True, help="Stock ticker, e.g. AAPL or TCS.NS")
    parser.add_argument("--period", default="5y", help="Period if start/end not set (default 5y)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--interval", default="1d", help="Data interval (default 1d)")
    args = parser.parse_args()

    try:
        df = fetch_stock_data(args.ticker, period=args.period, start=args.start,
                               end=args.end, interval=args.interval)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    path = save_data(df, args.ticker)
    print(f"Saved {len(df)} rows for {args.ticker} -> {path}")
    print(df.tail())


if __name__ == "__main__":
    main()
