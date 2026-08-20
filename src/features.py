"""
features.py
------------
Feature engineering for the Stock Price Predictor: technical indicators,
lag features, and calendar features. Designed to avoid look-ahead bias —
every feature at row t uses only information available up to and
including day t.

Usage:
    from src.features import build_features
    feat_df = build_features(raw_ohlcv_df)
"""

import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, EMA, RSI, MACD, and Bollinger Band features."""
    out = df.copy()
    close = out["Close"]

    out["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()
    out["SMA_50"] = SMAIndicator(close, window=50).sma_indicator()
    out["EMA_12"] = EMAIndicator(close, window=12).ema_indicator()
    out["EMA_26"] = EMAIndicator(close, window=26).ema_indicator()

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out["MACD"] = macd.macd()
    out["MACD_signal"] = macd.macd_signal()
    out["MACD_diff"] = macd.macd_diff()

    out["RSI_14"] = RSIIndicator(close, window=14).rsi()

    bb = BollingerBands(close, window=20, window_dev=2)
    out["BB_high"] = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_width"] = out["BB_high"] - out["BB_low"]

    # Volume-based
    out["Volume_change"] = out["Volume"].pct_change()
    out["Volume_SMA_20"] = out["Volume"].rolling(window=20).mean()

    # Volatility / returns
    out["Daily_return"] = close.pct_change()
    out["Volatility_10"] = out["Daily_return"].rolling(window=10).std()

    return out


def add_lag_features(df: pd.DataFrame, target_col: str = "Close", n_lags: int = 5) -> pd.DataFrame:
    """Add lagged closing price features: Close_lag1 ... Close_lagN."""
    out = df.copy()
    for lag in range(1, n_lags + 1):
        out[f"{target_col}_lag{lag}"] = out[target_col].shift(lag)
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week and month features (captures weak seasonality if present)."""
    out = df.copy()
    out["DayOfWeek"] = out.index.dayofweek
    out["Month"] = out.index.month
    return out


def build_features(df: pd.DataFrame, n_lags: int = 5, horizon: int = 1) -> pd.DataFrame:
    """
    Full feature pipeline: technical indicators + lag features + calendar features,
    plus the prediction target (next `horizon`-day closing price).

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV dataframe (output of data_pipeline.fetch_stock_data).
    n_lags : int
        Number of lagged close-price features to generate.
    horizon : int
        How many trading days ahead to predict. 1 = next day.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe with a 'Target' column, NaN rows dropped
        (NaNs arise from rolling-window warm-up and the shift for the target).
    """
    out = add_technical_indicators(df)
    out = add_lag_features(out, n_lags=n_lags)
    out = add_calendar_features(out)

    # Target: closing price `horizon` days into the future.
    out["Target"] = out["Close"].shift(-horizon)

    # Real market data occasionally has a zero-volume day (illiquid session,
    # exchange holiday that slipped through, etc.), which makes Volume_change
    # (a pct_change) blow up to +/-inf. Treat inf the same as a missing value
    # rather than letting it propagate into the scaler and crash training.
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    out = out.dropna()
    return out


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of model input feature columns (excludes Target, Ticker)."""
    exclude = {"Target", "Ticker"}
    return [c for c in df.columns if c not in exclude]
