"""
main.py — FastAPI backend for the Stock Price Predictor.

Endpoints:
    GET  /                          -> health check
    GET  /history?ticker=AAPL       -> historical OHLCV (fetches + caches on demand)
    POST /train                     -> train baseline + LSTM models for a ticker
    GET  /predict?ticker=AAPL&model=lstm&days=5  -> forecast next N days

Run:
    uvicorn api.main:app --reload --port 8000
"""

import os
import sys
import traceback
from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import fetch_stock_data, save_data, load_data
from src.features import build_features, get_feature_columns
from src.baseline_model import train_baseline
from src.lstm_model import train_lstm, make_sequences

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

app = FastAPI(title="Stock Price Predictor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    ticker: str
    period: str = "5y"
    epochs: int = 30
    window_size: int = 60


@app.get("/")
def root():
    return {"status": "ok", "service": "Stock Price Predictor API"}


@app.get("/history")
def get_history(ticker: str = Query(...), period: str = Query("2y")):
    """Fetch (and cache) historical OHLCV data for a ticker."""
    try:
        df = fetch_stock_data(ticker, period=period)
        save_data(df, ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch data: {e}")

    return {
        "ticker": ticker,
        "rows": len(df),
        "data": [
            {"date": str(idx.date()), "open": row.Open, "high": row.High,
             "low": row.Low, "close": row.Close, "volume": int(row.Volume)}
            for idx, row in df.iterrows()
        ],
    }


@app.post("/train")
def train_models(req: TrainRequest):
    """Train both the Linear Regression baseline and the LSTM model for a ticker."""
    try:
        df = fetch_stock_data(req.ticker, period=req.period)
        save_data(df, req.ticker)
        feat_df = build_features(df)

        lin_path = os.path.join(MODELS_DIR, f"{req.ticker.replace('.', '_')}_linear.joblib")
        lstm_path = os.path.join(MODELS_DIR, f"{req.ticker.replace('.', '_')}_lstm.keras")

        lin_result = train_baseline(feat_df, save_path=lin_path)
        lstm_result = train_lstm(
            feat_df, window_size=req.window_size, epochs=req.epochs,
            save_path=lstm_path, verbose=0,
        )

        return {
            "ticker": req.ticker,
            "linear_regression_metrics": lin_result["metrics"],
            "lstm_metrics": lstm_result["metrics"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@app.get("/predict")
def predict(
    ticker: str = Query(...),
    model: Literal["linear", "lstm"] = Query("lstm"),
    days: int = Query(5, ge=1, le=30),
):
    """
    Forecast the next `days` trading days of closing price for `ticker`.
    Requires the model to have been trained already via /train (or the CLI
    scripts). Uses iterative (recursive) multi-step forecasting: each
    predicted day feeds back in as input for the next.
    """
    safe = ticker.replace(".", "_")

    try:
        raw_df = load_data(ticker)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No cached data for '{ticker}'. Call /train or /history first.",
        )

    feat_df = build_features(raw_df)
    feature_cols = get_feature_columns(feat_df)

    if model == "linear":
        path = os.path.join(MODELS_DIR, f"{safe}_linear.joblib")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"No trained linear model for '{ticker}'. Call /train first.")
        bundle = joblib.load(path)
        lr_model, scaler = bundle["model"], bundle["scaler"]

        last_row = feat_df.iloc[[-1]][feature_cols]
        preds = []
        for _ in range(days):
            X_scaled = scaler.transform(last_row)
            pred = float(lr_model.predict(X_scaled)[0])
            preds.append(pred)
            # naive recursive update: shift lag features forward with the new prediction
            last_row = last_row.copy()
            if "Close_lag1" in last_row.columns:
                for lag in range(5, 1, -1):
                    if f"Close_lag{lag}" in last_row.columns and f"Close_lag{lag-1}" in last_row.columns:
                        last_row[f"Close_lag{lag}"] = last_row[f"Close_lag{lag-1}"]
                last_row["Close_lag1"] = pred

    else:  # lstm
        model_path = os.path.join(MODELS_DIR, f"{safe}_lstm.keras")
        meta_path = os.path.join(MODELS_DIR, f"{safe}_lstm_meta.joblib")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"No trained LSTM model for '{ticker}'. Call /train first.")

        import tensorflow as tf
        from src.lstm_model import relativize_features

        lstm_model = tf.keras.models.load_model(model_path)
        meta = joblib.load(meta_path)
        x_scaler, y_scaler = meta["x_scaler"], meta["y_scaler"]
        window_size = meta["window_size"]
        meta_feature_cols = meta["feature_cols"]  # already excludes "Close"

        rel_df = relativize_features(feat_df, meta_feature_cols)
        X_all = rel_df[meta_feature_cols].values
        X_scaled_all = x_scaler.transform(X_all)
        window = X_scaled_all[-window_size:].copy()

        last_close = float(feat_df["Close"].iloc[-1])
        preds = []
        for _ in range(days):
            X_input = window.reshape(1, window_size, -1)
            pred_return_scaled = lstm_model.predict(X_input, verbose=0)
            pred_return = float(y_scaler.inverse_transform(pred_return_scaled)[0][0])
            pred_price = last_close * (1 + pred_return)
            preds.append(pred_price)
            # Recursive step: reuse the last window row as a naive proxy for
            # the next day's (still-relativized) feature vector. This keeps
            # the sequence length constant for iterative multi-day forecasts;
            # treat forecasts beyond ~5-7 days as low-confidence (see README).
            window = np.vstack([window[1:], window[-1]])
            last_close = pred_price

    last_date = feat_df.index[-1]
    future_dates = [str((last_date + timedelta(days=i)).date()) for i in range(1, days + 1)]

    return {
        "ticker": ticker,
        "model": model,
        "last_actual_close": float(feat_df["Close"].iloc[-1]),
        "last_actual_date": str(last_date.date()),
        "forecast": [{"date": d, "predicted_close": round(p, 2)} for d, p in zip(future_dates, preds)],
    }
