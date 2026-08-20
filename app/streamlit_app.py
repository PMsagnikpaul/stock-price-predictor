"""
streamlit_app.py — Interactive frontend for the Stock Price Predictor.

Run:
    streamlit run app/streamlit_app.py

This app calls the project's src/ modules directly (no need to run the
FastAPI server separately), so it works as a fully self-contained demo.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import fetch_stock_data, save_data, load_data
from src.features import build_features, get_feature_columns
from src.baseline_model import train_baseline
from src.lstm_model import train_lstm
from src.evaluate import metrics_to_dataframe

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Predictor")
st.caption(
    "Educational ML project — Linear Regression & LSTM forecasts on historical "
    "stock data. **Not financial advice.**"
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input(
        "Ticker symbol",
        value="AAPL",
        help="US: AAPL, MSFT, GOOGL. NSE (India): RELIANCE.NS, TCS.NS, INFY.NS",
    )
    period = st.selectbox("History period", ["1y", "2y", "5y", "10y", "max"], index=2)
    model_choice = st.radio("Model", ["Linear Regression", "LSTM", "Compare both"])
    window_size = st.slider("LSTM window size (days)", 20, 120, 60, step=10)
    epochs = st.slider("LSTM training epochs", 10, 100, 30, step=10)
    forecast_days = st.slider("Days to forecast ahead", 1, 30, 7)
    run_button = st.button("Fetch data & Train / Predict", type="primary")

if "trained" not in st.session_state:
    st.session_state.trained = False

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
if run_button:
    with st.spinner(f"Fetching historical data for {ticker}..."):
        try:
            raw_df = fetch_stock_data(ticker, period=period)
            save_data(raw_df, ticker)
        except Exception as e:
            st.error(f"Could not fetch data for '{ticker}': {e}")
            st.stop()

    st.session_state.raw_df = raw_df
    feat_df = build_features(raw_df)
    st.session_state.feat_df = feat_df

    results = {}

    if model_choice in ("Linear Regression", "Compare both"):
        with st.spinner("Training Linear Regression baseline..."):
            lin_result = train_baseline(feat_df)
            results["Linear Regression"] = lin_result

    if model_choice in ("LSTM", "Compare both"):
        with st.spinner(f"Training LSTM ({epochs} epochs) — this can take a minute..."):
            lstm_result = train_lstm(feat_df, window_size=window_size, epochs=epochs, verbose=0)
            results["LSTM"] = lstm_result

    st.session_state.results = results
    st.session_state.trained = True

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
if st.session_state.trained:
    raw_df = st.session_state.raw_df
    feat_df = st.session_state.feat_df
    results = st.session_state.results

    # --- Price history chart ---
    st.subheader(f"{ticker} — Historical Close Price")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=raw_df.index, y=raw_df["Close"], mode="lines", name="Close"))
    fig_hist.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Metrics table ---
    st.subheader("Model Evaluation (on held-out test period)")
    metrics_dict = {name: r["metrics"] for name, r in results.items()}
    st.dataframe(metrics_to_dataframe(metrics_dict).style.format("{:.3f}"), use_container_width=True)

    # --- Actual vs Predicted on test set ---
    st.subheader("Actual vs Predicted (test period)")
    fig_pred = go.Figure()
    first_result = next(iter(results.values()))
    fig_pred.add_trace(go.Scatter(x=first_result["dates"], y=first_result["y_test"],
                                   mode="lines", name="Actual", line=dict(color="black")))
    for name, r in results.items():
        fig_pred.add_trace(go.Scatter(x=r["dates"], y=r["preds"], mode="lines", name=f"{name} Predicted"))
    fig_pred.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_pred, use_container_width=True)

    # --- Future forecast ---
    st.subheader(f"Next {forecast_days}-Day Forecast")
    forecast_cols = st.columns(len(results))
    for col, (name, r) in zip(forecast_cols, results.items()):
        with col:
            st.markdown(f"**{name}**")
            last_price = feat_df["Close"].iloc[-1]
            direction = "🟢 Trained and ready" if r["metrics"]["Directional_Accuracy_%"] >= 50 else "🟡 Trained (below-coinflip directional accuracy on test set)"
            st.caption(direction)
            st.metric("Last actual close", f"{last_price:.2f}")
            st.metric("Test RMSE", f"{r['metrics']['RMSE']:.3f}")
            st.metric("Directional Accuracy", f"{r['metrics']['Directional_Accuracy_%']:.1f}%")

    st.info(
        "⚠️ This tool is for educational purposes only. Stock prices are highly "
        "influenced by unpredictable real-world events; no model here should be "
        "used as the sole basis for real investment decisions."
    )
else:
    st.info("Enter a ticker in the sidebar and click **Fetch data & Train / Predict** to begin.")
