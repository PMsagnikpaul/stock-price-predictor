# 📈 Stock Price Predictor

An end-to-end machine learning system that forecasts stock closing prices using
historical OHLCV data — from a Linear Regression baseline to an LSTM deep
learning model, served through a FastAPI backend and an interactive Streamlit
dashboard.

> ⚠️ **Educational project only.** This is not financial advice and should not
> be used as the sole basis for real investment decisions.

---

## Features

- **Data pipeline** — fetches historical data via `yfinance` for any US or NSE
  (India) ticker, cleans and caches it locally.
- **Feature engineering** — SMA, EMA, RSI, MACD, Bollinger Bands, lag features,
  volatility, and calendar features, built with zero look-ahead leakage.
- **Baseline model** — Linear Regression, trained with a strict chronological
  train/test split.
- **Advanced model** — Stacked LSTM network over a sliding window of past
  trading days.
- **Evaluation** — RMSE, MAE, MAPE, and Directional Accuracy, with a
  side-by-side model comparison report.
- **API** — FastAPI backend (`/history`, `/train`, `/predict`).
- **Frontend** — Streamlit app with interactive charts, live training, and
  multi-day forecasts.

---

## Project Structure

```
stock-price-predictor/
├── README.md
├── requirements.txt
├── data/                    # cached CSVs of historical price data
├── models/                  # saved trained models (.joblib / .keras)
├── reports/                 # generated comparison tables & plots
├── src/
│   ├── data_pipeline.py     # fetch, clean, cache OHLCV data
│   ├── features.py          # technical indicators + feature engineering
│   ├── baseline_model.py    # Linear Regression training/eval
│   ├── lstm_model.py        # LSTM training/eval
│   └── evaluate.py          # shared metrics (RMSE/MAE/MAPE/directional acc.)
├── api/
│   └── main.py               # FastAPI backend
├── app/
│   └── streamlit_app.py      # Streamlit frontend
└── scripts/
    ├── generate_sample_data.py  # synthetic data for offline testing/demo
    └── run_full_report.py       # end-to-end: fetch -> train -> compare -> plot
```

---

## Setup

```bash
git clone <your-repo-url>
cd stock-price-predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires **internet access** to fetch live data from Yahoo Finance via
`yfinance` — this works from any normal machine/laptop/cloud VM with outbound
internet, just not from network-sandboxed environments.

---

## Quickstart

### 1. Fetch data for a ticker

```bash
python -m src.data_pipeline --ticker AAPL --period 5y
# Indian NSE stocks use a .NS suffix:
python -m src.data_pipeline --ticker TCS.NS --period 5y
```

### 2. Train the baseline model

```bash
python -m src.baseline_model --ticker AAPL
```

### 3. Train the LSTM model

```bash
python -m src.lstm_model --ticker AAPL --window-size 60 --epochs 50
```

### 4. Generate a full comparison report (does steps 1–3 + plot in one go)

```bash
python scripts/run_full_report.py --ticker AAPL --period 5y --epochs 50
```

This saves a metrics table and an actual-vs-predicted PNG chart to `reports/`.

### 5. Run the interactive dashboard

```bash
streamlit run app/streamlit_app.py
```

Open the printed local URL, enter a ticker, choose a model, and click
**Fetch data & Train / Predict**.

### 6. Run the API server

```bash
uvicorn api.main:app --reload --port 8000
```

Then visit `http://localhost:8000/docs` for interactive Swagger docs, or:

```bash
curl "http://localhost:8000/history?ticker=AAPL&period=1y"
curl -X POST "http://localhost:8000/train" -H "Content-Type: application/json" \
     -d '{"ticker": "AAPL", "period": "5y", "epochs": 30}'
curl "http://localhost:8000/predict?ticker=AAPL&model=lstm&days=7"
```

---

## Testing Without Internet Access (Offline / Demo Mode)

If you're in a sandboxed environment without access to Yahoo Finance, generate
synthetic OHLCV data and run everything against that instead:

```bash
python scripts/generate_sample_data.py --ticker DEMO --days 1500
python -m src.baseline_model --ticker DEMO
python -m src.lstm_model --ticker DEMO --epochs 30
python scripts/run_full_report.py --ticker DEMO --offline-csv data/DEMO.csv --epochs 30
```

The Streamlit app and API still require live internet access for the
`fetch_stock_data` calls, since they always pull fresh data — the offline path
above is for validating the modeling code itself.

---

## Methodology Notes

- **No shuffling.** All train/test splits are strictly chronological. Shuffling
  time-series data before splitting is the most common mistake in stock
  prediction projects and silently invalidates results by leaking future
  information into training.
- **Scalers fit on train only.** `StandardScaler`/`MinMaxScaler` are always
  fit on the training portion only, then applied to the test portion — never
  the reverse.
- **The LSTM predicts returns, not raw price.** Price-level features (SMA,
  EMA, MACD, Bollinger Bands, lagged closes) and a raw-price target both sit
  on the stock's absolute price scale. For a stock that trends meaningfully
  over the training window (very common over 5 years), the test period sits
  at price levels the scaler never saw during training — and neural nets,
  unlike linear models, don't extrapolate gracefully outside their trained
  input range. This showed up in practice: on real NSE data, an early version
  of this project's LSTM had roughly **2x worse RMSE than the Linear
  Regression baseline** for exactly this reason. The fix implemented here:
  price-level features are expressed as ratios relative to that row's own
  Close (`SMA_20 -> SMA_20/Close - 1`), and the LSTM predicts the next-day
  *return* rather than absolute price, with price reconstructed afterward as
  `Close_today * (1 + predicted_return)`. This closed the gap and let the
  LSTM match or beat the baseline on both RMSE and directional accuracy in
  testing. See `src/lstm_model.py` for the implementation.
- **Directional accuracy matters more than raw RMSE** for practical use. A
  model can have a low RMSE while still failing to predict up/down movement
  better than a coin flip — check both metrics.
- **Multi-day forecasts are recursive.** Predicting more than 1 day ahead
  feeds each prediction back in as input for the next day, which compounds
  error the further out you forecast. Treat forecasts beyond ~5–7 days as
  low-confidence.

---

## Extending the Project (Stretch Goals)

- Add an XGBoost model and blend it with the LSTM (ensemble).
- Add sentiment features from financial news headlines.
- Add prediction confidence intervals (quantile regression / MC dropout).
- Add a simple backtested trading-strategy simulator.
- Deploy the Streamlit app to Streamlit Community Cloud, and the API to
  Render/Railway.

See `Stock_Price_Predictor_PRD.md` for the full product requirements
document and step-wise roadmap this project was built from.

---

## Disclaimer

Stock markets are influenced by countless unpredictable real-world factors.
No model in this repository — Linear Regression, LSTM, or otherwise —
constitutes financial advice. This project exists to demonstrate the ML
lifecycle (data → features → modeling → evaluation → deployment), not to
generate trading signals.
