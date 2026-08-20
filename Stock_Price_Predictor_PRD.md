# Product Requirements Document (PRD)
## Stock Price Predictor

**Version:** 1.0
**Author:** Sagnik Paul
**Date:** August 2026
**Status:** Draft for Development

---

## 1. Executive Summary

Stock Price Predictor is a machine learning system that forecasts future stock prices using historical price and volume data. The project progresses from a baseline linear regression model to advanced sequential models (LSTM/GRU) and optionally ensemble/transformer-based approaches, culminating in a deployable web application with visualization and backtesting.

**Goal:** Build a working, demonstrable, and technically credible predictor suitable for a portfolio project, resume line, or hackathon submission — not a live trading system.

**Non-Goal:** This is not a financial advice tool. It will not be used for real capital allocation without extensive further validation, risk management, and regulatory review.

---

## 2. Problem Statement

Retail investors and students lack accessible tools to visualize and forecast short-term stock price trends using standard time-series ML techniques. Existing tools (Bloomberg terminals, institutional platforms) are expensive or inaccessible. There is an opportunity to build an educational, technically rigorous, open predictor that demonstrates the full ML lifecycle: data ingestion → feature engineering → modeling → evaluation → deployment.

---

## 3. Objectives & Success Metrics

| Objective | Success Metric |
|---|---|
| Build a working baseline model | Linear Regression trained and evaluated on ≥1 stock, RMSE/MAE reported |
| Build an advanced model | LSTM/GRU model outperforms baseline on same test set |
| Deployable product | Working web app (Streamlit/Flask/React) where user inputs a ticker and gets a forecast chart |
| Credibility | Backtested performance metrics (RMSE, MAPE, directional accuracy) documented |
| Portfolio-readiness | Clean GitHub repo, README, architecture diagram, optionally a demo video |

---

## 4. Target Users / Use Cases

1. **You (developer)** — resume/portfolio project, demonstrates ML + full-stack skills.
2. **Recruiters/evaluators** — judge technical depth via GitHub repo and live demo.
3. **Hobbyist end-user** — enters a stock ticker (e.g., RELIANCE.NS, AAPL) and views a predicted price trend for the next N days.
4. **Hackathon judges** (if repurposed) — evaluate novelty, correctness, UI polish.

---

## 5. Scope

### In Scope (v1 — Core Deliverable)
- Historical OHLCV (Open/High/Low/Close/Volume) data ingestion for one or more tickers
- Feature engineering (moving averages, RSI, MACD, lag features)
- Baseline model: Linear Regression
- Advanced model: LSTM (and optionally GRU/Bi-LSTM comparison)
- Train/test split with proper time-series validation (no data leakage)
- Evaluation: RMSE, MAE, MAPE, directional accuracy
- Visualization: actual vs predicted price chart
- Simple web interface: ticker input → forecast output

### In Scope (v2 — Stretch Goals)
- Multi-stock support with dropdown/search
- Sentiment features from financial news (NLP)
- Ensemble model (LSTM + XGBoost blend) — you already have XGBoost experience from your cholesterol project and LSTM/HMM experience from your investment platform, so this is a natural extension
- Confidence intervals / uncertainty bands on forecasts
- Backtesting module simulating a simple trading strategy
- Deployment to a public URL (Streamlit Cloud / Render / Vercel)

### Out of Scope
- Real-time tick-level trading execution
- Brokerage integration / order placement
- Guaranteed-accuracy or advisory claims
- Regulatory compliance (SEBI/SEC) — this is an educational tool only

---

## 6. System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Data Source     │────▶│  Data Pipeline     │────▶│  Feature Store /    │
│  (yfinance /     │     │  (fetch, clean,    │     │  Processed Dataset  │
│  Alpha Vantage / │     │  resample)         │     │  (CSV/Parquet/DB)   │
│  NSE/BSE API)    │     └──────────────────┘     └────────────────────┘
└─────────────────┘                                          │
                                                               ▼
                                                   ┌────────────────────┐
                                                   │  Feature Engineering│
                                                   │  (MA, RSI, MACD,    │
                                                   │  lag features,      │
                                                   │  scaling)           │
                                                   └────────────────────┘
                                                               │
                                    ┌──────────────────────────┼──────────────────────────┐
                                    ▼                          ▼                          ▼
                        ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
                        │ Baseline Model     │     │ Advanced Model    │     │ (Stretch) Ensemble │
                        │ Linear Regression  │     │ LSTM / GRU        │     │ LSTM + XGBoost     │
                        └───────────────────┘     └───────────────────┘     └───────────────────┘
                                    │                          │                          │
                                    └──────────────┬───────────┴──────────────────────────┘
                                                    ▼
                                        ┌────────────────────┐
                                        │  Evaluation Module  │
                                        │  RMSE/MAE/MAPE/Dir. │
                                        │  Accuracy, plots     │
                                        └────────────────────┘
                                                    │
                                                    ▼
                                        ┌────────────────────┐
                                        │  Serving Layer       │
                                        │  Flask/FastAPI API   │
                                        └────────────────────┘
                                                    │
                                                    ▼
                                        ┌────────────────────┐
                                        │  Frontend             │
                                        │  Streamlit / React    │
                                        │  Chart.js/Plotly       │
                                        └────────────────────┘
```

---

## 7. Technical Approach

### 7.1 Data Sources
- **`yfinance`** (Python library) — free, reliable, supports NSE (`.NS` suffix for Indian stocks like `RELIANCE.NS`, `TCS.NS`) and global tickers (`AAPL`, `GOOGL`)
- Alternative: Alpha Vantage API (free tier, rate-limited), NSE India public data
- Recommended for you: start with `yfinance` — zero cost, no API key friction

### 7.2 Data Pipeline
1. Fetch daily OHLCV data for chosen ticker(s), typically 3–10 years of history
2. Handle missing values (forward-fill for non-trading days)
3. Sort chronologically, check for stock splits/dividends adjustments (`Adj Close`)
4. Resample if needed (daily is standard; weekly for longer-horizon forecasts)

### 7.3 Feature Engineering
- **Lag features:** Close price at t-1, t-2, ..., t-n
- **Technical indicators:**
  - Simple/Exponential Moving Averages (SMA-20, SMA-50, EMA-12, EMA-26)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Volume-based features (volume change, OBV)
- **Date features:** day of week, month (captures seasonality if any)
- **Scaling:** MinMaxScaler or StandardScaler (fit only on training data to avoid leakage)

### 7.4 Modeling Strategy

**Phase 1 — Baseline: Linear Regression**
- Predict next-day closing price using lag features + technical indicators
- Fast to train, interpretable, sets a performance floor
- Use `scikit-learn`

**Phase 2 — Advanced: LSTM (Long Short-Term Memory)**
- Sequence model using a sliding window (e.g., past 60 days → predict next day)
- Architecture: 2 stacked LSTM layers (50–100 units) + Dropout (0.2) + Dense output layer
- Framework: TensorFlow/Keras or PyTorch (your choice — you've used Python ML pipelines before)
- This directly extends the LSTM experience from your ET AI Hackathon investment platform

**Phase 3 (Stretch) — Ensemble / Hybrid**
- Combine LSTM (captures temporal patterns) with XGBoost (captures nonlinear feature interactions) via weighted averaging or stacking
- Optionally add an HMM-based regime detector (bull/bear/sideways) as a feature — reusing your HMM regime detection module concept from the investment platform

### 7.5 Validation Strategy (Critical — Avoid Common Mistakes)
- **Never shuffle time-series data** — always use chronological train/test split (e.g., 80/20 by date)
- Use **walk-forward validation** (expanding or rolling window) for more robust evaluation than a single split
- Report metrics on a held-out test period the model never saw

### 7.6 Evaluation Metrics
| Metric | Purpose |
|---|---|
| RMSE (Root Mean Squared Error) | Penalizes large errors, standard regression metric |
| MAE (Mean Absolute Error) | Average magnitude of error, more interpretable |
| MAPE (Mean Absolute Percentage Error) | Scale-independent, good for comparing across stocks |
| Directional Accuracy | % of times model correctly predicts up/down movement — arguably more important than exact price for practical use |

---

## 8. Product Features (User-Facing)

| Feature | Priority | Description |
|---|---|---|
| Ticker input | P0 | User enters/selects a stock symbol |
| Historical chart | P0 | Show actual price history |
| Forecast chart | P0 | Overlay predicted vs actual on test period, plus N-day future forecast |
| Model comparison toggle | P1 | Switch between Linear Regression and LSTM predictions |
| Metrics dashboard | P1 | Display RMSE/MAE/MAPE/Directional Accuracy for transparency |
| Multi-stock support | P2 | Dropdown of popular NSE/S&P tickers |
| Confidence bands | P2 | Show prediction uncertainty range |
| Disclaimer banner | P0 | "Not financial advice — educational project only" (important for credibility and to avoid misuse) |

---

## 9. Tech Stack Recommendation

| Layer | Recommendation |
|---|---|
| Data fetching | `yfinance` (Python) |
| Data processing | `pandas`, `numpy` |
| Technical indicators | `ta` or `pandas-ta` library |
| Baseline ML | `scikit-learn` |
| Deep learning | `TensorFlow/Keras` (simpler for LSTM) or `PyTorch` |
| Backend API | `FastAPI` (lightweight, async, auto-docs) |
| Frontend | `Streamlit` (fastest for a solo dev to ship) OR `React + Chart.js/Plotly` (more polished, more work) |
| Visualization | `Plotly` (interactive charts) or `matplotlib` (static, simpler) |
| Deployment | Streamlit Community Cloud (free) / Render / Railway |
| Version control | GitHub, with clear README and architecture diagram |

**Recommendation for you specifically:** Given your bias toward complete, deployable outputs and prior experience, I'd suggest **Streamlit** for the frontend — it lets you ship a fully working interactive app in a fraction of the time of a React build, while still looking professional. You can always upgrade to React later if you want a portfolio-grade polish pass.

---

## 10. Step-Wise Development Plan

### Step 1 — Environment & Data Setup (Day 1)
- Set up Python environment (`venv` or `conda`)
- Install: `yfinance pandas numpy scikit-learn tensorflow matplotlib plotly ta streamlit fastapi`
- Fetch and save historical data for 1 chosen stock (e.g., `TCS.NS` or `AAPL`) to CSV
- Exploratory Data Analysis (EDA): plot price history, check for missing values, seasonality

### Step 2 — Feature Engineering (Day 1–2)
- Compute technical indicators (SMA, EMA, RSI, MACD)
- Create lag features
- Handle NaNs introduced by rolling window calculations
- Split data chronologically into train/test (e.g., 80/20)

### Step 3 — Baseline Model: Linear Regression (Day 2)
- Train `sklearn.linear_model.LinearRegression` on engineered features
- Predict next-day close on test set
- Compute RMSE, MAE, MAPE, directional accuracy
- Plot actual vs predicted

### Step 4 — Advanced Model: LSTM (Day 3–5)
- Reshape data into sequences (sliding window, e.g., 60-day lookback)
- Scale data with MinMaxScaler (fit on train only)
- Build LSTM architecture (Keras Sequential: LSTM → Dropout → LSTM → Dropout → Dense)
- Train with early stopping and validation split
- Evaluate on same test period as baseline; compare metrics directly

### Step 5 — Model Comparison & Selection (Day 5)
- Tabulate baseline vs LSTM metrics
- Visualize both predictions overlaid on actual prices
- Document findings (which performs better, why, trade-offs)

### Step 6 — Backend API (Day 6)
- Build FastAPI endpoint: `/predict?ticker=AAPL&days=30`
- Load trained model, run inference, return JSON with dates + predicted prices
- Add endpoint for historical data retrieval

### Step 7 — Frontend (Day 7–8)
- Build Streamlit app: ticker selector, date range picker, model toggle
- Interactive Plotly chart showing actual vs predicted
- Metrics panel
- Disclaimer banner

### Step 8 — Testing & Validation (Day 8–9)
- Test with multiple tickers (Indian: RELIANCE.NS, TCS.NS; US: AAPL, MSFT)
- Test edge cases (newly listed stocks with limited history, high-volatility stocks)
- Walk-forward validation for robustness check

### Step 9 — Deployment (Day 9–10)
- Deploy Streamlit app to Streamlit Community Cloud (free, simplest)
- Or containerize with Docker and deploy to Render/Railway
- Set up GitHub repo with clean README, requirements.txt, architecture diagram

### Step 10 — Documentation & Polish (Day 10)
- Write README: problem statement, approach, results, how to run locally
- Add architecture diagram (can reuse the one in Section 6)
- Optionally record a short demo video/GIF
- Add results table (RMSE/MAPE comparison) as proof of technical rigor

### Step 11 (Stretch) — Ensemble & Sentiment (Day 11+)
- Add XGBoost model, blend with LSTM
- Integrate news sentiment (e.g., via NewsAPI + simple sentiment classifier) as an additional feature
- Add confidence intervals using quantile regression or Monte Carlo dropout

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Stock prices are near-random-walk; models may show deceptively good-looking but low-value predictions | Emphasize directional accuracy and honest reporting of limitations in README; avoid overstating claims |
| Data leakage from improper train/test split | Strict chronological split, feature scaling fit only on train data |
| Overfitting LSTM on small datasets | Use dropout, early stopping, and cross-validate with walk-forward windows |
| API rate limits (yfinance/Alpha Vantage) | Cache fetched data locally as CSV, avoid redundant calls |
| Misuse as real financial advice | Prominent disclaimer in UI and README |

---

## 12. Deliverables Checklist

- [ ] GitHub repository with clean commit history
- [ ] `requirements.txt` / environment file
- [ ] Data pipeline scripts
- [ ] Baseline model (Linear Regression) notebook + saved model
- [ ] Advanced model (LSTM) notebook + saved model
- [ ] Evaluation report (metrics table + plots)
- [ ] FastAPI backend
- [ ] Streamlit (or React) frontend
- [ ] README with architecture diagram, setup instructions, results
- [ ] Deployed live demo link
- [ ] (Optional) Demo video/GIF

---

## 13. Next Steps

Pick one starting point and I can generate the actual code for it immediately:
1. Full project scaffold (folder structure + all starter files)
2. Data pipeline script (`yfinance` fetch + feature engineering)
3. Linear Regression baseline (full working notebook/script)
4. LSTM model (full working script)
5. Streamlit frontend (full working app)

Given your track record on the investment intelligence platform, you could likely compress Steps 1–5 into a single working session if I generate the code directly rather than you writing it step by step.
