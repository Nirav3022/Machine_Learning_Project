# TSLA Stock Analysis & Price Prediction (XGBoost)

A Jupyter notebook that explores Tesla (TSLA) historical stock price data and builds a short-term
close-price prediction model using gradient-boosted trees (XGBoost).

🔗 **GitHub Repository:** https://github.com/Nirav3022/Machine_Learning_Project/tree/main/TSLA-Stock-Price

---

## 📌 Project Overview

This project walks through a full mini data-science workflow on TSLA daily OHLCV
(Open, High, Low, Close, Adj Close, Volume) data:

1. **Data loading & cleaning** — reads `TSLA.csv`, standardizes column names, parses dates
2. **Exploratory visualization**
   - Interactive close-price line chart (Plotly)
   - Daily return distribution + rolling 30-day annualized volatility
   - Correlation heatmap across Open/High/Low/Close/Volume/Returns (Seaborn)
3. **Price prediction model**
   - Scales close price with `MinMaxScaler`
   - Builds sliding-window sequences (17-day lookback) as model input
   - Trains an `XGBRegressor` (1,000 estimators)
   - Evaluates with MAE and RMSE

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading & manipulation |
| `matplotlib`, `seaborn` | Static plots (correlation heatmap) |
| `plotly` | Interactive charts (price line, return/volatility) |
| `scikit-learn` | Scaling (`MinMaxScaler`) & metrics (MAE, RMSE) |
| `xgboost` | Gradient-boosted tree regression model |

## 📂 Files

- `TSLA_stock.ipynb` — main analysis & modeling notebook
- `TSLA.csv` — historical daily OHLCV data for TSLA (place alongside the notebook)

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost
jupyter notebook TSLA_stock.ipynb
```

> Note: update the `pd.read_csv(...)` path in the second cell to point to wherever `TSLA.csv`
> lives on your machine — the notebook currently references a local Windows path.

## 📊 What the Model Does

The model does **not** use fundamentals, news, or macro data — it learns purely from a rolling
17-day window of past scaled close prices to predict the next value. It's a technical/statistical
exercise in applying gradient boosting to time-series data, not a fundamentals-driven forecast.

## ⚠️ Limitations & Disclaimer

- The dataset is historical, not a live feed — predictions reflect patterns in the training
  window, not real-time market conditions.
- Tree-based models like XGBoost predict within the range of values seen during training, so they
  can underperform on strongly trending series (e.g., during sharp new highs).
- This project is for **educational and analytical purposes only** and is **not financial or
  investment advice**.

## 🙋 Author Notes

Feel free to fork this project, swap in a fresher CSV, or extend it with additional technical
indicators (moving averages, RSI, MACD, Bollinger Bands) and a rules-based trading strategy
backtest for a fuller technical-analysis workflow.
