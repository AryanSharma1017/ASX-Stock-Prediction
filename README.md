<div align="center">

# 🚀 Alpha Engine
### *A Multimodal Hybrid Deep Learning Framework for ASX Stock Prediction*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)]()
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)]()
[![Finance](https://img.shields.io/badge/Domain-Financial%20AI-green.svg)]()
[![Research](https://img.shields.io/badge/Research-HD%20Thesis-purple.svg)]()
[![Status](https://img.shields.io/badge/Project-Active-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

---

### 🧠 Multimodal Deep Learning for ASX Market Forecasting

*Technical Indicators + Financial NLP + Macroeconomic Signals + Temporal Deep Learning*

</div>

---

# 📌 Overview

Alpha Engine is a **research-grade multimodal financial forecasting framework** designed to predict directional movement of Australian Securities Exchange (ASX) stocks using deep learning and heterogeneous financial modalities.

The framework integrates:

- 📈 Historical stock market data
- 📊 Technical indicators
- 💬 Reddit financial sentiment
- 📰 Financial news sentiment
- 🌍 Macroeconomic market-regime indicators
- 🧠 Sequential deep learning architectures

The project investigates whether multimodal temporal fusion architectures can outperform traditional standalone machine learning approaches for ASX stock forecasting.

---

# 🎯 Research Objectives

<table>
<tr>
<th width="10%">RQ</th>
<th>Research Question</th>
</tr>

<tr>
<td><b>RQ1</b></td>
<td>How does integrating sentiment data affect ASX stock prediction performance?</td>
</tr>

<tr>
<td><b>RQ2</b></td>
<td>Do macroeconomic and geopolitical variables improve forecasting capability?</td>
</tr>

<tr>
<td><b>RQ3</b></td>
<td>Do multimodal sequential architectures outperform traditional ML models?</td>
</tr>

<tr>
<td><b>RQ4</b></td>
<td>How effectively can deep sequential models learn temporal financial dependencies?</td>
</tr>

</table>

---

# 🏗️ System Architecture

```text
                    ┌────────────────────┐
                    │  Yahoo Finance API │
                    └─────────┬──────────┘
                              │
                    Historical OHLCV Data
                              │
                              ▼
                    ┌────────────────────┐
                    │ Technical Features │
                    └─────────┬──────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼

┌────────────────┐  ┌────────────────┐  ┌──────────────────┐
│ Reddit Dataset │  │ News Headlines │  │ Macro Indicators │
└───────┬────────┘  └────────┬───────┘  └────────┬─────────┘
        │                    │                   │
        ▼                    ▼                   ▼

┌────────────────┐  ┌────────────────┐  ┌──────────────────┐
│ FinBERT NLP    │  │ FinBERT /      │  │ Market Regime    │
│ Sentiment      │  │ VADER Pipeline │  │ Features         │
└───────┬────────┘  └────────┬───────┘  └────────┬─────────┘
        │                    │                   │
        └────────────┬───────┴────────────┬──────┘
                     ▼                    ▼

              ┌─────────────────────────────┐
              │ Multimodal Feature Fusion   │
              └─────────────┬──────────────┘
                            ▼

          ┌──────────────────────────────────┐
          │ Sequential Deep Learning Models  │
          │ LSTM / Transformer / Hybrid      │
          └──────────────────────────────────┘
```

---

# 📂 Dataset Overview

## 📈 Financial Market Dataset

| Attribute | Details |
|---|---|
| Primary Ticker | `CBA.AX` |
| Market | Australian Securities Exchange |
| Source | Yahoo Finance (`yfinance`) |
| Period | 2015–2024 |
| Frequency | Daily |
| Problem Type | Binary Classification |

---

# 🧮 Engineered Financial Features

<details>
<summary><b>📊 Technical Indicators & Feature Engineering</b></summary>

---

## Momentum Features
- Daily Returns
- Log Returns
- Lag Returns (`t-1`, `t-2`, `t-3`, `t-5`)

## Trend Features
- Rolling Moving Averages
- MACD
- MACD Signal
- Ichimoku Indicators

## Volatility Features
- Bollinger Bands
- Rolling Volatility
- Rolling Standard Deviation

## Volume Features
- Volume Change
- Volume Moving Average

## Oscillators
- RSI (Relative Strength Index)

---

</details>

---

# 💬 Sentiment Analysis Pipelines

## 🧠 Reddit Financial Sentiment

| Component | Details |
|---|---|
| Source | `r/AusFinance`, `r/ASX_Bets` |
| NLP Model | `FinBERT` |
| Purpose | Retail investor sentiment modelling |
| Aggregation | Daily grouped aggregation |

### Generated Features

```python
[
    "sentiment_mean",
    "sentiment_std",
    "positive_ratio",
    "negative_ratio",
    "post_volume"
]
```

---

## 📰 Financial News Sentiment

| Component | Details |
|---|---|
| Source | The Guardian Open Platform API |
| Initial Model | VADER |
| Final Model | FinBERT |
| Observation | FinBERT > VADER for financial forecasting |

### Final News Features

```python
[
    "news_sentiment_mean",
    "news_sentiment_std",
    "news_positive_ratio",
    "news_negative_ratio",
    "news_headline_volume"
]
```

---

# 🌍 Macroeconomic Features

<details>
<summary><b>📉 Market Regime Indicators</b></summary>

| Variable | Description |
|---|---|
| `^VIX` | Global volatility / fear index |
| `^GSPC` | S&P500 spillover effects |
| `AUDUSD=X` | Currency sentiment |
| `CL=F` | Oil futures |
| `GC=F` | Gold futures |

### Engineered Macro Features

```python
[
    "vix_return",
    "sp500_return",
    "audusd_return",
    "oil_return",
    "gold_return"
]
```

</details>

---

# 🎯 Prediction Targets

| Target | Description |
|---|---|
| `Target_t1` | Next-day movement |
| `Target_t7` | 7-day future movement |

---

# 🔥 Major Research Discovery

> ✅ **Target_t7 consistently outperformed Target_t1 across nearly all architectures and feature groups.**

This suggests:
- medium-horizon forecasting contains stronger learnable temporal structure,
- while next-day prediction behaves closer to Efficient Market Hypothesis assumptions.

---

# 🤖 Implemented Models

## 📌 Traditional Machine Learning

| Model | Purpose |
|---|---|
| Logistic Regression | Linear interpretable baseline |
| XGBoost | Nonlinear ensemble baseline |

### Observation

```text
AUC ≈ 0.50–0.53
```

Traditional tabular methods struggled to capture temporal dependencies and multimodal interactions.

---

# 🧠 Sequential Deep Learning Models

<table>
<tr>
<th>Architecture</th>
<th>Description</th>
<th>Performance</th>
</tr>

<tr>
<td><b>Improved LSTM</b></td>
<td>Multi-layer temporal sequence model</td>
<td>Strong</td>
</tr>

<tr>
<td><b>LSTM + CNN</b></td>
<td>Local temporal motif extraction</td>
<td>Moderate</td>
</tr>

<tr>
<td><b>LSTM + Transformer</b></td>
<td>Sequential memory + temporal attention</td>
<td>Best Overall</td>
</tr>

<tr>
<td><b>LSTM + Informer</b></td>
<td>Efficient long-sequence attention</td>
<td>Weak Generalization</td>
</tr>

</table>

---

# 🏆 Current Best Model

<div align="center">

## 🥇 LSTM + Transformer Hybrid

</div>

| Configuration | Value |
|---|---|
| Target | `Target_t7` |
| Features | Stock + News FinBERT |
| Sequence Length | 60 |
| Hidden Size | 96 |
| Optimizer | AdamW |
| Epochs | 150 |

---

## 📊 Best Results

| Metric | Score |
|---|---|
| Validation AUC | **0.7075** |
| Test AUC | **0.6449** |

---

# 📉 Evaluation Metrics

```python
evaluation_metrics = [
    "ROC-AUC",
    "F1-Score",
    "Accuracy",
    "Confusion Matrix",
    "Threshold Analysis",
    "Probability Distribution Analysis"
]
```

---

# ⚠️ Important Methodological Finding

> Models demonstrated stronger ranking capability than calibrated binary directional classification.

Therefore:
- ROC-AUC became the primary evaluation metric,
- instead of raw accuracy alone.

---

# 🛡️ Leakage Prevention Strategy

<details open>
<summary><b>🔒 Research-Grade Temporal Integrity</b></summary>

### Strict Time-Series Principles

✅ Chronological splitting only  
✅ No random shuffling  
✅ Sentiment shifted by +1 day  
✅ Scaling fitted only on training data  
✅ Sequence generation after preprocessing  
✅ No future information leakage  

---

### Sentiment Alignment Rule

```python
sentiment(t-1) ---> predicts ---> stock movement(t)
```

This preserves realistic financial forecasting conditions.

</details>

---

# 🧪 Experimental Ablation Studies

## Evaluated Feature Groups

```python
[
    "Stock Only",
    "Stock + Reddit FinBERT",
    "Stock + News FinBERT",
    "Stock + Macro",
    "Stock + Reddit + News",
    "Stock + Macro + News",
    "Stock + Macro + Reddit + News"
]
```

---

# 📁 Project Structure

```bash
AlphaEngine/
│
├── data/
│   ├── CBA_target_ready_2015_present.csv
│   ├── reddit_daily_sentiment.csv
│   ├── news_daily_finbert_sentiment.csv
│   ├── macro_daily_features.csv
│   └── ...
│
├── notebooks/
│   ├── alphaengine.ipynb
│   └── ...
│
├── models/
│   ├── improved_lstm.py
│   ├── lstm_transformer.py
│   ├── lstm_cnn.py
│   └── informer_model.py
│
├── results/
│   ├── plots/
│   ├── metrics/
│   └── experiments/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# ⚙️ Tech Stack

<div align="center">

| Category | Technologies |
|---|---|
| Deep Learning | PyTorch |
| NLP | HuggingFace Transformers |
| Financial NLP | FinBERT |
| ML | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Financial Data | yfinance |

</div>

---

# 📌 Core Research Contributions

✅ ASX-specific multimodal forecasting framework  
✅ Temporal multimodal fusion methodology  
✅ Medium-horizon forecasting superiority discovery  
✅ FinBERT > VADER financial NLP finding  
✅ Sequential models > tabular models finding  
✅ Transformer-enhanced temporal forecasting improvements  
✅ Research-grade leakage-safe methodology  

---

# 🔮 Future Work

- Explainable AI (SHAP/LIME)
- Cross-stock generalization
- Portfolio optimization
- Event-aware transformers
- Attention visualization
- Regime-switching architectures
- Reinforcement learning integration

---

# 👨‍💻 Author

<div align="center">

## Aryan Sharma

Software Engineering — Artificial Intelligence  
Research Thesis Project  
Melbourne, Australia 🇦🇺

</div>

---

<div align="center">

### ⭐ If you found this project interesting, consider starring the repository.

</div>