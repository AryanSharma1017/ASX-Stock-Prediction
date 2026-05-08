# Alpha Engine
## A Multimodal Hybrid Deep Learning Framework for ASX Stock Prediction

Alpha Engine is a research-grade multimodal financial forecasting framework designed to predict directional movement of Australian Securities Exchange (ASX) stocks using deep learning, sentiment analysis, macroeconomic indicators, and temporal sequence modelling.

The project investigates whether integrating heterogeneous financial modalities — including historical market behaviour, technical indicators, Reddit sentiment, financial news sentiment, and macroeconomic signals — can improve predictive performance compared with traditional standalone machine learning approaches.

---

# Research Objectives

The project addresses the following core research questions:

- How does social and financial sentiment affect ASX stock prediction performance?
- Do multimodal architectures outperform traditional machine learning models?
- Can sequential deep learning architectures effectively model temporal financial dependencies?
- To what extent do macroeconomic and geopolitical signals improve market forecasting?

---

# Key Features

- Multimodal financial forecasting framework
- Sequential deep learning architectures
- Reddit financial sentiment analysis using FinBERT
- Financial news sentiment analysis
- Macroeconomic market regime integration
- Time-series safe preprocessing pipeline
- Leakage-safe experimental methodology
- Comparative architecture benchmarking
- Research-grade reproducibility

---

# Dataset Overview

## Primary Stock Dataset

Ticker used throughout experimentation:

- `CBA.AX` — Commonwealth Bank of Australia

Source:

- Yahoo Finance (`yfinance`)

Period:

- 2015–2024

Core OHLCV Features:

- Open
- High
- Low
- Close
- Volume

---

# Technical Feature Engineering

Implemented technical indicators include:

### Momentum Features
- Daily Returns
- Log Returns
- Lag Returns

### Trend Features
- Moving Averages
- MACD
- MACD Signal
- Ichimoku Indicators

### Volatility Features
- Bollinger Bands
- Rolling Standard Deviations
- Rolling Volatility

### Volume Features
- Volume Change
- Volume Moving Average

### Oscillators
- RSI (Relative Strength Index)

---

# Sentiment Analysis Pipeline

## Reddit Sentiment

### Sources
- r/AusFinance
- r/ASX_Bets

### Sentiment Model
- FinBERT (`ProsusAI/finbert`)

### Generated Features
- sentiment_mean
- sentiment_std
- positive_ratio
- negative_ratio
- post_volume

---

## Financial News Sentiment

### News Source
- The Guardian Open Platform API

### Sentiment Models Explored
- VADER
- FinBERT

### Final Observation
FinBERT significantly outperformed VADER for financial forecasting tasks due to domain-specific contextual understanding.

### Generated Features
- news_sentiment_mean
- news_sentiment_std
- news_positive_ratio
- news_negative_ratio
- news_headline_volume

---

# Macroeconomic Indicators

Integrated market-regime variables include:

- VIX (`^VIX`)
- S&P500 (`^GSPC`)
- AUD/USD (`AUDUSD=X`)
- Oil Futures (`CL=F`)
- Gold Futures (`GC=F`)

These features were selected to preserve broader ASX generalizability rather than company-specific dependency.

---

# Prediction Targets

Two binary classification targets were explored:

| Target | Description |
|---|---|
| Target_t1 | Next-day stock movement |
| Target_t7 | 7-day future movement |

### Major Research Finding

Target_t7 consistently demonstrated significantly stronger learnability and generalization performance compared with Target_t1 across nearly all architectures and feature combinations.

---

# Modelling Architectures

## Traditional Machine Learning
- Logistic Regression
- XGBoost

## Sequential Deep Learning
- Improved LSTM
- LSTM + CNN Hybrid
- LSTM + Transformer Hybrid
- Informer-inspired Architecture

---

# Current Best Model

## LSTM + Transformer Hybrid

### Feature Combination
- Technical Indicators
- News FinBERT Sentiment

### Target
- Target_t7

### Performance
- Best Validation AUC ≈ 0.7075
- Test AUC ≈ 0.6449

---

# Key Research Findings

- Sequential deep learning significantly outperformed traditional tabular machine learning.
- Transformer-enhanced architectures produced the strongest overall performance.
- FinBERT consistently outperformed VADER for financial sentiment extraction.
- Medium-horizon forecasting (Target_t7) was substantially more predictable than next-day forecasting.
- Reddit sentiment provided useful but noisy predictive information.
- Financial news sentiment produced stronger and more stable predictive lift.
- Strict temporal leakage prevention was critical for realistic financial forecasting.

---

# Leakage Prevention Strategy

The project follows strict research-grade time-series integrity principles:

- Chronological train/validation/test splitting
- No random shuffling
- Sentiment shifted by +1 day
- Scaling fitted only on training data
- Sequence generation after preprocessing
- No future information leakage

---

# Experimental Methodology

The project uses controlled multimodal ablation studies across multiple feature groups:

- Stock Only
- Stock + Reddit FinBERT
- Stock + News FinBERT
- Stock + Macro
- Stock + Reddit + News
- Stock + Macro + Reddit + News

Evaluation metrics include:

- ROC-AUC
- F1-score
- Accuracy
- Confusion Matrix
- Threshold Analysis

AUC was treated as the primary metric due to calibration instability in financial classification tasks.

---

# Project Structure

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
│   └── ...
│
├── results/
│   ├── plots/
│   ├── metrics/
│   └── ...
│
├── README.md
└── requirements.txt
```

---

# Technologies Used

## Machine Learning & Deep Learning
- PyTorch
- Scikit-learn
- XGBoost

## NLP & Sentiment Analysis
- Hugging Face Transformers
- FinBERT
- VADER

## Data Processing
- Pandas
- NumPy

## Visualization
- Matplotlib
- Seaborn

## Financial Data
- yfinance

---

# Future Work

Planned future research directions include:

- Explainable AI (SHAP/LIME)
- Cross-stock generalization
- Portfolio-level forecasting
- Event-aware transformers
- Attention visualization
- Market regime detection
- Reinforcement learning integration

---

# Research Contribution

Alpha Engine contributes toward ASX-specific multimodal financial forecasting by demonstrating:

- the importance of temporal sequence learning,
- the value of domain-specific financial NLP,
- the superiority of medium-horizon forecasting,
- and the effectiveness of multimodal transformer-enhanced architectures for financial prediction.

---

# Author

Aryan Sharma  
Software Engineering — Artificial Intelligence  
Research Project / Thesis Work
