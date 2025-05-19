# Stock Long/Short Prediction via 10-Q Embeddings & TCN

A time-series deep-learning pipeline that reads SEC Form 10-Q reports, encodes them with FinBERT, enriches with technical indicators, and trains a Temporal Convolutional Network (TCN) to predict long vs. short trading signals.

---

## 🚀 Overview

This project implements a **walk-forward cross-validation** framework to forecast stock direction (+1 = long, 0 = short) based on:
1. **Text embeddings** of quarterly 10-Q filings (via the `yiyanghkust/finbert-pretrain` model).
3. A **Temporal Convolutional Network (TCN)** classifier.

We compare performance to classical baselines (e.g. XGBoost) and show that combining document semantics with time-series architectures improves directional accuracy on large-cap equities.

---

## 📁 Repository Structure
