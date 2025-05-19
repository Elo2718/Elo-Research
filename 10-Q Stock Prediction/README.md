# Stock Long/Short Prediction via 10-Q Embeddings & TCN

A time-series deep-learning pipeline that reads SEC Form 10-Q reports, encodes them with FinBERT, enriches with technical indicators, and trains a Temporal Convolutional Network (TCN) to predict long vs. short trading signals.

---

## Overview

This project implements a **walk-forward cross-validation** framework to forecast stock direction (+1 = long, 0 = short) based on:
1. **Text embeddings** of quarterly 10-Q filings (via the `yiyanghkust/finbert-pretrain` model).  
2. **Price-based features** (EWMA, volatility, MACD, etc.).  
3. A **Temporal Convolutional Network (TCN)** classifier.

We compare performance to classical baselines (e.g. XGBoost) and show that combining document semantics with time-series architectures improves directional accuracy on large-cap equities.

---

## Repository Structure

```text
.
├── data/                           
│   ├── raw/                        # Input JSON & downloaded 10-Qs + price data
│   └── processed/                  # CSV tickers, cleaned price, embeddings, master CSV
├── src/                           
│   ├── jsonToCSV.py                # 1. Convert SEC JSON tickers → CSV
│   ├── large_cap_stocks.py         # 2. Generate list of large-cap tickers
│   ├── downloadStockData.py        # 3. Download historical prices via yfinance
│   ├── cleanStockPriceData.py      # 4. Clean & align raw price CSVs
│   ├── download10Q.py              # 5. Download 10-Q filings from SEC
│   ├── checkEmptyFolders.py        # 6. Verify no empty ticker folders
│   ├── clean10Q.py                 # 7. Preprocess 10-Q text for tokenization
│   ├── TokenizeCheck.py            # 8. Sanity-check that all .txt can be tokenized
│   ├── embedding.py                # 9. FinBERT tokenization & embedding of 10-Qs
│   ├── TripleBarrier.py            # 10. Label each date long/short via triple-barrier
│   ├── masterDataset.py            # 11. Merge embeddings + labels + features → master CSV
│   ├── gridSearch.py               # 12. Walk-forward grid search for TCN hyper-params
│   ├── train.py                    # 13. Train final TCN with best params
│   └── predict.py                  # 14. Run trained model on newest 10-Q → probabilities
├── models/                         # Saved checkpoints (e.g. tcn_best.pt)
├── reports/                        # LaTeX source & PDF of technical report
├── requirements.txt                # All Python dependencies
└── README.md                       # Project overview & instructions

# 1. Convert SEC JSON tickers → CSV
python src/jsonToCSV.py \
  --input  data/raw/company_tickers.json \
  --output data/processed/tickers.csv

# 2. Generate large-cap ticker list
python src/large_cap_stocks.py \
  --tickers data/processed/tickers.csv \
  --output data/processed/large_caps.txt

# 3. Download historical prices
python src/downloadStockData.py \
  --tickers data/processed/large_caps.txt \
  --output data/raw/prices/

# 4. Clean price CSVs
python src/cleanStockPriceData.py \
  --input  data/raw/prices/ \
  --output data/processed/prices_clean/

# 5. Download 10-Q filings
python src/download10Q.py \
  --tickers data/processed/large_caps.txt \
  --output data/raw/10Q/

# 6. Verify folder integrity
python src/checkEmptyFolders.py \
  --price-dir data/processed/prices_clean/ \
  --sec-dir   data/raw/10Q/

# 7. Clean 10-Q text
python src/clean10Q.py \
  --input  data/raw/10Q/ \
  --output data/processed/10Q_clean/

# 8. Tokenization check
python src/TokenizeCheck.py \
  --input data/processed/10Q_clean/

# 9. Generate embeddings
python src/embedding.py \
  --input  data/processed/10Q_clean/ \
  --output data/processed/embeddings/

# 10. Label via triple-barrier
python src/TripleBarrier.py \
  --prices data/processed/prices_clean/ \
  --output data/processed/labels.csv

# 11. Build master dataset
python src/masterDataset.py \
  --embeddings data/processed/embeddings/ \
  --labels     data/processed/labels.csv \
  --output     data/processed/master_dataset.csv

# 12. Hyperparam grid search (TCN)
python src/gridSearch.py \
  --data   data/processed/master_dataset.csv \
  --output models/tcn_best.pt

# 13. Train final TCN
python src/train.py \
  --data       data/processed/master_dataset.csv \
  --checkpoint models/tcn_best.pt \
  --output     models/tcn_final.pt

# 14. Predict on latest 10-Q
python src/predict.py \
  --model   models/tcn_final.pt \
  --ticker  AAPL \
  --output  predictions/AAPL_prob.csv



