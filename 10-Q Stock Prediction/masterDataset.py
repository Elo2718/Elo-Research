import os
import pandas as pd
import numpy as np
import bisect

# ==== CONFIG ====
PRICE_DIR = "StockPrice"
EMBED_DIR = "10-Q"
OUTPUT_PATH = "master_dataset.csv"
TEST_YEAR = 2025

# ---- Helper: Build Sorted Trading Calendar + Lookup ----
def build_trading_index(dates):
    td_list = sorted(pd.to_datetime(dates).unique())
    def next_trade(dt):
        pos = bisect.bisect_left(td_list, dt)
        return td_list[pos] if pos < len(td_list) else pd.NaT
    return td_list, next_trade

master_records = []

# ---- Loop Over Tickers ----
for ticker in sorted(os.listdir(EMBED_DIR)):
    emb_folder = os.path.join(EMBED_DIR, ticker, "Embedding")
    price_path = os.path.join(PRICE_DIR, ticker, f"{ticker}_price_labeled.csv")

    if not os.path.isdir(emb_folder) or not os.path.isfile(price_path):
        continue

    try:
        # Load labeled price data
        price_df = pd.read_csv(price_path, parse_dates=["Date"])
        price_df.rename(columns={"Date": "trade_date", "trade": "trade_target"}, inplace=True)
        price_df = price_df[price_df["trade_target"].isin([0, 1])]

        if len(price_df) < 10:
            print(f"⏭️ Skipping {ticker} due to insufficient labeled rows")
            continue

        _, next_trade = build_trading_index(price_df["trade_date"])
        emb_files = [f for f in os.listdir(emb_folder) if f.endswith(".csv")]
        valid_count = 0

        for fn in emb_files:
            try:
                filing_dt = pd.to_datetime(fn.replace(f"{ticker}-", "").replace(".csv", ""))
                trade_dt = next_trade(filing_dt)
                if pd.isna(trade_dt):
                    continue

                price_row = price_df[price_df["trade_date"] == trade_dt]
                if price_row.empty:
                    continue

                label = price_row["trade_target"].values[0]
                if pd.isna(label):
                    continue

                emb_df = pd.read_csv(os.path.join(emb_folder, fn), header=0)
                if emb_df.empty or emb_df.isna().all(axis=None):
                    continue

                emb_vals = emb_df.iloc[0].to_numpy()

                rec = {
                    "ticker": ticker,
                    "filing_date": filing_dt,
                    "trade_date": trade_dt,
                    "trade_target": int(label)
                }
                rec.update({f"emb_{i}": v for i, v in enumerate(emb_vals)})
                master_records.append(rec)
                valid_count += 1

            except Exception as e:
                print(f"⚠Error processing {ticker} {fn}: {e}")

        print(f"Finished {ticker}: {valid_count} matched 10-Q embeddings")

    except Exception as e:
        print(f"Failed loading price data for {ticker}: {e}")

# ---- Build and Save Master Dataset ----
master = pd.DataFrame(master_records)
master = master.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
master.to_csv(OUTPUT_PATH, index=False)

print(f"\nMaster dataset built and saved with {len(master)} rows to {OUTPUT_PATH}")


