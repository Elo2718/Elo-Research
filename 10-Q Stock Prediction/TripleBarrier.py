import os
import pandas as pd
import numpy as np

# ---- CONFIG ----
base_dir = "StockPrice"
horizon = 5                # short-term reaction window
threshold = 0.03           # 3% movement defines reaction
output_suffix = "_price_labeled.csv"

# ---- PROCESS ALL TICKERS ----
tickers = [t for t in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, t))]

for ticker in tickers:
    try:
        input_path = os.path.join(base_dir, ticker, f"{ticker}_price_cleaned.csv")
        output_path = os.path.join(base_dir, ticker, f"{ticker}{output_suffix}")

        # ---- LOAD ----
        df = pd.read_csv(input_path)
        df = df.dropna(subset=["Date", "Close"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        # ---- Initialize Labels ----
        df["trade"] = np.nan

        for i in range(len(df) - horizon):
            current_price = df["Close"].iloc[i]
            future = df["Close"].iloc[i + 1 : i + 1 + horizon]

            if np.isnan(current_price) or future.isna().any():
                continue

            max_return = (future.max() - current_price) / current_price
            min_return = (future.min() - current_price) / current_price

            if max_return >= threshold:
                df.iloc[i, df.columns.get_loc("trade")] = 1  # LONG
            elif min_return <= -threshold:
                df.iloc[i, df.columns.get_loc("trade")] = 0  # SHORT
            else:
                final = future.iloc[-1]
                df.iloc[i, df.columns.get_loc("trade")] = 1 if final > current_price else 0

        # ---- Clean and Save ----
        df = df.dropna(subset=["trade"])
        df["trade"] = df["trade"].astype(int)

        df.reset_index().to_csv(output_path, index=False)
        print(f"✅ Labeled and saved: {output_path}")

    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")

print("🎯 5-day reaction labeling complete.")








