import os
import pandas as pd
import yfinance as yf

# Load large cap tickers
large_cap_df = pd.read_csv("large_cap_stocks.csv")
tickers = large_cap_df["ticker"].dropna().unique()

# Base path for storing stock price data
base_path = "StockPrice"
os.makedirs(base_path, exist_ok=True)

for ticker in tickers:
    ticker = ticker.strip()
    if not ticker:
        continue

    # Define paths
    save_path = os.path.join(base_path, ticker)
    file_path = os.path.join(save_path, f"{ticker}_price.csv")

    # Skip if file already exists
    if os.path.isfile(file_path):
        print(f"✅ Skipping {ticker}, CSV already exists.")
        continue

    try:
        # Download price data
        data = yf.download(ticker, period='max', interval='1d', progress=False)
        if data.empty:
            print(f"⚠️ No stock data for {ticker}. Skipping.")
            continue

        # Append metadata
        info = yf.Ticker(ticker).info
        data["Company"] = info.get('longName', ticker)
        data["Sector"] = info.get('sector', 'Unknown')
        data["Industry"] = info.get('industry', 'Unknown')

        # Save to CSV
        os.makedirs(save_path, exist_ok=True)
        data.to_csv(file_path)
        print(f"📈 Downloaded stock data for {ticker}")

    except Exception as e:
        print(f"❌ Failed for {ticker}: {e}")
