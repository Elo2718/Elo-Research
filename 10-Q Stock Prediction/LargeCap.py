import pandas as pd
import yfinance as yf

# Load tickers and drop rows with missing tickers
df = pd.read_csv("company_tickers.csv").dropna(subset=["ticker"])
tickers = df["ticker"].tolist()

large_caps = []
non_large_caps = []
error_tickers = []

# Limit to the first 1000
for i, ticker in enumerate(tickers[:1000]):
    try:
        info = yf.Ticker(ticker).info
        market_cap = info.get("marketCap", 0)

        if market_cap and market_cap >= 10e9:
            large_caps.append({
                "ticker": ticker,
                "marketCap": market_cap,
                "company": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", "")
            })
            print(f"{ticker} Large Cap: ${market_cap:,}")
        else:
            non_large_caps.append(ticker)
            print(f"{ticker} Not Large Cap")
    except Exception as e:
        error_tickers.append(ticker)
        print(f"{ticker} Error: {e}")

# Save large caps to CSV
pd.DataFrame(large_caps).to_csv("large_cap_stocks.csv", index=False)

# Final summary
print(f"\nDone. Checked 1000 tickers.")
print(f"Large Caps Found: {len(large_caps)}")
print(f"⚠Errors Encountered: {len(error_tickers)}")

# Optional: print large caps
print("\n📈 Large Cap Tickers:")
print([stock["ticker"] for stock in large_caps])
