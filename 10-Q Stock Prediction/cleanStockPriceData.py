import os
import pandas as pd

dir = "StockPrice"

for ticker in os.listdir(dir):
    folder_path = os.path.join(dir, ticker)
    if not os.path.isdir(folder_path):
        continue

    csv_path = os.path.join(folder_path, f"{ticker}_price.csv")
    if not os.path.isfile(csv_path):
        continue

    try:
        # Read with multi-row header (2 rows: category + ticker)
        df = pd.read_csv(csv_path, header=[0, 1])

        # Rename the first column ("Price", "Ticker") -> just "Date"
        df.columns = ['Date'] + [col[0] for col in df.columns[1:]]

        # Parse date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(subset=["Date"], inplace=True)
        df.set_index("Date", inplace=True)

        # Save cleaned data
        cleaned_path = os.path.join(folder_path, f"{ticker}_price_cleaned.csv")
        df.to_csv(cleaned_path)
        print(f"✅ Cleaned data saved to {cleaned_path}")

    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")

