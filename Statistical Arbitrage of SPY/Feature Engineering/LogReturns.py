import pandas as pd
import numpy as np

# Load the dataset
file_path = "Mega_Data_With_Predicted_Volatility.csv"
data = pd.read_csv(file_path)

# Define the assets and price types
assets = ['SPY', 'T_Bill', 'USD_Index', 'VIX', 'Gold', 'Oil']
price_types = ['Close', 'High', 'Low', 'Open']

# Create a dictionary to hold the log returns and MACD columns for each asset and price type
log_return_columns = {}
macd_columns = {}

# Parameters for MACD
short_window = 12  # Fast EMA
long_window = 26   # Slow EMA
signal_window = 9  # Signal line EMA

# Compute log returns and MACD for each asset and price type
for asset in assets:
    for price_type in price_types:
        col_name = f"{asset}_{price_type}"  # Column name in the dataset
        if col_name in data.columns:
            # Compute log returns
            log_return_col = f"{asset}_{price_type}_LogReturn"
            data[log_return_col] = np.log(data[col_name] / data[col_name].shift(1))
            log_return_columns[log_return_col] = data[log_return_col]

            # Compute MACD components
            fast_ema_col = f"{asset}_{price_type}_FastEMA"
            slow_ema_col = f"{asset}_{price_type}_SlowEMA"
            macd_col = f"{asset}_{price_type}_MACD"
            signal_line_col = f"{asset}_{price_type}_SignalLine"
            macd_hist_col = f"{asset}_{price_type}_MACD_Histogram"

            # Calculate fast EMA, slow EMA, MACD line, signal line, and MACD histogram
            data[fast_ema_col] = data[col_name].ewm(span=short_window, adjust=False).mean()
            data[slow_ema_col] = data[col_name].ewm(span=long_window, adjust=False).mean()
            data[macd_col] = data[fast_ema_col] - data[slow_ema_col]
            data[signal_line_col] = data[macd_col].ewm(span=signal_window, adjust=False).mean()
            data[macd_hist_col] = data[macd_col] - data[signal_line_col]

            # Store MACD columns for reference
            macd_columns[macd_col] = data[macd_col]
            macd_columns[macd_hist_col] = data[macd_hist_col]

# Drop rows with NaN (first rows after calculating returns or MACD will be NaN)
data.dropna(inplace=True)

# Save the updated DataFrame to a new CSV file
output_file = "Mega_Data_LogReturn_MACD.csv"
data.to_csv(output_file, index=False)

print(f"Log returns and MACD components added and saved to {output_file}.")

