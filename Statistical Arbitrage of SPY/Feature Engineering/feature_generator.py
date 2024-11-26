import pandas as pd

# Load the data
df = pd.read_csv('Mega_Data.csv')

# List of assets
assets = ['SPY', 'T_Bill', 'USD_Index', 'VIX', 'Gold', 'Oil']

# List of asset data types (columns for each asset)
data_types = ['Open', 'High', 'Low', 'Close', 'Volume']  # Adjust if volume is not present for all

# Loop through each asset and its data type
for asset in assets:
    for data_type in data_types:
        column_name = f'{asset}_{data_type}'
        if column_name in df.columns:  # Ensure the column exists in the DataFrame
            # Calculate the 20-period rolling mean
            df[f'{column_name}_20_Period_Mean'] = df[column_name].rolling(window=20).mean()
            
            # Calculate the 20-period rolling standard deviation
            df[f'{column_name}_20_Period_Std'] = df[column_name].rolling(window=20).std()
            
            # Calculate the rolling Z-score
            df[f'{column_name}_Rolling_Z_Score'] = (
                (df[column_name] - df[f'{column_name}_20_Period_Mean']) / df[f'{column_name}_20_Period_Std']
            )

# Save the updated DataFrame
df.to_csv('Mega_Data_Updated.csv', index=False)
print("Data has been successfully updated with rolling statistics and saved.")
