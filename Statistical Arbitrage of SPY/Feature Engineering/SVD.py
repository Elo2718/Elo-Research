import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
file_path = 'Mega_Data_LogReturn_MACD.csv'
data = pd.read_csv(file_path)

# Columns to exclude from SVD
excluded_columns = ['Date', 'SPY_Open', 'SPY_Close', 'SPY_High', 'SPY_Low'] + [col for col in data.columns if '_Adj_Close' in col]

# Filter the dataset for SVD by excluding the specified columns
data_filtered = data.drop(columns=excluded_columns)

# Standardize the data (SVD requires normalized input)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_filtered)

# Perform SVD to retain 95% variance
svd = TruncatedSVD(n_components=min(data_scaled.shape[0], data_scaled.shape[1]) - 1, random_state=42)
svd.fit(data_scaled)

# Calculate the number of components to retain 95% variance
cumulative_variance = np.cumsum(svd.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

# Apply SVD with the optimal number of components
svd = TruncatedSVD(n_components=n_components_95, random_state=42)
data_svd = svd.fit_transform(data_scaled)

# Convert the reduced data back to a DataFrame
data_svd_df = pd.DataFrame(data_svd, columns=[f"Component_{i+1}" for i in range(n_components_95)])

# Add back the excluded columns and other important features
additional_columns = ['Date', 'SPY_Open', 'SPY_Close', 'SPY_High', 'SPY_Low', 'SPY_Close_Rolling_Z_Score', 'SPY_Close_LogReturn']
data_svd_df = pd.concat([data_svd_df, data[additional_columns].reset_index(drop=True)], axis=1)

# Reorder columns to have Date, SPY_Close_Rolling_Z_Score, and raw data at the beginning
columns_order = ['Date', 'SPY_Open', 'SPY_Close', 'SPY_High', 'SPY_Low', 'SPY_Close_Rolling_Z_Score', 'SPY_Close_LogReturn'] + [f"Component_{i+1}" for i in range(n_components_95)]
data_svd_df = data_svd_df[columns_order]

# Save the SVD-reduced data to a new CSV file
output_file = "Mega_Data_SVD.csv"
data_svd_df.to_csv(output_file, index=False)

print(f"SVD reduction completed. Retained {n_components_95} components explaining 95% variance.")
print(f"Reduced data saved to {output_file}.")
