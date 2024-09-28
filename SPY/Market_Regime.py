import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Load mega dataframe
data = pd.read_csv('mega_dataframe.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data["Year"] = data.index.year

# Define the features for market regime clustering
# Assuming that we have the necessary features in mega_dataframe.csv
features = data[["SPY_Percent_Close_Change", "VIX_Close", "SPY_Volume_Z_5" , "SPY_Oscillator_5", "SPY_Volume_Z_20", "SPY_Oscillator_20"]]

# Standardize the features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Apply Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
gmm.fit(features_normalized)

# Add the market regime labels to the dataframe
data['Market_Regime'] = gmm.predict(features_normalized)


for year in data['Year'].unique():
    yearly_data = data[data['Year'] == year]
    plt.figure(figsize=(12, 6))
    plt.scatter(yearly_data.index, yearly_data['SPY_Close'], c=yearly_data['Market_Regime'], cmap='viridis', s=10)
    plt.title(f"Market Regimes in {year}")
    plt.xlabel('Date')
    plt.ylabel('SPY Close Prices')
    plt.colorbar(label='Market Regime')
    plt.show()

# Plot market regimes over the entire dataset
plt.figure(figsize=(12, 6))
plt.scatter(data.index, data['SPY_Close'], c=data['Market_Regime'], cmap='viridis', s=10)
plt.title("Market Regimes Over Time")
plt.xlabel('Date')
plt.ylabel('SPY Close Prices')
plt.colorbar(label='Market Regime')
plt.show()

# Scatter plot of SPY_Return vs SPY_Volume_Z_5 colored by market regime
plt.figure(figsize=(10, 6))
plt.scatter(data['VIX_Close'], data["SPY_Percent_Close_Change"], c=data['Market_Regime'], cmap='viridis', s=10)
plt.title("Return vs. 5-Period Z-Score of SPY Volume")
plt.xlabel("Return (%)")
plt.ylabel("5-Period Z-Score of SPY Volume")
plt.colorbar(label='Market Regime')
plt.show()

# Print mean statistics of each market regime
print(data.groupby('Market_Regime').mean())
print(data.groupby('Market_Regime').describe())
