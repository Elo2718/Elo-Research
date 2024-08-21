import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('SPY.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date' , inplace= True)
data["Year"] = data.index.year

data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(window = 5).std()
data['Volume_Avg'] = data['Volume'].rolling(window = 5).mean()
data['Momentum'] = data['Return'].rolling(window = 5).mean()
data["High_Low_Dif"] = data["High"] - data['Low']
data["High_Low_Dif"] = data["High_Low_Dif"].rolling(window = 5).mean()
data['MA_5'] = data['Close'].rolling(window = 5).mean()
data['Oscillator'] = (data['Close'] - data['MA_5']) / data['MA_5']


data = data.dropna()

features = data[["Return" , "Volatility", "Volume_Avg", "Momentum", "High_Low_Dif", 'Oscillator']]

scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

gmm = GaussianMixture(n_components = 3, covariance_type = 'full' , random_state = 0)
gmm.fit(features_normalized)

data['Market_Regime'] = gmm.predict(features_normalized)

for year in data['Year'].unique():
    yearly_data = data[data['Year'] == year]
    plt.figure(figsize=(12 , 6))
    plt.scatter(yearly_data.index, yearly_data['Close'], c=yearly_data['Market_Regime'], cmap='viridis')
    plt.title(f"Market Regimes in {year}")
    plt.xlabel('Date')
    plt.ylabel('SPY Close Prices')
    plt.colorbar(label='Market Regime')
    plt.show()

plt.figure(figsize=(12 , 6))
plt.scatter(data.index, data['Close'], c = data['Market_Regime'], cmap = 'viridis')
plt.title("Market Regimes")
plt.xlabel('Date')
plt.ylabel('SPY Close Prices')
plt.colorbar(label='Market Regime')


plt.figure(figsize = (10, 6))
plt.scatter(data['Return'], data['Volatility'], c = data['Market_Regime'], cmap='viridis', s=10)
plt.title("Return vs. Volatility")
plt.xlabel("Return")
plt.ylabel("Volatility")
plt.colorbar(label='Market Regime')
plt.show()

print(data.groupby('Market_Regime').mean())
print(data.groupby('Market_Regime').describe())

data.to_csv("Market_Regime_SPY.csv", columns =['Close', 'Return', 'Volatility', 'Market_Regime'])