import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('SPY.csv')
data['Date'] = pd.to_datetime(data['Date'])
data_VIX = pd.read_csv("^VIX.csv")
data_VIX['Date'] = pd.to_datetime(data_VIX['Date'])
print(data_VIX)
data = pd.merge(data,data_VIX, on = "Date", how = 'inner')
print(data)

data['SPY_5_MA_Close'] = data['SPY_Close'].rolling(window=5).mean()
data['SPY_20_MA_Close'] = data['SPY_Close'].rolling(window=20).mean()


plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['SPY_Close'], label='SPY Close', color='blue')
plt.plot(data['Date'], data['SPY_5_MA_Close'], label='5-Day MA', color='orange', linestyle='--')
plt.plot(data['Date'], data['SPY_20_MA_Close'], label='20-Day MA', color='green', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SPY Close Prices with 5-Day and 20-Day Moving Averages')
plt.legend()
plt.show()

data['SPY_Volume_5_MA'] = data['SPY_Volume'].rolling(window = 5).mean()
data['SPY_Volume_5_SD'] = data['SPY_Volume'].rolling(window = 5).std()
data['SPY_Volume_Z_5'] = (data['SPY_Volume'] - data['SPY_Volume_5_MA']) / data["SPY_Volume_5_SD"]

data['SPY_Volume_20_MA'] = data['SPY_Volume'].rolling(window = 20).mean()
data['SPY_Volume_20_SD'] = data['SPY_Volume'].rolling(window = 20).std()
data['SPY_Volume_Z_20'] = (data['SPY_Volume'] - data["SPY_Volume_20_MA"]) / data["SPY_Volume_20_SD"]

data['SPY_Oscillator_5'] = (data['SPY_Close'] - data['SPY_5_MA_Close']) / data['SPY_5_MA_Close'] * 100
data['SPY_Oscillator_20'] = (data['SPY_Close'] - data['SPY_20_MA_Close']) / data['SPY_20_MA_Close'] * 100

plt.figure(figsize=(12, 8))

plt.plot(data['Date'], data['SPY_Oscillator_5'], label='SPY 5-Day Oscillator', color='blue', linestyle='--')
plt.plot(data['Date'], data['SPY_Oscillator_20'], label='SPY 20-Day Oscillator', color='green', linestyle='--')
plt.plot(data['Date'], data['SPY_Volume_Z_5'], label='5-Period Z-Score of SPY Volume', color='red', alpha=0.6)
plt.plot(data['Date'], data['SPY_Volume_Z_20'], label='20-Period Z-Score of SPY Volume', color='orange', alpha=0.6)

plt.xlabel('Date')
plt.ylabel('Oscillator (%) / Volume Z-Score')
plt.title('SPY Oscillators and Volume Z-Scores')
plt.legend()
plt.grid(True)


data['SPY_Percent_Close_Change'] = (data['SPY_Close'] - data['SPY_Close'].shift(1)) / data['SPY_Close'] * 100

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['SPY_Percent_Close_Change'], label="Percent Change in SPY Close", color='red')
plt.title('Percent Change in SPY Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Percent Change (%)')
plt.legend()
plt.grid(True)
plt.show()

data['SPY_Momentum_5'] = (data['SPY_Close'] - data['SPY_Close'].shift(5)) / data['SPY_Close'].shift(5) * 100
data['SPY_Momentum_20'] = (data['SPY_Close'] - data['SPY_Close'].shift(20)) / data['SPY_Close'].shift(20) * 100

data.dropna(inplace=True)

# Plot the SPY Momentum (5-period and 20-period percent change)
plt.figure(figsize=(10, 6))

plt.plot(data['Date'], data['SPY_Momentum_5'], label='SPY 5-Period Momentum (%)', color='blue')
plt.plot(data['Date'], data['SPY_Momentum_20'], label='SPY 20-Period Momentum (%)', color='green')

plt.xlabel('Date')
plt.ylabel('Momentum (%)')
plt.title('SPY 5-Period and 20-Period Momentum')
plt.legend()
plt.grid(True)
plt.show()

data['SPY_Percent_Change:High_To_Previous_Close'] = (data['SPY_High'] - data['SPY_Close'].shift(1)) / data['SPY_Close'] * 100

plt.figure(figsize=(10 , 6))
plt.plot(data['Date'], data['SPY_Percent_Change:High_To_Previous_Close'], label = "SPY_Percent High Change", color = 'red')
plt.title('Percent Change of High to Previous Close Over Time')
plt.xlabel("Date")
plt.ylabel('Percent Change')
plt.legend()

data['SPY_Percent_Change:Low_To_Previous_Close'] = (data['SPY_Low'] - data['SPY_Close'].shift(1)) / data['SPY_Close'] * 100

plt.figure(figsize=(10 , 6))
plt.plot(data['Date'], data['SPY_Percent_Change:Low_To_Previous_Close'], label = "Percent Low Change", color = 'red')
plt.title('Percent Change of Low to Previous Close Over Time')
plt.xlabel('Date')
plt.ylabel('Percent Change')
plt.legend()

print(data)
plt.show()

data['SPY_Percent_Close_Change_Label'] = data['SPY_Percent_Close_Change'].shift(-1)
data['SPY_Percent_Change:High_To_Previous_Close_Label'] = data['SPY_Percent_Change:High_To_Previous_Close'].shift(-1)
data['SPY_Percent_Change:Low_To_Previous_Close_Label'] = data['SPY_Percent_Change:Low_To_Previous_Close'].shift(-1)
data = data.replace('',np.nan)
data = data.fillna(np.nan)


data.to_csv('mega_dataframe.csv', index=False)