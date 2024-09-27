import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('mega_dataframe.csv')

# Drop any rows with missing values
df.dropna(inplace=True)

# Scatter plot of SPY_Percent_Close_Change vs VIX_Close
plt.figure(figsize=(10, 6))
plt.scatter(df['SPY_Percent_Close_Change'], df['VIX_Close'], color='blue', alpha=0.5)
plt.xlabel('SPY Percent Close Change')
plt.ylabel('VIX Close')
plt.title('SPY Percent Close Change vs VIX Close')
plt.grid(True)
plt.show()

# Scatter plot of SPY_Percent_Close_Change vs SPY_Volume_Z_5 (5-period Z-Score of SPY volume)
plt.figure(figsize=(10, 6))
plt.scatter(df['SPY_Percent_Close_Change'], df['SPY_Volume_Z_5'], color='green', alpha=0.5)
plt.xlabel('SPY Percent Close Change')
plt.ylabel('5-Period Z-Score of SPY Volume')
plt.title('SPY Percent Close Change vs 5-Period Z-Score of SPY Volume')
plt.grid(True)
plt.show()

# Scatter plot of SPY_Percent_Close_Change vs SPY_Volume_Z_20 (20-period Z-Score of SPY volume)
plt.figure(figsize=(10, 6))
plt.scatter(df['SPY_Percent_Close_Change'], df['SPY_Volume_Z_20'], color='red', alpha=0.5)
plt.xlabel('SPY Percent Close Change')
plt.ylabel('20-Period Z-Score of SPY Volume')
plt.title('SPY Percent Close Change vs 20-Period Z-Score of SPY Volume')
plt.grid(True)
plt.show()

# Scatter plot of SPY_Volume_Z_5 vs SPY_Volume_Z_20 to explore volume relationship
plt.figure(figsize=(10, 6))
plt.scatter(df['SPY_Volume_Z_5'], df['SPY_Volume_Z_20'], color='purple', alpha=0.5)
plt.xlabel('5-Period Z-Score of SPY Volume')
plt.ylabel('20-Period Z-Score of SPY Volume')
plt.title('5-Period vs 20-Period Z-Score of SPY Volume')
plt.grid(True)
plt.show()

# Scatter plot of SPY_Oscillator_5 vs SPY_Percent_Close_Change
plt.figure(figsize=(10, 6))
plt.scatter(df['SPY_Oscillator_5'], df['SPY_Percent_Close_Change'], color='blue', alpha=0.5)
plt.xlabel('SPY 5-Day Oscillator (%)')
plt.ylabel('SPY Percent Close Change (%)')
plt.title('SPY 5-Day Oscillator vs SPY Percent Close Change')
plt.grid(True)
plt.show()

# Scatter plot of SPY_Oscillator_20 vs SPY_Percent_Close_Change
plt.figure(figsize=(10, 6))
plt.scatter(df['SPY_Oscillator_20'], df['SPY_Percent_Close_Change'], color='green', alpha=0.5)
plt.xlabel('SPY 20-Day Oscillator (%)')
plt.ylabel('SPY Percent Close Change (%)')
plt.title('SPY 20-Day Oscillator vs SPY Percent Close Change')
plt.grid(True)
plt.show()

# Scatter plot of SPY_Oscillator_5 vs SPY_Volume_Z_5 (5-period Z-Score of SPY Volume)
plt.figure(figsize=(10, 6))
plt.scatter(df['SPY_Oscillator_5'], df['SPY_Volume_Z_5'], color='blue', alpha=0.5)
plt.xlabel('SPY 5-Day Oscillator (%)')
plt.ylabel('5-Period Z-Score of SPY Volume')
plt.title('SPY 5-Day Oscillator vs 5-Period Z-Score of SPY Volume')
plt.grid(True)
plt.show()

# Scatter plot of SPY_Oscillator_20 vs SPY_Volume_Z_20 (20-period Z-Score of SPY Volume)
plt.figure(figsize=(10, 6))
plt.scatter(df['SPY_Oscillator_20'], df['SPY_Volume_Z_20'], color='green', alpha=0.5)
plt.xlabel('SPY 20-Day Oscillator (%)')
plt.ylabel('20-Period Z-Score of SPY Volume')
plt.title('SPY 20-Day Oscillator vs 20-Period Z-Score of SPY Volume')
plt.grid(True)
plt.show()