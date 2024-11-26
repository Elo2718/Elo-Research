import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# Load your data
df = pd.read_csv('Mega_Data.csv')  # Update the file path if necessary

df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in datetime format

# Define assets and their types
assets = ['SPY', 'T_Bill', 'USD_Index', 'VIX', 'Gold', 'Oil']
types = ['Close', 'High', 'Low', 'Open']

# Calculate the 20-period rolling Z-score for each asset and type
for asset in assets:
    for asset_type in types:
        column_name = f"{asset}_{asset_type}"
        mean_col = f"{asset}_{asset_type}_20_Period_Mean"
        std_col = f"{asset}_{asset_type}_20_Period_Std"
        z_score_col = f"{asset}_{asset_type}_Rolling_Z_Score"

        # Calculate rolling mean and standard deviation
        df[mean_col] = df[column_name].rolling(window=20).mean()
        df[std_col] = df[column_name].rolling(window=20).std()

        # Calculate the rolling Z-score
        df[z_score_col] = (df[column_name] - df[mean_col]) / df[std_col]

# Include SPY Volume Z-score
volume_mean_col = 'SPY_Volume_20_Period_Mean'
volume_std_col = 'SPY_Volume_20_Period_Std'
volume_z_score_col = 'SPY_Volume_Rolling_Z_Score'

# Calculate rolling mean and standard deviation for SPY Volume
df[volume_mean_col] = df['SPY_Volume'].rolling(window=20).mean()
df[volume_std_col] = df['SPY_Volume'].rolling(window=20).std()

# Calculate the rolling Z-score for SPY Volume
df[volume_z_score_col] = (df['SPY_Volume'] - df[volume_mean_col]) / df[volume_std_col]

import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Define the ARIMA parameters for each feature
arima_parameters = {
    'SPY_Close': (5, 2, 0),
    'T_Bill_Close': (2, 1, 1),
    'USD_Index_Close': (0, 1, 0),
    'VIX_Close': (2, 1, 3),
    'Gold_Close': (3, 1, 3),
    'Oil_Close': (2, 1, 4),
    'SPY_High': (5, 2, 0),
    'T_Bill_High': (2, 1, 1),
    'USD_Index_High': (0, 1, 1),
    'VIX_High': (1, 1, 2),
    'Gold_High': (1, 1, 2),
    'Oil_High': (0, 1, 1),
    'SPY_Low': (5, 2, 0),
    'T_Bill_Low': (1, 1, 4),
    'USD_Index_Low': (3, 1, 2),
    'VIX_Low': (0, 1, 3),
    'Gold_Low': (1, 1, 2),
    'Oil_Low': (5, 1, 2),
    'SPY_Open': (5, 2, 0),
    'T_Bill_Open': (2, 1, 1),
    'USD_Index_Open': (0, 1, 1),
    'VIX_Open': (4, 1, 2),
    'Gold_Open': (2, 1, 1),
    'Oil_Open': (3, 1, 3),
    'SPY_Volume': (2, 1, 2)
}

# Iterate over each feature and apply ARIMA
for feature, (p, d, q) in arima_parameters.items():
    try:
        # Fit the ARIMA model
        model = ARIMA(df[feature], order=(p, d, q))
        model_fit = model.fit()

        # Make predictions
        predictions = model_fit.fittedvalues
        residuals = model_fit.resid

        # Add predictions and residuals to the original DataFrame
        df[f'{feature}_ARIMA_Predictions_{p}_{d}_{q}'] = predictions
        df[f'{feature}_ARIMA_Residuals_{p}_{d}_{q}'] = residuals

    except Exception as e:
        print(f"ARIMA model failed for {feature} with error: {e}")

# Calculate daily returns and rescale by multiplying by 100
df['Returns'] = df['SPY_Close'].pct_change().dropna() * 100

# Drop the first row with NaN values from the returns calculation
df = df.dropna()

# Initialize a list to store the predicted volatilities
predicted_volatilities = []

# Iterate over each time point to fit the GARCH model and forecast volatility
for i in range(1, len(df)):
    # Use data up to the current point
    returns_subset = df['Returns'][:i]
    
    # Fit the GARCH(1, 1) model
    model = arch_model(returns_subset, vol='Garch', p=1, q=1)
    garch_fit = model.fit(disp='off')  # Suppress output
    
    # Forecast the next day's volatility
    forecast = garch_fit.forecast(horizon=1)
    predicted_volatility = forecast.variance.values[-1][0] ** 0.5  # Take the square root to get volatility
    predicted_volatilities.append(predicted_volatility)

# Add the predicted volatilities to the DataFrame
# Align by adding a NaN for the first date where we couldn't forecast
df['Predicted_Volatility'] = [None] + predicted_volatilities

# Save the updated DataFrame to a CSV file
df.to_csv("Gay.csv", index=False)

print("Data with ARIMA features saved to 'Mega_Data_Added_ARIMA.csv' successfully!")


