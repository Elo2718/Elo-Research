import pandas as pd
from arch import arch_model

# Load your data (e.g., SPY returns)
df = pd.read_csv('Mega_Data_Added_ARIMA.csv')
df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in datetime format

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
df.to_csv("Mega_Data_With_Predicted_Volatility.csv", index=False)

# Print the updated DataFrame
print(df.head())
