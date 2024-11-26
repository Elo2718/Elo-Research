import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the backtest results
df = pd.read_csv('Backtest_Results_1260_Days.csv')
df.dropna(inplace=True)  # Drop any NaN values

# Extract necessary columns
close_prices = df['SPY_Close'].values
prob_up = df['Probability_Up'].values
prob_down = df['Probability_Down'].values

# Initial investment amount
money = float(input("Enter amount of money invested: "))

# Initialize arrays for the trading strategy
trading_account_close = np.zeros(len(df))
trading_account_close[0] = money

# Initialize holding account for comparison (Buy and Hold strategy)
holding_account = np.zeros(len(df))
holding_account[0] = money

# Assume a fixed number of shares bought at the first close price
shares_held = money / close_prices[0]

# Update the holding account over time
for i in range(1, len(df)):
    holding_account[i] = shares_held * close_prices[i]

# Initialize counters
loss_counter_close = 0
stop_loss_counter_close = 0

# Stop-out threshold (e.g., if account value drops below 80% of initial investment)
stop_out_threshold = 0.8 * money

# Perform backtest (Using Current Close and Next Close Prices)
for i in range(len(df) - 1):  # Iterate up to the second-to-last day
    try:
        # Stop-out logic
        if trading_account_close[i] < stop_out_threshold:
            stop_loss_counter_close += 1
            trading_account_close[i + 1] = trading_account_close[i]
            continue

        # Decision logic based on probabilities
        if prob_up[i] > 0.5:  # Go Long
            position_size = trading_account_close[i]
            change = position_size * (close_prices[i + 1] - close_prices[i]) / close_prices[i]

            # Apply stop-loss logic: Cap loss at 1% of the account value
            max_loss = 0.01 * trading_account_close[i]
            if change < -max_loss:
                change = -max_loss
                stop_loss_counter_close += 1

            if change < 0:
                loss_counter_close += 1

            trading_account_close[i + 1] = trading_account_close[i] + change

        else:  # Go Short
            position_size = trading_account_close[i]
            change = -position_size * (close_prices[i + 1] - close_prices[i]) / close_prices[i]

            # Apply stop-loss logic: Cap loss at 1% of the account value
            max_loss = 0.01 * trading_account_close[i]
            if change < -max_loss:
                change = -max_loss
                stop_loss_counter_close += 1

            if change < 0:
                loss_counter_close += 1

            trading_account_close[i + 1] = trading_account_close[i] + change

    except Exception as e:
        print(f"Error at index {i}: {e}")
        trading_account_close[i + 1] = trading_account_close[i]

# Calculate daily percentage returns for trading and holding accounts
trading_returns = np.diff(trading_account_close) / trading_account_close[:-1] * 100
holding_returns = np.diff(holding_account) / holding_account[:-1] * 100

# Define risk-free rate (daily, as a percentage)
Rf = 0.01  # Example daily risk-free rate

# Calculate excess returns
excess_trading_returns = trading_returns - Rf
excess_holding_returns = holding_returns - Rf

# Reshape for regression
X = excess_holding_returns.reshape(-1, 1)  # Market (holding account) excess returns
y = excess_trading_returns  # Portfolio (trading account) excess returns

# Fit the regression model
reg = LinearRegression()
reg.fit(X, y)

# Extract alpha and beta
alpha = reg.intercept_
beta = reg.coef_[0]

# Output the results
print(f"CAPM Alpha (Trading Account): {alpha:.4f}%")
print(f"CAPM Beta (Trading Account): {beta:.4f}")

# Plot the performance of the trading account (close) and holding account
plt.figure(figsize=(10, 6))
plt.plot(trading_account_close, label='Trading Account (Close)', alpha=0.8)
plt.plot(holding_account, label='Holding Account (Buy and Hold)', alpha=0.8, linestyle='--')
plt.title(f'Performance of Trading vs. Holding Accounts Over Time')
plt.xlabel('Trading Days')
plt.ylabel('Account Value ($)')
plt.legend()
plt.grid(True)
plt.show()
