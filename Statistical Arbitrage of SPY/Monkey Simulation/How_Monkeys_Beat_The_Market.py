import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as r

# Load data and set 'Date' as index
data = pd.read_csv('SPY_historical_data.csv')
data.set_index('Date', inplace=True)

# Extract closing prices
closing_prices = data['Close'].values

# Amount of money invested
money = float(input("Enter amount of money invested: "))

# Initialize hold account
hold_account = np.zeros(len(closing_prices))
hold_account[0] = money
position_size = hold_account[0] // closing_prices[0]
cash_left = hold_account[0] - position_size * closing_prices[0]

# Calculate hold account values
for i in range(1, len(hold_account)):
    hold_account[i] = position_size * closing_prices[i] + cash_left

# Number of trials
num_trials = int(input("Enter number of trials: "))

# Array that stores the results of each trial
all_trading_accounts = np.zeros((num_trials, len(closing_prices)))

# Size of position for every trade
trading_size = float(input("Enter size of trades in decimal: "))

# Number of early stops
stops = np.zeros(num_trials)  # Fixed the initialization

# Run Trials
for trial in range(num_trials):
    trading_account = np.zeros(len(closing_prices))
    trading_account[0] = money

    # Simulate Random Trades
    for i in range(1, len(trading_account)):
        trade = r.randrange(0, 2)
        if trade == 0:
            # Long position
            quantity = trading_account[i - 1] * trading_size // closing_prices[i - 1]
            change = quantity * closing_prices[i] - quantity * closing_prices[i - 1]
            if change < trading_account[i - 1] * -0.01:
                trading_account[i] = trading_account[i - 1] * 0.99
                stops[trial] += 1
            else:
                trading_account[i] = trading_account[i - 1] + change
        else:
            # Short position
            quantity = trading_account[i - 1] * trading_size // closing_prices[i - 1]
            change = quantity * closing_prices[i - 1] - quantity * closing_prices[i]
            if change < trading_account[i - 1] * -0.01:
                trading_account[i] = trading_account[i - 1] * 0.99
                stops[trial] += 1
            else:
                trading_account[i] = trading_account[i - 1] + change
    # Store results
    all_trading_accounts[trial] = trading_account

# Print the average and standard deviation of early stops
print(f"Average number of early stops: {np.mean(stops)}")
print(f"Standard deviation of early stops: {np.std(stops)}")

# Plot results
plt.figure(figsize=(14, 7))
for trial in range(num_trials):
    plt.plot(all_trading_accounts[trial], label=f'Trial {trial + 1}', alpha=0.3)

plt.plot(hold_account, label='Hold Account', color='black', linewidth=2)
plt.title(f'${money} Over Time Across {num_trials} Trials')
plt.xlabel('Time')
plt.ylabel('Account Value')
plt.show()