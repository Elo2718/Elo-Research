import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Load your data
df = pd.read_csv('Mega_Data_With_Predicted_Volatility.csv')

# Drop rows with NaN values (resulting from rolling calculations)
df = df.dropna()

# Ensure the 'Date' column is in datetime format and the data is sorted by date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# Select features for the HMM model
selected_features = ['SPY_Close_Rolling_Z_Score']

# Split the data into training and testing sets
train_data = df.iloc[-504:-252]  # Use the 252 prior data points for training
test_data = df.iloc[-252:]  # Use the most recent 252 data points for testing

# Prepare the feature matrices
X_train = train_data[selected_features].values
X_test = test_data[selected_features].values

# Fit the HMM model on the training data
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
model.fit(X_train)

# Predict the hidden states for the testing data
hidden_states = model.predict(X_test)

# Add the hidden states to the test DataFrame
test_data['Hidden_State'] = hidden_states

# Calculate the average returns and standard deviations for each state
state_metrics = {}
for state in range(model.n_components):
    state_data = test_data[test_data['Hidden_State'] == state]['SPY_Close_Rolling_Z_Score']
    avg_return = state_data.mean()
    std_dev = state_data.std()
    state_metrics[state] = {"avg_return": avg_return, "std_dev": std_dev}

# Identify the states based on the criteria
bullish_state = max(state_metrics, key=lambda x: state_metrics[x]["avg_return"])
bearish_state = max(state_metrics, key=lambda x: state_metrics[x]["std_dev"])
sideways_state = [state for state in state_metrics if state != bullish_state and state != bearish_state][0]

# Reassign the hidden states consistently
state_mapping = {
    bullish_state: 0,  # Assign 0 for bullish
    bearish_state: 1,  # Assign 1 for bearish
    sideways_state: 2  # Assign 2 for sideways
}

# Map the Hidden_State to the consistent labels
test_data['Hidden_State'] = test_data['Hidden_State'].map(state_mapping)

# Update the color mapping based on the new state assignments
state_colors = {
    0: "green",   # Bullish state
    1: "red",     # Bearish state
    2: "yellow"   # Sideways state
}

# Create a directory to save the plots
os.makedirs("HMM_Test_Plots", exist_ok=True)

# Plot the classified data
fig, ax = plt.subplots(figsize=(14, 7))
for state in range(model.n_components):
    state_df = test_data[test_data['Hidden_State'] == state]
    ax.scatter(state_df['Date'], state_df['SPY_Close'], label=f"{'Bullish' if state == 0 else 'Bearish' if state == 1 else 'Sideways'}", color=state_colors[state], alpha=0.6)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.set_xlabel("Date")
ax.set_ylabel("SPY Close Price")
ax.legend()
ax.set_title("Market States for the Most Recent 252 Trading Days with HMM")

# Save the plot
plt.savefig("HMM_Test_Plots/HMM_Market_States_252_Days.png")
plt.close()  # Close the plot to free memory

# Save the updated test DataFrame to a new CSV file
test_data.to_csv("HMM_Test_Data_252_Days.csv", index=False)

print("Test data with HMM states saved successfully to 'HMM_Test_Data_252_Days.csv'!")
print("Plot saved in the 'HMM_Test_Plots' directory!")
