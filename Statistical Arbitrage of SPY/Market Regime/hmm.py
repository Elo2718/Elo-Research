import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Load your data
df = pd.read_csv('Mega_Data_SVD.csv')

# Drop rows with NaN values (resulting from rolling calculations)
df = df.dropna()

# Select features for the HMM model
selected_features = [
    'SPY_Close_Rolling_Z_Score',  # Adjust as per your column names
        # Include SPY_Close_LogReturn in features
]

X = df[selected_features].values

# Fit the HMM model
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)  # Set n_components=3 for three states
model.fit(X)

# Predict the hidden states
hidden_states = model.predict(X)

# Add the hidden states to the DataFrame
df['Date'] = pd.to_datetime(df['Date'])
df['Hidden_State'] = hidden_states

# Calculate the average returns for each state
state_metrics = {}
for state in range(model.n_components):
    state_data = df[df['Hidden_State'] == state]['SPY_Close_LogReturn']  # Use 'SPY_Close_LogReturn'
    avg_return = state_data.mean()
    state_metrics[state] = avg_return

# Identify the states based on returns
upward_state = max(state_metrics, key=state_metrics.get)  # State with the highest average return
downward_state = min(state_metrics, key=state_metrics.get)  # State with the lowest average return
sideways_state = [state for state in state_metrics if state not in [upward_state, downward_state]][0]  # Remaining state

# Reassign the hidden states consistently
state_mapping = {
    upward_state: 0,  # Assign 0 for upward movement
    downward_state: 1,  # Assign 1 for downward movement
    sideways_state: 2   # Assign 2 for sideways movement
}

# Map the Hidden_State to the consistent labels
df['Hidden_State'] = df['Hidden_State'].map(state_mapping)

# Update the color mapping based on the new state assignments
state_colors = {
    0: "green",   # Upward state
    1: "red",     # Downward state
    2: "yellow"   # Sideways state
}

# Create a directory to save the plots
os.makedirs("Market_State_Plots", exist_ok=True)

# Plot year-by-year candlestick charts with corrected colors and save them
years = df['Date'].dt.year.unique()
for year in years:
    yearly_df = df[df['Date'].dt.year == year]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    for state in state_mapping.values():
        state_df = yearly_df[yearly_df['Hidden_State'] == state]
        label = "Upward" if state == 0 else "Downward" if state == 1 else "Sideways"
        ax.scatter(state_df['Date'], state_df['SPY_Close'], label=label, color=state_colors[state], alpha=0.6)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xlabel("Month")
    ax.set_ylabel("SPY Close Price")
    ax.legend()
    ax.set_title(f"Market States in {year} with SPY Close Prices")
    
    # Save the figure
    plt.savefig(f"Market_State_Plots/Market_States_{year}.png")
    plt.close()  # Close the plot to free memory

# Plot the entire graph with corrected colors and save it
fig, ax = plt.subplots(figsize=(16, 8))
for state in state_mapping.values():
    state_df = df[df['Hidden_State'] == state]
    label = "Upward" if state == 0 else "Downward" if state == 1 else "Sideways"
    ax.scatter(state_df['Date'], state_df['SPY_Close'], label=label, color=state_colors[state], alpha=0.6)

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlabel("Year")
ax.set_ylabel("SPY Close Price")
ax.legend()
ax.set_title("Market States Over Entire Period with SPY Close Prices")

# Save the entire period plot
plt.savefig("Market_State_Plots/Market_States_Entire_Period.png")
plt.close()  # Close the plot to free memory

# Save the updated DataFrame to a new CSV file
df.to_csv("Mega_Data_Completed.csv", index=False)

print("Dataset with HMM states (Up/Down/Sideways) and color mapping saved successfully!")
print("All plots saved in the 'Market_State_Plots' directory!")
