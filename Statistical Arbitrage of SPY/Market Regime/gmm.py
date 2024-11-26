import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Load the dataset
df = pd.read_csv('Mega_Data_SVD.csv')

# Drop rows with NaN values (from rolling calculations or missing data)
df = df.dropna()

# Ensure the Date column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Select features for the GMM model
selected_features = [
    'SPY_Close_Rolling_Z_Score'
]

X = df[selected_features].values

# Fit the GMM model
n_components = 3  # Number of market regimes
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(X)

# Predict the hidden states (market regimes)
hidden_states = gmm.predict(X)

# Add the hidden states to the DataFrame
df['Hidden_State'] = hidden_states

# Calculate the average returns and standard deviations for each state
state_metrics = {}
for state in range(n_components):
    state_data = df[df['Hidden_State'] == state]['SPY_Close_LogReturn']
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
df['Hidden_State'] = df['Hidden_State'].map(state_mapping)

# Update the color mapping based on the new state assignments
state_colors = {
    0: "green",   # Bullish state
    1: "red",     # Bearish state
    2: "yellow"   # Sideways state
}

# Create a directory to save the plots
os.makedirs("Market_State_Plots_GMM", exist_ok=True)

# Ensure all years are covered, even if no data exists for some years
full_years = range(df['Date'].dt.year.min(), df['Date'].dt.year.max() + 1)

for year in full_years:
    yearly_df = df[df['Date'].dt.year == year]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if not yearly_df.empty:  # Check if there is data for this year
        for state in range(n_components):
            state_df = yearly_df[yearly_df['Hidden_State'] == state]
            ax.scatter(
                state_df['Date'], state_df['SPY_Close'], 
                label=f"{'Bullish' if state == 0 else 'Bearish' if state == 1 else 'Sideways'}", 
                color=state_colors[state], alpha=0.6
            )
    else:
        print(f"No data available for year {year}. Skipping plot.")
    
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xlabel("Month")
    ax.set_ylabel("SPY Close Price")
    ax.legend()
    ax.set_title(f"Market States in {year} with SPY Close Prices (GMM)")

    # Save the figure
    plt.savefig(f"Market_State_Plots_GMM/Market_States_{year}.png")
    plt.close()  # Close the plot to free memory

# Plot the entire graph for the full period with SPY Close Prices and save it
fig, ax = plt.subplots(figsize=(16, 8))
for state in range(n_components):
    state_df = df[df['Hidden_State'] == state]
    ax.scatter(
        state_df['Date'], state_df['SPY_Close'], 
        label=f"{'Bullish' if state == 0 else 'Bearish' if state == 1 else 'Sideways'}", 
        color=state_colors[state], alpha=0.6
    )

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlabel("Year")
ax.set_ylabel("SPY Close Price")
ax.legend()
ax.set_title("Market States Over Entire Period with SPY Close Prices (GMM)")

# Save the entire period plot
plt.savefig("Market_State_Plots_GMM/Market_States_Entire_Period.png")
plt.close()  # Close the plot to free memory

# Save the updated DataFrame to a new CSV file
output_file = "Mega_Data_GMM_Completed.csv"
df.to_csv(output_file, index=False)

print("Dataset with GMM clusters and color mapping saved successfully!")
print("All plots saved in the 'Market_State_Plots_GMM' directory!")
