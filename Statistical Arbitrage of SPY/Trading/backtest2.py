import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the main dataset
df = pd.read_csv('Mega_Data_Completed.csv')

# Calculate the percentage change in SPY_Close
df['Close_Change'] = (df['SPY_Close'].pct_change()) * 100

# Shift the percentage change back to align with the next day for prediction
df['Next_Day_Close_Target'] = np.where(df['Close_Change'].shift(-1) > 0, 1, 0)

# Drop rows with NaN values that may have been introduced by the shifting operation
df = df.dropna()

# Convert Hidden_State to categorical if it's not already
df['Hidden_State'] = df['Hidden_State'].astype('category')

# Select features: all 'Component_' columns plus 'Hidden_State'
features = [col for col in df.columns if col.startswith('Component_')] + ['Hidden_State']

# Define X and y for modeling
X = df[features]
y = df['Next_Day_Close_Target']

# Initialize the XGBClassifier with the preferred parameters
xgb_model = XGBClassifier(
    n_estimators=750,         # Number of boosting rounds
    learning_rate=0.05,       # Step size shrinkage
    max_depth=3,              # Maximum depth of each tree
    min_child_weight=1,       # Minimum sum of instance weight (hessian) needed in a child
    gamma=0.5,                # Minimum loss reduction required to make a further partition
    subsample=0.8,            # Fraction of samples used for fitting individual trees
    colsample_bytree=0.8,     # Fraction of features used per tree
    reg_alpha=1,              # L1 regularization term on weights
    reg_lambda=1.5,           # L2 regularization term on weights
    scale_pos_weight=4,       # Balancing of positive and negative weights
    enable_categorical=True,  # Enable handling of categorical features
    use_label_encoder=False,  # Suppress unnecessary warning
    eval_metric='logloss'     # Evaluation metric for classification
)

# Backtest for the last 1260 days with retraining every 252 days
backtest_days = 1260
retrain_interval = 252

# Prepare a DataFrame to store predictions
backtest_results = pd.DataFrame(
    index=df.index[-backtest_days:],
    columns=['Date', 'SPY_Open', 'SPY_High', 'SPY_Low', 'SPY_Close', 'Actual', 'Predicted', 'Probability_Up', 'Probability_Down']
)

# Populate the backtest results with Date and SPY columns
backtest_results['Date'] = df['Date'].iloc[-backtest_days:]
backtest_results['SPY_Open'] = df['SPY_Open'].iloc[-backtest_days:]
backtest_results['SPY_High'] = df['SPY_High'].iloc[-backtest_days:]
backtest_results['SPY_Low'] = df['SPY_Low'].iloc[-backtest_days:]
backtest_results['SPY_Close'] = df['SPY_Close'].iloc[-backtest_days:]

# Loop through the backtest period
for start_idx in range(len(df) - backtest_days, len(df), retrain_interval):
    end_idx = start_idx + retrain_interval
    if end_idx > len(df):
        end_idx = len(df)

    # Define training and testing sets
    train_X = X.iloc[:start_idx]
    train_y = y.iloc[:start_idx]
    test_X = X.iloc[start_idx:end_idx]
    test_y = y.iloc[start_idx:end_idx]

    # Train the model
    xgb_model.fit(train_X, train_y)

    # Predict probabilities and classes for the testing set
    prob_predictions = xgb_model.predict_proba(test_X)
    class_predictions = xgb_model.predict(test_X)

    # Store results
    backtest_results.loc[test_X.index, 'Actual'] = test_y
    backtest_results.loc[test_X.index, 'Predicted'] = class_predictions
    backtest_results.loc[test_X.index, 'Probability_Up'] = prob_predictions[:, 1]
    backtest_results.loc[test_X.index, 'Probability_Down'] = prob_predictions[:, 0]

# Save backtest results to a CSV
backtest_results.to_csv('Backtest_Results_1260_Days.csv', index=False)

# Ensure Actual and Predicted are numeric and drop any NaN values
actual = backtest_results['Actual'].dropna().astype(int)
predicted = backtest_results['Predicted'].dropna().astype(int)

# Ensure they are of the same length
if len(actual) != len(predicted):
    raise ValueError("Actual and Predicted values have mismatched lengths.")

# Calculate accuracy
accuracy = accuracy_score(actual, predicted)
print(f"Backtest Accuracy: {accuracy:.4f}")

# Output summary of results
print("Backtest results saved to 'Backtest_Results_1260_Days.csv'.")
