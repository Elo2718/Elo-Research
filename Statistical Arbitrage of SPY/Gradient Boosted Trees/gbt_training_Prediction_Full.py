import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
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

# Use TimeSeriesSplit for time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

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

# Train the model using cross-validation
accuracies = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model
    xgb_model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = xgb_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Fold Accuracy: {accuracy:.4f}")

# Output the average cross-validation accuracy
cv_accuracy = np.mean(accuracies)
print(f"Cross-Validation Accuracy: {cv_accuracy:.4f}")

# Fit the model to the entire dataset
xgb_model.fit(X, y)

# Predict on the entire dataset
y_pred_full = xgb_model.predict(X)

# Calculate and output the overall accuracy
overall_accuracy = accuracy_score(y, y_pred_full)
print("Overall Accuracy on the entire dataset:", overall_accuracy)
