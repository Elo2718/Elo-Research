import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Load your data
df = pd.read_csv('Gay.csv')

# Create the target variables
df['SPY_Close_Target_Reg'] = df['SPY_Close'].shift(-1)  # For regression
df['SPY_Close_Target_Clf'] = np.where(df['SPY_Close'] > df['SPY_Close'].shift(1), 1, 0)  # For classification


# Select features for the HMM model
selected_features = ['SPY_Close_Rolling_Z_Score']

# Fit the HMM model
X_train_hmm = df[selected_features].iloc[:-1].values
hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
hmm_model.fit(X_train_hmm)

# Predict hidden states and add to DataFrame
hidden_states = hmm_model.predict(df[selected_features].values)
df['Hidden_State'] = hidden_states

# Convert 'Hidden_State' to a categorical variable
df['Hidden_State'] = df['Hidden_State'].astype('category')

# Define features for the models, including 'Hidden_State' as a feature
features = [col for col in df.columns if col not in ['Date', 'Returns', 'SPY_Close_Target_Reg', 'SPY_Close_Target_Clf'] and "Adj_Close" not in col]

# Split the data into training and testing sets (exclude the last row for training)
X_train = df[features].iloc[:-1]  # All data except the last row
y_train_reg = df['SPY_Close_Target_Reg'].iloc[:-1]  # All target values for regression except the last
y_train_clf = df['SPY_Close_Target_Clf'].iloc[:-1]  # All target values for classification except the last

X_test = df[features].iloc[[-1]]  # The last row for testing
y_test_reg = df['SPY_Close_Target_Reg'].iloc[-1]  # The last target value for regression
y_test_clf = df['SPY_Close_Target_Clf'].iloc[-1]  # The last target value for classification

# -------------------- REGRESSION MODEL --------------------
# Initialize and train the XGBRegressor with categorical data support
model_reg = XGBRegressor(
    n_estimators=100, learning_rate=0.6, max_depth=9, min_child_weight=6,
    gamma=0, subsample=1.0, colsample_bytree=1.0, reg_alpha=1, reg_lambda=1,
    enable_categorical=True  # Enable support for categorical features
)
model_reg.fit(X_train, y_train_reg)

# -------------------- CLASSIFICATION MODEL --------------------
# Initialize and train the XGBClassifier
model_clf = XGBClassifier(
    n_estimators=110, learning_rate=0.7, max_depth=9, min_child_weight=6,
    gamma=0, subsample=1.0, colsample_bytree=1.0, reg_alpha=1, reg_lambda=1,
    enable_categorical=True, use_label_encoder=False  # Enable categorical feature handling
)
model_clf.fit(X_train, y_train_clf)

# Make predictions for the last row
predicted_close = model_reg.predict(X_test)[0]  # Regression prediction
predicted_probabilities = model_clf.predict_proba(X_test)[0]  # Classification probabilities

# Extract the probabilities
probability_down = predicted_probabilities[0]  # Probability of down
probability_up = predicted_probabilities[1]    # Probability of up

# Output the results
print("Actual Close for the last row:", df['SPY_Close'].iloc[-1])
print("Predicted Close for the last row:", predicted_close)
print("Probability of Down for the last row:", probability_down)
print("Probability of Up for the last row:", probability_up)

# You can also create a DataFrame to save the results if needed
last_row_prediction = pd.DataFrame({
    'Date': [df['Date'].iloc[-1]],  # Date of the last row
    'SPY_Open': [df['SPY_Open'].iloc[-1]],  # Open price of the last row
    'SPY_Close': [df['SPY_Close'].iloc[-1]],  # Actual close price of the last row
    'Predicted_Close': [predicted_close],  # Predicted close price
    'Probability_Down': [probability_down],  # Probability of down
    'Probability_Up': [probability_up]      # Probability of up
})

# Save the result to a CSV file
last_row_prediction.to_csv('last_row_prediction.csv', index=False)
