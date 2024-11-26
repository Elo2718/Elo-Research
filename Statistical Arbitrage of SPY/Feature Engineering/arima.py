import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv('Mega_Data_Added.csv')

# List of features to apply ARIMA on, including SPY_Volume
arima_features = ['SPY_Close', 'T_Bill_Close', 'USD_Index_Close', 'VIX_Close', 'Gold_Close', 'Oil_Close',
                  'SPY_High', 'T_Bill_High', 'USD_Index_High', 'VIX_High', 'Gold_High', 'Oil_High',
                  'SPY_Low', 'T_Bill_Low', 'USD_Index_Low', 'VIX_Low', 'Gold_Low', 'Oil_Low',
                  'SPY_Open', 'T_Bill_Open', 'USD_Index_Open', 'VIX_Open', 'Gold_Open', 'Oil_Open',
                  'SPY_Volume']  # Added SPY_Volume

# Apply ARIMA modeling to each feature
for feature in arima_features:
    try:
        # Drop missing values for the current feature
        df[feature].dropna(inplace=True)

        # Use auto_arima to find the best (p, d, q) parameters
        auto_model = pm.auto_arima(df[feature], seasonal=False, trace=True)
        p, d, q = auto_model.order

        # Fit the ARIMA model using statsmodels
        arima_model = ARIMA(df[feature], order=(p, d, q)).fit()

        # Calculate the number of initial observations to discard
        discard_count = max(p, d, q)

        # Generate in-sample predictions starting from the adjusted point
        predictions = arima_model.predict(start=discard_count, end=len(df)-1)
        df[f'{feature}_ARIMA_Predictions_{p}_{d}_{q}'] = pd.Series(predictions, index=df.index[discard_count:])

        # Calculate residuals: difference between the actual values and the ARIMA predictions
        df[f'{feature}_ARIMA_Residuals_{p}_{d}_{q}'] = df[feature] - df[f'{feature}_ARIMA_Predictions_{p}_{d}_{q}']

    except Exception as e:
        print(f"Failed to fit ARIMA model for {feature}: {e}")

# Save the updated DataFrame to a CSV file
df.to_csv('Mega_Data_Added_ARIMA.csv', index=False)

print("Data with ARIMA features saved to 'Mega_Data_Added_ARIMA.csv' successfully!")

