import pandas as pd

# Load your main dataset
df = pd.read_csv('Mega_Data.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
df.set_index('Date', inplace=True)

# Load and prepare additional data
spy_data = pd.read_csv('SPY_aditional_data.csv')
t_bill_data = pd.read_csv('T_Bill_30_aditional_data.csv')
usd_index_data = pd.read_csv('US_Dollar_Index_aditional_data.csv')
vix_data = pd.read_csv('VIX_aditional_data.csv')

# Prepare SPY data
spy_data = spy_data.drop(index=[0, 1]).reset_index(drop=True)
spy_data.rename(columns={
    'Price': 'Date',
    'Adj Close': 'SPY_Adj_Close',
    'Close': 'SPY_Close',
    'High': 'SPY_High',
    'Low': 'SPY_Low',
    'Open': 'SPY_Open',
    'Volume': 'SPY_Volume'
}, inplace=True)
spy_data['Date'] = pd.to_datetime(spy_data['Date'], errors='coerce').dt.date
spy_data.set_index('Date', inplace=True)

# Prepare T-Bill data
t_bill_data = t_bill_data.drop(index=[0, 1]).reset_index(drop=True)
t_bill_data.rename(columns={
    'Price': 'Date',
    'Adj Close': 'T_Bill_Adj_Close',
    'Close': 'T_Bill_Close',
    'High': 'T_Bill_High',
    'Low': 'T_Bill_Low',
    'Open': 'T_Bill_Open'
}, inplace=True)
t_bill_data = t_bill_data.drop(columns='Volume')
t_bill_data['Date'] = pd.to_datetime(t_bill_data['Date'], errors='coerce').dt.date
t_bill_data.set_index('Date', inplace=True)

# Prepare USD Index data
usd_index_data = usd_index_data.drop(index=[0, 1]).reset_index(drop=True)
usd_index_data.rename(columns={
    'Price': 'Date',
    'Adj Close': 'USD_Index_Adj_Close',
    'Close': 'USD_Index_Close',
    'High': 'USD_Index_High',
    'Low': 'USD_Index_Low',
    'Open': 'USD_Index_Open'
}, inplace=True)
usd_index_data = usd_index_data.drop(columns='Volume')
usd_index_data['Date'] = pd.to_datetime(usd_index_data['Date'], errors='coerce').dt.date
usd_index_data.set_index('Date', inplace=True)

# Prepare VIX data
vix_data = vix_data.drop(index=[0, 1]).reset_index(drop=True)
vix_data.rename(columns={
    'Price': 'Date',
    'Adj Close': 'VIX_Adj_Close',
    'Close': 'VIX_Close',
    'High': 'VIX_High',
    'Low': 'VIX_Low',
    'Open': 'VIX_Open'
}, inplace=True)
vix_data = vix_data.drop(columns='Volume')
vix_data['Date'] = pd.to_datetime(vix_data['Date'], errors='coerce').dt.date
vix_data.set_index('Date', inplace=True)

# Merge the additional data into the main DataFrame
df = df.merge(spy_data, how='outer', left_index=True, right_index=True)
df = df.merge(t_bill_data, how='outer', left_index=True, right_index=True)
df = df.merge(usd_index_data, how='outer', left_index=True, right_index=True)
df = df.merge(vix_data, how='outer', left_index=True, right_index=True)

df.to_csv('Mega_Data.csv')

# Now `df` contains the updated data with additional information

