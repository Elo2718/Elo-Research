import pandas as pd

spy_data = pd.read_csv('Raw Data/SPY_historical_data.csv')
t_bill_data = pd.read_csv('Raw Data/T_Bill_30_historical_data.csv')
usd_index_data = pd.read_csv('Raw Data/US_Dollar_Index_historical_data.csv')
vix_data = pd.read_csv('Raw Data/VIX_historical_data.csv')
gold_data_1 = pd.read_csv('Raw Data/Gold_Futures_Historical_Data_Part1.csv')
gold_data_2 = pd.read_csv('Raw Data/Gold_Futures_Historical_Data_Part2.csv')
oil_data_1 = pd.read_csv('Raw Data/Crude_Oil_Futures_Historical_Data_Part1.csv')
oil_data_2 = pd.read_csv('Raw Data/Crude_Oil_Futures_Historical_Data_Part2.csv')

spy_data = spy_data.drop(index = [0,1]).reset_index(drop = True)
spy_data.rename(columns={'Price' : 'Date'}, inplace = True)
spy_data.rename(columns={'Adj Close' : 'SPY_Adj_Close'}, inplace = True)
spy_data.rename(columns={'Close' : 'SPY_Close'}, inplace = True)
spy_data.rename(columns={'High' : 'SPY_High'}, inplace = True)
spy_data.rename(columns={'Low' : 'SPY_Low'}, inplace = True)
spy_data.rename(columns={'Open' : 'SPY_Open'}, inplace = True)
spy_data.rename(columns={'Volume' : 'SPY_Volume'}, inplace = True)
spy_data['Date'] = pd.to_datetime(spy_data['Date'] , errors = 'coerce').dt.date
spy_data.set_index('Date', inplace=True)

t_bill_data = t_bill_data.drop(index = [0,1]).reset_index(drop = True)
t_bill_data.rename(columns={'Price' : 'Date'}, inplace = True)
t_bill_data.rename(columns={'Adj Close' : 'T_Bill_Adj_Close'}, inplace = True)
t_bill_data.rename(columns={'Close' : 'T_Bill_Close'}, inplace = True)
t_bill_data.rename(columns={'High' : 'T_Bill_High'}, inplace = True)
t_bill_data.rename(columns={'Low' : 'T_Bill_Low'}, inplace = True)
t_bill_data.rename(columns={'Open' : 'T_Bill_Open'}, inplace = True)
t_bill_data = t_bill_data.drop(columns='Volume')
t_bill_data['Date'] = pd.to_datetime(t_bill_data['Date'] , errors = 'coerce').dt.date
t_bill_data.set_index('Date', inplace=True)

usd_index_data = usd_index_data.drop(index = [0,1]).reset_index(drop = True)
usd_index_data.rename(columns={'Price' : 'Date'}, inplace = True)
usd_index_data.rename(columns={'Adj Close' : 'USD_Index_Adj_Close'}, inplace = True)
usd_index_data.rename(columns={'Close' : 'USD_Index_Close'}, inplace = True)
usd_index_data.rename(columns={'High' : 'USD_Index_High'}, inplace = True)
usd_index_data.rename(columns={'Low' : 'USD_Index_Low'}, inplace = True)
usd_index_data.rename(columns={'Open' : 'USD_Index_Open'}, inplace = True)
usd_index_data = usd_index_data.drop(columns='Volume')
usd_index_data['Date'] = pd.to_datetime(usd_index_data['Date'] , errors = 'coerce').dt.date
usd_index_data.set_index('Date', inplace=True)

vix_data = vix_data.drop(index = [0,1]).reset_index(drop = True)
vix_data.rename(columns={'Price' : 'Date'}, inplace = True)
vix_data.rename(columns={'Adj Close' : 'VIX_Adj_Close'}, inplace = True)
vix_data.rename(columns={'Close' : 'VIX_Close'}, inplace = True)
vix_data.rename(columns={'High' : 'VIX_High'}, inplace = True)
vix_data.rename(columns={'Low' : 'VIX_Low'}, inplace = True)
vix_data.rename(columns={'Open' : 'VIX_Open'}, inplace = True)
vix_data = vix_data.drop(columns='Volume')
vix_data['Date'] = pd.to_datetime(vix_data['Date'] , errors = 'coerce').dt.date
vix_data.set_index('Date', inplace=True)

gold_data = pd.concat([gold_data_1,gold_data_2], ignore_index = True)
gold_data.rename(columns={'Price' : 'Gold_Close'}, inplace = True)
gold_data.rename(columns={'High' : 'Gold_High'}, inplace = True)
gold_data.rename(columns={'Low' : 'Gold_Low'}, inplace = True)
gold_data.rename(columns={'Open' : 'Gold_Open'}, inplace = True)
gold_data = gold_data.drop(columns='Vol.')
gold_data = gold_data.drop(columns='Change %')
gold_data['Date'] = pd.to_datetime(gold_data['Date'] , errors = 'coerce').dt.date
gold_data.set_index('Date', inplace=True)

oil_data = pd.concat([oil_data_1,oil_data_2], ignore_index = True)
oil_data.rename(columns={'Price' : 'Oil_Close'}, inplace = True)
oil_data.rename(columns={'High' : 'Oil_High'}, inplace = True)
oil_data.rename(columns={'Low' : 'Oil_Low'}, inplace = True)
oil_data.rename(columns={'Open' : 'Oil_Open'}, inplace = True)
oil_data = oil_data.drop(columns='Vol.')
oil_data = oil_data.drop(columns='Change %')
oil_data['Date'] = pd.to_datetime(oil_data['Date'] , errors = 'coerce').dt.date
oil_data.set_index('Date', inplace=True)


mega_data = spy_data.merge(t_bill_data , on = 'Date' , how  = 'inner').merge(usd_index_data , on = 'Date' , how = 'inner').merge(vix_data , on = 'Date' , how = 'inner').merge(gold_data , on = 'Date' , how = 'inner').merge(oil_data , on = 'Date' , how = 'inner')

gold_columns = ['Gold_Close', 'Gold_High', 'Gold_Low', 'Gold_Open']

# Remove commas from the columns
for column in gold_columns:
    # Check for values with commas
    has_commas = mega_data[column].str.contains(',', regex=False, na=False)
    
    # Only replace commas in rows where they exist
    mega_data.loc[has_commas, column] = mega_data.loc[has_commas, column].str.replace(',', '', regex=False)
    
    # Convert the cleaned column to numeric
    mega_data[column] = pd.to_numeric(mega_data[column], errors='coerce')

# Check for non-numeric values in each Gold column
#for column in gold_columns:
    #non_numeric_values = mega_data[pd.to_numeric(mega_data[column], errors='coerce').isna() & ~mega_data[column].isna()]
    #if not non_numeric_values.empty:
        #print(f"Non-numeric values found in column {column}:")
        #print(non_numeric_values[[column]])

print(mega_data.isna().sum())

mega_data.to_csv('Mega_Data.csv')

