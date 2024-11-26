import yfinance as yf
import quandl
import pandas as pd

quandl.ApiConfig.api_key = '2SpebsLmKScc-guKPyuD'

ticker_symbol1 = 'SPY'
data1 = yf.download(ticker_symbol1, start='1993-1-01', end = '2024-11-20', interval = '1d')
data1.to_csv('SPY_data.csv')

#ticker_symbol2 = '^TYX'
#data2 = yf.download(ticker_symbol2, start = '2024-11-19', end = '2024-11-20', interval = '1d')
#data2.to_csv('T_Bill_30_aditional_data.csv')

#ticker_symbol3 = '^VIX'
#data3 = yf.download(ticker_symbol3, start = '2024-11-19', end = '2024-11-20', interval = '1d')
#data3.to_csv('VIX_aditional_data.csv')

#ticker_symbol4 = 'DX-Y.NYB'
#data4 = yf.download(ticker_symbol4, start = '2024-11-19', end = '2024-11-20', interval = '1d')
#data4.to_csv('US_Dollar_Index_aditional_data.csv')
