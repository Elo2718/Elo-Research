import pandas as pd

# Load the dataset
cpiData = pd.read_csv("CPI.csv")

# Convert the 'Year' column to a datetime format
cpiData["Date"] = pd.to_datetime(cpiData['Year'].astype(str) + '-01-01')

# Melt the DataFrame to have a single column for dates and another for the inflation rate
cpiData_melted = cpiData.melt(id_vars=['Date'], value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                       var_name='Month', value_name='Inflation')

# Create a date for the 1st day of each month
month_numbers = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
cpiData_melted['Date'] = cpiData_melted.apply(lambda row: row['Date'].replace(month=month_numbers[row['Month']]), axis=1)

# Drop the 'Month' column
cpiData_melted.drop(columns=['Month'], inplace=True)

# Set the 'Date' column as the index
cpiData_melted.set_index('Date', inplace=True)

# Create a daily date range covering the full range of the data
full_date_range = pd.date_range(start=cpiData_melted.index.min(), end=cpiData_melted.index.max(), freq='D')

# Reindex the DataFrame to this daily date range, forward-filling the missing values
cpiData_daily = cpiData_melted.reindex(full_date_range).ffill()

# Reset the index so 'Date' is no longer the index
cpiData_daily.reset_index(inplace=True)
cpiData_daily.rename(columns={'index': 'Date'}, inplace=True)

print(cpiData_daily)
