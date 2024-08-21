import pandas as pd

data1 = pd.read_excel("CPI.xlsx")
data2 = pd.read_excel("PPI.xlsx")

data1.to_csv("CPI.csv", index = False)
data2.to_csv("PPI.csv", index = False)















