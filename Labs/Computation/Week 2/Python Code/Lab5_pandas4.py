import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pylab

#Problem 1
data = pd.read_csv('DJIA.csv', index_col = ['DATE'], na_values=["."])
data.VALUE = pd.Series(data.VALUE)
data.index = pd.to_datetime(data.index)
data["VALUE"] = data["VALUE"].fillna(np.nan).astype(float)

#Problem 2
paychecks = pd.read_csv('paychecks.csv', header = None)
fri1 = pd.date_range(start = '03/13/2008', periods = 93, freq = 'WOM-1FRI')
fri3 = pd.date_range(start = '03/13/2008', periods = 93, freq = 'WOM-3FRI')
fri1 = fri1.union(fri3)[:93]
paychecks.index = fri1

#Problem 4
finances = pd.read_csv('finances.csv')
finances.index = pd.period_range("1978-09", "1999-06", freq ="Q-DEC")

#Problem 5
traffic = pd.read_csv('website_traffic.csv')
a = traffic["ENTER"] = pd.to_datetime(traffic["ENTER"])
b= traffic["LEAVE"] = pd.to_datetime(traffic["LEAVE"])
c = b - a
traffic["DURATION"] = c
print(np.mean(c))

traffic["HOURS"] = 1
traffic.index = pd.to_datetime(traffic["ENTER"])
traffic = traffic.drop("IP", 1)
traffic = traffic.drop("ENTER", 1)
traffic = traffic.drop("LEAVE", 1)
traffic = traffic.drop("DURATION", 1)
print(traffic.resample('H').sum())

#Problem 6
data["DIFFERENCE"] = data - data.shift(1)
maxdaygain = np.max(data["DIFFERENCE"])
mindaygain = np.min(data["DIFFERENCE"])

maxdayindex = np.where(data["DIFFERENCE"]==maxdaygain)
mindayindex = np.where(data["DIFFERENCE"]==mindaygain)
print("Day with largest gain: ", str(data.index.values[maxdayindex])[2:12])
print("Day with largest loss: ", str(data.index.values[mindayindex])[2:12])

data = data.resample('M').sum()

maxmonthgain = np.max(data["DIFFERENCE"])
minmonthgain = np.min(data["DIFFERENCE"])
maxmonthindex = np.where(data["DIFFERENCE"]==maxmonthgain)
minmonthindex = np.where(data["DIFFERENCE"]==minmonthgain)
print("Month with largest gain: ", str(data.index.values[maxmonthindex])[2:9])
print("Month with largest loss: ", str(data.index.values[minmonthindex])[2:9])

#Problem 7
data = pd.read_csv('DJIA.csv', index_col = ['DATE'], na_values=["."])
data.VALUE = pd.Series(data.VALUE)
data.index = pd.to_datetime(data.index)
data["VALUE"] = data["VALUE"].fillna(np.nan).astype(float)
plt.plot(data, label = "Original")
plt.title("Dow Jones Industrial Average")

plt.plot(data.rolling(window = 30, min_periods = 20).mean(), label = "Rolling")
plt.plot(data.rolling(window = 365, min_periods = 20).mean())
plt.plot(data.ewm(span = 30, min_periods = 20).mean(), label = "EWMA")
plt.plot(data.ewm(span = 365, min_periods = 20).mean())
plt.legend(loc = 'lower right')
plt.show()
