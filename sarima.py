import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.Series.rolling(timeseries, window=12).mean()
    rolstd = pd.Series.rolling(timeseries, window=12).std()

    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()


def ts_of_grid(ds, t_range, flow, x, y):
    grid_data = [ m[flow][y][x] for m in ds ]
    return pd.Series(grid_data, t_range)

f = h5py.File('NYC14_M16x8_T60_NewEnd.h5', 'r')

data = f['data']

# Last ten days of data is preserved for testing
training_data = data[:-240]
testing_data = data[-240:]

# Time ranges of two sets of data
training_time_range = pd.date_range('4/1/2014', periods=4152, freq='H')
testing_time_range = pd.date_range('9/21/2014', periods=240, freq='H')

'''
ts_diff = ts - ts.shift(1)
ts_diff.dropna(inplace=True)

ts_seasonal_diff = ts - ts.shift(12)
ts_seasonal_diff.dropna(inplace=True)
'''
'''
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(ts_seasonal_diff.iloc[13:], lags=12, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(ts_seasonal_diff.iloc[13:], lags=12, ax=ax2)

plt.show()
'''

'''
rmse_sum = 0
count = 0

for x in range(2, 6):
    for y in range(2, 6):
        count += 1
'''
x, y = 4, 4

ts = ts_of_grid(training_data, training_time_range, 1, x, y)

mod = sm.tsa.statespace.SARIMAX(ts, trend='n', order=(0,1,0), seasonal_order=(1,1,1,24))
results = mod.fit()

ts_predicted = results.predict(start = 4152, end= 4391, dynamic= True)
ts_true = ts_of_grid(testing_data, testing_time_range, 1, x, y)

rmse = (1.0/240.0*sum((ts_true-ts_predicted)**2))**0.5
print "RMSE:\t" + str(rmse)


plt.plot(ts, label="Original")
plt.plot(ts_predicted, label="Predicted")
plt.plot(ts_true, label="Observed")
plt.legend(loc='best')
plt.title('Time Series Prediction')

plt.show()
