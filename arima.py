import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from statsmodels.tsa.arima_model import ARIMA

f = h5py.File('NYC14_M16x8_T60_NewEnd.h5', 'r')

data = f['data']

# Last ten days of data is preserved for testing
training_data = data[:-240]
testing_data = data[-240:]

# Time ranges of two sets of data
training_time_range = pd.date_range('4/1/2014', periods=4152, freq='H')
testing_time_range = pd.date_range('9/21/2014', periods=240, freq='H')

# Generate the timeseries
def ts_of_grid(ds, t_range, flow, x, y):
    grid_data = [ m[flow][y][x] for m in ds ]
    return pd.Series(grid_data, t_range)

def ts_diff_of(ts):
    moving_avg = pd.Series.rolling(ts, 12).mean()
    '''
    ew_avg = pd.ewma(ts, halflife=12)
    '''
    ts_ewma_diff = ts - moving_avg
    ts_diff = ts - ts.shift(1)
    ts_diff.dropna(inplace=True)
    return ts_diff

# Test the stationarity of a timeseries
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.Series.rolling(timeseries, window=24).mean()
    rolstd = pd.Series.rolling(timeseries, window=24).std()

    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=True)

sum_sq_err = 0.0
for fl in [0, 1]:
    print "Processing inflow..." if fl == 0 else "Processing outflow..."
    for x in range(8):
        for y in range(16):
            print "Grid:\t(" + str(x+1) + ", " + str(y+1) + ")"
            ts = ts_of_grid(training_data, training_time_range, fl, x, y)
            moving_avg = pd.Series.rolling(ts, 12).mean()
            ts_moving_avg_diff = ts - moving_avg
            ts_moving_avg_diff.dropna(inplace=True)
            ts_diff = ts_diff_of(ts)
            model = ARIMA(ts, order=(2,1,2))
            results_ARIMA = model.fit(disp=-1)
            sum_sq_err += sum((results_ARIMA.fittedvalues-ts_diff)**2)

rmse = (sum_sq_err/(2*8*16*240))**0.5
print rmse
'''
x, y = 4, 4
ts = ts_of_grid(training_data, training_time_range, 0, x, y)

moving_avg = pd.Series.rolling(ts, 12).mean()
ts_moving_avg_diff = ts - moving_avg
ts_moving_avg_diff.dropna(inplace=True)

ts_diff = ts_diff_of(ts)


# test_stationarity(ts_moving_avg_diff)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid



#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_diff, nlags=20)
lag_pacf = pacf(ts_diff, nlags=20, method='ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts, order=(2,1,2))
results_ARIMA = model.fit(disp=-1)

prediction = results_ARIMA.predict(start=4152, end=4391, dynamic=True)
true_val = ts_diff_of(ts_of_grid(testing_data, testing_time_range, 0, x, y))

plt.plot(ts_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RMSE: %.4f'% ((1.0/len(results_ARIMA.fittedvalues))*sum((results_ARIMA.fittedvalues-ts_diff)**2))**0.5)

plt.plot(prediction)
plt.plot(true_val)
plt.show()

#cannot divide by zero
#ts_log = np.log(ts)
'''
