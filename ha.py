import h5py
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from pandas.tseries.offsets import *

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

# Return the HA value
def get_historical_average(ts, time):
    t = time
    start_time = pd.Timestamp(2014, 4, 1)
    end_time = pd.Timestamp(2014, 9, 20, 23)
    sum = 0
    count = 0
    while t >= start_time:
        if t > end_time:
            t = t - DateOffset(weeks=1)
            continue
        sum += ts[t]
        count += 1
        t = t - DateOffset(weeks=1)
    return sum / count

sum_sq_err = 0.0
for fl in [0, 1]:
    print "Processing inflow..." if fl == 0 else "Processing outflow..."
    for x in range(8):
        for y in range(16):
            print "Grid:\t(" + str(x+1) + ", " + str(y+1) + ")"
            training_ts_flow = ts_of_grid(training_data, training_time_range, fl, x, y)
            testing_ts_flow = ts_of_grid(testing_data, testing_time_range, fl, x, y)
            for t in testing_time_range:
                hist_val = get_historical_average(training_ts_flow, t)
                true_val = testing_ts_flow[t]
                sum_sq_err += (hist_val - true_val)**2

rmse = (sum_sq_err / (2*8*16*240))**0.5
print "RMSE:\t" + str(rmse)
