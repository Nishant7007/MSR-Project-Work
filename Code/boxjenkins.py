from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from loadSeries import *

def test_stationarity(timeseries, window = 12, cutoff = 0.01):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag = 20 )
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)
    
    print(dfoutput)

def convert_daily_weekly(df):
    i=0
    data=[]
    while i+7<len(df):
        week = df[i:i+7]
        week = sum(week)/7
        data.append(week)
        i=i+7
    return pd.DataFrame(data)

df=bangalore_mandi
df=convert_daily_weekly(df)
#first_diff = df - df.shift(1)
#first_diff=first_diff[1:]
# ses_diff=first_diff-first_diff.shift(52)
# ses_diff=ses_diff[52:]
#test_stationarity(first_diff)
#first_diff = df - df.shift(1)
#first_diff = first_diff.dropna(inplace = False)
#test_stationarity(first_diff, window = 52)


# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(first_diff, lags=160, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(first_diff, lags=160, ax=ax2)
    # plt.show()

import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.arima.auto_arima(df, start_p=0, start_q=0,
                         test='adf',
                         max_p=2, max_q=2, m=52,
                         start_P=0,max_P=2, start_Q=0,
                         max_Q=2, seasonal=True,
                         start_d=0,max_d=1, start_D=0,
                         max_D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=False,random=True,n_fits=10)

print(smodel.summary())



#kalyani price Fit ARIMA: order=(1, 0, 0) seasonal_order=(1, 0, 1, 7); AIC=58437.678, 
#BIC=58470.006, Fit time=9.074 seconds

#kalyani_arrival Fit ARIMA: order=(2, 0, 0) seasonal_order=(0, 0, 0, 7); AIC=10368.845,
# BIC=10394.707, Fit time=0.819 seconds

#bijnaur price Fit ARIMA: order=(2, 0, 1) seasonal_order=(0, 0, 1, 7); AIC=61442.554, 
#BIC=61481.346, Fit time=13.749 seconds

# bijnaur_Arrival Fit ARIMA: order=(1, 0, 0) seasonal_order=(1, 0, 1, 7); AIC=26040.215,
#  BIC=26072.542, Fit time=9.301 seconds


