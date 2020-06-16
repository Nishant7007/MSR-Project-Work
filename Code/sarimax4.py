#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:26:34 2019

@author: ictd
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pmdarima as pm

from loadSeries import *
series=mumbai_retail
near=lasalgaon_mandi

def convert_daily_weekly(df):
    i=0
    data=[]
    while i+7<len(df):
        week = df[i:i+7]
        week = sum(week)/7
        data.append(week)
        i=i+7
    return np.array(data)

def convert_weekly_daily(predicted):
    daily=[]
    for i in predicted:
        daily.extend([i,i,i,i,i,i,i])
    return daily


trains = series
train = [x for x in trains]
train=convert_daily_weekly(train)
near = near
near = [x for x in near]
near=convert_daily_weekly(near)
n = len(train)
p=220
n=450
print(n)


end_index = p
forecasted = list(train[ : p])

while end_index != n:
    print(end_index)
    history = train[ : end_index]
    near_history = near[ : end_index]
    model=pm.arima.auto_arima(history, exogenous=pd.DataFrame(near_history), start_p=0, d=None, start_q=0, max_p=2, max_d=1, max_q=2,start_P=0, D=None, start_Q=0, max_P=2, max_D=1, max_Q=2, suppress_warnings =True,seasonal=True) 
    print(model.summary())
    print('3')
    if end_index + 4 < n:
        predictions = model.predict(4,exogenous=pd.DataFrame(near[end_index:end_index+4]))
        forecasted = list(forecasted + predictions.tolist())
    else:
        vals = n - end_index
        predictions = model.predict(vals,exogenous=pd.DataFrame(near[end_index:end_index+vals]))
        forecasted = list(forecasted + predictions.tolist())
    prev_index = end_index
    end_index = min(end_index + 4, n)
print('End: ' + str(end_index))
print(len(forecasted))
forecasted=convert_weekly_daily(forecasted)
p=len(forecasted)
dk=pd.DataFrame(forecasted)
dk['date']=pd.date_range('2006-01-01',periods=p)
dk.columns=['price','date']
dk=dk[['date','price']]
dk.to_csv('../Results/Retail/Mumbai/mumbai_retail_sarimax.csv')
