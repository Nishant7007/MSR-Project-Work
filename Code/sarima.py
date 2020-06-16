import numpy as np
import pandas as pd
import statsmodels.api as sm
import pmdarima as pm

from loadSeries import *

series=kalyani_mandi

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
n = len(train)
end_index = 678
forecasted = list(train[ : 678])
print(n)
while end_index != n:
    print(end_index)
    history = train[ : end_index]
    print('1')
    model = sm.tsa.statespace.SARIMAX(endog=history, exog=None, order=(0,1,2), seasonal_order=(1,0,1,52),initialization='approximate_diffuse')
    print('2')
    res = model.fit(disp=False)
    print('3')
    if end_index + 4 < n:
        predictions = res.forecast(4)
        forecasted = list(forecasted + predictions.tolist())
    else:
        vals = n - end_index
        predictions = res.forecast(vals)
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
dk.to_csv('../Results/2019/kalyani_mandi_2019_sarima.csv')




















