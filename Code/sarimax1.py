
"""
Created on Tue Dec 17 22:26:34 2019

@author: ictd
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


series=pd.read_csv('/home/nishant/study/researchwork/MSThesis-Time_Series_Forecasting-master/Data/website/Retail/lucknow_retail.csv',header=None)[1]
near1=pd.read_csv('/home/nishant/study/researchwork/MSThesis-Time_Series_Forecasting-master/Data/Processed/lucknow_mandi.csv',header=None)[1]
near2=pd.read_csv('/home/nishant/study/researchwork/MSThesis-Time_Series_Forecasting-master/Data/Processed/inflation.csv',header=None)[1]

near=np.vstack((near1,near2)).T
print(series.shape)
# def convert_daily_weekly(df):
#     i=0
#     data=[]
#     while i+7<len(df):
#         week = df[i:i+7]
#         week = sum(week)/7
#         data.append(week)
#         i=i+7
#     return np.array(data)

# def convert_weekly_daily(predicted):
#     daily=[]
#     for i in predicted:
#         daily.extend([i,i,i,i,i,i,i])
#     return daily


# trains = series
# train = [x for x in trains]
# train=convert_daily_weekly(train)
# near = near
# near = [x for x in near]
# near=convert_daily_weekly(near)
# n = len(train)
# n=450
# end_index = 218
# print(n)
# forecasted = list(train[ : end_index])
# while end_index != n:
#     print(end_index)
#     history = train[ : end_index]
#     near_history = near[ : end_index]
#     print('1')
#     model = sm.tsa.statespace.SARIMAX(endog=history, exog=None, order=(0,1,2), seasonal_order=(1,0,1,52),initialization='approximate_diffuse')
    
#     print('2')
#     res = model.fit(disp=False)
#     print('3')
#     if end_index + 4 < n:
#         predictions = res.forecast(4,exog=None)
#         forecasted = list(forecasted + predictions.tolist())
#     else:
#         vals = n - end_index
#         predictions = res.forecast(vals,exog=None)
#         forecasted = list(forecasted + predictions.tolist())
#     prev_index = end_index
#     end_index = min(end_index + 4, n)
# print('End: ' + str(end_index))
# print(len(forecasted))
# forecasted=convert_weekly_daily(forecasted)                                                                                                                                                                                                                                                                                                                                                                                                                             
# p=len(forecasted)
# dk=pd.DataFrame(forecasted)
# dk['date']=pd.date_range('2006-01-01',periods=p)
# dk.columns=['price','date']
# dk=dk[['date','price']]
# dk.to_csv('../Results/Arrival/Kalyani/kalyani_arrival_sarimax1.csv')


model = sm.tsa.statespace.SARIMAX(endog=series, exog=None, order=(4,1,6), seasonal_order=(3,0,1,7),initialization='approximate_diffuse',enforce_stationarity=False)
res = model.fit(disp=False)
predictions = res.forecast(30,exog=None)
dk=pd.DataFrame(predictions)
dk['date']=pd.date_range('2020-01-01',periods=30)
#dk['date']=pd.date_range('2020-01-01')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    -01-01',periods=30)
#dk.columns=['price','date']
dk=dk[['date',0]]
print(dk)
plt.plot(dk[0])
plt.show()
# dk.to_csv('/home/nishant/study/researchwork/MSThesis-Time_Series_Forecasting-master/Data/website/Arrival/kalyani_arrival_2020.csv',index=False,header=False)