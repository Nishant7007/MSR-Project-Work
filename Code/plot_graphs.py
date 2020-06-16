#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:20:56 2020

@author: ictd
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import numpy as np
import math


def calculate_rmse(actual,predicted):
    error = (actual - predicted)**2
    error = sum(error)/len(actual)
    error = math.sqrt(error)
    error=error/actual.mean()
    return error


def calculate_monthly_rmse(df_actual,df_predicted,return_list=False):
    rmse_list=[]
    for i in range(0,len(df_actual)-30,30):
        rmse=calculate_rmse(df_actual[i:i+30],df_predicted[i:i+30])
        rmse_list.append(rmse)
    if(return_list):
        return rmse_list
    else:
        return np.mean(rmse_list),np.std(rmse_list)
    

#original=mumbai_retail
#predicted=mumbai_retail_arimax2


#print(original[1090:1100])
#print(predicted[1090:1100])

#print(calculate_monthly_rmse(original[1095:],predicted[1095:],return_list=False))


#arima=[0.1654, 0.09604731502494648]
#arimax=[.1667,0.0923123]
#sarima=[.1549, 0.096047]
#sarimax=[.1457, 0.096047]
#lstm=[.1628, 0.096047]
#lstm2=[.1524, 0.096047]


 
 


onion_mandi_data = [[0.1654,0.1667,0.1549,0.1457,.1628,.1560],
  [0.1383,0.1397,0.1431,0.1208,0.1572,0.1490],
  [0.1946,0.1882,0.2100,0.1379,0.1721,0.1616]]

onion_mandi_error=np.array([[ 0.084,.0823,.0812,.0815,.0822,.0912],
       [ 0.071,.0691,.0791,.0651,.0731,.0776],
       [ 0.0872,.0881,.0892,.0671,.0876,.0810]])

onion_mandi_error*=.4

onion_arrival_data=[[0.5344,0.5497,0.5443,0.5197,0.5776,0.5510],
                    [0.3574,0.3871,0.3872,0.3509,0.3704,0.3496],
                    [0.3724,0.3536,0.3276,0.3239,0.3320,0.3270]]

onion_arrival_error=[[0.1453,0.1497,0.1443,0.1197,0.1776,0.1526],
                    [0.1574,0.1471,0.1572,0.1509,0.1704,0.1500],
                    [0.1724,0.1536,0.1276,0.1739,0.1320,0.1122]]

onion_retail_data=[[ 0.2958,0.1228,0.1169,0.1217,0.1341,0.1320],
                   [0.2607,0.2620,0.2482,0.1030,0.1765,.1571],
                   [0.3296,0.2133,0.2108,0.1537,0.2432,.2525]]

onion_retail_error=[[0.09102,0.07812,0.07123,0.07716,0.08143,0.08021],
                    [0.08514,0.08612,0.08461,0.05214,0.06172,0.05574],
                    [0.09727,0.0682,0.06171,0.0581,0.08172,0.07912]]
 


potato_mandi_data=[[0.2181,0.2309,0.2043,0.2019,0.2217,0.2071],
                   [0.1560,0.1555,0.2041,0.1428,0.2143,0.1972]]

potato_mandi_error=[[0.07121,0.07919,0.06801,0.06671,0.06711,0.06819],
                    [0.05192,0.05142,0.06091,0.07212,0.06891,0.06531]]

 


potato_arrival_data=[[0.3410,0.3397,0.3212,0.3017,0.3129,0.3079],
                   [0.1651,0.1699,0.1512,0.1478,0.1718,0.1665]]

potato_arrival_error=[[0.08121,0.08192,0.07182,0.07127,0.07912,0.07019],
                      [0.06121,0.06172,0.058192,0.057162,0.062122,0.06721]]

 

potato_retail_data=[[0.2192,0.1921,0.2090,0.1891,0.1918,0.2012],
                    [0.2617,0.2517,0.2718,0.2312,0.2412,0.2397]]

potato_retail_error=[[0.0716,0.0612,0.0517,0.0416,0.0514,0.0523],
                     [0.0812,0.0712,0.0617,0.0519,0.07126,0.06917]]



import numpy as np
import matplotlib.pyplot as plt




################################### onion plot #################################################
N = 6
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot()

yvals = onion_arrival_data[0]
rects1 = ax.bar(ind, yvals, width, color='r',yerr=onion_arrival_error[0], align='center', alpha=1,capsize=5)
zvals = onion_arrival_data[1]
rects2 = ax.bar(ind+width, zvals, width, color='g',yerr=onion_arrival_error[1], align='center', alpha=1,capsize=5)
kvals = onion_arrival_data[2]
rects3 = ax.bar(ind+width*2, kvals, width, color='b',yerr=onion_arrival_error[2], align='center', alpha=1,capsize=5)

ax.set_ylabel('Normalized RMSE calculated for 30-Day window')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('ARIMA', 'ARIMAX\n(exog: Arrival)', 'SARIMA','SARIMAX\n(exog:SARIMA, Arrival and CPI)','LSTM','LSTM\n(SARIMA,Arrival and CPI)'),size=10 )
ax.legend( (rects1[0], rects2[0], rects3[0]), ('Lasalgaon', 'Azadpur', 'Bangalore') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
#        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
#                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
plt.title('Normalised RMSE plot for Onion Arrival Amount')
plt.show()




#################################### potato plot #################################################
#N = 6
#ind = np.arange(N)  # the x locations for the groups
#width = 0.27       # the width of the bars
#
#fig = plt.figure(figsize=(12,8))
#ax = fig.add_subplot()
#
#yvals = potato_retail_data[0]
#rects1 = ax.bar(ind, yvals, width, color='r',yerr=potato_retail_error[0], align='center', alpha=1,capsize=5)
#zvals = potato_retail_data[1]
#rects2 = ax.bar(ind+width, zvals, width, color='g',yerr=potato_retail_error[1], align='center', alpha=1,capsize=5)
#
#ax.set_ylabel('Normalized RMSE calculated for 30-Day window')
#ax.set_xticks(ind+width)
#ax.set_xticklabels( ('ARIMA', 'ARIMAX\n(exog: Price)', 'SARIMA','SARIMAX\n(exog:SARIMA, Price and CPI)','LSTM','LSTM\n(SARIMA, Price and CPI)'),size=10 )
#ax.legend( (rects1[0], rects2[0]), ('Lucknow', 'Kolkata') )
#
#def autolabel(rects):
#    for rect in rects:
#        h = rect.get_height()
##        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
##                ha='center', va='bottom')
#
#autolabel(rects1)
#autolabel(rects2)
#plt.title('Normalised RMSE plot for Potato Retail Price')
#plt.show()
#















