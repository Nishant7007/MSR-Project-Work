#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:10:16 2020

@author: ictd
"""

from loadSeries import *
import numpy as np
import pandas as pd
from scipy.stats import variation
import matplotlib.pyplot as plt
from datetime import timedelta


def get_retail_series(df,code):
    df=df[df[0]>='2006-01-01']
    df=df[df[1]==code]
    df.drop_duplicates(subset=[0],keep='first',inplace=True)
#    df[1]=df[1]*100
    return df

def find_max_missing(dx):
    c_m_d = 0
    max_m_d = 0
    for i in range(len(dx)):
        if dx.iloc[i][2] == 0:
            c_m_d+=1
        else:
            max_m_d = max(max_m_d, c_m_d)
            c_m_d = 0
#    print("max mising: ",max_m_d)
    return max_m_d    

def remove_anomalous(price_series,anomaly_series):
    ratio_list=[]
    for i in range(len(anomaly_series)):
#        print(i)
        start=anomaly_series.iloc[i][0]
        end=anomaly_series.iloc[i][1]
        dx=price_series[(price_series[0]>start) & (price_series[0]<end)]
        dx=dx[dx[2]!=0.0]
        if(len(dx[dx[2]!=0.0])>26):
            ratio=dx[2].max()/dx[2].min()
#            print("ratio: ",ratio)
            ratio_list.append(ratio)
#    anomaly_series=anomaly_series[(anomaly_series[2]=='hoarding') | (anomaly_series[2]=='weather')]
    anomaly_series=anomaly_series[anomaly_series[2]!='no']
    print(len(anomaly_series))
    final_anomalies=[]
    c=0
    for i in range(len(anomaly_series)):
        start=anomaly_series.iloc[i][0]
        end=anomaly_series.iloc[i][1]
        dx=price_series[(price_series[0]>start) & (price_series[0]<end)]
#        print(dx)
#        dx=dx[dx[1]!=0.0]
#        print(dx)
        if(len(dx[dx[2]!=0.0])>26):
#            dx=dx[dx[2]!=0.0]
#            ratio_list.append(dx[2].max()/dx[2].min())
#            print(len(dx))
            c+=1
            final_anomalies.append([start,end,anomaly_series.iloc[i][2]])
        else:
            max_m_d=find_max_missing(dx)
            if(max_m_d<7):
                c+=1
                final_anomalies.append([start,end,anomaly_series.iloc[i][2]])
    print("c: ",c)
    print("final anomalies: ",len(final_anomalies))
    return pd.DataFrame(final_anomalies),np.median(np.array(ratio_list))
#    pd.DataFrame(final_anomalies).to_csv('/home/ictd/nishant/research/Project/Data/Anomaly/new anomalies/mumbai_low_anomalies.csv',header=None,index=False)





def mark_normal(price_series,anomaly_series):
    print('mark normal anomalies')
    print(len(anomaly_series))
    anomaly_series=anomaly_series.append({0:'2020-01-01',1:'2020-01-02',2:'supply'},ignore_index=True)
#    print(len(anomaly_series))
    anomaly_series[0]=pd.to_datetime(anomaly_series[0])
    anomaly_series[1]=pd.to_datetime(anomaly_series[1])
    price_series[0]=pd.to_datetime(price_series[0])
    initial=pd.to_datetime('2006-01-01')
    final=pd.to_datetime('2019-12-31')
    start=initial
    end=initial+timedelta(43)
    c=0
    l=[]
    l.append([anomaly_series.iloc[0][0].strftime("%Y-%m-%d"),anomaly_series.iloc[0][1].strftime("%Y-%m-%d"),anomaly_series.iloc[0][2]])
    for i in range(1,len(anomaly_series)):
        s=anomaly_series.iloc[i-1][1]+timedelta(1)
        e=anomaly_series.iloc[i][0]-timedelta(1)
        if((e-s).days<43):
            l.append([anomaly_series.iloc[i][0].strftime("%Y-%m-%d"),anomaly_series.iloc[i][1].strftime("%Y-%m-%d"),anomaly_series.iloc[i][2]])
            continue
        e1=s+timedelta(43)
        while(e1<e):
#            print(s,e1)
            dx=price_series[(price_series[0]>=s) & (price_series[0]<=e1)]
#            dx=dx[dx[2]!=0.0]
            max_m_d=find_max_missing(dx)
#            print('max missing:', max_m_d)
#            print('length of dx:',len(dx))
            if(len(dx[dx[2]!=0.0])>13):
#                print('days more than 26 started')
                dx=dx[dx[2]!=0.0]
                ratio=dx[2].max()/dx[2].min()
#                print(dx[2].max(),dx[2].min())
#                print(dx[2].idxmax())
#                print("ratio:",ratio)
                if(ratio<=1.4):
                    c+=1
                    l.append([s.strftime("%Y-%m-%d"),e1.strftime("%Y-%m-%d"),'no'])
                    e1=e1+timedelta(43)
                    s=s+timedelta(43)
#                    l.append([s.strftime("%Y-%m-%d"),e1.strftime("%Y-%m-%d"),'no'])
                else:
                    e1=e1+timedelta(1)
                    s=s+timedelta(1)
            elif(max_m_d<=16):
#                print('missing days less than 7 started')
                dx=dx[dx[2]!=0.0]
                ratio=dx[2].max()/dx[2].min()
#                print("ratio:",ratio)
                if(ratio<=1.4):
                    c+=1
                    e1=e1+timedelta(43)
                    s=s+timedelta(43)
                    l.append([s.strftime("%Y-%m-%d"),e1.strftime("%Y-%m-%d"),'no'])
#                    l.append([anomaly_series.iloc[i][0].strftime("%Y-%m-%d"),anomaly_series.iloc[i][1].strftime("%Y-%m-%d"),anomaly_series.iloc[i][2]])
                else:
                    e1=e1+timedelta(1)
                    s=s+timedelta(1)
#                    l.append([anomaly_series.iloc[i][0].strftime("%Y-%m-%d"),anomaly_series.iloc[i][1].strftime("%Y-%m-%d"),anomaly_series.iloc[i][2]])
#                print('missing days less than 7 ended')

            else:
                e1=e+timedelta(1)
                s=s+timedelta(1)
        l.append([anomaly_series.iloc[i][0].strftime("%Y-%m-%d"),anomaly_series.iloc[i][1].strftime("%Y-%m-%d"),anomaly_series.iloc[i][2]])
#    print(c)
#    print(l)
    data=pd.DataFrame(l[:-1])
#    print(data)
    print(len(data))
    data1=data[data[2]=='no']
    print(len(data1))
    #print(len(data))
    data.to_csv('/home/ictd/nishant/research/Project/Data/Anomaly/Dummy/262020/'+str(city)+str(anomaly_type)+'anomalies.csv',header=None,index=False)
#
    
retail_series=pd.read_csv('/home/ictd/nishant/research/Project/Data/original/Raw/mandi_data.csv',header=None)
 
city_code = {'kolkata': 37, 'lucknow': 40, 'lasalgaon': 44, 'bangalore': 7, 'azadpur': 16}
city = 'kolkata'
code = city_code[city]
anomaly_type='high'
          
df=get_retail_series(retail_series,code)
#df=retail_series
anomaly_series=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/oldanomalies/normal_h_w_kolkata11.csv',header=None)
   
#mark_normal(df,anomaly_series)
  
    
#df=get_retail_series(retail_series,37)
#anomaly_series=kolkata_high_anomaly

anomalous,rl=remove_anomalous(df,anomaly_series)
print('median of ratio: ',rl)
#print(anomalous)
mark_normal(df,anomalous)
#print(rl)



#df=azadpur_high_anomaly
#
#df=df[df[2]!='no']
#
#df.to_csv('/home/ictd/nishant/research/Project/Data/Anomaly/Dummy/Anomalous/azadpur_high_anomaly.csv',header=None,index=False)





#            l.append([anomaly_series.iloc[i][0].strftime("%Y-%m-%d"),anomaly_series.iloc[i][1].strftime("%Y-%m-%d"),anomaly_series.iloc[i][2]])

#                print('days more than 26 ended')
#            elif(max_m_d<7):
#                print('missing days less than 7 started')
#                print('max missing:', max_m_d)
#                dx=dx[dx[2]!=0.0]
#                ratio=dx[2].max()/dx[2].min()
##                print("ratio:",ratio)
#                if(ratio<=2.5):
#                    c+=1
#                    e1=e1+timedelta(43)
#                    s=s+timedelta(43)
#                    l.append([s.strftime("%Y-%m-%d"),e1.strftime("%Y-%m-%d"),'no'])
#                    l.append([anomaly_series.iloc[i][0].strftime("%Y-%m-%d"),anomaly_series.iloc[i][1].strftime("%Y-%m-%d"),anomaly_series.iloc[i][2]])
#                else:
#                    e1=e1+timedelta(1)
#                    s=s+timedelta(1)
#                    l.append([anomaly_series.iloc[i][0].strftime("%Y-%m-%d"),anomaly_series.iloc[i][1].strftime("%Y-%m-%d"),anomaly_series.iloc[i][2]])
#                print('missing days less than 7 ended')
                






