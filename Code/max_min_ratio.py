#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:30:05 2020

@author: ictd
"""

import pandas as pd
from loadSeries import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
retail_series=pd.read_csv('/home/ictd/nishant/research/Project/Data/original/Raw/mandi.csv',header=None)

from statistics import median


def get_medians(l,percentile_list):
    l.sort()
    result=[]
    for i in percentile_list:
        index=int((i*len(l))/100)
        result.append(l[index])
    return result

def get_retail_series(df,code):
    df=df[df[0]>='2006-01-01']
    df=df[df[2]==code]
    df.drop_duplicates(subset=[0],keep='first',inplace=True)
    df[1]=df[1]*100
    return df

def max_min_all(anomaly_series,price_series):
    anomaly_series.reset_index(inplace=True)
    max_min_ratio_list=[]
    max_min_diff_list=[]
    for i in range(0,len(anomaly_series)):
        start=anomaly_series.iloc[i][0]
        end=anomaly_series.iloc[i][1]
        dx=df[(price_series[0]>=start)& (price_series[0]<=end)]
        dx=dx[dx[2]!=0.0][2]
        if(len(dx)>26):
            x=np.max(dx)/np.min(dx)
            y=np.max(dx)-np.min(dx)
            max_min_ratio_list.append(x)
            max_min_diff_list.append(y)
    #max_min_ratio_list= [5 if x>5 else x for x in max_min_ratio_list]
    print('All events')
    print(sum(max_min_ratio_list)/len(max_min_ratio_list))
    print(sum(max_min_diff_list)/len(max_min_diff_list))
    return max_min_ratio_list,max_min_diff_list

def max_min_anomalous(anomaly_series,price_series):
#    anomaly_series=anomaly_series[anomaly_series[2] != 'no']
    anomaly_series=anomaly_series[anomaly_series[2] != 'no']
    anomaly_series.reset_index(inplace=True)
    max_min_ratio_list=[]
    max_min_diff_list=[]
    for i in range(0,len(anomaly_series)):
        start=anomaly_series.iloc[i][0]
        end=anomaly_series.iloc[i][1]
        dx=df[(price_series[0]>=start)& (price_series[0]<=end)]
        dx=dx[dx[2]!=0.0][2]
        if(len(dx)>26):
            x=np.max(dx)/np.min(dx)
            y=np.max(dx)-np.min(dx)
            max_min_ratio_list.append(x)

            max_min_diff_list.append(y)
            #print(y)
    #max_min_ratio_list= [5 if x>5 else x for x in max_min_ratio_list]
    print('Anomalous')
    print(median(max_min_ratio_list))
    print(median(max_min_diff_list))
#    print(len(max_min_ratio_list))
#    print(get_medians(max_min_ratio_list,[10,20,30,40,50,60,70,80,90]))
    return max_min_ratio_list,max_min_diff_list


def max_min_non_anomalous(anomaly_series,price_series):
#    anomaly_series=anomaly_series[anomaly_series[2] != 'no']
    anomaly_series=anomaly_series[anomaly_series[2] != 'no']
    anomaly_series.reset_index(inplace=True)
    max_min_ratio_list=[]
    max_min_diff_list=[]
    for i in range(0,len(anomaly_series)-1):
        start=anomaly_series.iloc[i][1]
        end=anomaly_series.iloc[i+1][0]
        dx=df[(price_series[0]>=start)& (price_series[0]<=end)]
        #dx=dx[dx[2]!=0.0][2]
        dx=dx[2]
        for j in range(0,len(dx),43):
            dk=dx[j:j+43]
            dk=dk[dk != 0]
            #print(len(dk))
            if(len(dk)>26):
                x=np.max(dk)/np.min(dk)
                y=np.max(dk)-np.min(dk)
                #print(y)
                max_min_ratio_list.append(x)
                max_min_diff_list.append(y)
    #max_min_ratio_list= [5 if x>5 else x for x in max_min_ratio_list]
    print('Non Anomalous')
    print(median(max_min_ratio_list))
    print(median(max_min_diff_list))
#    print(len(max_min_ratio_list))
#    print(get_medians(max_min_ratio_list,[10,20,30,40,50,60,70,80,90]))
    return max_min_ratio_list,max_min_diff_list

def max_min_normal(anomaly_series,price_series):
#    anomaly_series=anomaly_series[anomaly_series[2] == 'no']
    anomaly_series=anomaly_series[anomaly_series[2] == 'no']
    anomaly_series.reset_index(inplace=True)
    max_min_ratio_list=[]
    max_min_diff_list=[]
    for i in range(0,len(anomaly_series)):
        start=anomaly_series.iloc[i][0]
        end=anomaly_series.iloc[i][1]
        dx=df[(price_series[0]>=start)& (price_series[0]<=end)]
        dx=dx[dx[2]!=0.0][2]
        if(len(dx)>26):
            x=np.max(dx)/np.min(dx)
            y=np.max(dx)-np.min(dx)
            max_min_ratio_list.append(x)
            max_min_diff_list.append(y)
    #max_min_ratio_list= [5 if x>5 else x for x in max_min_ratio_list]
    print('Normal')
    print(sum(max_min_ratio_list)/len(max_min_ratio_list))
    print(sum(max_min_diff_list)/len(max_min_diff_list))
    return max_min_ratio_list,max_min_diff_list
    
    
def plot_distribution(list1,list2,list3,list4,title,xlabelname):
    
    mean1=sum(list1)/len(list1)
    mean2=sum(list2)/len(list2)
    mean3=sum(list3)/len(list3)
    mean4=sum(list4)/len(list4)
    
    plt.figure(figsize=(6, 6))
    sns.distplot(list1, hist=True,kde=False,bins=40,color = 'red', label='Anomalous, Mean: '+str(mean1)[:4])
#    sns.distplot(list2, hist=False,kde=True,color = 'green', label='Normal, Mean: '+str(mean2)[:4])
    #sns.distplot(list3,hist=True ,kde=False,color = 'blue', label='Non Anom, Mean: '+str(mean3)[:4],axlabel=xlabelname)
#    sns.distplot(list4,hist=False ,kde=True,color = 'violet', label='all events, Mean: '+str(mean4)[:4],axlabel='Mean value')
#    plt.title('Bangalore ')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title(title)
    plt.show()
#    x, bins, p=plt.hist(max_min_ratio_list_anomalous, density=True)
#    print(x)
#    for item in p:
#        item.set_height(item.get_height()/sum(x))
#    sns.distplot(max_min_ratio_list_anomalous,bins=50,kde=True,hist=True,color='r')
#    sns.distplot(max_min_ratio_list_normal,bins=50,kde=True,hist=True)
#    plt.show()


name='Kolkata'
name1='Low'
df=get_retail_series(retail_series,37)
anomaly_series=kolkata_high_anomaly
#print(df)



max_min_ratio_list_anomalous,max_min_diff_list_anomalous=max_min_anomalous(anomaly_series,df)
max_min_ratio_list_non_anomalous,max_min_diff_list_non_anomalous=max_min_non_anomalous(anomaly_series,df)
#max_min_ratio_list_normal,max_min_diff_list_normal=max_min_normal(anomaly_series,df)
#max_min_ratio_list_all,max_min_diff_list_all=max_min_all(anomaly_series,df)


#plot_distribution(max_min_ratio_list_anomalous,max_min_ratio_list_non_anomalous,max_min_ratio_list_normal,max_min_ratio_list_all,'Probability distribution Max/Min: ' +str(name)+ ' ('+str(name1)+ ' Price)','Max/Min')
#
#plot_distribution(max_min_diff_list_anomalous,max_min_diff_list_non_anomalous,max_min_diff_list_normal,max_min_diff_list_all,'Probability distribution Max-Min: ' +str(name)+  ' ('+str(name1)+ ' Price)','Max-Min')




#
#bins=50
#interval=np.linspace(1,5,bins+1)
#count=np.zeros(len(interval))
#gap=(5-1)/bins
#for i in max_min_ratio_list_normal:
#    #print(i)
#    x=int((i-1)/gap)
#    count[x]+=1
#count=count/count.sum()
#plt.figure(figsize=(6, 6))
#plt.plot(interval,count)
#interval=np.linspace(1,5,bins+1)
#count=np.zeros(len(interval))
#gap=(5-1)/bins
#for i in max_min_ratio_list_anomalous:
#    #print(i)
#    x=int((i-1)/gap)
#    count[x]+=1
#count=count/count.sum()
#
#plt.plot(interval,count,color='red')
#plt.show()






