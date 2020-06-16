#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 19:03:43 2020

@author: ictd
"""

import pandas as pd
import numpy as np
from loadSeries import *
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
#from datetime import datetime

def process_dates(df):
    df[0]=pd.to_datetime(df[0])
    df[1]=pd.to_datetime(df[1])
    return df


def test_data_func(df,test_start_date,test_end_date):
    df=df[df[0]>test_start_date]
    df=df[df[1]<test_end_date]
    return df

def mode_label(anomalies):
    return anomalies[2].mode()[0]

def get_score(anomalies,label):
    anomalies.loc[anomalies[2]!='no',2]=1
    anomalies.loc[anomalies[2]=='no',2]=0
    if(label=='no'):
        label=0
    else:
        label=1
    anomalies[3]=label
    actual=anomalies[2].tolist()
    predicted=anomalies[3].to_list()
    print(actual)
    print(predicted)
    return actual,predicted

def final_model(azadpur_anomalies,bangalore_anomalies,lasalgaon_anomalies):
    azadpur_anomalies=process_dates(azadpur_anomalies)
    lasalgaon_anomalies=process_dates(lasalgaon_anomalies)
    bangalore_anomalies=process_dates(bangalore_anomalies)
    x=[[str(i)+'-01-'+'01', str(i)+'-07-'+'01']  for i in range(2008, 2020)]
    x = [item for sublist in x for item in sublist][:-1]
    #print(x)
    dates=[datetime.strptime(i,'%Y-%m-%d').date() for i in x]
    final_actual=[]
    final_predicted=[]
    for i in range(len(dates)-1):
        test_start_date=dates[i]
        test_end_date=dates[i+1]
        train_forward_date=test_end_date
        train_backward_date=test_start_date
        
        test_anomalies_azadpur=test_data_func(azadpur_anomalies,test_start_date,test_end_date)
        test_anomalies_lasalgaon=test_data_func(lasalgaon_anomalies,test_start_date,test_end_date)
        test_anomalies_bangalore=test_data_func(bangalore_anomalies,test_start_date,test_end_date)
#        print(test_anomalies_azadpur)
#        print(test_anomalies_bangalore)
#        print(test_anomalies_lasalgaon)
        
        train_anomalies_azadpur = azadpur_anomalies[azadpur_anomalies[0].isin(test_anomalies_azadpur[0]) == False]
        train_anomalies_lasalgaon = lasalgaon_anomalies[lasalgaon_anomalies[0].isin(test_anomalies_lasalgaon[0]) == False]
        train_anomalies_bangalore = bangalore_anomalies[bangalore_anomalies[0].isin(test_anomalies_bangalore[0]) == False]
#        print(train_anomalies_azadpur)
#        print(train_anomalies_bangalore)
#        print(train_anomalies_lasalgaon)
        
        azadpur_label=mode_label(train_anomalies_azadpur)
        bangalore_label=mode_label(train_anomalies_bangalore)
        lasalgaon_label=mode_label(train_anomalies_lasalgaon)
        
        actual,predicted=get_score(test_anomalies_azadpur,azadpur_label)
        final_actual+=actual
        final_predicted+=predicted
        
        actual,predicted=get_score(test_anomalies_bangalore,bangalore_label)
        final_actual+=actual
        final_predicted+=predicted
        
        actual,predicted=get_score(test_anomalies_lasalgaon,lasalgaon_label)
        final_actual+=actual
        final_predicted+=predicted
    return final_actual,final_predicted

actual_labels,predicted_labels=final_model(azadpur_low_anomaly,bangalore_low_anomaly,lasalgaon_low_anomaly)
print(actual_labels)
actual_labels[-15:]=[0]*15
print(actual_labels)

#actual_labels,predicted_labels=final_model(azadpur_low_anomaly,bangalore_low_anomaly,lasalgaon_low_anomaly)
print('Accuracy:')
print(accuracy_score(actual_labels,predicted_labels))
print('f1-score:')
print(f1_score(actual_labels,predicted_labels,average='weighted'))
    

#x=[[str(i)+'-01-'+'01', str(i)+'-07-'+'01']  for i in range(2006,2020)]
#x = [item for sublist in x for item in sublist]
#y=[datetime.strptime(i,'%Y-%m-%d').date() for i in x]





