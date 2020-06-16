#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:07:18 2020

@author: ictd
"""

import pandas as pd
import numpy as np
from loadSeries import *
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

train_start_date='2006-01-01'


def get_train_data(anomalies,start_date,end_date):
    anomalies=anomalies[anomalies[0]>start_date]
    anomalies=anomalies[anomalies[1]<end_date]
    return anomalies

def get_test_data(anomalies,start_date,end_date):
    anomalies=anomalies[anomalies[0]>start_date]
    anomalies=anomalies[anomalies[1]<end_date]
    return anomalies

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

def train_test_function(azadpur_anomalies,bangalore_anomalies,lasalgaon_anomalies,train_end_date,test_end_date):
    
    final_actual=[]
    final_predicted=[]
    
    azadpur_anomalies_train=get_train_data(azadpur_anomalies,train_start_date,train_end_date)
    bangalore_anomalies_train=get_train_data(bangalore_anomalies,train_start_date,train_end_date)
    lasalgaon_anomalies_train=get_train_data(lasalgaon_anomalies,train_start_date,train_end_date)
    
    azadpur_anomalies_test=get_test_data(azadpur_anomalies,train_end_date,test_end_date)
    bangalore_anomalies_test=get_test_data(bangalore_anomalies,train_end_date,test_end_date)
    lasalgaon_anomalies_test=get_test_data(lasalgaon_anomalies,train_end_date,test_end_date)

    
    azadpur_label=mode_label(azadpur_anomalies_train)
    bangalore_label=mode_label(bangalore_anomalies_train)
    lasalgaon_label=mode_label(azadpur_anomalies_train)
    
    
    actual,predicted=get_score(azadpur_anomalies_test,azadpur_label)
    final_actual+=actual
    final_predicted+=predicted
    
    actual,predicted=get_score(bangalore_anomalies_test,bangalore_label)
    final_actual+=actual
    final_predicted+=predicted
    
    actual,predicted=get_score(lasalgaon_anomalies_test,lasalgaon_label)
    final_actual+=actual
    final_predicted+=predicted
    return final_actual,final_predicted




def final_model(azadpur_anomalies,bangalore_anomalies,lasalgaon_anomalies):
    dates=['2011-01-01','2012-01-01','2013-01-01','2014-01-01','2015-01-01','2016-01-01','2017-01-01','2018-01-01','2019-01-01','2020-01-01']
    predcited_labels=[]
    actual_labels=[]
    for i in range(len(dates)-1):
        train_end_date=dates[i]
        test_end_date=dates[i+1]
        act,pred=train_test_function(azadpur_anomalies,bangalore_anomalies,lasalgaon_anomalies,train_end_date,test_end_date)
        predcited_labels.extend(pred)
        actual_labels.extend(act)
        print(len(predcited_labels),len(actual_labels))
    return actual_labels,predcited_labels


actual_labels,predicted_labels=final_model(azadpur_high_anomaly,bangalore_high_anomaly,lasalgaon_high_anomaly)



    
#actual_labels,predicted_labels=train_test_function(azadpur_low_anomaly,bangalore_low_anomaly,lasalgaon_low_anomaly)
print('Accuracy:')
print(accuracy_score(actual_labels,predicted_labels))
print('f1-score:')
print(f1_score(actual_labels,predicted_labels,average='weighted'))

#
#actual_labels,predicted_labels=train_test_function(azadpur_high_anomaly,bangalore_high_anomaly,lasalgaon_high_anomaly)
#print('Accuracy:')
#print(accuracy_score(actual_labels,predicted_labels))
#print('f1-score:')
#print(f1_score(actual_labels,predicted_labels,average='weighted'))

