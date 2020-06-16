#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:28:15 2020

@author: ictd
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 22:20:49 2020

@author: ictd
"""

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
#    print('anomalies[2].mode()[0]')
#    print(anomalies[2].mode())
    return anomalies[2].mode()[0]

def get_score(anomalies,label):
    #anomalies=anomalies[anomalies[2]!=' Normal_train']
#    print(anomalies)
    #anomalies=anomalies[anomalies[2]==' Hoarding']
#    anomalies.loc[anomalies[2]!=' Weather',2]=1
#    anomalies.loc[anomalies[2]==' Weather',2]=0
#    print(anomalies)
    anomalies['pred']=label
#    print(anomalies)
    actual=anomalies[2].tolist()
    predicted=anomalies['pred'].to_list()
#    print(actual)
#    print(predicted)
#    print(anomalies)
#    print('actual')
#    print(actual)
#    print('predicted')
#    print(predicted)
    return actual,predicted

def final_model(kolkata_anomalies,lucknow_anomalies):
    kolkata_anomalies=process_dates(kolkata_anomalies)
    lucknow_anomalies=process_dates(lucknow_anomalies)
    #bangalore_anomalies=process_dates(bangalore_anomalies)
    x=[[str(i)+'-01-'+'01', str(i)+'-07-'+'01']  for i in range(2006, 2020)]
    x = [item for sublist in x for item in sublist][:-1]
    #print(x)
    dates=[datetime.strptime(i,'%Y-%m-%d').date() for i in x]
    final_actual=[]
    final_predicted=[]
    kolkata_anomalies=kolkata_anomalies[kolkata_anomalies[2]!='no']
    lucknow_anomalies=lucknow_anomalies[lucknow_anomalies[2]!='no']
    #bangalore_anomalies=bangalore_anomalies[bangalore_anomalies[2]!=' Normal_train']
    print(kolkata_anomalies)
    for i in range(len(dates)-1):
        test_start_date=dates[i]
        test_end_date=dates[i+1]
        train_forward_date=test_end_date
        train_backward_date=test_start_date
        
        test_anomalies_kolkata=test_data_func(kolkata_anomalies,test_start_date,test_end_date)
        test_anomalies_lucknow=test_data_func(lucknow_anomalies,test_start_date,test_end_date)
        #test_anomalies_bangalore=test_data_func(bangalore_anomalies,test_start_date,test_end_date)
#        print(test_anomalies_kolkata)
#        print(test_anomalies_bangalore)
#        print(test_anomalies_lucknow)
        
        train_anomalies_kolkata = kolkata_anomalies[kolkata_anomalies[0].isin(test_anomalies_kolkata[0]) == False]
        train_anomalies_lucknow = lucknow_anomalies[lucknow_anomalies[0].isin(test_anomalies_lucknow[0]) == False]
        #train_anomalies_bangalore = bangalore_anomalies[bangalore_anomalies[0].isin(test_anomalies_bangalore[0]) == False]
#        print(train_anomalies_kolkata)
#        print(train_anomalies_bangalore)
#        print(train_anomalies_lucknow)
        
        kolkata_label=mode_label(train_anomalies_kolkata)
        #bangalore_label=mode_label(train_anomalies_bangalore)
        lucknow_label=mode_label(train_anomalies_lucknow)
        
        actual,predicted=get_score(test_anomalies_kolkata,kolkata_label)
        final_actual+=actual
        final_predicted+=predicted
        
#        actual,predicted=get_score(test_anomalies_bangalore,bangalore_label)
#        final_actual+=actual
#        final_predicted+=predicted
        
        actual,predicted=get_score(test_anomalies_lucknow,lucknow_label)
        final_actual+=actual
        final_predicted+=predicted
    return final_actual,final_predicted

actual_labels,predicted_labels=final_model(kolkata_high_anomaly,lucknow_high_anomaly)


#actual_labels,predicted_labels=final_model(kolkata_low_anomaly,bangalore_low_anomaly,lucknow_low_anomaly)
print('Accuracy:')
print(len(actual_labels),len(predicted_labels))
print(actual_labels)
print(predicted_labels)
print(accuracy_score(actual_labels,predicted_labels))
print('f1-score:')
print(f1_score(actual_labels,predicted_labels,average='weighted'))
    

#x=[[str(i)+'-01-'+'01', str(i)+'-07-'+'01']  for i in range(2006,2020)]
#x = [item for sublist in x for item in sublist]
#y=[datetime.strptime(i,'%Y-%m-%d').date() for i in x]







