#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:24:54 2020

@author: ictd
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:25:03 2020

@author: ictd
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 21:40:08 2020

@author: ictd
"""


"""
Created on Wed Jan 22 15:44:28 2020

@author: ictd
"""

import pandas as pd
import numpy as np
from loadSeries import *
from datetime import timedelta
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import sklearn
#np.random.seed(42)

n_estimators = [int(x) for x in np.linspace(start = 2, stop = 10, num = 5)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(2, 8, num =4)]
max_depth.append(None)
min_samples_split = [int(x) for x in range(2,4)]
min_samples_leaf = [int(x) for x in range(2,4)]
bootstrap = [True,False]
#oob_score=[True,False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf = RandomForestClassifier()

#azadpur_mandi_arimax2.index=pd.to_datetime(azadpur_mandi_arimax2[0])
#azadpur_mandi_arimax2=azadpur_mandi_arimax2[1]
#
#bangalore_mandi_arimax2.index=pd.to_datetime(bangalore_mandi_arimax2[0])
#bangalore_mandi_arimax2=bangalore_mandi_arimax2[1]
#
#lasalgaon_mandi_arimax2.index=pd.to_datetime(lasalgaon_mandi_arimax2[0])
#lasalgaon_mandi_arimax2=lasalgaon_mandi_arimax2[1]
#
#azadpur_arrival_arimax2.index=pd.to_datetime(azadpur_arrival_arimax2[0])
#azadpur_arrival_arimax2=azadpur_arrival_arimax2[1]
#
#bangalore_arrival_arimax2.index=pd.to_datetime(bangalore_arrival_arimax2[0])
#bangalore_arrival_arimax2=bangalore_arrival_arimax2[1]
#
#lasalgaon_arrival_arimax2.index=pd.to_datetime(lasalgaon_arrival_arimax2[0])
#lasalgaon_arrival_arimax2=lasalgaon_arrival_arimax2[1]

#delhi_retail_arimax2.index=pd.to_datetime(delhi_retail_arimax2[0])
#delhi_retail_arimax2=delhi_retail_arimax2[1]

#bangalore_retail_arimax2.index=pd.to_datetime(bangalore_retail_arimax2[0])
#bangalore_retail_arimax2=bangalore_retail_arimax2[1]
#
#mumbai_retail_arimax2.index=pd.to_datetime(mumbai_retail_arimax2[0])
#mumbai_retail_arimax2=mumbai_retail_arimax2[1]


def Normalise(arr):
  '''
  Normalise each sample
  '''
  m = arr.mean()
  am = arr.min()
  aM = arr.max()
  arr -= m
  arr /= (aM - am)
  return arr


kalyani_mandi=Normalise(kalyani_mandi)
lucknow_mandi=Normalise(kalyani_mandi)

kalyani_arrival=Normalise(kalyani_arrival)
lucknow_arrival=Normalise(lucknow_arrival)

kolkata_retail=Normalise(kolkata_retail)
lucknow_retail=Normalise(lucknow_retail)

kalyani_mandi_arimax2=Normalise(kalyani_mandi_arimax2)
lucknow_mandi_arimax2=Normalise(kalyani_mandi_arimax2)

kalyani_arrival_arimax2=Normalise(kalyani_arrival_arimax2)
lucknow_arrival_arimax2=Normalise(lucknow_arrival_arimax2)

kolkata_retail_arimax2=Normalise(kolkata_retail_arimax2)
lucknow_retail_arimax2=Normalise(lucknow_retail_arimax2)


def newlabels(df):
    #print(df)
    df.loc[df[2]!='no',2]=1
    df.loc[df[2]=='no',2]=0
    #print(df)
    return df


def process_train_data(anomalies,start_date):
    anomalies=anomalies[anomalies[1]<=start_date]
    return anomalies

def process_test_data(anomalies,start_date,last_date):
    anomalies=anomalies[anomalies[1]>=start_date]
    anomalies=anomalies[anomalies[1]<=last_date]
    return anomalies


def prepare(anomalies,priceserieslist,forecastedpriceserieslist,days):
    x = []
    y=[]
    anomalies.reset_index(inplace=True)
    for i in range(0,len(anomalies)):
        p=[]
        mid_date=datetime.strptime(anomalies[0][i], '%Y-%m-%d')+timedelta(21)
        start_date=mid_date-timedelta(int(days/2))
        end_date=mid_date+timedelta(int(days/2))
#        #print(start_date)
#        #print(end_date)
        for j in range(len(priceserieslist)):
            ##print(j)
            current=(np.array(priceserieslist[j][start_date:end_date])).tolist()
            previous=(np.array(forecastedpriceserieslist[j][start_date-timedelta(21):start_date])).tolist()
            if(len(previous)<21):
                k=21-len(previous)+1
                temp=[0]*k
                previous=temp+previous
#            #print('len of current:',len(current))
#            #print('len of previous:',len(previous))
            previous+=current
#            #print('len of previous11:',len(previous))
            p+=previous
        #for j in range(len(forecastedpriceserieslist)):
            #p+=(np.array(forecastedpriceserieslist[j][start_date-timedelta(21):start_date])).tolist()
            ##print(np.array(p).shape))
        x.append(np.array(p))
        ##print()
        y.append(anomalies[2][i])
    ##print(np.array(x).shape)
    return np.array(x),np.array(y)

def prepare_train_data(azadpur_low_anomaly,azadpur_mandi,azadpur_mandi_residual,lucknow_low_anomaly,
                       lasalgaon_mandi,lasalgaon_mandi_residual,days):
    x1,y1=prepare(azadpur_low_anomaly,azadpur_mandi,azadpur_mandi_residual,days)
    #print(x1.shape)
    x2,y2=prepare(lucknow_low_anomaly,lasalgaon_mandi,lasalgaon_mandi_residual,days)
    #print(x2.shape)
#    x3,y3=prepare(bangalore_low_anomaly,bangalore_mandi,bangalore_mandi_residual,days)
#    #print(x3.shape)
    
#    #print(x1.shape,x2.shape,x3.shape)
    
    l_c=[0,0]
    l_x=[x1,x2]
    l_y=[y1,y2]
#    print(x1.shape,x2.shape,x3.shape)
    for i in range(len(l_c)):
        if(len(l_x[i])>0):
            l_c[i]=1
#    print(l_c)
    i=0
    while(i<len(l_c)):
        if(l_c[i]==1):
            x_all=l_x[i]
            y_all=l_y[i]
            i+=1
            break
    while(i<len(l_c)):
        if(l_c[i]==1):
            x_all=np.concatenate((x_all,l_x[i]),axis=0)
            y_all=np.concatenate((y_all,l_y[i]),axis=0)
        i+=1
#    x_all=np.concatenate((x1,x2),axis=0)
#    y_all=np.concatenate((y1,y2),axis=0)
    return x_all,y_all

def train_model(x_train,y_train):
#    #print('train model')
    ##print(y_train)
    n_estimators = [int(x) for x in np.linspace(start = 5, stop = 100, num = 20)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(2, 5, num =2)]
    max_depth.append(None)
    min_samples_split = [int(x) for x in range(2,10)]
    min_samples_leaf = [int(x) for x in range(2,10)]
    bootstrap = [True,False]
    #oob_score=[True,False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
#    rf = RandomForestClassifier()
#    model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=0,n_jobs=-1)# Fit the random search model
    model = RandomForestClassifier(max_depth=6, random_state=0,n_estimators=20)
    model.fit(x_train, y_train)    
    return model
    
def train_test_function(days,start_date,last_date,train_series1,train_series2,
                        test_series1,test_series2,residual1,residual2,
                        anomaly1,anomaly2):

#    #print(train_series1[0][-30:])
#    #print(test_series1[0][-30:])

    
    #print('Processing train data:')
    train_anomaly1=process_train_data(anomaly1,start_date)
    train_anomaly2=process_train_data(anomaly2,start_date)
    #train_anomaly3=process_train_data(anomaly3,start_date)
    
    #print('processing test data:')
    test_anomaly1=process_test_data(anomaly1,start_date,last_date)
    test_anomaly1.index=np.arange(0, len(test_anomaly1))
    test_anomaly2=process_test_data(anomaly2,start_date,last_date)
    test_anomaly2.index=np.arange(0, len(test_anomaly2))
#    test_anomaly3=process_test_data(anomaly3,start_date,last_date)
#    test_anomaly3.index=np.arange(0, len(test_anomaly3))

    #print('Prepare train data for random forest')
    x_train,y_train=prepare_train_data(train_anomaly1,train_series1,residual1,
       train_anomaly2,train_series2,residual2,days)

    #print('Prepare test data for random forest')
    x_test,y_test=prepare_train_data(test_anomaly1,test_series1,residual1,
       test_anomaly2,test_series2,residual2,days)
    
        
    
    #print('Running model')
    model=train_model(x_train,y_train)
    pred=model.predict(x_test)
    ##print(x_test)
#    #print(y_test)
#    #print(pred)
#    #print("F1-score")
#    #print(f1_score(y_test,pred,average='weighted'))
#    #print("Accuracy")
#    #print(accuracy_score(y_test,pred))
    ##print(model.predict_proba(y_test))
    return y_test,pred
    
 
def final_model(days,azadpur_mandi,lasalgaon_mandi,
                azadpur_mandi_arimax2,lasalgaon_mandi_arimax2,
                azadpur_residual,lasalgaon_residual,
                azadpur_low_anomaly,lucknow_low_anomaly):
    #print('new labels as 1/0:')
    ##print(str(anomaly1[2][0]))
    #print(azadpur_low_anomaly)
    azadpur_low_anomaly=newlabels(azadpur_low_anomaly)
    #print(azadpur_low_anomaly)
    #print('labelling done')
    #bangalore_low_anomaly=newlabels(bangalore_low_anomaly)
    lucknow_low_anomaly=newlabels(lucknow_low_anomaly)
#    #print(azadpur_low_anomaly)
#    #print(bangalore_low_anomaly)
#    #print(lucknow_low_anomaly)
    dates=['2011-01-01','2012-01-01','2013-01-01','2014-01-01','2015-01-01','2016-01-01','2017-01-01','2018-01-01','2019-01-01']
    predcited_labels=[]
    actual_labels=[]
    for i in range(len(dates)-1):
        start_date=dates[i]
        last_date=dates[i+1]
        act,pred=train_test_function(days,start_date,last_date,azadpur_mandi,lasalgaon_mandi,
                        azadpur_mandi_arimax2,lasalgaon_mandi_arimax2,
                        azadpur_residual,lasalgaon_residual,
                        azadpur_low_anomaly,lucknow_low_anomaly)
        ##print(act)
        ##print(pred)
        predcited_labels.extend(pred)
        actual_labels.extend(act)
    return actual_labels,predcited_labels
        

#lucknow_low_anomaly=lucknow_low_anomaly
#bangalore_low_anomaly=kolkata_low_anomaly

azadpur_mandi=kalyani_mandi
lasalgaon_mandi=lucknow_mandi

azadpur_arrival=kalyani_arrival
lasalgaon_arrival=lucknow_arrival

delhi_retail=kolkata_retail
mumbai_retail=lucknow_retail

azadpur_mandi_arimax2=kalyani_mandi_arimax2
lasalgaon_mandi_arimax2=lucknow_mandi_arimax2

azadpur_arrival_arimax2=kalyani_arrival_arimax2
lasalgaon_arrival_arimax2=lucknow_arrival_arimax2

delhi_retail_arimax2=kolkata_retail_arimax2
mumbai_retail_arimax2=lucknow_retail_arimax2

##print(kolkata_low_anomaly)
# 

days=43
#print('------------------------Mandi Price----------------------------------')     
#actual_labels,predicted_labels=final_model(days,[azadpur_mandi],[lasalgaon_mandi],
#                        [azadpur_mandi_arimax2],[lasalgaon_mandi_arimax2],
#                        [azadpur_mandi-azadpur_mandi_arimax2],
#                        [lasalgaon_mandi-lasalgaon_mandi_arimax2],kolkata_low_anomaly,
#                        lucknow_low_anomaly)




#print('-------------------------Arrival-------------------------------------------')
#actual_labels,predicted_labels=final_model(days,[azadpur_arrival],[lasalgaon_arrival],
#                        [azadpur_arrival_arimax2],[lasalgaon_arrival_arimax2],
#                        [azadpur_arrival-azadpur_arrival_arimax2],[lasalgaon_arrival-lasalgaon_arrival_arimax2],
#                        azadpur_low_anomaly,lucknow_low_anomaly)

#
#
##
#print('-------------------Arrival and Mandi----------------------------')
#actual_labels,predicted_labels=final_model(days,[azadpur_arrival,azadpur_mandi],[lasalgaon_arrival,lasalgaon_mandi],
#                        [azadpur_arrival_arimax2,azadpur_mandi_arimax2],[lasalgaon_arrival_arimax2,lasalgaon_mandi_arimax2],
#                        [azadpur_mandi-azadpur_mandi_arimax2,azadpur_arrival-azadpur_arrival_arimax2],
#                        [lasalgaon_mandi-lasalgaon_mandi_arimax2,lasalgaon_arrival-lasalgaon_arrival_arimax2],
#                        azadpur_low_anomaly,lucknow_low_anomaly)


#print('----------------------Retail----------------------------------------------')
#actual_labels,predicted_labels=final_model(days,[delhi_retail],[mumbai_retail],
#                        [delhi_retail_arimax2],[mumbai_retail_arimax2],
#                        [delhi_retail-delhi_retail_arimax2],
#                        [mumbai_retail-mumbai_retail_arimax2],
#                        azadpur_low_anomaly,lucknow_low_anomaly)


#
#print('----------------Mandi and Retail---------------------------------------------')
#actual_labels,predicted_labels=final_model(days,[delhi_retail,azadpur_mandi],[mumbai_retail,lasalgaon_mandi],
#                        [delhi_retail_arimax2,azadpur_mandi_arimax2],[mumbai_retail_arimax2,lasalgaon_mandi_arimax2],
#                        [azadpur_mandi-azadpur_mandi_arimax2,delhi_retail-delhi_retail_arimax2],
#                        [lasalgaon_mandi-lasalgaon_mandi_arimax2,mumbai_retail-mumbai_retail_arimax2],
#                        azadpur_low_anomaly,lucknow_low_anomaly)

#
#print('----------------arrival and Retail---------------------------------------------')
#actual_labels,predicted_labels=final_model(days,[delhi_retail,azadpur_arrival],[mumbai_retail,lasalgaon_arrival],
#                        [delhi_retail_arimax2,azadpur_arrival_arimax2],[mumbai_retail_arimax2,lasalgaon_arrival_arimax2],
#                        [azadpur_arrival-azadpur_arrival_arimax2,delhi_retail-delhi_retail_arimax2],
#                        [lasalgaon_arrival-lasalgaon_arrival_arimax2,mumbai_retail-mumbai_retail_arimax2],
#                        azadpur_low_anomaly,lucknow_low_anomaly)

print('----------------arrival and Retail and mandi---------------------------------------------')
actual_labels,predicted_labels=final_model(days,[delhi_retail,azadpur_arrival,azadpur_mandi],[mumbai_retail,lasalgaon_arrival,lasalgaon_mandi],
                        [delhi_retail_arimax2,azadpur_arrival_arimax2,azadpur_mandi_arimax2],[mumbai_retail_arimax2,lasalgaon_arrival_arimax2,lasalgaon_mandi_arimax2],
                        [azadpur_arrival-azadpur_arrival_arimax2,delhi_retail-delhi_retail_arimax2,azadpur_mandi-azadpur_mandi_arimax2],
                        [lasalgaon_arrival-lasalgaon_arrival_arimax2,mumbai_retail-mumbai_retail_arimax2,lasalgaon_mandi-lasalgaon_mandi_arimax2],
                        azadpur_low_anomaly,lucknow_low_anomaly)




print(str(accuracy_score(actual_labels,predicted_labels))[:6],str(f1_score(actual_labels,predicted_labels,average='weighted'))[:6])



#print('final results')
###print(actual_labels)
###print(predicted_labels)
#print('Accuracy:')
#print(accuracy_score(actual_labels,predicted_labels))
#print('f1-score:')
#print(f1_score(actual_labels,predicted_labels,average='weighted'))
#from sklearn.metrics import confusion_matrix as cm
#print(cm(actual_labels,predicted_labels))