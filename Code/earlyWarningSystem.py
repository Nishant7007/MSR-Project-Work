
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
#
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

azadpur_mandi=Normalise(azadpur_mandi)
lasalgaon_mandi=Normalise(lasalgaon_mandi)
bangalore_mandi=Normalise(bangalore_mandi)

azadpur_arrival=Normalise(azadpur_arrival)
bangalore_arrival=Normalise(bangalore_arrival)
lasalgaon_arrival=Normalise(lasalgaon_arrival)

delhi_retail=Normalise(delhi_retail)
bangalore_retail=Normalise(bangalore_retail)
mumbai_retail=Normalise(mumbai_retail)

azadpur_mandi_arimax2=Normalise(azadpur_mandi_arimax2)
lasalgaon_mandi_arimax2=Normalise(lasalgaon_mandi_arimax2)
bangalore_mandi_arimax2=Normalise(bangalore_mandi_arimax2)


azadpur_arrival_arimax2=Normalise(azadpur_arrival_arimax2)
bangalore_arrival_arimax2=Normalise(bangalore_arrival_arimax2)
lasalgaon_arrival_arimax2=Normalise(lasalgaon_arrival_arimax2)


delhi_retail_arimax2=Normalise(delhi_retail_arimax2)
bangalore_retail_arimax2=Normalise(bangalore_retail_arimax2)
mumbai_retail_arimax2=Normalise(mumbai_retail_arimax2)

def newlabels(df):
    df.loc[df[2]!='no',2]=1
    df.loc[df[2]=='no',2]=0
    #print(df)
    #print(df)
    return df


def process_train_data(anomalies,start_date):
    anomalies=anomalies[anomalies[1]<=start_date]
    return anomalies

def process_test_data(anomalies,start_date,last_date):
    anomalies=anomalies[anomalies[1]>=start_date]
    anomalies=anomalies[anomalies[1]<=last_date]
    return anomalies


def prepare(anomalies,priceserieslist,days):
    x = []
    y=[]
    for i in range(0,len(anomalies)):
        p=[]
        mid_date=datetime.strptime(anomalies[0][i], '%Y-%m-%d')+timedelta(21)
        start_date=mid_date-timedelta(int(days/2))
        end_date=mid_date+timedelta(int(days/2))
        #print(start_date,end_date)
        for j in range(len(priceserieslist)):
            #print(i,j)
            p+=(np.array(priceserieslist[j][start_date:end_date])).tolist()
        x.append(np.array(p))
        y.append(anomalies[2][i])
    return np.array(x),np.array(y)

def prepare_train_data(azadpur_high_anomaly,azadpur_mandi,lasalgaon_high_anomaly,
                       lasalgaon_mandi,bangalore_high_anomaly,bangalore_mandi,days):
    x1,y1=prepare(azadpur_high_anomaly,azadpur_mandi,days)
#    print('x1')
    x2,y2=prepare(lasalgaon_high_anomaly,lasalgaon_mandi,days)
#    print('x2')
    x3,y3=prepare(bangalore_high_anomaly,bangalore_mandi,days)
#    print('x3')
#    print(x1.shape,x2.shape,x3.shape)
    x_all=np.concatenate((x1,x2,x3),axis=0)
    y_all=np.concatenate((y1,y2,y3),axis=0)
    return x_all,y_all

def train_model(x_train,y_train):
#    print('train model')
    #print(y_train)
    n_estimators = [int(x) for x in np.linspace(start = 5, stop = 1000, num = 200)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(2, 400, num =200)]
    max_depth.append(None)
    min_samples_split = [int(x) for x in range(2,200)]
    min_samples_leaf = [int(x) for x in range(2,200)]
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
    
def train_test_function(days,start_date,last_date,train_series1,train_series2,train_series3,
                        test_series1,test_series2,test_series3,anomaly1,anomaly2,anomaly3):

#    print(train_series1[0][-30:])
#    print(test_series1[0][-30:])

    
#    print('Processing train data:')
    train_anomaly1=process_train_data(anomaly1,start_date)
    train_anomaly2=process_train_data(anomaly2,start_date)
    train_anomaly3=process_train_data(anomaly3,start_date)
    
#    print('processing test data:')
    test_anomaly1=process_test_data(anomaly1,start_date,last_date)
    test_anomaly1.index=np.arange(0, len(test_anomaly1))
    test_anomaly2=process_test_data(anomaly2,start_date,last_date)
    test_anomaly2.index=np.arange(0, len(test_anomaly2))
    test_anomaly3=process_test_data(anomaly3,start_date,last_date)
    test_anomaly3.index=np.arange(0, len(test_anomaly3))

#    print('Prepare train data for random forest')
    x_train,y_train=prepare_train_data(train_anomaly1,train_series1,
       train_anomaly2,train_series2,train_anomaly3,train_series3,days)

#    print('Prepare test data for random forest')
    x_test,y_test=prepare_train_data(test_anomaly1,test_series1,
       test_anomaly2,test_series2,test_anomaly3,test_series3,days)
    
        
    
#    print('Running model')
    model=train_model(x_train,y_train)
    pred=model.predict(x_test)
    return y_test,pred
    
 
def final_model(days,azadpur_mandi,bangalore_mandi,lasalgaon_mandi,
                azadpur_mandi_arimax2,bangalore_mandi_arimax2,lasalgaon_mandi_arimax2,
                azadpur_high_anomaly,bangalore_high_anomaly,lasalgaon_high_anomaly):
#    print('new labels as 1/0:')
    #print(str(anomaly1[2][0]))
    azadpur_high_anomaly=newlabels(azadpur_high_anomaly)
    bangalore_high_anomaly=newlabels(bangalore_high_anomaly)
    lasalgaon_high_anomaly=newlabels(lasalgaon_high_anomaly)
#    print(azadpur_high_anomaly)
#    print(bangalore_high_anomaly)
#    print(lasalgaon_high_anomaly)
    dates=['2011-01-01','2012-01-01','2013-01-01','2014-01-01','2015-01-01','2016-01-01','2017-01-01','2018-01-01','2019-01-01','2020-01-01']
    predcited_labels=[]
    actual_labels=[]
    for i in range(len(dates)-1):
        start_date=dates[i]
        last_date=dates[i+1]
        act,pred=train_test_function(days,start_date,last_date,azadpur_mandi,bangalore_mandi,lasalgaon_mandi,
                        azadpur_mandi_arimax2,bangalore_mandi_arimax2,lasalgaon_mandi_arimax2,
                        azadpur_high_anomaly,bangalore_high_anomaly,lasalgaon_high_anomaly)
        #print(act)
        #print(pred)
        predcited_labels.extend(pred)
        actual_labels.extend(act)
    return actual_labels,predcited_labels
        
        

#print('------------------------Mandi Price----------------------------------')     
#actual_labels,predicted_labels=final_model(43,[azadpur_mandi],[bangalore_mandi],[lasalgaon_mandi],
#                        [azadpur_mandi_arimax2],[bangalore_mandi_arimax2],[lasalgaon_mandi_arimax2],
#                        azadpur_high_anomaly,bangalore_high_anomaly,lasalgaon_high_anomaly)


#
#print('-------------------------Arrival-------------------------------------------')
#actual_labels,predicted_labels=final_model(43,[azadpur_arrival],[bangalore_arrival],[lasalgaon_arrival],
#                        [azadpur_arrival_arimax2],[bangalore_arrival_arimax2],[lasalgaon_arrival_arimax2],
#                        azadpur_high_anomaly,bangalore_high_anomaly,lasalgaon_high_anomaly)

#print('-------------------Arrival and Mandi----------------------------')
#actual_labels,predicted_labels=final_model(43,[azadpur_arrival,azadpur_mandi],[bangalore_arrival,bangalore_mandi],[lasalgaon_arrival,lasalgaon_mandi],
#                        [azadpur_arrival_arimax2,azadpur_mandi_arimax2],[bangalore_arrival_arimax2,bangalore_mandi_arimax2],[lasalgaon_arrival_arimax2,lasalgaon_mandi_arimax2],
#                        azadpur_high_anomaly,bangalore_high_anomaly,lasalgaon_high_anomaly)


print('----------------------Retail----------------------------------------------')
actual_labels,predicted_labels=final_model(43,[delhi_retail],[bangalore_retail],[mumbai_retail],
                        [delhi_retail_arimax2],[bangalore_retail_arimax2],[mumbai_retail_arimax2],
                        azadpur_high_anomaly,bangalore_high_anomaly,lasalgaon_high_anomaly)


#print('----------------Mandi and Retail---------------------------------------------')
#actual_labels,predicted_labels=final_model(43,[delhi_retail,azadpur_mandi],[bangalore_retail,bangalore_mandi],[mumbai_retail,lasalgaon_mandi],
#                        [delhi_retail_arimax2,azadpur_mandi_arimax2],[bangalore_retail_arimax2,bangalore_mandi_arimax2],[mumbai_retail_arimax2,lasalgaon_mandi_arimax2],
#                        azadpur_high_anomaly,bangalore_high_anomaly,lasalgaon_high_anomaly)

#print('----------------Retail and arrival---------------------------------------------')
#actual_labels,predicted_labels=final_model(43,[delhi_retail,azadpur_arrival],[bangalore_retail,bangalore_arrival],[mumbai_retail,lasalgaon_arrival],
#                        [delhi_retail_arimax2,azadpur_arrival_arimax2],[bangalore_retail_arimax2,bangalore_arrival_arimax2],[mumbai_retail_arimax2,lasalgaon_arrival_arimax2],
#                        azadpur_high_anomaly,bangalore_high_anomaly,lasalgaon_high_anomaly)

#print('----------------Retail and arrival and mandi---------------------------------------------')
#actual_labels,predicted_labels=final_model(43,[delhi_retail,azadpur_arrival,azadpur_mandi],[bangalore_retail,bangalore_arrival,bangalore_mandi],[mumbai_retail,lasalgaon_arrival,lasalgaon_mandi],
#                        [delhi_retail_arimax2,azadpur_arrival_arimax2,azadpur_mandi_arimax2],[bangalore_retail_arimax2,bangalore_arrival_arimax2,bangalore_mandi_arimax2],[mumbai_retail_arimax2,lasalgaon_arrival_arimax2,lasalgaon_mandi_arimax2],
#                        azadpur_high_anomaly,bangalore_high_anomaly,lasalgaon_high_anomaly)


#print(actual_labels)
#print(accuracy_score(actual_labels,predicted_labels))
#print('final results')
#print(actual_labels)
#print(predicted_labels)
#print('Accuracy:')
print(str(accuracy_score(actual_labels,predicted_labels))[:6],str(f1_score(actual_labels,predicted_labels,average='weighted'))[:6])
#print('f1-score:')
#print(f1_score(actual_labels,predicted_labels,average='weighted'))
#from sklearn.metrics import confusion_matrix as cm
#print(cm(actual_labels,predicted_labels))