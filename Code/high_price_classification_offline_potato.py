#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 01:19:58 2020

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
from sklearn.metrics import confusion_matrix

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

#kalyani_mandi_arimax2.index=pd.to_datetime(kalyani_mandi_arimax2[0])
#kalyani_mandi_arimax2=kalyani_mandi_arimax2[1]
#
#bangalore_mandi_arimax2.index=pd.to_datetime(bangalore_mandi_arimax2[0])
#bangalore_mandi_arimax2=bangalore_mandi_arimax2[1]
#
#lucknow_mandi_arimax2.index=pd.to_datetime(lucknow_mandi_arimax2[0])
#lucknow_mandi_arimax2=lucknow_mandi_arimax2[1]
#
#kalyani_arrival_arimax2.index=pd.to_datetime(kalyani_arrival_arimax2[0])
#kalyani_arrival_arimax2=kalyani_arrival_arimax2[1]
#
#bangalore_arrival_arimax2.index=pd.to_datetime(bangalore_arrival_arimax2[0])
#bangalore_arrival_arimax2=bangalore_arrival_arimax2[1]
#
#lucknow_arrival_arimax2.index=pd.to_datetime(lucknow_arrival_arimax2[0])
#lucknow_arrival_arimax2=lucknow_arrival_arimax2[1]


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
lucknow_mandi=Normalise(lucknow_mandi)
bangalore_mandi=Normalise(bangalore_mandi)

kalyani_arrival=Normalise(kalyani_arrival)
bangalore_arrival=Normalise(bangalore_arrival)
lucknow_arrival=Normalise(lucknow_arrival)

kolkata_retail=Normalise(kolkata_retail)
bangalore_retail=Normalise(bangalore_retail)
lucknow_retail=Normalise(lucknow_retail)

kalyani_mandi_arimax2=Normalise(kalyani_mandi_arimax2)
lucknow_mandi_arimax2=Normalise(lucknow_mandi_arimax2)
bangalore_mandi_arimax2=Normalise(bangalore_mandi_arimax2)


kalyani_arrival_arimax2=Normalise(kalyani_arrival_arimax2)
bangalore_arrival_arimax2=Normalise(bangalore_arrival_arimax2)
lucknow_arrival_arimax2=Normalise(lucknow_arrival_arimax2)



def adjust_anomaly_window(anomalies,series):
    anomalies_copy = anomalies
    for i in range(0,len(anomalies_copy)):
#        print(anomalies_copy.iloc[i])
        anomaly_period = series[anomalies_copy[0][i]:anomalies_copy[1][i]]
        mid_date_index = anomaly_period[10:31].idxmax()
#        print(type(anomalies_copy[0][i]))
        anomalies_copy[0][i] = mid_date_index - timedelta(days=21)
        anomalies_copy[1][i] = mid_date_index + timedelta(days=21)
        anomalies_copy[0][i] = datetime.strftime(anomalies_copy[0][i],'%Y-%m-%d')
        anomalies_copy[1][i] = datetime.strftime(anomalies_copy[1][i],'%Y-%m-%d')
    return anomalies_copy

def get_anomalies(path,series):
	anomalies = pd.read_csv(path, header=None, index_col=None)
	#print(path)
	#print(anomalies)
	anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[0]]
	anomalies[1] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[1]]
	anomalies = adjust_anomaly_window(anomalies,series)
	return anomalies

def get_anomalies_year(anomalies):
	mid_date_labels=[]
	for i in range(0,len(anomalies[0])):
		mid_date_labels.append(datetime.strftime(datetime.strptime(anomalies[0][i],'%Y-%m-%d')+timedelta(days=21),'%Y-%m-%d'))
	return mid_date_labels



def newlabels(anomalies):
  # print len(anomalies[anomalies[2] != ' Normal_train']), len(oldlabels)
  labels = []
  #k=0
  #print('anomalies')
  #print(anomalies)
  an=anomalies[anomalies[2]!='no']
  an.reset_index(inplace=True)
  for i in range(0,len(an)):
    if(an[2][i] != 'hoarding'):
      labels.append(0)
      #print k,oldlabels[k]
      #k = k+1
    else:
      labels.append(1)
  #print('new labels are:')
  #print(labels)
  return labels




def prepare(anomalies,labels,priceserieslist):
    x = []
    for i in range(0,len(anomalies)):
        p=[]
        for j in range(0,len(priceserieslist)):
            p += ((np.array(priceserieslist[j][anomalies[0][i]:anomalies[1][i]].tolist()))).tolist()
            #print(len(p))
        x.append(np.array(p))
		#print(p)
	#print(pd.DataFrame(x).shape)
    return np.array(x),np.array(labels)


def getKey(item):
	return item[0]

def partition(xseries,yseries,year,months):
	# min_month = datetime.strptime(min(year),'%Y-%m-%d')
	# max_month = datetime.strptime(max(year),'%Y-%m-%d')
	combined_series = zip(year,xseries,yseries)
	combined_series = sorted(combined_series,key=getKey)
	train = []
	train_labels = []
	fixed = datetime.strptime('2006-01-01','%Y-%m-%d')
	i=0
	while(fixed < datetime.strptime('2019-12-31','%Y-%m-%d')):
		currx=[]
		curry=[]
		for anomaly in combined_series:
			i += 1
			if(datetime.strptime(anomaly[0],'%Y-%m-%d') > fixed and datetime.strptime(anomaly[0],'%Y-%m-%d')- fixed <= timedelta(days=months*30)):
				currx.append(anomaly[1])
				curry.append(anomaly[2])
		train.append(currx)
		train_labels.append(curry)
		fixed = fixed +timedelta(days = months*30)
	# print fixed
	# train.append(currx)
	# train_labels.append(curry)
	return np.array(train),np.array(train_labels)

def get_score(xtrain,xtest,ytrain,ytest):
	#scaler = preprocessing.StandardScaler().fit(xtrain)
	#xtrain = scaler.transform(xtrain)
	#xtest = scaler.transform(xtest)
    model = RandomForestClassifier(max_depth=4, random_state=0,n_estimators=10)
    xtrain=np.array(xtrain)
    ytrain=np.array(ytrain)
    xtrain=np.nan_to_num(xtrain)
    #print(np.isnan(np.min(xtrain)))
#    print(np.isinf(np.min(xtrain)))
#    print(xtrain.min())
    #print(xtrain.max())
    #print(len(xtrain))
    #print(type(xtrain[0]))
    #print(ytrain.shape)
    #xtrain.fillna(xtrain.mean())
#    where_are_NaNs = np.isnan(xtrain)
#    xtrain[where_are_NaNs] = 0
#    
#    where_are_NaNs = np.isnan(ytrain)
#    ytrain[where_are_NaNs] = 0
    
#    print('x shape,y shape')
#    print(xtrain.shape,ytrain.shape)
    model.fit(xtrain,ytrain)
    test_pred = np.array(model.predict(xtest))
	# ytest = np.array(ytest)
	# if(test_pred[0] == ytest[0]):
	# 	return 1
	# else:
	# 	return 0
    return test_pred


def train_test_function(align_m,align_d,data_m,data_d,names,lucknow_anomaly,kolkata_anomaly):
    anomalieslucknow = lucknow_anomaly
    #print('anomalieslucknow')
	#print(anomalieslucknow)
    anomalieskolkata = kolkata_anomaly
    #anomalieslucknow = get_anomalies('/home/nishant/study/researchwork/Onion/information/normal_h_w_lucknow.csv',align_l)
    
#    anomalieskolkata=anomalieskolkata[anomalieskolkata1[2]!='no']
#    anomalieslucknow=anomalieslucknow[anomalieslucknow1[2]!='no']
#    anomaliesbangalore=anomaliesbangalore[anomaliesbangalore1[2]!='no']
#    
    kolkatalabelsnew = newlabels(anomalieskolkata)
	#lucknowlabelsnew = newlabels(anomalieslucknow)
    lucknowlabelsnew = newlabels(anomalieslucknow)
    total_anomalies=len(kolkatalabelsnew)+len(lucknowlabelsnew)
    anomalieslucknow=adjust_anomaly_window(anomalieslucknow,align_m)
    anomalieskolkata=adjust_anomaly_window(anomalieskolkata,align_d)    
	#print('lucknowlabelsnew')
	#print(lucknowlabelsnew)
    kolkata_anomalies_year = get_anomalies_year(anomalieskolkata)
    lucknow_anomalies_year = get_anomalies_year(anomalieslucknow)
	#lucknow_anomalies_year = get_anomalies_year(anomalieslucknow)
    x1,y1 = prepare(anomalieskolkata,kolkatalabelsnew,data_d)
    x2,y2 = prepare(anomalieslucknow,lucknowlabelsnew,data_m)
    #x3,y3 = prepare(anomalieslucknow,lucknowlabelsnew,data_l)
#    print('x1 shape',x1.shape)
#    print('x2 shape',x2.shape)
#    print('x3 shape',x3.shape)
    xall = np.array(x1.tolist()+x2.tolist())
    yall = np.array(y1.tolist()+y2.tolist())
#    print(xall.shape,yall.shape)
    xall_new =[]
    yall_new = []
    yearall_new = []
    yearall = np.array(kolkata_anomalies_year+lucknow_anomalies_year)
	# for x in range(0,len(xall)):
	# 	print len(xall[x]),yall[x]
    for y in range(0,len(yall)):
        if( yall[y] == 1):
            xall_new.append(xall[y])
            yall_new.append(1)
            yearall_new.append(yearall[y])
        elif (yall[y] == 0):
            xall_new.append(xall[y])
            yall_new.append(0)
            yearall_new.append(yearall[y])

	# xall_new = np.array(xall_new)
	# yall_new = np.array(yall_new)
    assert(len(xall_new) == len(yearall_new))
    total_data, total_labels = partition(xall_new,yall_new,yearall_new,6)
    predicted = []
    actual_labels = []
    for i in range(0,len(total_data)):
        if( len(total_data[i]) != 0):
            test_split = total_data[i]
            test_labels = total_labels[i]
            actual_labels = actual_labels + test_labels
            train_split = []
            train_labels_split = []
            for j in range(0,len(total_data)):
                if( j != i):
                    train_split = train_split + total_data[j]
                    train_labels_split = train_labels_split+total_labels[j]
            pred_test = get_score(train_split,test_split,train_labels_split,test_labels)
            predicted = predicted + pred_test.tolist()
    predicted_labels = np.array(predicted)
    actual_labels= np.array(actual_labels)
	# print len(actual_labels)
#    print('accuracy')
#    print(len(predicted))
#    print(actual_labels)
#    print(predicted)
    
#    print(sum(predicted == actual_labels)/total_anomalies, f1_score(actual_labels,predicted,labels=[0,1],average="weighted"))
	#plot_confusion_matrix(actual_labels, predicted, ['Anomaly','Not Anomaly'],'title_name',cmap=plt.cm.Blues)
	# print actual_labels
	# print predicted
#    print('f1 score:')
#    print(f1_score(actual_labels,predicted,labels=[0,1],average="weighted"))
#    cm=confusion_matrix(actual_labels,predicted)
#    print(cm)
#	ax= plt.subplot()
#	sns.heatmap(cm, annot=True, ax = ax,fmt='g'); #annot=True to annotate cells
    print(str(accuracy_score(actual_labels,predicted_labels))[:6],str(f1_score(actual_labels,predicted_labels,average='weighted'))[:6])

#
#	#labels, title and ticks
#	ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
#	ax.set_title('Confusion Matrix'); 
#	ax.xaxis.set_ticklabels(['Non-Anomalous', 'Anomalous'])
#	ax.yaxis.set_ticklabels(['Non-Anomalous ', 'Anomalous '])
#	#plt.title('Arima')
#	#plt.title('Sarima')
#	#plt.title('Sarimax')
#	#plt.title('Arimax')
#	#plt.title('Arimax-Original')
#	plt.title(str(names))
#	plt.show()
    #print(fs(actual_labels,predicted))
    #print(f(actual_labels, predicted, labels=None, pos_label=1, average=’binary’, sample_weight=None))

#print(retailpriceserieslucknow)



#
#print('mandi')
#train_test_function(lucknow_mandi,kalyani_mandi,[lucknow_mandi],[kalyani_mandi],'mandi price',lucknow_high_anomaly,kolkata_high_anomaly)
#
#print('Arrival')
#train_test_function(lucknow_arrival,kalyani_arrival,[lucknow_arrival],[kalyani_arrival],'arrival',lucknow_high_anomaly,kolkata_high_anomaly)
##1
###
#print('retail price')
#train_test_function(lucknow_retail,kolkata_retail,[lucknow_retail],[kolkata_retail],'retail price',lucknow_high_anomaly,kolkata_high_anomaly)
####
#print('Arrival')
#train_test_function(lucknow_arrival,kalyani_arrival,[lucknow_arrival],[kalyani_arrival],'arrival',lucknow_high_anomaly,kolkata_high_anomaly)
#
#print('retail and mandi')
#train_test_function(lucknow_retail,kolkata_retail,[lucknow_mandi],[kalyani_mandi],'retail and mandi price',lucknow_high_anomaly,kolkata_high_anomaly)
#
#print('mandi and retail')
#train_test_function(lucknow_mandi,kalyani_mandi,[lucknow_retail],[kolkata_retail],'mandi and retail',lucknow_high_anomaly,kolkata_high_anomaly)
#
#print('arrival and mandi')
#train_test_function(lucknow_arrival,kalyani_arrival,[lucknow_mandi],[kalyani_mandi],'arrival and mandi price',lucknow_high_anomaly,kolkata_high_anomaly)
##
#print('mandi and arrival')
#train_test_function(lucknow_mandi,kalyani_mandi,[lucknow_arrival],[kalyani_arrival],'mandi and arrival',lucknow_high_anomaly,kolkata_high_anomaly)
##
#print('arrival and retail')
#train_test_function(lucknow_arrival,kalyani_arrival,[lucknow_retail],[kolkata_retail],'arrival and retail',lucknow_high_anomaly,kolkata_high_anomaly)
##
#print('retail and arrival')
#train_test_function(lucknow_retail,kolkata_retail,[lucknow_arrival],[kalyani_arrival],'retail and arrival',lucknow_high_anomaly,kolkata_high_anomaly)

#print('mandi and residual')
#train_test_function(lucknow_mandi,kalyani_mandi,[lucknow_mandi-lucknow_mandi_arimax2],[kalyani_mandi-kalyani_mandi_arimax2],'mandi and residuals',lucknow_high_anomaly,kolkata_high_anomaly)
##
#print('residual and mandi')
#train_test_function(lucknow_mandi-lucknow_mandi_arimax2,kalyani_mandi-kalyani_mandi_arimax2,[lucknow_mandi,lucknow_retail],[kalyani_mandi,kolkata_retail],'arrival, retail-mandi',lucknow_high_anomaly,kolkata_high_anomaly)
##
#
#
##print('arrival, retail-mandi')
##train_test_function(lucknow_retail-lucknow_mandi,bangalore_retail-bangalore_mandi,kolkata_retail-kalyani_mandi,[lucknow_arrival],[bangalore_arrival],[kalyani_arrival],'arrival, retail-mandi',lucknow_high_anomaly,kolkata_high_anomaly)
#
#
#print('arrival, retail and mandi')
#train_test_function(lucknow_arrival,kalyani_arrival,[lucknow_mandi,lucknow_retail],[kalyani_mandi,kolkata_retail],'arrival, retail and mandi',lucknow_high_anomaly,kolkata_high_anomaly)
