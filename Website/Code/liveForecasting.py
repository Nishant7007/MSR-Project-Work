import pandas as pd
import numpy as np
import pickle
from liveParamaterFinder import *
from datetime import date
date_format = "%Y-%m-%d"
from pmdarima.preprocessing import FourierFeaturizer

def priceActual():
	for k,v in priceDic.items():
		print(k,v)
		series = pd.read_csv('../Data/Final/Wholesale/'+k,names = [0.1],index_col=0,header=None)
		near_file = v[:-4]+'.npy'
		near = np.load('../Data/Results/Near/'+near_file)
		daysToForecast = 30
		series_train = np.squeeze(series.values)
		n = len(series_train)
		near_train =  near[:n]
		near_test = near[-30:]
		trans = FourierFeaturizer(365.25, 1)
		y_prime, exogen = trans.fit_transform(series_train)
		exogen = exogen.mul(pd.Series(series_train),axis=0)
		futureExog =  trans.transform(y = series_train, n_periods = 30)
		futureExog = pd.DataFrame(futureExog[1])
		futureExog = futureExog.mul(pd.Series(near_test),axis=0)
		exogen['near'] = near_train
		futureExog['near'] = near_test
		#print('MODel searching')
		model=pm.arima.auto_arima(series_train, exogenous = exogen, start_p=0, d=None, start_q=0, max_p=3, max_d=1, max_q=3,start_P=0, D=None, start_Q=0, max_P=2, max_D=1, max_Q=2,suppress_warnings =True,seasonal=True,max_order=4,m=7,stepwise=True) 
		model.fit(series_train)
		pred = (model.predict(daysToForecast,exogenous = futureExog))
		series = np.concatenate((series_train,pred),axis=0)
		series = pd.DataFrame(series)
		series.index = pd.date_range(start = '2006-01-01',periods = len(series))
		fileName = '../Data/Results/Actual/'+str(k)
		#print(fileName)
		series.to_csv(fileName)


def arrivalActual():
	for k,v in arrivalDic.items():
		print(k,v)
		series = pd.read_csv('../Data/Final/Wholesale/'+k,names = [0.1],index_col=0,header=None)
		near_file = v[:-4]+'.npy'
		near = np.load('../Data/Results/Near/'+near_file)
		daysToForecast = 30
		series_train = np.squeeze(series.values)
		n = len(series_train)
		near_train =  near[:n]
		near_test = near[-30:] 
		trans = FourierFeaturizer(365.25, 1)
		y_prime, exogen = trans.fit_transform(series_train)
		exogen = exogen.mul(pd.Series(series_train),axis=0)
		futureExog =  trans.transform(y = series_train, n_periods = 30)
		futureExog = pd.DataFrame(futureExog[1])
		futureExog = futureExog.mul(pd.Series(near_test),axis=0)
		exogen['near'] = near_train
		futureExog['near'] = near_test
		#print('MODEL SEARCHING')
		model=pm.arima.auto_arima(series_train, exogenous = exogen, start_p=0, d=None, start_q=0, max_p=3, max_d=1, max_q=3,start_P=0, D=None, start_Q=0, max_P=2, max_D=1, max_Q=2,suppress_warnings =True,seasonal=True,max_order=4,m=7,stepwise=True)
		model.fit(series_train)
		pred = (model.predict(daysToForecast,exogenous = futureExog))
		series = np.concatenate((series_train,pred),axis=0)
		series = pd.DataFrame(series)
		series.index = pd.date_range(start = '2006-01-01',periods = len(series))
		fileName = '../Data/Results/Actual/'+str(k)
		#print(fileName)
		series.to_csv('../Data/Results/Actual/'+str(k))

def retailActual():
	for k,v in retailDic.items():
		print(k,v)
		series = pd.read_csv('../Data/Final/Retail/'+k,names = [0.1],index_col=0,header=None)
		near_file = v[:-4]+'.npy'
		near = np.load('../Data/Results/Near/'+near_file)
		daysToForecast = 30
		series_train = np.squeeze(series.values)
		n = len(series_train)
		near_train =  near[:n]
		near_test = near[-30:] 
		trans = FourierFeaturizer(365.25, 1)
		y_prime, exogen = trans.fit_transform(series_train)
		exogen = exogen.mul(pd.Series(series_train),axis=0)
		futureExog =  trans.transform(y = series_train, n_periods = 30)
		futureExog = pd.DataFrame(futureExog[1])
		futureExog = futureExog.mul(pd.Series(near_test),axis=0)
		exogen['near'] = near_train
		futureExog['near'] = near_test
		#print('MODEL SEARCHING')
		model=pm.arima.auto_arima(series_train, exogenous = exogen, start_p=0, d=None, start_q=0, max_p=3, max_d=1, max_q=3,start_P=0, D=None, start_Q=0, max_P=2, max_D=1, max_Q=2,suppress_warnings =True,seasonal=True,max_order=4,m=7,stepwise=True)
		model.fit(series_train)
		pred = (model.predict(daysToForecast,exogenous = futureExog))
		series = np.concatenate((series_train,pred),axis=0)
		series = pd.DataFrame(series)
		series.index = pd.date_range(start = '2006-01-01',periods = len(series))
		fileName = '../Data/Results/Actual/'+str(k)
		#print(fileName)
		series.to_csv(fileName)                                                     

