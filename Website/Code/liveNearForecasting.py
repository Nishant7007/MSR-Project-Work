import pandas as pd
import numpy as np
import pickle
from liveParamaterFinder import *
from datetime import date
date_format = "%Y-%m-%d"
from pmdarima.preprocessing import FourierFeaturizer

#{'AzadpurPrice.csv': [1i0, (1, 1, 1), (1, 1, 1), 1], 'BangalorePrice.csv': [10, (1, 1, 1), (1, 1, 1), 1], 'BijnaurPrice.csv': [10, (1, 1, 1), (1, 1, 1), 1], 'LasalgaonPrice.csv': [10, (1, 1, 1), (1, 1, 1), 1], 'KalyaniPrice.csv': [10, (1, 1, 1), (1, 1, 1), 1]}
#{'AzadpurPrice.csv': 'KeshopurPrice.csv', 'BangalorePrice.csv': 'DavangerePrice.csv', 'BijnaurPrice.csv': 'LucknowPrice.csv', 'KalyaniPrice.csv': 'KalnaPrice.csv', 'LasalgaonPrice.csv': 'YeolaPrice.csv'}

priceResultsDict = pickle.load(open('price.pickle','rb'))
#print(priceResultsDict)
#print(priceDic)

arrivalResultsDict = pickle.load(open('arrival.pickle','rb'))
#print(arrivalResultsDict)
#print(arrivalDic)

retailResultsDict = pickle.load(open('retail.pickle','rb'))
#print(retailResultsDict)
#print(retailDic)


def priceNear():        
        for k,v in priceDic.items():
                print(k,v)
                series = pd.read_csv('../Data/Final/Wholesale/'+k,names = [0.1],index_col=0,header=None)
                near = pd.read_csv('../Data/Final/Wholesale/'+v,names = [0.1],index_col=0,header=None)
                a = datetime.strptime(series.index.max(), date_format)
                b = datetime.strptime(near.index.max(), date_format)
                x = (a-b).days
                daysToForecast = 30+x
                near_train = (np.squeeze(near.values))
                #print(len(near_train))
                trans = FourierFeaturizer(365.25, 1)
                y_prime, exogen = trans.fit_transform(near_train)
                futureExog =  trans.transform(y = near_train, n_periods = daysToForecast)
                exogen = exogen.mul(pd.Series(near_train),axis=0)
                print('MODEL SEARCHING')
                model=pm.arima.auto_arima(near_train, exogenous = exogen, start_p=0, d=None, start_q=0, max_p=3, max_d=1, max_q=3,start_P=0, D=None, start_Q=0, max_P=2, max_D=1, max_Q=2,suppress_warnings =True,seasonal=True,max_order=5,m=7,stepwise=True)
                #print(model.aic())
                model.fit(near_train)
                pred = model.predict(daysToForecast,exogenous = futureExog)
                #print(pred)
                near = (np.concatenate((near_train , np.array(pred)),axis = 0))
                #print(near.shape)
                fileName ='../Data/Results/Near/'+ v[:-4]+'.npy'
                #print(fileName)
                np.save(fileName,near)


def arrivalNear():        
        for k,v in arrivalDic.items():
                print(k,v)
                series = pd.read_csv('../Data/Final/Wholesale/'+k,names = [0.1],index_col=0,header=None)
                near = pd.read_csv('../Data/Final/Wholesale/'+v,names = [0.1],index_col=0,header=None)
                a = datetime.strptime(series.index.max(), date_format)
                b = datetime.strptime(near.index.max(), date_format)
                x = (a-b).days
                daysToForecast = 30+x
                near_train = (np.squeeze(near.values))
                #print(len(near_train))
                trans = FourierFeaturizer(365.25, 1)
                y_prime, exogen = trans.fit_transform(near_train)
                futureExog =  trans.transform(y = near_train, n_periods = daysToForecast)
                exogen = exogen.mul(pd.Series(near_train),axis=0)
                #print('MODEL SEARCHING')
                model=pm.arima.auto_arima(near_train, exogenous = exogen, start_p=0, d=None, start_q=0, max_p=3, max_d=1, max_q=3,start_P=0, D=None, start_Q=0, max_P=2, max_D=1, max_Q=2,suppress_warnings =True,seasonal=True,max_order=4,m=7,stepwise=True)
                #print(model.aic())
                model.fit(near_train)
                pred = model.predict(daysToForecast,exogenous = futureExog)
                #print(pred)
                near = (np.concatenate((near_train , np.array(pred)),axis = 0))
                #print(near.shape)
                fileName ='../Data/Results/Near/'+ v[:-4]+'.npy'
                #print(fileName)
                np.save(fileName,near)

def retailNear():
	for k,v in retailDic.items():
		print(k,v)
		series = pd.read_csv('../Data/Final/Retail/'+k,names = [0.1],index_col=0,header=None)
		near = pd.read_csv('../Data/Final/Wholesale/'+v,names = [0.1],index_col=0,header=None)
		#retailDict = loadAllRetailSeries(retailSeries,myDicts['centreCodes'])
		a = datetime.strptime(series.index.max(),'%Y-%m-%d')
		b = datetime.strptime(near.index.max(),'%Y-%m-%d')
		#print(a,b)
		x = (a-b).days
		daysToForecast = 30+x
		near_train = (np.squeeze(near.values))
		#print(len(near_train))
		trans = FourierFeaturizer(365.25, 1)
		y_prime, exogen = trans.fit_transform(near_train)
		futureExog =  trans.transform(y = near_train, n_periods = daysToForecast)
		exogen = exogen.mul(pd.Series(near_train),axis=0)
		#print('MODEL SEARCHING')
		model=pm.arima.auto_arima(near_train, exogenous = exogen, start_p=0, d=None, start_q=0, max_p=3, max_d=1, max_q=3,start_P=0, D=None, start_Q=0, max_P=2, max_D=1, max_Q=2,suppress_warnings =True,seasonal=True,max_order=4,m=7,stepwise=True)
		#print(model.aic())
		model.fit(near_train)
		pred = model.predict(daysToForecast,exogenous = futureExog)
		#print(pred)
		near = (np.concatenate((near_train , np.array(pred)),axis = 0))
		#print(near.shape)
		fileName = '../Data/Results/Near/'+v[:-4]+'.npy'
		#print(fileName)
		np.save(fileName,near)













#priceNear()
#arrivalNear()
#retailNear()
