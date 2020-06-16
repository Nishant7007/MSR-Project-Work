import pickle
import pandas as pd 
from datetime import datetime
import numpy as np
from os import listdir
cts = pickle.load(open ("myDicts.p","rb"))

date_format = "%Y-%m-%d"
startDate = "2006-01-01"

myDicts = pickle.load( open ("myDicts.p", "rb"))


def loadMandiSeries(name):
	df = pd.read_csv('../Data/Results/Actual/'+name+'Price.csv',header=None)
	df = df[df[0]>=startDate]
	df.drop_duplicates(subset = 0 , inplace = True)
	df.reset_index(inplace = True , drop = True)
	recent_date = df[0].max()
	l = (datetime.strptime(recent_date, date_format) - datetime.strptime('2006-01-01', date_format)).days	
	idx = pd.date_range('2006-01-01', periods = l)
	df.index = pd.DatetimeIndex(df[0])
	df = df[[1]]
	df = df.reindex(idx, fill_value = 0)
	df.replace(0,np.nan,inplace = True)
	df = df.interpolate(method = 'linear',limit_direction = 'both')
	df = df[[1]]
	return df


def loadAllMandiSeries(Dicts):
	mandiDict = {}
	for v in Dicts.values():
		name = v[0]
		#print(name)
		mandiDict[name] = loadMandiSeries(name)
	return (mandiDict)	

def loadArrivalSeries(name):
	df = pd.read_csv('../Data/Results/Actual/'+name+'Arrival.csv',header=None)
	df = df[df[0]>=startDate]
	df.drop_duplicates(subset = 0, inplace = True)
	df.reset_index(inplace = True, drop = True)
	recent_date = df[0].max()
	l = (datetime.strptime(recent_date, date_format) - datetime.strptime('2006-01-01', date_format)).days	
	idx = pd.date_range('2006-01-01', periods = l)
	df.index = pd.DatetimeIndex(df[0])
	df = df[[1]]
	df = df.reindex(idx, fill_value = 0)
	df.replace(0,np.nan,inplace = True)
	df = df.interpolate(method = 'linear',limit_direction = 'both')
	df = df[[1]]
	return df


def loadAllArrivalSeries(Dicts):
	arrivalDict = {}
	for v in Dicts.values():
		name = v[0]
		arrivalDict[name] = loadArrivalSeries(name)
	return (arrivalDict)	


def loadRetailSeries(name):
        df = pd.read_csv('../Data/Results/Actual/'+name+'Retail.csv',header=None)
        df = df[df[0]>=startDate]
        df.drop_duplicates(subset = 0 , inplace = True)
        df.reset_index(inplace = True , drop = True)
        recent_date = df[0].max()
        l = (datetime.strptime(recent_date, date_format) - datetime.strptime('2006-01-01', date_format)).days
        idx = pd.date_range('2006-01-01', periods = l)
        df.index = pd.DatetimeIndex(df[0])
        df = df[[1]]
        df = df.reindex(idx, fill_value = 0)
        df.replace(0,np.nan,inplace = True)
        df = df.interpolate(method = 'linear',limit_direction = 'both')
        df = df[[1]]
        return df


def loadAllRetailSeries(Dicts):
        retailDict = {}
        for v in Dicts.values():
                name = v[0]
                retailDict[name] = loadRetailSeries(name)
        return (retailDict)


def AnomalySeries(dicts):
	for v in dicts.values():
		name = v[0]
		fileName = '../Data/Anomaly/'+str(name)+'Anomalies.csv'
		df = pd.read_csv(fileName,header=None,names =['start','end','class'])
		print(name)
		print(df)


retailDict = loadAllRetailSeries(myDicts['centreCodes'])
priceDict = loadAllMandiSeries(myDicts['mandiCodes'])
arrivalDict = loadAllArrivalSeries(myDicts['mandiCodes'])

AnomalySeries(myDicts['centreCodes'])

azadpurArrival = arrivalDict.get('Azadpur')
kalyaniArrival = arrivalDict.get('Kalyani')
bengaloreArrival = arrivalDict.get('Bangalore')
bijnaurArrival = arrivalDict.get('Bijnaur')
LasalgaonArrival = arrivalDict.get('Lasalgaon')


azadpurPrice = priceDict.get('Azadpur')
bijnaruPrice = priceDict.get('Bijnaur')
bengalorePrice = priceDict.get('Bangalore')
kalyaniPrice = priceDict.get('Kalyani')
LasalgaonPrice = priceDict.get('Lasalgaon')



delhiRetail = retailDict.get('DELHI')
lucknowRetail = retailDict.get('LUCKNOW')
bangaloreRetail = retailDict.get('Bengaluru')
kolkataRetail = retailDict.get('KOLKATA')
mumbaiRetail = retailDict.get('MUMBAI')


#print('azadpurArrival')
#print(azadpurArrival.tail())
#print('kalyaniArrival')
#print(kalyaniArrival.tail())
#print('bengaloreArrival')
#print(bengaloreArrival.tail())
#print('bijnaurArrival')
#print(bijnaurArrival.tail())
#print('LasalgaonArrival')
#print(LasalgaonArrival.tail())


#print('azadpurPrice')
#print(azadpurPrice.tail())
#print('bijnaruPrice')
#print(bijnaruPrice.tail())
#print('bengalorePrice')
#print(bengalorePrice.tail())
#print('kalyaniPrice')
#print(kalyaniPrice.tail())
#print('LasalgaonPrice')
#print(LasalgaonPrice.tail())


#print('delhiRetail')
#print(delhiRetail.tail())
#print('lucknowRetail')
#print(lucknowRetail.tail())
#print('bangaloreRetail')
#print(bangaloreRetail.tail())
#print('kolkataRetail')
#print(kolkataRetail.tail())
#print('mumbaiRetail')
#print(mumbaiRetail.tail())






#print(myDicts['mandiCodes'])





