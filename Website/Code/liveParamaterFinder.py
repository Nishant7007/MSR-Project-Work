import pandas as pd
from sarimaxForecasting import *
import pickle
from liveLoadSeries import *
#'LUCKNOW', 40    'DELHI', 16    'Bengaluru', 7   'KOLKATA', 37   'MUMBAI', 44


#with open('myDicts.p','rb') as f:
#	myDicts = pickle.load(f)
#print(myDicts)


priceDic = { 'AzadpurPrice.csv' : 'KeshopurPrice.csv',
		'BijnaurPrice.csv' : 'LucknowPrice.csv',
		'KalyaniPrice.csv' : 'KalnaPrice.csv',
		'BangalorePrice.csv' : 'DavangerePrice.csv',
		'LasalgaonPrice.csv' : 'YeolaPrice.csv'
}

arrivalDic = { 'AzadpurArrival.csv' : 'KeshopurArrival.csv',
                'BijnaurArrival.csv' : 'LucknowArrival.csv',
                'KalyaniArrival.csv' : 'KalnaArrival.csv',
                'BangaloreArrival.csv' : 'DavangereArrival.csv',
                'LasalgaonArrival.csv' : 'YeolaArrival.csv'
}

retailDic = { 'DELHIRetail.csv' : 'AzadpurPrice.csv',
                'LUCKNOWRetail.csv' : 'BijnaurPrice.csv',
                'KOLKATARetail.csv' : 'KalyaniPrice.csv',
                'BengaluruRetail.csv' : 'BangalorePrice.csv',
                'MUMBAIRetail.csv' : 'LasalgaonPrice.csv'
}


import pandas as pd
import matplotlib.pyplot as plt
# import pmdarima as pm
import statsmodels.api as sm

def findPriceParameters():
	print('Mandi Price')
	priceResultsDic = {}
	for k,v in priceDic.items():
		#print(k)
		series = pd.read_csv('../Data/Final/Wholesale/'+str(k),names = [0.1],index_col=0,header=None)
		near = pd.read_csv('../Data/Final/Wholesale/'+str(v),names = [0.1],index_col=0,header=None)
		priceResultsDic[k] = forecast(series,near)
	with open('price.pickle', 'wb') as handle:
		pickle.dump(priceResultsDic, handle)

def findArrivalParameters():
	print('Arrival')
	arrivalResultsDic = {}
	for k,v in arrivalDic.items():
		#print(k)
		series = pd.read_csv('../Data/Final/Wholesale/'+str(k),names = [0.1],index_col=0,header=None)
		near = pd.read_csv('../Data/Final/Wholesale/'+str(v),names = [0.1],index_col=0,header=None)
		arrivalResultsDic[k] = forecast(series,near)
	with open('arrival.pickle', 'wb') as handle:
		pickle.dump(arrivalResultsDic, handle)

def findRetailParameters():
	print('Retail')
	retailDict = loadAllRetailSeries(retailSeries,myDicts['centreCodes'])
	retailResultsDic = {}
	for k in retailDict.keys():
		#print(k,retailDic[k])	
		near_csv = retailDic[k]
		near = pd.read_csv('../Data/Final/Wholesale/'+near_csv,names = [0.1],index_col=0,header=None)
		series = retailDict[k]
		retailResultsDic[str(k)] = forecast(series,near)	
	with open('retail.pickle', 'wb') as handle:
		pickle.dump(retailResultsDic, handle)


#findRetailParameters()
#findPriceParameters()
#findArrivalParameters()
