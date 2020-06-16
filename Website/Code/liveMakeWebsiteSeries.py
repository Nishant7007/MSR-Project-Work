from os import listdir
import pandas as pd
from liveCommon import *
import numpy as np
import pickle

# states = [f for f in mandiCodes.keys()]

# print(mandiCodes)
from datetime import datetime
date_format = "%Y-%m-%d"

myDicts = pickle.load( open ("myDicts.p", "rb"))


#print(myDicts['mandiCodes'])
#print(myDicts)
def makeArrivalFinalSeries(dict):
        for v in dict.values():
                fileName = v[0]
                dfRaw = pd.read_csv('../Data/Original/Wholesale/'+str(fileName)+'Arrival.csv',header=None,names=['date','info'])
                dfProcessed = pd.read_csv('../Data/Final/Wholesale/'+str(fileName)+'Arrival.csv',header=None,names=['date','values'])
                dfProcessed['info'] = dfRaw['info']
                dfProcessed['info'] = dfProcessed['info'].map(lambda x: 0 if x==0.0 else 1)
                dfSmoothed = dfProcessed.copy()
                dfSmoothed['values'] = dfSmoothed['values'].rolling(14).mean()
                dfProcessed.to_csv('../Data/Plot/Wholesale/'+str(fileName)+'ArrivalYear.csv',header=False,index=False)
                dfSmoothed.to_csv('../Data/Plot/Wholesale/'+str(fileName)+'ArrivalFull.csv',header=False,index=False)
                df30 = pd.read_csv('../Data/Results/Actual/'+str(fileName)+'Arrival.csv',names=['date','values'])
                df30.to_csv('../Data/Plot/Wholesale/'+str(fileName)+'ArrivalPredicted.csv',header=False,index=False)


def makePriceFinalSeries(dict):
        for v in dict.values():
                fileName = v[0]
                dfRaw = pd.read_csv('../Data/Original/Wholesale/'+str(fileName)+'Price.csv',header=None,names=['date','info'])
                dfProcessed = pd.read_csv('../Data/Final/Wholesale/'+str(fileName)+'Price.csv',header=None,names=['date','values'])
                dfProcessed['info'] = dfRaw['info']
                dfProcessed['info'] = dfProcessed['info'].map(lambda x: 0 if x==0.0 else 1)
                dfSmoothed = dfProcessed.copy()
                dfSmoothed['values'] = dfSmoothed['values'].rolling(14).mean()
                dfProcessed.to_csv('../Data/Plot/Wholesale/'+str(fileName)+'PriceYear.csv',header=False,index=False)
                dfSmoothed.to_csv('../Data/Plot/Wholesale/'+str(fileName)+'PriceFull.csv',header=False,index=False)
                df30 = pd.read_csv('../Data/Results/Actual/'+str(fileName)+'Price.csv',names=['date','values'])
                df30.to_csv('../Data/Plot/Wholesale/'+str(fileName)+'PricePredicted.csv',header=False,index=False)



def makeRetailFinalSeries(dict):
        for v in dict.values():
                fileName = v[0]
                dfRaw = pd.read_csv('../Data/Original/Retail/'+str(fileName)+'Retail.csv',header=None,names=['date','info'])
                dfProcessed = pd.read_csv('../Data/Final/Retail/'+str(fileName)+'Retail.csv',header=None,names=['date','values'])
                dfProcessed['info'] = dfRaw['info']
                dfProcessed['info'] = dfProcessed['info'].map(lambda x: 0 if x==0.0 else 1)
                dfSmoothed = dfProcessed.copy()
                dfSmoothed['values'] = dfSmoothed['values'].rolling(14).mean()
                dfProcessed.to_csv('../Data/Plot/Retail/'+str(fileName)+'RetailYear.csv',header=False,index=False)
                dfSmoothed.to_csv('../Data/Plot/Retail/'+str(fileName)+'RetailFull.csv',header=False,index=False)
                df30 = pd.read_csv('../Data/Results/Actual/'+str(fileName)+'Retail.csv',names=['date','values'])
                df30.to_csv('../Data/Plot/Retail/'+str(fileName)+'RetailPredicted.csv',header=False,index=False)


#makeRetailFinalSeries(myDicts['centreCodes'])

#makePriceFinalSeries(myDicts['mandiCodes'])

#makeArrivalFinalSeries(myDicts['mandiCodes'])

