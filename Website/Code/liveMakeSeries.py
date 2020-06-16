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

def fillMissingDatesArrivalandMandi(df):
	# print('inside missing value function:')
	df.columns = [0,1] 
	# print(len(df))
	recent_date = df[0].max()
	# print(recent_date)
	l = (datetime.strptime(recent_date, date_format) - datetime.strptime('2006-01-01', date_format)).days
	# print(l)
	df = df.drop_duplicates(subset = 0, keep = 'first')
	df = df.reset_index()
	idx = pd.date_range('2006-01-01', periods = l)
	#print(idx)
	df.index = pd.DatetimeIndex(df[0])
	df = df[[1]] 
	#print(df)
	#print (df.index.name)
	df = df.reindex(idx, fill_value = 0)
	df.replace(0,np.nan,inplace = True)
	# print(df)
	df = df.interpolate(method = 'linear',limit_direction = 'both')
	df = df[[1]]
	# print(df)
	return df


def makeArrivalSeries(dicts):
	# print(dicts)
	states = [i for i in dicts.keys()]
	for state in states:
		#print(state)
		fileName = '../Data/Processed/Wholesale/'+str(state)+'.csv'
		df = pd.read_csv(fileName,header = None)
		df = df[df[1] == dicts[state][1]]
		df.drop_duplicates(subset = 4,keep='first')
		arrival = pd.DataFrame(df[[0,2]])
		arrival = fillMissingDatesArrivalandMandi(arrival)
		fileName = '../Data/Final/Wholesale/'+str(dicts[state][0])+'Arrival.csv'
		arrival.to_csv(fileName, header = False)		

def makePriceSeries(dicts):
	# print(dicts)
	states = [i for i in dicts.keys()]
	for state in states:
		# print(state)
		fileName = '../Data/Processed/Wholesale/'+str(state)+'.csv'
		# print(fileName)
		df = pd.read_csv(fileName,header = None)
		# print(df.head())
		df = df[df[1] == dicts[state][1]]
		# print(df.head())
		# print(len(df))
		df.drop_duplicates(subset = 4,keep='first')
		# print(len(df))
		price = pd.DataFrame(df[[0,7]])
		# print(len(price))
		price = fillMissingDatesArrivalandMandi(price)
		# print('after fill missing:',len(price))
		fileName = '../Data/Final/Wholesale/'+str(dicts[state][0])+'Price.csv'
		# print(fileName)
		price.to_csv(fileName, header = False)


		
def makeRetailSeries(dicts):
	mainFile = pd.read_csv('../Data/Final/Retail/retailData.csv',header = None,engine='python',error_bad_lines=False)
	# print(mainFile)
	# print(mainFile.columns)
	fileNames = [f for f in listdir('../Data/Processed/Retail') if not f.startswith('.')]
	#print(fileNames)
	for file in fileNames:
		# print(file)
		fileName = '../Data/Processed/Retail/' + str(file)
		print(fileName)
		df = pd.read_csv(fileName,engine='python',sep='[,]',names =[0,1,2])
		#print(df.head())
		df = df[1:]
		#print(df.head())
		df[0] = df[0].map(lambda x:x.lstrip('\"'))
		df[2] = df[2].map(lambda x:x.rstrip('\"'))		
		#print(df.head())
		df.reset_index(inplace = True, drop = True)
		# print(df.columns)
		codesDict = dicts
		for each in codesDict.keys():
			if(df.iloc[0][1] == codesDict[each][0]):
				df[1] = codesDict[each][1]
		# print(len(mainFile))
		mainFile = pd.concat([mainFile,df],sort = True)
		# print(mainFile.columns)
	mainFile.to_csv('../Data/Final/Retail/retailData.csv',header = False,index = False)
