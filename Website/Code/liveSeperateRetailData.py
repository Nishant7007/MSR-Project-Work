import pandas as pd
import pickle
import numpy as np

def seperateRetailSeries():
	df = pd.read_csv('../Data/Final/Retail/retailData.csv',header=None)
	#print(df)
	myDicts = pickle.load(open ("myDicts.p","rb"))
	dict = myDicts['centreCodes']
	for v in dict.values():
		name = v[0]
		code = v[1]
		fileName = '../Data/Final/Retail/'+str(name)+'Retail.csv'
		#print(fileName)
		dx = df[df[1]==code]
		dx = dx[[0,2]]
		dx.columns = [0,1]
		dx = dx.set_index(0,drop=True)
		lastDate = dx.index.max()
		dx = dx.loc[~dx.index.duplicated(keep='first')]
		idx = pd.date_range('2006-01-01', lastDate)
		dx.index = pd.DatetimeIndex(dx.index)
		dx = dx.reindex(idx, fill_value=0)
		
		dx[1].replace(0,np.nan,inplace=True)
		dx[1].interpolate(method='linear',inplace=True,limit_direction='both')
		#print(dx.head())
		dx.to_csv(fileName,header=False)
#seperateRetailSeries()
