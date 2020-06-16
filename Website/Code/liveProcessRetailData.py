import pandas as pd
import pickle
import numpy as np

#{'centreCodes': {'NCT of Delhi': ['DELHI', 16], 'Maharashtra': ['MUMBAI', 44], 'West Bengal': ['KOLKATA', 37], 'Karnataka': ['Bengaluru', 7], 'Uttar Pradesh': ['LUCKNOW', 40]}

def processRetailData():
	df = pd.read_csv('../Data/Final/Retail/retailData.csv',header=None,engine='python', error_bad_lines=False)
	#print(len(df))
	#print(df.head())
	df[2] = df[2]*100
	myDicts = pickle.load(open ("myDicts.p","rb"))
	for v in myDicts['centreCodes'].values():
		name = v[0]
		code = v[1]
		dx = df[df[1]==code][[0,2]]
		dx.index = dx[0]
		dx = dx[[2]]
		dx = dx.loc[~dx.index.duplicated(keep='last')]
		#print(dx.head())
		idx = pd.date_range('2006-01-01', str(dx.index.max()))
		dx.index = pd.DatetimeIndex(dx.index)
		dx = dx.reindex(idx, fill_value=np.nan)
		dx.columns = [1]
		dx[1].replace(0.0,np.nan,inplace=True)
		dx = dx.interpolate(method='linear')
		dx = dx.bfill(axis ='rows') 
		dx.to_csv('../Data/Final/Retail/'+str(name)+'Retail.csv',header=False)
	#print('RETAIL PROCESSING DONE!!!!')
