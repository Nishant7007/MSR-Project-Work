import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima.preprocessing import FourierFeaturizer
import itertools 
from random import choice as ch
import statsmodels.api as sm
import pandas as pd


pList = [i for i in range(8)]
qList = [i for i in range(8)]
dList = [i for i in range(2)]
PList = [i for i in range(5)]
QList = [i for i in range(5)]
DList = [i for i in range(2)]

nonSeasonal = list(itertools.product(pList, dList, qList))
seasonal = list(itertools.product(PList, DList, QList))


def forecast(series,near):

	train_near = np.squeeze(near.values[:-30])
	train_series = np.squeeze(series.values[:len(train_near)])

	test_near = np.squeeze(near.values[-30:])
	test_series = np.squeeze(series.values[-30:])
	df = pd.DataFrame(columns=['aic','nonSeasonal','seasonal','k'])
	for i in range(25):
		print('value of i is:',i)
		nonSeasonalParams = ch(nonSeasonal)
		seasonalParams = ch(seasonal)
		val = sum(list(nonSeasonalParams)) + sum(list(seasonalParams))
		#try:
		if(val>8):
			continue
		seasonalParams = seasonalParams + (7,)
		#print(nonSeasonalParams,seasonalParams)
		try:
			trans = FourierFeaturizer(365.25, 1)
			y_prime, exogen = trans.fit_transform(train_series)
			#exogen = exogen.mul(pd.Series(train_series),axis=0)
			exogen['near'] = train_near
			model = sm.tsa.statespace.SARIMAX(endog = train_series, exog = train_near, order = nonSeasonalParams, seasonal_order = seasonalParams,initialization='approximate_diffuse',enforce_stationarity=False) 
			res = model.fit(disp=False)
			#print(res.aic)
			to_append = [res.aic,nonSeasonalParams,seasonalParams,1]
			a_series = pd.Series(to_append, index = df.columns)
			df = df.append(a_series, ignore_index=True)
		except:
			print('inside except block now.....')
			x = pd.Series([10000000,(1,1,1),(1,1,1),1],index = df.columns)
			df = df.append(x, ignore_index=True)
	x = pd.Series([10000000,(1,1,1),(1,1,1),1],index = df.columns)
	df = df.append(x, ignore_index=True)
	print('df is :',df)
	dx = (df[df.aic == df.aic.min()])
	dx.reset_index(inplace=True,drop=True)
	print('baest parameters are:',dx)
	value = [dx.iloc[0][0],dx.iloc[0][1],dx.iloc[0][2],dx.iloc[0][3]]
	print('final parameters are:',value)
	return value
