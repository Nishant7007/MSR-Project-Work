import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima.preprocessing import FourierFeaturizer

series = pd.read_csv('../Data/Final/Wholesale/LasalgaonPrice.csv',names = [0.1],index_col=0,header=None)
near = pd.read_csv('../Data/Final/Wholesale/YeolaPrice.csv',names = [0.1],index_col=0,header=None)

train_near = np.squeeze(near.values[:-30])
train_series = np.squeeze(series.values[:len(train_near)])

test_near = np.squeeze(near.values[-30:])
test_series = np.squeeze(series.values[-30:])

for k in range(1,4):
	print(k)
	trans = FourierFeaturizer(365.25, k)
	y_prime, exogen = trans.fit_transform(train_series)
	exogen = exogen.mul(pd.Series(train_series),axis=0)
	exogen['near'] = train_near
	model=pm.arima.auto_arima(train_series, exogenous=pd.DataFrame(exogen), start_p=0, d=None, start_q=0, max_p=5, max_d=2, max_q=5,start_P=0, D=None, start_Q=0, max_P=5, max_D=2, max_Q=5,suppress_warnings =True,seasonal=True,max_order=8,m=7,stepwise=True,error_action="ignore")
	print(model.summary())
