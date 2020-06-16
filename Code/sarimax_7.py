#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:37:09 2019

@author: ictd
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:11:55 2019

@author: ictd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from loadSeries import *


series=lasalgaon_mandi
near=yeola_mandi
print(near)
n = len(series)
print(n)
m=len(near)
print(m)
end_index =300
forecasted = list(series[ :3000])
while end_index < n:
    print(end_index)
    history = series[: end_index]
    model=pm.arima.auto_arima(history, exogenous=pd.DataFrame(near[:end_index]), start_p=0, d=None, start_q=0, max_p=3, max_d=1, max_q=3,start_P=0, D=None, start_Q=0, max_P=3, max_D=1, max_Q=3,suppress_warnings =True,seasonal=True,max_order=6,m=7) 
    print(model.summary())
    if end_index + 30 < n:
        predictions = model.predict(30,exogenous=pd.DataFrame(near[end_index:end_index+30]))
        forecasted = list(forecasted + predictions.tolist())
    else:
        vals = n - end_index
        print('vals',vals)
        predictions = model.predict(vals,exogenous=pd.DataFrame(near[end_index:end_index+vals]))
        forecasted = list(forecasted + predictions.tolist())
    end_index = min(end_index + 30, n)
print('End: ' + str(end_index))
print(len(forecasted))
p=len(forecasted)
dk=pd.DataFrame(forecasted)
dk['date']=pd.date_range('2006-01-01',periods=p)
dk.columns=['price','date']
dk.index=dk['date']
dk=dk[['price']]
dk.to_csv('../Results/Mandi/lasalgaon_mandi_sarimax7days2.csv')