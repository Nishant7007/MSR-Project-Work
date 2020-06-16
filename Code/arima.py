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
 
series=lucknow_arrival
n=len(series)
print()
end_index =4748
forecasted = list(series[ :4748])
while end_index < n:
    print(end_index)
    history = series[: end_index]
    model=pm.arima.auto_arima(history, exogenous=None, start_p=0, d=None, start_q=0, max_p=10, max_d=2, max_q=10,suppress_warnings =True,seasonal=False)
    print(model.summary())
    if end_index + 30 < n:
        predictions = model.predict(30,exogenous=None)
        forecasted = list(forecasted + predictions.tolist())
    else:
        vals = n - end_index
        print('vals',vals)
        predictions = model.predict(vals,exogenous=None)
        forecasted = list(forecasted + predictions.tolist())
    end_index = min(end_index + 30, n)
print('End: ' + str(end_index))
print(len(forecasted))
p=len(forecasted)
dk=pd.DataFrame(forecasted)
dk['date']=pd.date_range('2006-01-01',periods=p)
dk.columns=['price','date']
dk=dk[['date','price']]
dk.to_csv('../Results/lucknow_arrival_2019_arima.csv')
