#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:50:56 2020

@author: ictd
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:21:41 2020

@author: ictd
"""

from loadSeries import *
import numpy as np
import pandas as pd
from scipy.stats import variation
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#print(variation(lucknow_mandi))

#print(variation(bangalore _mandi))
    

def variation_in_data(anomalies,priceseries):
    anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[0]]
    anomalies[1] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[1]]
    anomalies[0]=pd.to_datetime(anomalies[0])
    anomalies[1]=pd.to_datetime(anomalies[1])
    final_list=[]
    for i in range(len(anomalies)):
        price=priceseries[priceseries.index>=anomalies[0][i]]
        price=price[price.index<=anomalies[1][i]]
        final_list.append([anomalies[2][i],variation(price)])
    #return pd.DataFrame(np.array(final_list))
    print(len(anomalies[anomalies[2]!=' Normal_train'])/len(anomalies))
    return 0
#final=variation_in_data(lucknow_high_anomaly,lucknow_mandi)
#print(final)




def baseline_model(anomalies,priceseries):
    anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[0]]
    anomalies[1] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[1]]
    anomalies[0]=pd.to_datetime(anomalies[0])
    anomalies[1]=pd.to_datetime(anomalies[1])
    anomalies.loc[anomalies[2]!='no',2]=1
    anomalies.loc[anomalies[2]=='no',2]=0
    actual_labels=[]
    predicted_labels=[]
    #print(anomalies)
    variation_list=[]
    for i in range(len(anomalies)):
        price=np.array(priceseries[anomalies[0][i]:anomalies[1][i]])
        x=variation(price)
        variation_list.append(x)
        if(x<0.05):
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)
        actual_labels.append(anomalies[2][i])
    variation_list.sort()
    print(variation_list)
    print(len(actual_labels),len(predicted_labels))
    return actual_labels,predicted_labels

#print(baseline_model(kolkata_low_anomaly,delhi_retail))

def final_model(delhi_anomalies,mumbai_anomalies,priceseriesdelhi,priceseriesmumbai):
    final_actual=[]
    final_predicted=[]
    
    actual,predicted=baseline_model(delhi_anomalies,priceseriesdelhi)
    final_actual+=actual
    final_predicted+=predicted

    actual,predicted=baseline_model(mumbai_anomalies,priceseriesmumbai)
    final_actual+=actual
    final_predicted+=predicted
    print('Accuracy:')
    print(accuracy_score(final_actual,final_predicted))
    print('f1-score:')
    print(f1_score(final_actual,final_predicted,average='weighted'))
    
final_model(kolkata_low_anomaly,lucknow_low_anomaly,kalyani_mandi,lucknow_mandi)






