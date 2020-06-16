#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:21:13 2019

@author: ictd
"""
import pandas as pd
import numpy as np
from datetime import datetime
START='2006-01-01'
END='2018-12-31'
RP=2
CENTREID = 1
  

mandi_info = pd.read_csv('../Data/information/mandis.csv')

dict_centreid_mandicode = mandi_info.groupby('centreid')['mandicode'].apply(list).to_dict()
dict_mandicode_mandiname = mandi_info.groupby('mandicode')['mandiname'].apply(list).to_dict()
dict_mandicode_statecode = mandi_info.groupby('mandicode')['statecode'].apply(list).to_dict()
dict_mandicode_centreid = mandi_info.groupby('mandicode')['centreid'].apply(list).to_dict()
dict_mandiname_mandicode = mandi_info.groupby('mandiname')['mandicode'].apply(list).to_dict()

centre_info = pd.read_csv('../Data/information/centres.csv')
dict_centreid_centrename = centre_info.groupby('centreid')['centrename'].apply(list).to_dict()
dict_centreid_statecode = centre_info.groupby('centreid')['statecode'].apply(list).to_dict()
dict_statecode_centreid = centre_info.groupby('statecode')['centreid'].apply(list).to_dict()
dict_centrename_centreid = centre_info.groupby('centrename')['centreid'].apply(list).to_dict()

state_info = pd.read_csv('../Data/information/states.csv')
dict_statecode_statename = state_info.groupby('statecode')['state'].apply(list).to_dict() 
dict_statename_statecode = state_info.groupby('state')['statecode'].apply(list).to_dict()

def load_retail_data():
  RP = 2
  CENTREID = 1
  retailP = pd.read_csv('../Data/Processed/retailoniondata.csv', header=None)
  retailP = retailP[retailP[RP] != 0]
  retailP = retailP[np.isfinite(retailP[RP])]
  retailP = retailP[retailP[0] >= START]
  retailP = retailP[retailP[0] <= END]
  retailP = retailP.drop_duplicates(subset=[0, 1], keep='last')
  retailP = retailP[retailP[0] >= START]
  retailP = retailP[retailP[0] <= END]
  return retailP
retailP = load_retail_data()


def CreateCentreSeries(Centre, RetailPandas):
  rc = RetailPandas[RetailPandas[1] == Centre]
  rc.groupby(0, group_keys=False).mean()
  rc = rc.sort_values([0], ascending=[True])
  rc[3] = rc.apply(lambda row: datetime.strptime(row[0], '%Y-%m-%d'), axis=1)
  rc.drop(rc.columns[[0, 1]], axis=1, inplace=True)
  rc.set_index(3, inplace=True)
  rc.index.names = [None]
  idx = pd.date_range(START, END)
  rc = rc.reindex(idx, fill_value=0)
  return rc * 100

def RemoveNaNFront(series):
  index = 0
  while True:
    if(not np.isfinite(series[index])):
      index += 1
    else:
      break
  if(index < len(series)):
    for i in range(0, index):
      series[i] = series[index]
  return series

def getcenter(centrename):
  code = dict_centrename_centreid[centrename][0]
  series = CreateCentreSeries(code,retailP)
  price = series[RP]   
  price = price.replace(0.0, np.NaN, regex=True)
  price = price.interpolate(method='pchip')
  price = RemoveNaNFront(price)
  return price


def load_wholesale_data():
  WP = 7
  WA = 2
  wholeSalePA = pd.read_csv('../Data/Processed/wholesaledataprocessed.csv', header=None)
  wholeSalePA = wholeSalePA[np.isfinite(wholeSalePA[WA])]
  wholeSalePA = wholeSalePA[np.isfinite(wholeSalePA[WP])]
  wholeSalePA = wholeSalePA[wholeSalePA[0] >= START]
  wholeSalePA = wholeSalePA[wholeSalePA[0] <= END]
  wholeSalePA = wholeSalePA.drop_duplicates(subset=[0, 1], keep='last')
  return wholeSalePA

def CreateMandiSeries(Mandi, MandiPandas):
  mc = MandiPandas[MandiPandas[1] == Mandi]
  mc = mc.sort_values([0], ascending=[True])
  mc[8] = mc.apply(lambda row: datetime.strptime(row[0], '%Y-%m-%d'), axis=1)
  mc.drop(mc.columns[[0, 1, 3, 4, 5, 6]], axis=1, inplace=True)
  mc.set_index(8, inplace=True)
  mc.index.names = [None]
  idx = pd.date_range(START, END)
  mc = mc.reindex(idx, fill_value=0)
  return mc



wholeSalePA = load_wholesale_data()

# function to get the mandi arrival series or mandi price series for a particular mandiname based on whether the price variable if False or True
def getmandi(mandiname,price):
  if price:
    switch = 7
  else:
    switch = 2 
  mcode = dict_mandiname_mandicode[mandiname][0]
  series = CreateMandiSeries(mcode,wholeSalePA)
  arrival = series[switch]   
  arrival = arrival.replace(0.0, np.NaN, regex=True)
  arrival = arrival.interpolate(method='pchip')
  arrival = RemoveNaNFront(arrival)
  return arrival

#print(getcenter('DELHI'))
#azadpur_mandi=getmandi('Azadpur',True)
#lasalgaon_mandi=getmandi('Lasalgaon',True)
#bangalore_mandi=getmandi('Bangalore',True)
#azadpur_arrival=getmandi('Azadpur',False)
#lasalgaon_arrival=getmandi('Lasalgaon',False)
#bangalore_arrival=getmandi('Bangalore',False)
#bangalore_retail=getcenter('BENGALURU')
#delhi_retail=getcenter('DELHI')
#mumbai_retail=getcenter('MUMBAI')

mumbai_retail=pd.read_csv('../Data/Processed/mumbai_retail.csv',header=None)
mumbai_retail.index=pd.to_datetime(mumbai_retail[0])
mumbai_retail=mumbai_retail[1]

delhi_retail=pd.read_csv('../Data/Processed/delhi_retail.csv',header=None)
delhi_retail.index=pd.to_datetime(delhi_retail[0])
delhi_retail=delhi_retail[1]

bangalore_retail=pd.read_csv('../Data/Processed/bangalore_retail.csv',header=None)
bangalore_retail.index=pd.to_datetime(bangalore_retail[0])
bangalore_retail=bangalore_retail[1]

lasalgaon_mandi=pd.read_csv('../Data/Processed/lasalgaon_mandi.csv',header=None)
lasalgaon_mandi.index=pd.to_datetime(lasalgaon_mandi[0])
lasalgaon_mandi=lasalgaon_mandi[1]

bangalore_mandi=pd.read_csv('../Data/Processed/bangalore_mandi.csv',header=None)
bangalore_mandi.index=pd.to_datetime(bangalore_mandi[0])
bangalore_mandi=bangalore_mandi[1]

azadpur_mandi=pd.read_csv('../Data/Processed/azadpur_mandi.csv',header=None)
azadpur_mandi.index=pd.to_datetime(azadpur_mandi[0])
azadpur_mandi=azadpur_mandi[1]


lasalgaon_arrival=pd.read_csv('../Data/Processed/lasalgaon_arrival.csv',header=None)
lasalgaon_arrival.index=pd.to_datetime(lasalgaon_arrival[0])
lasalgaon_arrival=lasalgaon_arrival[1]

bangalore_arrival=pd.read_csv('../Data/Processed/bangalore_arrival.csv',header=None)
bangalore_arrival.index=pd.to_datetime(bangalore_arrival[0])
bangalore_arrival=bangalore_arrival[1]

azadpur_arrival=pd.read_csv('../Data/Processed/azadpur_arrival.csv',header=None)
azadpur_arrival.index=pd.to_datetime(azadpur_arrival[0])
azadpur_arrival=azadpur_arrival[1]

col_names1=['date','price']
col_names2=['date','arrival']

##ARRIVAL

azadpur_arrival_arima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Azadpur/azadpur_arrival_arima.csv',names=col_names2,header=0)
azadpur_arrival_arimax1=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Azadpur/azadpur_arrival_arimax.csv',names=col_names2,header=0)

azadpur_arrival_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Azadpur/azadpur_arrival_arimax2.csv',header=None)



azadpur_arrival_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Azadpur/azadpur_arrival_sarima.csv',header=None)
azadpur_arrival_sarima=azadpur_arrival_sarima[1]

azadpur_arrival_sarimax=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Azadpur/azadpur_arrival_sarima.csv',names=col_names2,header=0)
#azadpur_arrival_lstm=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Azadpur/azadpur_arrival_lstm.csv',names=col_names2,header=0)


lasalgaon_arrival_arima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Lasalgaon/lasalgaon_arrival_arima.csv',names=col_names2,header=0)
lasalgaon_arrival_arimax1=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Lasalgaon/lasalgaon_arrival_arimax.csv',names=col_names2,header=0)
lasalgaon_arrival_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Lasalgaon/lasalgaon_arrival_sarima_arimax.csv',header=None)
lasalgaon_arrival_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Lasalgaon/lasalgaon_arrival_sarima.csv',names=col_names2,header=0)
lasalgaon_arrival_sarimax=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Lasalgaon/lasalgaon_arrival_sarima.csv',names=col_names2,header=0)
#lasalgaon_arrival_lstm=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Lasalgaon/lasalgaon_arrival_lstm.csv',names=col_names2,header=0)

bangalore_arrival_arima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Bangalore/bangalore_arrival_arima.csv',names=col_names2,header=0)
bangalore_arrival_arimax1=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Bangalore/bangalore_arrival_arimax.csv',names=col_names2,header=0)
bangalore_arrival_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Bangalore/bangalore_arrival_sarima_arimax.csv',header=None)

bangalore_arrival_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Bangalore/bangalore_arrival_sarima.csv',header=None)
bangalore_arrival_sarima=bangalore_arrival_sarima[1]

bangalore_arrival_sarimax=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Bangalore/bangalore_arrival_sarima.csv',names=col_names2,header=0)
#bangalore_arrival_lstm=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/bangalore/bangalore_arrival_lstm.csv',names=col_names2,header=0)

#RETAIL

bangalore_retail_arima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Bangalore/bangalore_retail_arima.csv',names=col_names1,header=0)
bangalore_retail_arimax1=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Bangalore/bangalore_retail_arimax.csv',names=col_names1,header=0)

bangalore_retail_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Bangalore/bangalore_retail_arimax2.csv',header=None)
bangalore_retail_arimax2.index=pd.DatetimeIndex(bangalore_retail_arimax2[0])
bangalore_retail_arimax2=bangalore_retail_arimax2[1]

bangalore_retail_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Bangalore/bangalore_retail_sarima.csv',header=None)
bangalore_retail_sarima=pd.DatetimeIndex(bangalore_retail_sarima[1])
bangalore_retail_sarimax=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Bangalore/bangalore_retail_sarimax.csv',names=col_names1,header=0)
#bangalore_retail_lstm=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Bangalore/bangalore_retail_lstm.csv',names=col_names1,header=0)

delhi_retail_arima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Delhi/delhi_retail_arima.csv',names=col_names1,header=0)
delhi_retail_arimax1=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Delhi/delhi_retail_arimax.csv',names=col_names1,header=0)

delhi_retail_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Delhi/delhi_retail_arimax2.csv',header=None)
delhi_retail_arimax2.index=pd.DatetimeIndex(delhi_retail_arimax2[0])
delhi_retail_arimax2=delhi_retail_arimax2[1]

delhi_retail_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Delhi/delhi_retail_sarima.csv',header=None)
delhi_retail_sarima=delhi_retail_sarima[1]
delhi_retail_sarimax=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Delhi/delhi_retail_sarimax.csv',names=col_names1,header=0)
#delhi_retail_lstm=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/delhi/delhi_retail_lstm.csv',names=col_names1,header=0)

mumbai_retail_arima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Mumbai/mumbai_retail_arima.csv',names=col_names1,header=0)
mumbai_retail_arimax1=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Mumbai/mumbai_retail_arimax.csv',names=col_names1,header=0)

mumbai_retail_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Mumbai/mumbai_retail_arimax2.csv',header=None)
mumbai_retail_arimax2.index=pd.DatetimeIndex(mumbai_retail_arimax2[0])
mumbai_retail_arimax2=mumbai_retail_arimax2[1]

mumbai_retail_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Mumbai/mumbai_retail_sarima.csv',header=None)
mumbai_retail_sarima=mumbai_retail_sarima[1]
mumbai_retail_sarimax=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Mumbai/mumbai_retail_sarimax.csv',names=col_names1,header=0)
#mumbai_retail_lstm=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/mumbai/mumbai_retail_lstm.csv',names=col_names1,header=0)



#MANDI

azadpur_mandi_arima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Azadpur/azadpur_mandi_arima.csv',names=col_names1,header=0)
azadpur_mandi_arimax1=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Azadpur/azadpur_mandi_arimax.csv',names=col_names1,header=0)
azadpur_mandi_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Azadpur/azadpur_mandi_sarima_arimax.csv',header=None)

azadpur_mandi_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Azadpur/azadpur_mandi_sarima.csv',header=None)
azadpur_mandi_sarima.index=pd.DatetimeIndex(azadpur_mandi_sarima[0])
azadpur_mandi_sarima=azadpur_mandi_sarima[1]

azadpur_mandi_sarimax=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Azadpur/azadpur_mandi_sarimax.csv',names=col_names1,header=0)
azadpur_mandi_lstm=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Azadpur/azadpur_mandi_lstm.csv',names=col_names1,header=0)

lasalgaon_mandi_arima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Lasalgaon/lasalgaon_mandi_arima.csv',names=col_names1,header=0)


lasalgaon_mandi_arimax1=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Lasalgaon/lasalgaon_mandi_arimax.csv',names=col_names1,header=0)
lasalgaon_mandi_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Lasalgaon/lasalgaon_mandi_sarima_arimax.csv',header=None)

lasalgaon_mandi_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Lasalgaon/lasalgaon_mandi_sarima.csv',header=None)
lasalgaon_mandi_sarima=pd.to_datetime(lasalgaon_mandi_sarima[1])

lasalgaon_mandi_sarimax=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Lasalgaon/lasalgaon_mandi_sarimax.csv',names=col_names1,header=0)
lasalgaon_mandi_lstm=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Lasalgaon/lasalgaon_mandi_lstm.csv',names=col_names1,header=0)

bangalore_mandi_arima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Bangalore/bangalore_mandi_arima.csv',names=col_names1,header=0)
bangalore_mandi_arimax1=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Bangalore/bangalore_mandi_arimax.csv',names=col_names1,header=0)
bangalore_mandi_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Bangalore/bangalore_mandi_sarima_arimax.csv',header=None)

bangalore_mandi_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Bangalore/bangalore_mandi_sarima.csv',header=None)
bangalore_mandi_sarima.index=pd.DatetimeIndex(bangalore_mandi_sarima[0])
bangalore_mandi_sarima=bangalore_mandi_sarima[1]

bangalore_mandi_sarimax=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Bangalore/bangalore_mandi_sarimax.csv',names=col_names1,header=0)
bangalore_mandi_lstm=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Bangalore/bangalore_mandi_lstm.csv',names=col_names1,header=0)


#--------------- Neighbouring Mandis onion----------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------- 
#MANDI

yeola_mandi=pd.read_csv('../Data/Processed/yeola_mandi.csv',header=None)
yeola_mandi.index=yeola_mandi[0]
yeola_mandi=yeola_mandi[1]

keshopur_mandi=pd.read_csv('../Data/Processed/keshopur_mandi.csv',header=None)
keshopur_mandi.index=keshopur_mandi[0]
keshopur_mandi=keshopur_mandi[1]

davangere_mandi=pd.read_csv('../Data/Processed/davangere_mandi.csv',header=None)
davangere_mandi.index=davangere_mandi[0]
davangere_mandi=davangere_mandi[1]

#ARRIVAL

yeola_arrival=pd.read_csv('../Data/Processed/yeola_arrival.csv',header=None)
yeola_arrival.index=yeola_arrival[0]
yeola_arrival=yeola_arrival[1]

keshopur_arrival=pd.read_csv('../Data/Processed/keshopur_arrival.csv',header=None)
keshopur_arrival.index=keshopur_arrival[0]
keshopur_arrival=keshopur_arrival[1]

davangere_arrival=pd.read_csv('../Data/Processed/davangere_mandi.csv',header=None)
davangere_arrival.index=davangere_arrival[0]
davangere_arrival=davangere_arrival[1]


#Anomaly DATA

azadpur_low_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/azadpurAnomalyNoAnomaly.csv',header=None)
#azadpur_low_anomaly[0]=pd.to_datetime(azadpur_low_anomaly[0])
#azadpur_low_anomaly[1]=pd.to_datetime(azadpur_low_anomaly[1])

lasalgaon_low_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/lasalgaonAnomalyNoAnomaly.csv',header=None)
#lasalgaon_low_anomaly[0]=pd.to_datetime(lasalgaon_low_anomaly[0])
#lasalgaon_low_anomaly[1]=pd.to_datetime(lasalgaon_low_anomaly[1])


bangalore_low_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/bangaloreAnomalyNoAnomaly.csv',header=None)
#bangalore_low_anomaly[0]=pd.to_datetime(bangalore_low_anomaly[0])
#bangalore_low_anomaly[1]=pd.to_datetime(bangalore_low_anomaly[1])

azadpur_high_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/normal_h_w_delhi.csv',header=None)
lasalgaon_high_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/normal_h_w_mumbai1.csv',header=None)
bangalore_high_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/normal_h_w_bangalore.csv',header=None)





#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        #;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
                #[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]
#Potato_Data
                
kalna_mandi=pd.read_csv('../Data/Processed/kalna_mandi.csv',header=None)[1]
kalna_arrival=pd.read_csv('../Data/Processed/kalna_arrival.csv',header=None)[1]

kalyani_mandi=pd.read_csv('../Data/Processed/kalyani_mandi.csv',header=None)
kalyani_mandi.index=pd.DatetimeIndex(kalyani_mandi[0])
kalyani_mandi=kalyani_mandi[1]


kalyani_arrival=pd.read_csv('../Data/Processed/kalyani_arrival.csv',header=None)
kalyani_arrival.index=pd.DatetimeIndex(kalyani_arrival[0])
kalyani_arrival=kalyani_arrival[1]

lucknow_mandi=pd.read_csv('../Data/Processed/lucknow_mandi.csv',header=None)
lucknow_mandi.index=pd.DatetimeIndex(lucknow_mandi[0])
lucknow_mandi=lucknow_mandi[1]

lucknow_arrival=pd.read_csv('../Data/Processed/lucknow_arrival.csv',header=None)
lucknow_arrival.index=pd.DatetimeIndex(lucknow_arrival[0])
lucknow_arrival=lucknow_arrival[1]


bijnaur_mandi=pd.read_csv('../Data/Processed/bijnaur_mandi.csv',header=None)
bijnaur_arrival=pd.read_csv('../Data/Processed/bijnaur_arrival.csv',header=None)

kolkata_retail=pd.read_csv('../Data/Processed/kolkata_retail.csv',header=None)
kolkata_retail.index=pd.DatetimeIndex(kolkata_retail[0])
kolkata_retail=kolkata_retail[1]

lucknow_retail=pd.read_csv('../Data/Processed/lucknow_retail.csv',header=None)
lucknow_retail.index=pd.DatetimeIndex(lucknow_retail[0])
lucknow_retail=lucknow_retail[1]

#-----------------------POTATO RETAIL RESULTS----------------------------------
lucknow_retail_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Lucknow/lucknow_retail_sarima.csv',header=None)
lucknow_retail_sarima=lucknow_retail_sarima[1]

kolkata_retail_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Kolkata/kolkata_retail_sarima.csv',header=None)
kolkata_retail_sarima=kolkata_retail_sarima[1]

#lucknow_retail_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Lucknow/lucknow_retail_arimax2.csv',header=None)
#lucknow_retail_arimax2.index=pd.DatetimeIndex(lucknow_retail_arimax2[0])
#lucknow_retail_arimax2=lucknow_retail_arimax2[1]
#
#kolkata_retail_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Kolkata/kolkata_retail_arimax2.csv',header=None)
#kolkata_retail_arimax2.index=pd.DatetimeIndex(kolkata_retail_arimax2[0])
#kolkata_retail_arimax2=kolkata_retail_arimax2[1]

#-----------------------POTATO MANDI RESULTS----------------------------------
#lucknow_mandi_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Lucknow/lucknow_mandi_sarima.csv',header=None)
#lucknow_mandi_sarima.index=pd.to_datetime(lucknow_mandi_sarima)
#lucknow_mandi_sarima=lucknow_mandi_sarima[1]

kalyani_mandi_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Kalyani/kalyani_mandi_sarima.csv',header=None)
kalyani_mandi_sarima=kalyani_mandi_sarima[1]

#lucknow_mandi_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Lucknow/lucknow_mandi_arimax2.csv',header=None)
#lucknow_mandi_arimax2.index=pd.DatetimeIndex(lucknow_mandi_arimax2[0])
#lucknow_mandi_arimax2=lucknow_retail_arimax2[1]

#kalyani_mandi_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Kalyani/kalyani_mandi_arimax2.csv',header=None)
#kalyani_mandi_arimax2.index=pd.DatetimeIndex(kalyani_mandi_arimax2[0])
#kalyani_mandi_arimax2=kalyani_mandi_arimax2[1]


#-----------------------POTATO ARRIVAL RESULTS----------------------------------
lucknow_arrival_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Lucknow/lucknow_arrival_sarima.csv',header=None)
lucknow_arrival_sarima=lucknow_arrival_sarima[1]

kalyani_arrival_sarima=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Kalyani/kalyani_arrival_sarima.csv',header=None)
kalyani_arrival_sarima=kalyani_arrival_sarima[1]


#lucknow_arrival_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Lucknow/lucknow_arrival_arimax2.csv',header=None)
#lucknow_arrival_arimax2.index=pd.DatetimeIndex(lucknow_arrival_arimax2[0])
#lucknow_arrival_arimax2=lucknow_arrival_arimax2[1]
#
#kalyani_arrival_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Kalyani/kalyani_arrival_arimax2.csv',header=None)
#kalyani_arrival_arimax2.index=pd.DatetimeIndex(kalyani_arrival_arimax2[0])
#kalyani_arrival_arimax2=kalyani_arrival_arimax2[1]

#-----------------------------------------------------------------------------------#

kolkata_low_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/kolkataAnomalyNoAnomaly.csv',header=None)
lucknow_low_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/lucknowAnomalyNoAnomaly.csv',header=None)

#kolkata_high_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/normal_h_w_lucknow.csv',header=None)
lucknow_high_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/normal_h_w_lucknow.csv',header=None)
kolkata_high_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/normal_h_w_kolkata11.csv',header=None)

#-----------------------------------------------------
#-----------------------------------------------------
#---------------------------------------------
#------------------------------------------------#
#---------------------------------------------------------
#  ARIMAX SERIES 
 

lasalgaon_mandi_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Lasalgaon/lasalgaon_mandi_arimax2.csv',header=None)
lasalgaon_mandi_arimax2.index=pd.DatetimeIndex(lasalgaon_mandi_arimax2[0])
lasalgaon_mandi_arimax2=lasalgaon_mandi_arimax2[1]

bangalore_mandi_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Bangalore/bangalore_mandi_arimax2.csv',header=None)
bangalore_mandi_arimax2.index=pd.DatetimeIndex(bangalore_mandi_arimax2[0])
bangalore_mandi_arimax2=bangalore_mandi_arimax2[1]

azadpur_mandi_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Azadpur/azadpur_mandi_arimax2.csv',header=None)
azadpur_mandi_arimax2.index=pd.DatetimeIndex(azadpur_mandi_arimax2[0])
azadpur_mandi_arimax2=azadpur_mandi_arimax2[1]


lucknow_mandi_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Lucknow/lucknow_mandi_arimax2.csv',header=None)
lucknow_mandi_arimax2.index=pd.DatetimeIndex(lucknow_mandi_arimax2[0])
lucknow_mandi_arimax2=lucknow_mandi_arimax2[1]
 
kalyani_mandi_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Mandi/Kalyani/kalyani_mandi_arimax2.csv',header=None)
kalyani_mandi_arimax2.index=pd.DatetimeIndex(kalyani_mandi_arimax2[0])
kalyani_mandi_arimax2=kalyani_mandi_arimax2[1]




lasalgaon_arrival_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Lasalgaon/lasalgaon_arrival_arimax2.csv',header=None)
lasalgaon_arrival_arimax2.index=pd.DatetimeIndex(lasalgaon_arrival_arimax2[0])
lasalgaon_arrival_arimax2=lasalgaon_arrival_arimax2[1]

bangalore_arrival_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Bangalore/bangalore_arrival_arimax2.csv',header=None)
bangalore_arrival_arimax2.index=pd.DatetimeIndex(bangalore_arrival_arimax2[0])
bangalore_arrival_arimax2=bangalore_arrival_arimax2[1]

azadpur_arrival_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Azadpur/azadpur_arrival_arimax2.csv',header=None)
azadpur_arrival_arimax2.index=pd.DatetimeIndex(azadpur_arrival_arimax2[0])
azadpur_arrival_arimax2=azadpur_arrival_arimax2[1]


lucknow_arrival_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Lucknow/lucknow_arrival_arimax2.csv',header=None)
lucknow_arrival_arimax2.index=pd.DatetimeIndex(lucknow_arrival_arimax2[0])
lucknow_arrival_arimax2=lucknow_arrival_arimax2[1]

kalyani_arrival_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Arrival/Kalyani/kalyani_arrival_arimax2.csv',header=None)
kalyani_arrival_arimax2.index=pd.DatetimeIndex(kalyani_arrival_arimax2[0])
kalyani_arrival_arimax2=kalyani_arrival_arimax2[1]



mumbai_retail_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Mumbai/mumbai_retail_arimax2.csv',header=None)
mumbai_retail_arimax2.index=pd.DatetimeIndex(mumbai_retail_arimax2[0])
mumbai_retail_arimax2=mumbai_retail_arimax2[1]

bangalore_retail_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Bangalore/bangalore_retail_arimax2.csv',header=None)
bangalore_retail_arimax2.index=pd.DatetimeIndex(bangalore_retail_arimax2[0])
bangalore_retail_arimax2=bangalore_retail_arimax2[1]

delhi_retail_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Delhi/delhi_retail_arimax2.csv',header=None)
delhi_retail_arimax2.index=pd.DatetimeIndex(delhi_retail_arimax2[0])
delhi_retail_arimax2=delhi_retail_arimax2[1]


lucknow_retail_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Lucknow/lucknow_retail_arimax2.csv',header=None)
lucknow_retail_arimax2.index=pd.DatetimeIndex(lucknow_retail_arimax2[0])
lucknow_retail_arimax2=lucknow_retail_arimax2[1]

kolkata_retail_arimax2=pd.read_csv('/home/ictd/nishant/research/Project/Results/Retail/Kolkata/kolkata_retail_arimax2.csv',header=None)
kolkata_retail_arimax2.index=pd.DatetimeIndex(kolkata_retail_arimax2[0])
kolkata_retail_arimax2=kolkata_retail_arimax2[1]


#print(kolkata_low_anomaly)





########### DUMMY ANOMALIES #################################################################
#-------------------
#----------------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------
#-------------------------------------------------------------
#------------------------------------------------
#-------------------------------------------------------------
#-----------------------------------------------------------------------
#----------------------------------------------------------------------------



kolkata_low_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/Dummy/272020/kolkatalowanomalies.csv',header=None)
lucknow_low_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/Dummy/272020/lucknowlowanomalies.csv',header=None)


kolkata_high_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/Dummy/272020/kolkatahighanomalies.csv',header=None)
lucknow_high_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/Dummy/272020/lucknowhighanomalies.csv',header=None)

lasalgaon_high_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/Dummy/272020/lasalgaonhighanomalies.csv',header=None)
bangalore_high_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/Dummy/272020/bangalorehighanomalies.csv',header=None)
azadpur_high_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/Dummy/272020/azadpurhighanomalies.csv',header=None)

lasalgaon_low_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/Dummy/272020/lasalgaonlowanomalies.csv',header=None)
bangalore_low_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/Dummy/272020/bangalorelowanomalies.csv',header=None)
azadpur_low_anomaly=pd.read_csv('/home/ictd/nishant/research/Project/Data/Anomaly/Dummy/272020/azadpurlowanomalies.csv',header=None)







