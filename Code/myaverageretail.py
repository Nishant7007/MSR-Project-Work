# Code to upload the retail price data for various centers
from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib.pyplot as plt
import math
START='2006-01-01'
END='2018-12-31'
RP=2
CENTREID = 1

mandi_info = pd.read_csv('/home/ictd/nishant/research/low_price_Anomaly/Data/information/mandis.csv')

# dictionary: centerid_mandicode,
# 				mandicode_mandiname,
# 				mandicode_statecode,
# 				mandicode_centreid,
# 				mandiname_mandicode
dict_centreid_mandicode = mandi_info.groupby('centreid')['mandicode'].apply(list).to_dict()
dict_mandicode_mandiname = mandi_info.groupby('mandicode')['mandiname'].apply(list).to_dict()
dict_mandicode_statecode = mandi_info.groupby('mandicode')['statecode'].apply(list).to_dict()
dict_mandicode_centreid = mandi_info.groupby('mandicode')['centreid'].apply(list).to_dict()
dict_mandiname_mandicode = mandi_info.groupby('mandiname')['mandicode'].apply(list).to_dict()

# dictionary: centreid_centrename
# 			centreid_statecode
# 			statecode_centreid
# 			centrename_centreid
centre_info = pd.read_csv('/home/ictd/nishant/research/low_price_Anomaly/Data/information/centres.csv')
dict_centreid_centrename = centre_info.groupby('centreid')['centrename'].apply(list).to_dict()
dict_centreid_statecode = centre_info.groupby('centreid')['statecode'].apply(list).to_dict()
dict_statecode_centreid = centre_info.groupby('statecode')['centreid'].apply(list).to_dict()
dict_centrename_centreid = centre_info.groupby('centrename')['centreid'].apply(list).to_dict()

# dictionary: statecode_statename
# 			statename_statecode
state_info = pd.read_csv('/home/ictd/nishant/research/low_price_Anomaly/Data/information/states.csv')
dict_statecode_statename = state_info.groupby('statecode')['state'].apply(list).to_dict() 
dict_statename_statecode = state_info.groupby('state')['statecode'].apply(list).to_dict()
#print('1')




#Getting the statecode and centreIDs for the top 1o states
top_10_states = [ 'Maharashtra', 'Karnataka' , 'Madhya Pradesh', 'Bihar', 'Gujarat', 
'Rajasthan', 'Haryana' , 'Andhra Pradesh' , 'Telangana' , 'Uttar Pradesh']
#relevant_centres_id is the list of all the centresids for top 10 states
relevant_centres_id = []
for state in top_10_states:
  statecode = dict_statename_statecode[state]
  centreids = dict_statecode_centreid[statecode[0]]
  #print(state,statecode,centreids)
  relevant_centres_id = relevant_centres_id + centreids
#print('2')



def load_retail_data():
  RP = 2
  CENTREID = 1
  START='2006-01-01'
  END='2018-12-31'
  #The retailoniondate.csv file has been upfated with the new date: 01/10/2017 -- 31,12/2018 
  retailP = pd.read_csv(CONSTANTS['ORIGINALRETAIL'], header=None)
  retailP = retailP[retailP[RP] != 0]
  retailP = retailP[np.isfinite(retailP[RP])]
  retailP = retailP[retailP[0] >= START]
  retailP = retailP[retailP[0] <= END]
  retailP = retailP.drop_duplicates(subset=[0, 1], keep='last')
  retailP = retailP[retailP[0] >= START]
  retailP = retailP[retailP[0] <= END]
  return retailP
retailP = load_retail_data()
#print(retailP)
#print('3')


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
#print('4')


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
#print('5')

from os import listdir
imagenames = [f for f in listdir('/home/ictd/nishant/research/Low_Price/Data/Wholesale/Processed/bigmandis10')]

# Function to get interpolated retail price series for the particular center
def getcenter(centrename):
  code = dict_centrename_centreid[centrename][0]
  series = CreateCentreSeries(code,retailP)
  price = series[RP]   
  price = price.replace(0.0, np.NaN, regex=True)
  #price = price.interpolate(method='pchip',limit_direction='both')
  price = price.interpolate(method='pchip')
  price = RemoveNaNFront(price)
  return price
  #print(price)
#print('6')

# I think this is useless but not sure
def getcenter2(centrename):
  code = dict_centrename_centreid[centrename][0]
  series = CreateCentreSeries(code,retailP)
  price = series[RP]   
  return price
#print('7')

#  just to check  :  print(getcenter('LUCKNOW'))

def give_df_imagenames_center(imagenames):
  centreSeries = []
  RP=2
  for imagename in imagenames:
    if len(imagename.split('_'))>1:
      imagename = imagename.replace('.','_')
      print(imagename)
      [statename,centrename,mandiname,_] = imagename.split('_')
      price = getcenter(centrename)
      centreSeries.append(price)
      #print(centreSeries)

  centreDF = pd.DataFrame()
  for i in range(0, len(centreSeries)):
    centreDF[i] = centreSeries[i]
  
  return centreDF
# just to check   ::  print(give_df_imagenames_center(imagenames))
#print('8')

def give_average_of_df(mandiDF):
  meanseries = mandiDF.mean(axis=1)
  meanseries = meanseries.replace(0.0, np.NaN, regex=True)
  meanseries = meanseries.interpolate(method='pchip',)
  mandiarrivalseries = RemoveNaNFront(meanseries)
  return mandiarrivalseries
#print('9')


centreDF = give_df_imagenames_center(imagenames)
retailpriceseries = give_average_of_df(centreDF)
#just to check : print(centreDF)
#just to check : print(retailpriceseries)
#print('10')

retailpriceexpected = retailpriceseries.rolling(window=30,center=True).mean()
retailpriceexpected = retailpriceexpected.groupby([retailpriceexpected.index.month, retailpriceexpected.index.day]).mean()
idx = pd.date_range(START, END)
data = [ (retailpriceexpected[index.month][index.day]) for index in idx]
expectedretailprice = pd.Series(data, index=idx)
#print(expectedretailprice)
