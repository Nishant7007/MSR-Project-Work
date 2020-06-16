import time
import datetime
import os
import csv
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import pickle
from liveWholesaleCrawler import *
from liveRetailCrawler import *
from liveCommon import *
import pickle
from liveSystemFunctions import *
from liveProcessData import *
from liveMakeSeries import *
from liveLoadSeries import *
from liveFormatWholesaleData import *
from liveProcessRetailData import *
from liveNearForecasting import *
from liveForecasting import *
from liveParamaterFinder import *
import sys 
from liveSeperateRetailData import *
from liveMakeWebsiteSeries import *
from liveMakeWebsiteRawSeries import *


myDicts = pickle.load(open ("myDicts.p","rb"))

print('Dictionay is:',myDicts)

print('**************************************DOWNLOADING WHOLESALE DATA*******************************************')
x1 = myDicts['monthsDict']
returnedWholesaleDict = wholesaleDataCrawler(myDicts['monthsDict'])
myDicts['monthsDict'] = returnedWholesaleDict
y1 = myDicts['monthsDict']

print('**********************************DOWNLOADING RETAIL DATA*************************************************')
##x2 = myDicts['monthsDictCentres']
##returnedWholesaleDict = retailDataCrawler(myDicts['monthsDictCentres'])
##myDicts['monthsDictCentres'] = returnedWholesaleDict
##y2 = myDicts['monthsDictCentres']

print('*************************************FORMATING WHOLESALE DATA********************************************')
formatData()
print('*************************************PROCESS WHOLESALE DATA*********************************************')
processWholesaleData(states)

print('*************************************PROCESS RETAIL DATA**************************************************')
processRetailData()
print('***************************************MAKE PRICE SERIES**************************************************')
makePriceSeries(myDicts['mandiCodes'])
#makePriceSeries(myDicts['nearMandiCodes'])
print('***************************************MAKE ARRIVAL SERIES************************************************')
makeArrivalSeries(myDicts['mandiCodes'])
#makeArrivalSeries(myDicts['nearMandiCodes'])
print('*****************************MAKE NEIGHBOURING PRICE SERIES************************************************')
makePriceSeries(myDicts['nearMandiCodes'])

print('***************************MAKE NEIGHBOURING ARRIVAL SERIES************************************************')
makeArrivalSeries(myDicts['nearMandiCodes'])
print('------MAKE RETAIL SERIES-------------')
makeRetailSeries(myDicts['centreCodes'])
print('--------SEPERATING RETAIL SERIES CENTREWISE----------')
#seperateRetailSeries()
#print(myDicts)
#myDicts = pickle.load(open ("myDicts.p","rb"))

#print('LOADING RETAIL SERIES')
#retailDict = loadAllRetailSeries(retailSeries,myDicts['centreCodes'])

#print('LOADING MANDI SERIES')
#mandiDict = loadAllMandiSeries(myDicts['mandiCodes'])

#print('LOADING NEIGHBOURING MANDI SERIES')
#nearMandiDict = loadAllMandiSeries(myDicts['nearMandiCodes'])

#print('LOADING ARRIVAL SERIES')
#arrivalDict = loadAllArrivalSeries(myDicts['mandiCodes'])

#print('LOADING NEIGHBOURING ARRIVAL SERIES')
#nearArrivalDict = loadAllArrivalSeries(myDicts['nearMandiCodes'])

# print(nearMandiDict)

# print(nearArrivalDict)

print('*********************************FORECASTING NEIGHBOURING DATA******************************************')

if(x1!=y1):
        print('New wholesale Data Downloaded, So forecasting for near mandis')
        #priceNear()
        #arrivalNear()
else:
        print('no new data downloaded, So no forecasting will be done')


##if(x2!=y2):
  ##      print('New retail Data Downloaded, So forecasting for near centres')
        #retailNear()
##else:
  ##      print('no new data downloaded, So no forecasting will be done *****************************************')

print('*********************************FORECASTING ACTUAL SERIES************************************************')

if(x1!=y1):
        print('New wholesale Data Downloaded, So forecasting for actual mandis')
        #priceActual()
        #arrivalActual()
else:
        print('no new data downloaded, So no forecasting will be done')


##if(x2!=y2):
##        print('New retail Data Downloaded, So forecasting for actual centres')
        #retailActual()
##else:
##        print('no new data downloaded, So no forecasting will be done')
print('-----ACTUAL DATA FORECASTING DONE---------------------------------')

print('********************MAKING RAW SERIES FOR PLOTTING THE DATA IN WEBSITE************************************')

print('----MAKE PRICE RAW SERIES--------')
makePriceRawSeries(myDicts['mandiCodes'])
print('-----MAKE ARRIVAL RAW SERIES-------')
makeArrivalRawSeries(myDicts['mandiCodes'])
print('--------MAKE RETAIL RAW SERIES------------')
makeRetailRawSeries(myDicts['centreCodes'])
print('--------SEPERATE RETAIL RAW SERIES---------')
seperateRetailRawSeries(myDicts['centreCodes'])

print('***********************MAKING PROCESSED SERIES FOR PLOTTING THE DATA IN WEBSITE****************************')
makeRetailFinalSeries(myDicts['centreCodes'])

makePriceFinalSeries(myDicts['mandiCodes'])

makeArrivalFinalSeries(myDicts['mandiCodes'])





print(myDicts)

file1 = open('myDicts.p', 'wb')
pickle.dump(myDicts, file1)

