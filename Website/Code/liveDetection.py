from liveLoadSeries import *

retailDict = loadAllRetailSeries(myDicts['centreCodes'])
mandiDict = loadAllMandiSeries(myDicts['mandiCodes'])
arrivalDict = loadAllArrivalSeries(myDicts['mandiCodes'])


print(retailDict)
print(mandiDict)
print(arrivalDict)
