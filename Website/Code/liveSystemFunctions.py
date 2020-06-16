
from liveWholesaleCrawler import *
from liveRetailCrawler import *

monthList = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

def wholesaleDataCrawler(Dict):
    centres, months, years = ([],[],[])
    for k in Dict.keys():
        centres.append(k)
        months.append(Dict[k][0])
        years.append(Dict[k][1])
    wholesaleResult = extractWholesaleData(centres, months,years)
    #print(wholesaleResult)
    newDict = {}
    for k in wholesaleResult.keys():
        if(wholesaleResult[k][0] == 1):
            indexValue = monthList.index(wholesaleResult[k][1])
            if((indexValue + 1) == len(monthList)):
                m = monthList[0]
                y = wholesaleResult[k][2] + 1
            else:
                m = monthList[indexValue+1]
                y = wholesaleResult[k][2]
        else:
            m = wholesaleResult[k][1]
            y = wholesaleResult[k][2]
        newDict[k] = [m,y]
        #print(newDict)
    return newDict

def retailDataCrawler(Dict):
    centres, months, years = ([],[],[])
    for k in Dict.keys():
        centres.append(k)
        months.append(Dict[k][0])
        years.append(Dict[k][1])
    retailResult = extractRetailData(centres, months,years)
    #print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    #print(retailResult)
    #print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
    newDict = {}
    for k in retailResult.keys():
        if(retailResult[k][0] == 1):
            indexValue = monthList.index(retailResult[k][1])
            if((indexValue + 1) == len(monthList)):
                m = monthList[0]
                y = retailResult[k][2] + 1
            else:
                m = monthList[indexValue+1]
                y = retailResult[k][2]
        else:
            m = retailResult[k][1]
            y = retailResult[k][2]
        newDict[k] = [m,y]
        #print('1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
        #print(newDict)
        #print('2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222')
    return newDict

