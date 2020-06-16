import pickle 
import pandas as pd
from os import listdir
from datetime import datetime
date_format = "%Y-%m-%d"

myDicts = pickle.load(open ("myDicts.p","rb"))
#print(myDicts)


def fillMissingDatesArrivalandMandi(df):
        # print('inside missing value function:')
        df.columns = [0,1]
        # print(len(df))
        recent_date = df[0].max()
        # print(recent_date)
        l = (datetime.strptime(recent_date, date_format) - datetime.strptime('2006-01-01', date_format)).days
        # print(l)
        df = df.drop_duplicates(subset = 0, keep = 'first')
        df = df.reset_index()
        idx = pd.date_range('2006-01-01', periods = l)
        #print(idx)
        df.index = pd.DatetimeIndex(df[0])
        df = df[[1]]
        #print(df)
        #print (df.index.name)
        df = df.reindex(idx, fill_value = 0)
        #df.replace(0,np.nan,inplace = True)
        # print(df)
        #df = df.interpolate(method = 'linear',limit_direction = 'both')
        df = df[[1]]
        # print(df)
        return df

def makeArrivalRawSeries(dicts):
        # print(dicts)
        states = [i for i in dicts.keys()]
        for state in states:
                #print(state)
                fileName = '../Data/Processed/Wholesale/'+str(state)+'.csv'
                df = pd.read_csv(fileName,header = None)
                df = df[df[1] == dicts[state][1]]
                df.drop_duplicates(subset = 4,keep='first')
                arrival = pd.DataFrame(df[[0,2]])
                arrival = fillMissingDatesArrivalandMandi(arrival)
                fileName = '../Data/Original/Wholesale/'+str(dicts[state][0])+'Arrival.csv'
                arrival.to_csv(fileName, header = False)


def makePriceRawSeries(dicts):
        # print(dicts)
        states = [i for i in dicts.keys()]
        for state in states:
                # print(state)
                fileName = '../Data/Processed/Wholesale/'+str(state)+'.csv'
                # print(fileName)
                df = pd.read_csv(fileName,header = None)
                # print(df.head())
                df = df[df[1] == dicts[state][1]]
                # print(df.head())
                # print(len(df))
                df.drop_duplicates(subset = 4,keep='first')
                # print(len(df))
                price = pd.DataFrame(df[[0,7]])
                # print(len(price))
                price = fillMissingDatesArrivalandMandi(price)
                # print('after fill missing:',len(price))
                fileName = '../Data/Original/Wholesale/'+str(dicts[state][0])+'Price.csv'
                # print(fileName)
                price.to_csv(fileName, header = False)

def makeRetailRawSeries(dicts):
        mainFile = pd.read_csv('../Data/Original/Retail/retailData.csv',header = None,engine='python',error_bad_lines=False)
        # print(mainFile)
        # print(mainFile.columns)
        fileNames = [f for f in listdir('../Data/Processed/Retail') if not f.startswith('.')]
        #print(fileNames)
        for file in fileNames:
                # print(file)
                fileName = '../Data/Processed/Retail/' + str(file)
                #print(fileName)
                df = pd.read_csv(fileName,engine='python',sep='[,]',names =[0,1,2])
                #print(df.head())
                df = df[1:]
                #print(df.head())
                df[0] = df[0].map(lambda x:x.lstrip('\"'))
                df[2] = df[2].map(lambda x:x.rstrip('\"'))
                #print(df.head())
                df.reset_index(inplace = True, drop = True)
                # print(df.columns)
                codesDict = dicts
                for each in codesDict.keys():
                        if(df.iloc[0][1] == codesDict[each][0]):
                                df[1] = codesDict[each][1]
                # print(len(mainFile))
                mainFile = pd.concat([mainFile,df],sort = True)
        mainFile.to_csv('../Data/Original/Retail/retailData.csv',header = False,index = False)


#makeRetailSeries(myDicts['centreCodes'])

def seperateRetailRawSeries(dict):
	df = pd.read_csv('../Data/Original/Retail/retailData.csv',header = None,engine='python',error_bad_lines=False)
	#print(df.head())
	for v in dict.values():
		code = v[1]
		centre = v[0]
		#print(code,centre)
		dx = df[df[1]==code]
		dx.drop_duplicates(subset=[0],inplace=True)
		dx.index = dx[0]
		dx = dx[[2]]
		idx = pd.date_range('2006-01-01', dx.index.max())
		dx.index = pd.DatetimeIndex(dx.index)		
		dx = dx.reindex(idx, fill_value=0)
		#print(dx.head())
		dx.to_csv('../Data/Original/Retail/'+str(centre)+'Retail.csv',header=None)


#print('MAKE PRICE RAW SERIES')
#makePriceRawSeries(myDicts['mandiCodes'])
#print('MAKE ARRIVAL RAW SERIES')
#makeArrivalRawSeries(myDicts['mandiCodes'])
#print('MAKE RETAIL RAW SERIES')
#makeRetailRawSeries(myDicts['centreCodes'])
#print('SEPERATE RETAIL RAW SERIES')
#seperateRetailRawSeries(myDicts['centreCodes'])		
