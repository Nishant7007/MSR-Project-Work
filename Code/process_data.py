from datetime import datetime
import pandas as pd
import numpy as np
import scipy
#from constants import CONSTANTS
import matplotlib.pyplot as plt
import math
from os import listdir
import datetime as datetime

def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

mandi_info = pd.read_csv('../Data/Information/mandis.csv')
dict_centreid_mandicode = mandi_info.groupby('centreid')['mandicode'].apply(list).to_dict()
dict_mandicode_mandiname = mandi_info.groupby('mandicode')['mandiname'].apply(list).to_dict()
dict_mandicode_statecode = mandi_info.groupby('mandicode')['statecode'].apply(list).to_dict()
dict_mandicode_centreid = mandi_info.groupby('mandicode')['centreid'].apply(list).to_dict()
dict_mandiname_mandicode = mandi_info.groupby('mandiname')['mandicode'].apply(list).to_dict()

centre_info = pd.read_csv('../Data/Information/centres.csv')
dict_centreid_centrename = centre_info.groupby('centreid')['centrename'].apply(list).to_dict()
dict_centreid_statecode = centre_info.groupby('centreid')['statecode'].apply(list).to_dict()
dict_statecode_centreid = centre_info.groupby('statecode')['centreid'].apply(list).to_dict()
dict_centrename_centreid = centre_info.groupby('centrename')['centreid'].apply(list).to_dict()

state_info = pd.read_csv('../Data/Information/states.csv')
dict_statecode_statename = state_info.groupby('statecode')['state'].apply(list).to_dict() 
dict_statename_statecode = state_info.groupby('state')['statecode'].apply(list).to_dict() 

files = [f for f in listdir('../Data/Original/Wholesale/maha2020')]

code=-1

newfile = open('maha202000.csv','w')
lines = []
print(files)

for j in range(len(files)):
	file=files[j]
	with open('../Data/Original/Wholesale/maha2020/'+file) as f:
		#print(file)
		content=f.readlines()
		#print(len(content))
		for i in range(1,len(content)):
			temp = content[i].strip().split(',')
			print('content: ',temp)
			if(len(temp) > 8):
			    temp[0:2] = [''.join(temp[0:2])]
			mandi = temp[0]
			#print(len(temp))
			date = temp[1]
			if(len(temp) < 8):
			    continue
			if date != '':
				print(date)
				date=date+str('20')
				print(date)
				date = datetime.datetime.strptime(date,'%d/%m/%Y').strftime('%Y-%m-%d')
				arrival = temp[2]
				variety = temp[3]
				minp = temp[4]
				maxp = temp[5]
				modalp = temp[6]
			if not isInt(minp):
				minp = '0'
			if not isInt(maxp):
				maxp = '0'
			if not isInt(modalp):
				modalp = '0'
			if mandi != '':
			    #print(mandi)
			    if mandi in dict_mandiname_mandicode.keys():
			    	code = dict_mandiname_mandicode[mandi]
			    else:
			    	code = -1
			    # print(2,mandi)
			#print(code)
			if code != -1 and minp != 'NR':
				mystr=date+','+str(code[0])+','+arrival+',NR,'+variety+','+minp+','+maxp+','+modalp+'\n'
				print(mystr)
				lines.append(mystr)

lines.sort()

for line in lines:
	print(line)
	newfile.write(line)
newfile.close()
print('done')