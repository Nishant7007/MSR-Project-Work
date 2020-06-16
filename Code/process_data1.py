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

files = [f for f in listdir('../Data/Original/Wholesale/Uttar Pradesh')]

code=-1

newfile = open('upfile.csv','w')
lines = []
#print(files)

for j in range(len(files)):
	file=files[j]
	with open('../Data/Original/Wholesale/Uttar Pradesh/'+file) as f:
		print(file)
		content=f.readlines()
		#print(len(content))
		for i in range(1,len(content)):
			temp = content[i].strip().split(',')
			mandi=temp[0]
			if mandi != '':
			    #print(mandi)
			    if mandi in dict_mandiname_mandicode.keys():
			    	code = dict_mandiname_mandicode[mandi][0]
			    else:
			    	code = -1
	    	#print(code)

			# if(temp[1] == ''):
				#print(1)
			date = temp[1]
			#print(date)
			if date != '':
				# if(len(date)<10):
				# 	date=date+str('20')
				#print(date)
				date = datetime.datetime.strptime(date,'%d/%m/%Y').strftime('%Y-%m-%d')
			arrival = temp[2]
			variety = temp[3]
			minp = temp[4]
			maxp = temp[5]
			modalp = temp[6]

			mystr=date+','+str(code)+','+arrival+',NR,'+variety+','+minp+','+maxp+','+modalp+'\n'
			lines.append(mystr)
			#print(mystr)

lines.sort()

for line in lines:
	print(line)
	newfile.write(line)
newfile.close()
print('done')