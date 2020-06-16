from sklearn.metrics import f1_score
from datetime import timedelta
from datetime import datetime
import pandas as pd
import numpy as np
#from constants import CONSTANTS
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score as fs
import seaborn as sns
import matplotlib.pyplot as plt     



import os
cwd = os.getcwd()

'''
'''




#delhilabels = [2,4,1,3,1,2,2,2,3,4,1,2,2,1,4,2,5,5,2,2,3,1,5,4,2,5,5,5,3,5,3,5,2,2,5,2,2,5,5,5,2,5,5,5,2,2,2,3,1,5,1,2]
#lucknowlabels = [2,1,1,2,2,2,5,4,3,1,5,5,5,3,2,2,5,5,4,3,4,5,4,2,5,5,5,5,2,2,3,2,2,5,3,2,5,2]
#mumbailabels = [2,2,2,3,5,1,2,5,2,5,2,2,2,4,2,3,2,3,3,1,1,2,5,5,3,3,2,5,3,5,5,5,2,5,5,5,2,5,2,5,3,2,5,2,5,3,2,1,5,5,2,1,2,2,2,1,5,5,2,2]
#bangalorelabels = [2,2,2,5,2,2,2,2,2,2,5,2,5,5,2,5,5,2,2,5,2,5,2,5,2,5,2,5,5,2,2,2,2,5,5,2]
#print(mumbailabels)
'''
['BHUBANESHWAR']
['DELHI']
['LUCKNOW']
['MUMBAI']
['PATNA']
'''
#print(len(delhilabels),len(mumbailabels),len(lucknowlabels),len(bangalorelabels))





def whiten(series):
  '''
  Whitening Function
  Formula is
    W[x x.T] = E(D^(-1/2))E.T
  Here x: is the observed series
  Read here more:
  https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf
  '''
  import scipy
  EigenValues, EigenVectors = np.linalg.eig(series.cov())
  D = [[0.0 for i in range(0, len(EigenValues))] for j in range(0, len(EigenValues))]
  for i in range(0, len(EigenValues)):
    D[i][i] = EigenValues[i]
  DInverse = np.linalg.matrix_power(D, -1)
  DInverseSqRoot = scipy.linalg.sqrtm(D)
  V = np.dot(np.dot(EigenVectors, DInverseSqRoot), EigenVectors.T)
  series = series.apply(lambda row: np.dot(V, row.T).T, axis=1)
  return series

def whiten_series_list(list):
	for i in range(0,len(list)):
		mean = list[i].mean()
		list[i] -= mean
	temp = pd.DataFrame()
	for i in range(0,len(list)):
		temp[i] = list[i]
	temp = whiten(temp)
	newlist = [temp[i] for i in range(0,len(list))]
	return newlist

# from reading_timeseries import retailP, mandiP, mandiA, retailPM, mandiPM, mandiAM
# retailpriceseriesmumbai = retailP[3]
# retailpriceseriesdelhi = retailP[1]
# retailpriceserieslucknow = retailP[2]
# print retailpriceseriesmumbai
from myaverageretail import getcenter
retailpriceseriesmumbai = getcenter('MUMBAI')
#print('retailpriceseriesmumbai')
retailpriceseriesdelhi = getcenter('DELHI')
retailpriceserieslucknow = getcenter('LUCKNOW')
retailpriceseriesbhub = getcenter('BHUBANESHWAR')
retailpriceseriespatna = getcenter('PATNA')
retailpriceseriesbangalore = getcenter('DELHI')

# [retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai] = whiten_series_list([retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai])

from myaveragemandi import getmandi
mandipriceseriesdelhi = getmandi('Azadpur',True)
mandiarrivalseriesdelhi = getmandi('Azadpur',False)
mandipriceserieslucknow = getmandi('Bahraich',True)
mandiarrivalserieslucknow = getmandi('Bahraich',False)
mandipriceseriesmumbai = getmandi('Lasalgaon',True)
mandiarrivalseriesmumbai = getmandi('Lasalgaon',False)
mandipriceseriesbangalore = getmandi('Bangalore',True)
mandiarrivalseriesbangalore = getmandi('Bangalore',False)

from myaveragemandi import mandipriceseries
from myaveragemandi import mandiarrivalseries
mandipriceseriesmumbai = mandipriceseries
mandiarrivalseriesmumbai = mandiarrivalseries

arimalasalgaon=pd.read_csv('/home/nishant/study/researchwork/Onion/code/forecasting(1)/forecasting/lasalgaon_arima1.csv',header=None)
arimabangalore=pd.read_csv('/home/nishant/study/researchwork/Onion/code/forecasting(1)/forecasting/bangalore_arima1.csv',header=None)
arimaazadpur=pd.read_csv('/home/nishant/study/researchwork/Onion/code/forecasting(1)/forecasting/azadpur_arima1.csv',header=None)

arimaxlasalgaon=pd.read_csv('/home/nishant/study/researchwork/Onion/code/forecastingModels/lasalgaon_arimax.csv',header=None)
arimaxbangalore=pd.read_csv('/home/nishant/study/researchwork/Onion/code/forecastingModels/bangalore_arimax.csv',header=None)
arimaxazadpur=pd.read_csv('/home/nishant/study/researchwork/Onion/code/forecastingModels/lasalgaon_arimax.csv',header=None)

sarimalasalgaon=pd.read_csv('/home/nishant/study/researchwork/Onion/code/forecastingModels/lasalgaon_sarima.csv',header=None)
sarimabangalore=pd.read_csv('/home/nishant/study/researchwork/Onion/code/forecastingModels/bangalore_sarima.csv',header=None)
sarimaazadpur=pd.read_csv('/home/nishant/study/researchwork/Onion/code/forecastingModels/azadpur_sarima.csv',header=None)

sarimaxlasalgaon=pd.read_csv('/home/nishant/study/researchwork/Onion/code/forecastingModels/lasalgaon_sarimax.csv',header=None)
sarimaxbangalore=pd.read_csv('/home/nishant/study/researchwork/Onion/code/forecastingModels/bangalore_sarimax.csv',header=None)
sarimaxazadpur=pd.read_csv('/home/nishant/study/researchwork/Onion/code/forecastingModels/azadpur_sarimax.csv',header=None)


arimalasalgaon.index=pd.date_range('2006-01-01','2018-12-31')
arimalasalgaon=arimalasalgaon[1]
arimabangalore.index=pd.date_range('2006-01-01','2018-12-31')
arimabangalore=arimabangalore[1]
arimaazadpur.index=pd.date_range('2006-01-01','2018-12-31')
arimaazadpur=arimaazadpur[1]

arimaxlasalgaon.index=pd.date_range('2006-01-01','2018-12-31')
arimaxlasalgaon=arimaxlasalgaon[1]
arimaxbangalore.index=pd.date_range('2006-01-01','2018-12-31')
arimaxbangalore=arimaxbangalore[1]
arimaxazadpur.index=pd.date_range('2006-01-01','2018-12-31')
arimaxazadpur=arimaxazadpur[1]

sarimalasalgaon.index=pd.date_range('2006-01-01','2018-12-31')
sarimalasalgaon=sarimalasalgaon[1]
sarimabangalore.index=pd.date_range('2006-01-01','2018-12-31')
sarimabangalore=sarimabangalore[1]
sarimaazadpur.index=pd.date_range('2006-01-01','2018-12-31')
sarimaazadpur=sarimaazadpur[1]

sarimaxlasalgaon.index=pd.date_range('2006-01-01','2018-12-31')
sarimaxlasalgaon=sarimaxlasalgaon[1]
sarimaxbangalore.index=pd.date_range('2006-01-01','2018-12-31')
sarimaxbangalore=sarimaxbangalore[1]
sarimaxazadpur.index=pd.date_range('2006-01-01','2018-12-31')
sarimaxazadpur=sarimaxazadpur[1]




#print('mandipriceseriesmumbai')
# [mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseriesmumbai] = whiten_series_list([mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseries])
# [mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesmumbai] = whiten_series_list([mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseries])
# mandipriceseriesdelhi = mandiP[3]
# mandipriceserieslucknow = mandiP[4]
# mandipriceseriesmumbai = mandiP[5]
# mandiarrivalseriesdelhi = mandiA[3]
# mandiarrivalserieslucknow = mandiA[4]
# mandiarrivalseriesmumbai = mandiA[5]
# print mandipriceseriesdelhi

def Normalise(arr):
  '''
  Normalise each sample
  '''
  m = arr.mean()
  am = arr.min()
  aM = arr.max()
  arr -= m
  arr /= (aM - am)
  return arr

def get_derivative (series):
	#print('series')
	#print(series)
	derivative_series = series
	derivative_series[0] = series[1] - series[0]
	for i in range(1,len(series)-1):
		derivative_series[i] = (series[i+1]-series[i-1])/2.0
	derivative_series[len(series)-1] = series[len(series)-1]-series[len(series)-2]
	#print(derivative_series)
	return derivative_series

#print('mandipriceseriesmumbai')
#print(mandipriceseriesmumbai.shape)
#print(mandipriceseriesmumbai)
mandipriceseriesmumbai_derivative = get_derivative(mandipriceseriesmumbai)
mandipriceseriesdelhi_derivative = get_derivative(mandipriceseriesdelhi)
mandipriceserieslucknow_derivative = get_derivative(mandipriceserieslucknow)
retailpriceseriesmumbai_derivative = get_derivative(retailpriceseriesmumbai)
retailpriceseriesdelhi_derivative = get_derivative(retailpriceseriesmumbai)
retailpriceserieslucknow_derivative = get_derivative(retailpriceseriesmumbai)

retailpriceseriesbangalore_derivative = get_derivative(retailpriceseriesbangalore)
mandipriceseriesbangalore_derivative = get_derivative(mandipriceseriesbangalore)

def adjust_anomaly_window(anomalies,series):
	#print(anomalies)
	for i in range(0,len(anomalies)):
		anomaly_period = series[anomalies[0][i]:anomalies[1][i]]
		mid_date_index = anomaly_period[10:31].idxmax()
		# print type(mid_date_index),mid_date_index
		# mid_date_index - timedelta(days=21)
		anomalies[0][i] = mid_date_index - timedelta(days=21)
		anomalies[1][i] = mid_date_index + timedelta(days=21)
		anomalies[0][i] = datetime.strftime(anomalies[0][i],'%Y-%m-%d')
		anomalies[1][i] = datetime.strftime(anomalies[1][i],'%Y-%m-%d')
	return anomalies

def get_anomalies(path,series):
	anomalies = pd.read_csv(path, header=None, index_col=None)
	#print(path)
	#print(anomalies)
	anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[0]]
	anomalies[1] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[1]]
	anomalies = adjust_anomaly_window(anomalies,series)
	return anomalies

def get_anomalies_year(anomalies):
	mid_date_labels=[]
	for i in range(0,len(anomalies[0])):
		mid_date_labels.append(datetime.strftime(datetime.strptime(anomalies[0][i],'%Y-%m-%d')+timedelta(days=21),'%Y-%m-%d'))
	return mid_date_labels



def newlabels(anomalies):
  # print len(anomalies[anomalies[2] != ' Normal_train']), len(oldlabels)
  labels = []
  #k=0
  #print('anomalies')
  #print(anomalies)
  for i in range(0,len(anomalies)):
    if(anomalies[2][i] != 'no'):
      labels.append(1)
      #print k,oldlabels[k]
      #k = k+1
    else:
      labels.append(0)
  #print('new labels are:')
  #print(labels)
  return labels




def prepare(anomalies,labels,priceserieslist):
	x = []
	for i in range(0,len(anomalies)):
		p=[]
		for j in range(0,len(priceserieslist)):
			# p += (Normalise(np.array(priceserieslist[j][anomalies[0][i]:anomalies[1][i]].tolist()))).tolist()
			p += ((np.array(priceserieslist[j][anomalies[0][i]:anomalies[1][i]].tolist()))).tolist()

			# if(i==0):
			# 	print anomalies[0][i], anomalies[1][i]
		x.append(np.array(p))
		#print(p)
	#print(pd.DataFrame(x).shape)
	return np.array(x),np.array(labels)


def getKey(item):
	return item[0]

def partition(xseries,yseries,year,months):
	# min_month = datetime.strptime(min(year),'%Y-%m-%d')
	# max_month = datetime.strptime(max(year),'%Y-%m-%d')
	combined_series = zip(year,xseries,yseries)
	combined_series = sorted(combined_series,key=getKey)
	train = []
	train_labels = []
	fixed = datetime.strptime('2006-01-01','%Y-%m-%d')
	i=0
	while(fixed < datetime.strptime('2018-12-31','%Y-%m-%d')):
		currx=[]
		curry=[]
		for anomaly in combined_series:
			i += 1
			if(datetime.strptime(anomaly[0],'%Y-%m-%d') > fixed and datetime.strptime(anomaly[0],'%Y-%m-%d')- fixed <= timedelta(days=months*30)):
				currx.append(anomaly[1])
				curry.append(anomaly[2])
		train.append(currx)
		train_labels.append(curry)
		fixed = fixed +timedelta(days = months*30)
	# print fixed
	# train.append(currx)
	# train_labels.append(curry)
	return np.array(train),np.array(train_labels)

def get_score(xtrain,xtest,ytrain,ytest):
	scaler = preprocessing.StandardScaler().fit(xtrain)
	xtrain = scaler.transform(xtrain)
	xtest = scaler.transform(xtest)
	model = RandomForestClassifier(max_depth=2, random_state=0,n_estimators=10)
	# model = SVC( kernel='rbf', C=0.8)
	model.fit(xtrain,ytrain)
	test_pred = np.array(model.predict(xtest))
	# ytest = np.array(ytest)
	# if(test_pred[0] == ytest[0]):
	# 	return 1
	# else:
	# 	return 0
	return test_pred


def train_test_function(align_m,align_d,align_b,data_m,data_d,data_b,names):
	anomaliesmumbai = get_anomalies('/home/nishant/Downloads/lasalgaonAnomalyNonAnomaly.csv',align_m)
	#print('anomaliesmumbai')
	#print(anomaliesmumbai)
	anomaliesdelhi = get_anomalies('/home/nishant/Downloads/azadpurAnomalyNoAnomaly.csv',align_d)
	#anomalieslucknow = get_anomalies('/home/nishant/study/researchwork/Onion/information/normal_h_w_lucknow.csv',align_l)
	anomaliesbangalore = get_anomalies('/home/nishant/Downloads/bangaloreAnomalyNoAnomaly.csv',align_b)

	delhilabelsnew = newlabels(anomaliesdelhi)
	#lucknowlabelsnew = newlabels(anomalieslucknow)
	mumbailabelsnew = newlabels(anomaliesmumbai)
	bangalorelabelsnew = newlabels(anomaliesbangalore)
	#print('mumbailabelsnew')
	#print(mumbailabelsnew)


	delhi_anomalies_year = get_anomalies_year(anomaliesdelhi)
	mumbai_anomalies_year = get_anomalies_year(anomaliesmumbai)
	#lucknow_anomalies_year = get_anomalies_year(anomalieslucknow)
	bangalore_anomalies_year = get_anomalies_year(anomaliesbangalore)

	x1,y1 = prepare(anomaliesdelhi,delhilabelsnew,data_d)
	x2,y2 = prepare(anomaliesmumbai,mumbailabelsnew,data_m)
	#x3,y3 = prepare(anomalieslucknow,lucknowlabelsnew,data_l)
	x3,y3 = prepare(anomaliesbangalore,bangalorelabelsnew,data_b)

	xall = np.array(x1.tolist()+x2.tolist()+x3.tolist())
	yall = np.array(y1.tolist()+y2.tolist()+y3.tolist())
	xall_new =[]
	yall_new = []
	yearall_new = []
	yearall = np.array(delhi_anomalies_year+mumbai_anomalies_year+bangalore_anomalies_year)
	# for x in range(0,len(xall)):
	# 	print len(xall[x]),yall[x]
	for y in range(0,len(yall)):
		if( yall[y] == 1):
			xall_new.append(xall[y])
			yall_new.append(1)
			yearall_new.append(yearall[y])
		elif (yall[y] == 0):
			xall_new.append(xall[y])
			yall_new.append(0)
			yearall_new.append(yearall[y])

	# xall_new = np.array(xall_new)
	# yall_new = np.array(yall_new)
	assert(len(xall_new) == len(yearall_new))
	total_data, total_labels = partition(xall_new,yall_new,yearall_new,6)
	predicted = []
	actual_labels = []
	for i in range(0,len(total_data)):
		if( len(total_data[i]) != 0):
			test_split = total_data[i]
			test_labels = total_labels[i]
			actual_labels = actual_labels + test_labels
			train_split = []
			train_labels_split = []
			for j in range(0,len(total_data)):
				if( j != i):
					train_split = train_split + total_data[j]
					train_labels_split = train_labels_split+total_labels[j]
			pred_test = get_score(train_split,test_split,train_labels_split,test_labels)
			predicted = predicted + pred_test.tolist()
	predicted = np.array(predicted)
	actual_labels= np.array(actual_labels)
	# print len(actual_labels)
	print('accuracy')
	print(len(predicted))
	print(sum(predicted == actual_labels)/136)
	#plot_confusion_matrix(actual_labels, predicted, ['Anomaly','Not Anomaly'],'title_name',cmap=plt.cm.Blues)
	# print actual_labels
	# print predicted
	print('f1 score:')
	print(f1_score(actual_labels,predicted,labels=[0,1],average="macro"))
	cm=confusion_matrix(actual_labels,predicted)
	ax= plt.subplot()
	sns.heatmap(cm, annot=True, ax = ax,fmt='g'); #annot=True to annotate cells

	#labels, title and ticks
	ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
	ax.set_title('Confusion Matrix'); 
	ax.xaxis.set_ticklabels(['Non-Anomalous', 'Anomalous'])
	ax.yaxis.set_ticklabels(['Non-Anomalous ', 'Anomalous '])
	#plt.title('Arima')
	#plt.title('Sarima')
	#plt.title('Sarimax')
	#plt.title('Arimax')
	#plt.title('Arimax-Original')
	plt.title(str(names))
	plt.show()
    #print(fs(actual_labels,predicted))
    #print(f(actual_labels, predicted, labels=None, pos_label=1, average=’binary’, sample_weight=None))

#print(retailpriceseriesmumbai)
print('retail price')
train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceseriesbangalore,[retailpriceseriesmumbai],[retailpriceseriesdelhi],[retailpriceseriesbangalore],'retail price')
print('1')

# print('Arima')
# train_test_function(arimalasalgaon,arimabangalore,arimaazadpur,[arimalasalgaon],[arimabangalore],[arimaazadpur],'Arima')

# print('Arimax')
# train_test_function(arimaxlasalgaon,arimaxbangalore,arimaxazadpur,[arimaxlasalgaon],[arimaxbangalore],[arimaxazadpur],'Arimax')


# print('Sarima')
# train_test_function(sarimalasalgaon,sarimabangalore,sarimalasalgaon,[sarimalasalgaon],[sarimabangalore],[sarimalasalgaon],'Sarima')

# print('Sarimax')
# train_test_function(sarimaxlasalgaon,sarimaxbangalore,sarimaxazadpur,[sarimaxlasalgaon],[sarimaxbangalore],[sarimaxazadpur],'Sarimax')

# print('Lstm')
# train_test_function(arimaxlasalgaon,sarimabangalore,sarimaxazadpur,[arimaxlasalgaon],[sarimabangalore],[sarimaxazadpur],'Lstm')


# print('Arima-original')
# train_test_function(arimalasalgaon-mandipriceseriesmumbai,arimabangalore-mandipriceseriesbangalore,arimaazadpur-mandipriceseriesdelhi,[arimalasalgaon-mandipriceseriesmumbai],[arimabangalore-mandipriceseriesbangalore],[arimaazadpur-mandipriceseriesdelhi],'Arima-original')


# print('Arimax-original')
# train_test_function(arimaxlasalgaon-mandipriceseriesmumbai,arimaxbangalore-mandipriceseriesbangalore,arimaxazadpur-mandipriceseriesdelhi,[arimaxlasalgaon-mandipriceseriesmumbai],[arimaxbangalore-mandipriceseriesbangalore],[arimaxazadpur-mandipriceseriesdelhi],'Arimax-original')

# print('Sarima-original')
# train_test_function(sarimalasalgaon-mandipriceseriesmumbai,sarimabangalore-mandipriceseriesbangalore,sarimalasalgaon-mandipriceseriesmumbai,[sarimalasalgaon-mandipriceseriesmumbai],[sarimabangalore-mandipriceseriesbangalore],[sarimalasalgaon-mandipriceseriesmumbai],'Sarima-original')

# print('Sarimax-original')
# train_test_function(sarimaxlasalgaon-mandipriceseriesmumbai,sarimaxbangalore-mandipriceseriesbangalore,sarimaxazadpur-mandipriceseriesdelhi,[sarimaxlasalgaon-mandipriceseriesmumbai],[sarimaxbangalore-mandipriceseriesbangalore],[sarimaxazadpur-mandipriceseriesdelhi],'Sarimax-original')

# print('Lstm-Original')
# train_test_function(arimaxlasalgaon-mandipriceseriesmumbai,sarimabangalore-mandipriceseriesbangalore,sarimaxazadpur-mandipriceseriesdelhi,[arimaxlasalgaon-mandipriceseriesmumbai],[sarimabangalore-mandipriceseriesbangalore],[sarimaxazadpur-mandipriceseriesdelhi],'Lstm-original')


#print('retail and mandi')
train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesbangalore,[mandipriceseriesmumbai],[mandipriceseriesdelhi],[mandipriceserieslucknow],[mandipriceseriesbangalore])
print('2')
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesbangalore,[retailpriceseriesmumbai,mandipriceseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow],[retailpriceseriesbangalore,mandipriceseriesbangalore])
# print('4')
#train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesbangalore,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow,mandiarrivalserieslucknow],[retailpriceseriesbangalore-mandipriceseriesbangalore,mandiarrivalseriesbangalore])
# print('5')
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesbangalore,[retailpriceseriesmumbai-mandipriceseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow],[retailpriceseriesbangalore-mandipriceseriesbangalore])
# print('6')
#train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesbangalore,[retailpriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandiarrivalserieslucknow],[retailpriceseriesbangalore,mandiarrivalseriesbangalore])
# print('7')
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesbangalore,[retailpriceseriesmumbai,mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow,mandiarrivalserieslucknow],[retailpriceseriesbangalore,mandipriceseriesbangalore,mandiarrivalseriesbangalore])
# print('8')
#train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesbangalore,[retailpriceseriesmumbai/mandipriceseriesmumbai],[retailpriceseriesdelhi/mandipriceseriesdelhi],[retailpriceserieslucknow/mandipriceserieslucknow],[retailpriceseriesbangalore/mandipriceseriesbangalore])
# print('9')
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesbangalore,[retailpriceseriesmumbai,mandipriceseriesmumbai_derivative,retailpriceseriesmumbai_derivative],[retailpriceseriesdelhi,mandipriceseriesdelhi_derivative,retailpriceseriesdelhi_derivative],[retailpriceserieslucknow,mandipriceserieslucknow_derivative,retailpriceserieslucknow_derivative],[retailpriceseriesbangalore,mandipriceseriesbangalore_derivative,retailpriceseriesbangalore_derivative])
print('retail price and mandi price')
#train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceseriesbangalore,[mandipriceseriesmumbai,mandipriceseriesmumbai_derivative,retailpriceseriesmumbai_derivative],[mandipriceseriesdelhi,mandipriceseriesdelhi_derivative,retailpriceseriesdelhi_derivative],[mandipriceseriesbangalore,mandipriceseriesbangalore_derivative,retailpriceseriesbangalore_derivative],'retail price and mandi price')
#print('10')
#train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesbangalore,[retailpriceseriesmumbai,mandipriceseriesmumbai,mandipriceseriesmumbai_derivative,retailpriceseriesmumbai_derivative],[retailpriceseriesdelhi,mandipriceseriesdelhi,mandipriceseriesdelhi_derivative,retailpriceseriesdelhi_derivative],[retailpriceserieslucknow,mandipriceserieslucknow,mandipriceserieslucknow_derivative,retailpriceserieslucknow_derivative],[retailpriceseriesbangalore,mandipriceseriesbangalore,mandipriceseriesbangalore_derivative,retailpriceseriesbangalore_derivative])
#print('11')
#train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesbangalore,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandiarrivalseriesmumbai,mandipriceseriesmumbai_derivative,retailpriceseriesmumbai_derivative],[retailpriceseriesdelhi-mandipriceseriesdelhi,mandiarrivalseriesdelhi,mandipriceseriesdelhi_derivative,retailpriceseriesdelhi_derivative],[retailpriceserieslucknow-mandipriceserieslucknow,mandiarrivalserieslucknow,mandipriceserieslucknow_derivative,retailpriceserieslucknow_derivative],[retailpriceseriesbangalore-mandipriceseriesbangalore,mandiarrivalseriesbangalore,mandipriceseriesbangalore_derivative,retailpriceseriesbangalore_derivative])
#print('12')
#train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesbangalore,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandipriceseriesmumbai_derivative,retailpriceseriesmumbai_derivative],[retailpriceseriesdelhi-mandipriceseriesdelhi,mandipriceseriesdelhi_derivative,retailpriceseriesdelhi_derivative],[retailpriceserieslucknow-mandipriceserieslucknow,mandipriceserieslucknow_derivative,retailpriceserieslucknow_derivative],[retailpriceseriesbangalore-mandipriceseriesbangalore,mandipriceseriesbangalore_derivative,retailpriceseriesbangalore_derivative])
#rint('13')
#train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesbangalore,[retailpriceseriesmumbai,mandiarrivalseriesmumbai,mandipriceseriesmumbai_derivative,retailpriceseriesmumbai_derivative],[retailpriceseriesdelhi,mandiarrivalseriesdelhi,mandipriceseriesdelhi_derivative,retailpriceseriesdelhi_derivative],[retailpriceserieslucknow,mandiarrivalserieslucknow,mandipriceserieslucknow_derivative,retailpriceserieslucknow_derivative],[retailpriceseriesbangalore,mandiarrivalseriesbangalore,mandipriceseriesbangalore_derivative,retailpriceseriesbangalore_derivative])
# print('14')
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesbangalore,[retailpriceseriesmumbai,mandipriceseriesmumbai,mandiarrivalseriesmumbai,mandipriceseriesmumbai_derivative,retailpriceseriesmumbai_derivative],[retailpriceseriesdelhi,mandipriceseriesdelhi,mandiarrivalseriesdelhi,mandipriceseriesdelhi_derivative,retailpriceseriesdelhi_derivative],[retailpriceserieslucknow,mandipriceserieslucknow,mandiarrivalserieslucknow,mandipriceserieslucknow_derivative,retailpriceserieslucknow_derivative],[retailpriceseriesbangalore,mandipriceseriesbangalore,mandiarrivalseriesbangalore,mandipriceseriesbangalore_derivative,retailpriceseriesbangalore_derivative])
# print('15')
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesbangalore,[retailpriceseriesmumbai/mandipriceseriesmumbai,mandipriceseriesmumbai_derivative,retailpriceseriesmumbai_derivative],[retailpriceseriesdelhi/mandipriceseriesdelhi,mandipriceseriesdelhi_derivative,retailpriceseriesdelhi_derivative],[retailpriceserieslucknow/mandipriceserieslucknow,mandipriceserieslucknow_derivative,retailpriceserieslucknow_derivative],[retailpriceseriesbangalore/mandipriceseriesbangalore,mandipriceseriesbangalore_derivative,retailpriceseriesbangalore_derivative])

#print('16')
#train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseriesbangalore,[retailpriceseriesmumbai],[retailpriceseriesdelhi],[retailpriceserieslucknow],[retailpriceseriesbangalore])
print('mandi price')
#train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceseriesbangalore,[mandipriceseriesmumbai],[mandipriceseriesdelhi],[mandipriceseriesbangalore],'mandi price')
# # print('17')

# print('retail price')
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceseriesbangalore,[retailpriceseriesmumbai],[retailpriceseriesdelhi],[retailpriceseriesbangalore],'retail price')

# #train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseriesbangalore,[retailpriceseriesmumbai,mandipriceseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow],[retailpriceseriesbangalore,mandipriceseriesbangalore])
print('mandi price, retail price, arrival')
train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceseriesbangalore,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceseriesbangalore-mandipriceseriesbangalore,mandiarrivalseriesbangalore],'mandi price, retail price, arrival')
# print('18')
#train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseriesbangalore,[retailpriceseriesmumbai-mandipriceseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow],[retailpriceseriesbangalore-mandipriceseriesbangalore])
#print('19')
#train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseriesbangalore,[retailpriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandiarrivalserieslucknow],[retailpriceseriesbangalore,mandiarrivalseriesbangalore])
print('20')
train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseriesbangalore,[retailpriceseriesmumbai,mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow,mandiarrivalserieslucknow],[retailpriceseriesbangalore,mandipriceseriesbangalore,mandiarrivalseriesbangalore])
#print('21')
#train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseriesbangalore,[retailpriceseriesmumbai/mandipriceseriesmumbai],[retailpriceseriesdelhi/mandipriceseriesdelhi],[retailpriceserieslucknow/mandipriceserieslucknow],[retailpriceseriesbangalore/mandipriceseriesbangalore])
#print('29')
#train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,retailpriceseriesbangalore-mandipriceseriesbangalore,[retailpriceseriesmumbai/mandipriceseriesmumbai],[retailpriceseriesdelhi/mandipriceseriesdelhi],[retailpriceserieslucknow/mandipriceserieslucknow],[retailpriceseriesbangalore/mandipriceseriesbangalore])


# # Change the argmax to idxmin for running the part below
#print('30')
#train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesbangalore,[retailpriceseriesmumbai],[retailpriceseriesdelhi],[retailpriceserieslucknow],[retailpriceseriesbangalore])
#print('31')
#train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesbangalore,[mandipriceseriesmumbai],[mandipriceseriesdelhi],[mandipriceserieslucknow],[mandipriceseriesbangalore])
#print('32')
#train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesbangalore,[retailpriceseriesmumbai,mandipriceseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow],[retailpriceseriesbangalore,mandipriceseriesbangalore])
# print('33')
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesbangalore,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow,mandiarrivalserieslucknow],[retailpriceseriesbangalore-mandipriceseriesbangalore,mandiarrivalseriesbangalore])
# print('arrival, retail-mandi')
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalseriesbangalore,[retailpriceseriesmumbai-mandipriceseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi],[retailpriceseriesbangalore-mandipriceseriesbangalore],'arrival, retail-mandi')
# #print('35')
#train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesbangalore,[retailpriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandiarrivalserieslucknow],[retailpriceseriesbangalore,mandiarrivalseriesbangalore])
#print('36')
#train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesbangalore,[retailpriceseriesmumbai,mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow,mandiarrivalserieslucknow],[retailpriceseriesbangalore,mandipriceseriesbangalore,mandiarrivalseriesbangalore])
#print('37')
#train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesbangalore,[retailpriceseriesmumbai/mandipriceseriesmumbai],[retailpriceseriesdelhi/mandipriceseriesdelhi],[retailpriceserieslucknow/mandipriceserieslucknow],[retailpriceseriesbangalore/mandipriceseriesbangalore])


#anomalie = get_anomalies('data/anomaly/normal_h_w_mumbai.csv',align_m)
#print(anomalie)
#

#anomalies = pd.read_csv('data/anomaly/mumbaicheck.csv', header=None, index_col=None)
#print(anomalies)


#residuals and mandi price series
# print('Arima residuals and mandi price')
# train_test_function(arimalasalgaon-mandipriceseriesmumbai,arimabangalore-mandipriceseriesbangalore,arimaazadpur-mandipriceseriesdelhi,[mandipriceseriesmumbai],[mandipriceseriesbangalore],[mandipriceseriesdelhi],'Arima residuals and mandi price')

# print('Arimax residuals and mandi price')
# train_test_function(arimaxlasalgaon-mandipriceseriesmumbai,arimaxbangalore-mandipriceseriesbangalore,arimaxazadpur-mandipriceseriesdelhi,[mandipriceseriesmumbai],[mandipriceseriesbangalore],[mandipriceseriesdelhi],'Arimax residuals and mandi price')

# print('Sarima residuals and mandi price')
# train_test_function(sarimalasalgaon-mandipriceseriesmumbai,sarimabangalore-mandipriceseriesbangalore,sarimaxazadpur-mandipriceseriesdelhi,[mandipriceseriesmumbai],[mandipriceseriesbangalore],[mandipriceseriesdelhi],'Sarima residuals and mandi price')

print('Sarimax residuals and mandi price')
train_test_function(sarimaxlasalgaon-mandipriceseriesmumbai,sarimaxbangalore-mandipriceseriesbangalore,sarimaxlasalgaon-mandipriceseriesmumbai,[mandipriceseriesmumbai],[mandipriceseriesbangalore],[mandipriceseriesmumbai],'Sarimax residuals and mandi price')

# print('LSTM residuals and mandi price')
# train_test_function(arimaxlasalgaon-mandipriceseriesmumbai,sarimabangalore-mandipriceseriesbangalore,sarimaxazadpur-mandipriceseriesdelhi,[mandipriceseriesmumbai],[mandipriceseriesbangalore],[mandipriceseriesdelhi],'LSTM residuals and mandi price')


#residuals and retail price series
# print('Arima residuals and retail price')
# train_test_function(arimalasalgaon-mandipriceseriesmumbai,arimabangalore-mandipriceseriesbangalore,arimaazadpur-mandipriceseriesdelhi,[retailpriceseriesmumbai],[retailpriceseriesbangalore],[retailpriceseriesdelhi],'Arima residuals and retail price')

# print('Arimax residuals and retail price')
# train_test_function(arimaxlasalgaon-mandipriceseriesmumbai,arimaxbangalore-mandipriceseriesbangalore,arimaxazadpur-mandipriceseriesdelhi,[retailpriceseriesmumbai],[retailpriceseriesbangalore],[retailpriceseriesdelhi],'Arimax residuals and retail price')

# print('Sarima residuals and retail price')
# train_test_function(sarimalasalgaon-mandipriceseriesmumbai,sarimabangalore-mandipriceseriesbangalore,sarimaxazadpur-mandipriceseriesdelhi,[retailpriceseriesmumbai],[retailpriceseriesbangalore],[retailpriceseriesdelhi],'Sarima residuals and retail price')

# print('Sarimax residuals and retail price')
# train_test_function(sarimaxlasalgaon-mandipriceseriesmumbai,sarimaxbangalore-mandipriceseriesbangalore,sarimaxlasalgaon-mandipriceseriesmumbai,[retailpriceseriesmumbai],[retailpriceseriesbangalore],[retailpriceseriesmumbai],'Sarimax residuals and retail price')

# print('LSTM residuals and retail price')
# train_test_function(arimaxlasalgaon-mandipriceseriesmumbai,sarimabangalore-mandipriceseriesbangalore,sarimaxazadpur-mandipriceseriesdelhi,[retailpriceseriesmumbai],[retailpriceseriesbangalore],[retailpriceseriesdelhi],'LSTM residuals and retail price')



#residuals and retail Arrival series
# print('Arima residuals and Arrival')
# train_test_function(arimalasalgaon-mandipriceseriesmumbai,arimabangalore-mandipriceseriesbangalore,arimaazadpur-mandipriceseriesdelhi,[mandiarrivalseriesmumbai],[mandiarrivalseriesbangalore],[mandiarrivalseriesdelhi],'Arima residuals and Arrival')

# print('Arimax residuals and Arrival')
# train_test_function(arimaxlasalgaon-mandipriceseriesmumbai,arimaxbangalore-mandipriceseriesbangalore,arimaxazadpur-mandipriceseriesdelhi,[mandiarrivalseriesmumbai],[mandiarrivalseriesbangalore],[mandiarrivalseriesdelhi],'Arimax residuals and Arrival')

# print('Sarima residuals and Arrival')
# train_test_function(sarimalasalgaon-mandipriceseriesmumbai,sarimabangalore-mandipriceseriesbangalore,sarimaxazadpur-mandipriceseriesdelhi,[mandiarrivalseriesmumbai],[mandiarrivalseriesbangalore],[mandiarrivalseriesdelhi],'Sarima residuals and Arrival')

# print('Sarimax residuals and Arrival')
# train_test_function(sarimaxlasalgaon-mandipriceseriesmumbai,sarimaxbangalore-mandipriceseriesbangalore,sarimaxlasalgaon-mandipriceseriesmumbai,[mandiarrivalseriesmumbai],[mandiarrivalseriesbangalore],[mandiarrivalseriesmumbai],'Sarimax residuals and Arrival')

print('LSTM residuals and Arrival')
train_test_function(arimaxlasalgaon-mandipriceseriesmumbai,sarimabangalore-mandipriceseriesbangalore,sarimaxazadpur-mandipriceseriesdelhi,[mandiarrivalseriesmumbai],[mandiarrivalseriesbangalore],[mandiarrivalseriesdelhi],'LSTM residuals and Arrival')
print('arrival, retail-mandi')
train_test_function(sarimaxlasalgaon-mandipriceseriesmumbai,sarimaxbangalore-mandipriceseriesbangalore,sarimaxazadpur-mandipriceseriesdelhi,[retailpriceseriesmumbai-mandipriceseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi],[retailpriceseriesbangalore-mandipriceseriesbangalore],'arrival, retail-mandi')
