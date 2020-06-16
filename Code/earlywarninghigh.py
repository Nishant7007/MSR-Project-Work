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

def whiten(series):
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


from myaverageretail import getcenter
retailpriceseriesmumbai = getcenter('MUMBAI')
retailpriceseriesdelhi = getcenter('DELHI')
retailpriceseriesbangalore = getcenter('BENGALURU')

from myaveragemandi import getmandi
mandipriceseriesdelhi = getmandi('Azadpur',True)
mandiarrivalseriesdelhi = getmandi('Azadpur',False)
#mandipriceserieslucknow = getmandi('Bahraich',True)
#mandiarrivalserieslucknow = getmandi('Bahraich',False)
mandipriceseriesmumbai = getmandi('Lasalgaon',True)
mandiarrivalseriesmumbai = getmandi('Lasalgaon',False)
mandipriceseriesbangalore = getmandi('Bangalore',True)
mandiarrivalseriesbangalore = getmandi('Bangalore',False)

arimalasalgaon=pd.read_csv('/home/ictd/nishant/research/Low_Price/Code/re/lasalgaon_arima1.csv',header=None)
arimabangalore=pd.read_csv('/home/ictd/nishant/research/Low_Price/Code/re/bangalore_arima1.csv',header=None)
arimaazadpur=pd.read_csv('/home/ictd/nishant/research/Low_Price/Code/re/azadpur_arima1.csv',header=None)

arimaxlasalgaon=pd.read_csv('/home/ictd/nishant/research/Low_Price/Code/re/lasalgaon_arimax.csv',header=None)
arimaxbangalore=pd.read_csv('/home/ictd/nishant/research/Low_Price/Code/re/bangalore_arimax.csv',header=None)
arimaxazadpur=pd.read_csv('/home/ictd/nishant/research/Low_Price/Code/re/lasalgaon_arimax.csv',header=None)

sarimalasalgaon=pd.read_csv('/home/ictd/nishant/research/Low_Price/Code/re/lasalgaon_sarima.csv',header=None)
sarimabangalore=pd.read_csv('/home/ictd/nishant/research/Low_Price/Code/re/bangalore_sarima.csv',header=None)
sarimaazadpur=pd.read_csv('/home/ictd/nishant/research/Low_Price/Code/re/azadpur_sarima.csv',header=None)

sarimaxlasalgaon=pd.read_csv('/home/ictd/nishant/research/Low_Price/Code/re/lasalgaon_sarimax.csv',header=None)
sarimaxbangalore=pd.read_csv('/home/ictd/nishant/research/Low_Price/Code/re/bangalore_sarimax.csv',header=None)
sarimaxazadpur=pd.read_csv('/home/ictd/nishant/research/Low_Price/Code/re/azadpur_sarimax.csv',header=None)

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

mandipriceseriesmumbai_derivative = get_derivative(mandipriceseriesmumbai)
mandipriceseriesdelhi_derivative = get_derivative(mandipriceseriesdelhi)
#mandipriceserieslucknow_derivative = get_derivative(mandipriceserieslucknow)
retailpriceseriesmumbai_derivative = get_derivative(retailpriceseriesmumbai)
retailpriceseriesdelhi_derivative = get_derivative(retailpriceseriesmumbai)
#retailpriceserieslucknow_derivative = get_derivative(retailpriceseriesmumbai)
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
			##print(p)
			# if(i==0):
			# 	print anomalies[0][i], anomalies[1][i]
		x.append(np.array(p))
		#print(p)
	#print(np.array(labels).shape)
	#print(np.array(x).shape)
	return np.array(x),np.array(labels)

def getKey(item):
	return item[0]

def partition(xseries,yseries,year,months):
	# min_month = datetime.strptime(min(year),'%Y-%m-%d')
	# max_month = datetime.strptime(max(year),'%Y-%m-%d')
	combined_series = zip(year,xseries,yseries)
	combined_series = sorted(combined_series,key=getKey)
	#print(combined_series)
	train = []
	train_labels = []
	fixed = datetime.strptime('2006-01-01','%Y-%m-%d')
	i=0
	while(fixed < datetime.strptime('2018-12-31','%Y-%m-%d')):
		currx=[]
		curry=[]
		for anomaly in combined_series:
			i += 1
			#print(anomaly[0])
			#print(datetime.strptime(anomaly[0],'%Y-%m-%d')- fixed)
			if(datetime.strptime(anomaly[0],'%Y-%m-%d') > fixed and datetime.strptime(anomaly[0],'%Y-%m-%d')- fixed <= timedelta(days=months*30)):
				#print(anomaly[1])
				currx.append(anomaly[1])
				curry.append(anomaly[2])
			#print(len(currx))
		train.append(currx)
		train_labels.append(curry)
		fixed = fixed +timedelta(days = months*30)
		#print(fixed)
	# print fixed
	# train.append(currx)
	# train_labels.append(curry)
	#print(train_labels)
	return np.array(train),np.array(train_labels)

def get_score(xtrain,xtest,ytrain,ytest):
	#print('xtest')
	#print(xtest)
	scaler = preprocessing.StandardScaler().fit(xtrain)
	xtrain = scaler.transform(xtrain)
	xtest = scaler.transform(xtest)
	model = RandomForestClassifier(max_depth=2, random_state=0,n_estimators=10)
	# model = SVC( kernel='rbf', C=0.8)
	model.fit(xtrain,ytrain)
	#print(xtest.shape)
	test_pred = np.array(model.predict(xtest))
	#print(model.get_params())
	#print(test_pred)
	# ytest = np.array(ytest)
	# if(test_pred[0] == ytest[0]):
	# 	return 1
	# else:
	# 	return 0
	return test_pred

def train_test_function(align_m,align_d,align_b,data_m,data_d,data_b,names):
	anomaliesmumbai = get_anomalies('/home/ictd/nishant/research/Low_Price/Data/Anomaly/lasalgaonAnomalyNonAnomaly.csv',align_m)
	#print('anomaliesmumbai')
	#print(anomaliesmumbai)
	anomaliesdelhi = get_anomalies('/home/ictd/nishant/research/Low_Price/Data/Anomaly/azadpurAnomalyNoAnomaly.csv',align_d)
	#anomalieslucknow = get_anomalies('/home/nishant/study/researchwork/Onion/information/normal_h_w_lucknow.csv',align_l)
	anomaliesbangalore = get_anomalies('/home/ictd/nishant/research/Low_Price/Data/Anomaly/bangaloreAnomalyNoAnomaly.csv',align_b)

	delhilabelsnew = newlabels(anomaliesdelhi)
	#lucknowlabelsnew = newlabels(anomalieslucknow)
	mumbailabelsnew = newlabels(anomaliesmumbai)
	bangalorelabelsnew = newlabels(anomaliesbangalore)
	#print(anomaliesdelhi)


	delhi_anomalies_year = get_anomalies_year(anomaliesdelhi)
	mumbai_anomalies_year = get_anomalies_year(anomaliesmumbai)
	#lucknow_anomalies_year = get_anomalies_year(anomalieslucknow)
	bangalore_anomalies_year = get_anomalies_year(anomaliesbangalore)
	#print(delhi_anomalies_year)

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
	#print(total_data)
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
	#print(predicted)
	#print(actual_labels)
	# print len(actual_labels)
	print('accuracy')
	#print(len(predicted))
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

print('mandi price')
#train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceseriesbangalore,
	#[mandipriceseriesmumbai],[mandipriceseriesdelhi],[mandipriceseriesbangalore],'mandi price')



#def split_data(xall,xall):



def earlywarning(align_m,align_d,align_b,data_m,data_d,data_b,names):
	anomaliesmumbai = get_anomalies('/home/ictd/nishant/research/Low_Price/Data/Anomaly/normal_h_w_mumbai1.csv',align_m)
	#print('anomaliesmumbai')
	#print(anomaliesmumbai)
	anomaliesdelhi = get_anomalies('/home/ictd/nishant/research/Low_Price/Data/Anomaly/normal_h_w_delhi.csv',align_d)
	#anomalieslucknow = get_anomalies('/home/nishant/study/researchwork/Onion/information/normal_h_w_lucknow.csv',align_l)
	anomaliesbangalore = get_anomalies('/home/ictd/nishant/research/Low_Price/Data/Anomaly/normal_h_w_bangalore.csv',align_b)

	delhilabelsnew = newlabels(anomaliesdelhi)
	#lucknowlabelsnew = newlabels(anomalieslucknow)
	mumbailabelsnew = newlabels(anomaliesmumbai)
	bangalorelabelsnew = newlabels(anomaliesbangalore)
	#print(anomaliesdelhi)


	delhi_anomalies_year = get_anomalies_year(anomaliesdelhi)
	mumbai_anomalies_year = get_anomalies_year(anomaliesmumbai)
	#lucknow_anomalies_year = get_anomalies_year(anomalieslucknow)
	bangalore_anomalies_year = get_anomalies_year(anomaliesbangalore)
	x1,y1 = prepare(anomaliesdelhi,delhilabelsnew,data_d)
	x2,y2 = prepare(anomaliesmumbai,mumbailabelsnew,data_m)
	#x3,y3 = prepare(anomalieslucknow,lucknowlabelsnew,data_l)
	x3,y3 = prepare(anomaliesbangalore,bangalorelabelsnew,data_b)
	#print(x1)
	xall = np.array(x1.tolist()+x2.tolist()+x3.tolist())
	yall = np.array(y1.tolist()+y2.tolist()+y3.tolist())
	print(xall.shape)
	print(yall.shape)
	rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
	#print(xall)
	# #x_test=np.array([[196.10874511, 223.82062745, 225.58968628, 237.20515686, 243.89742157,
 # 228.05128922, 220.34935539, 214.8253223,  255.08733885, 272.45633058,
 # 263.77183471, 243.11408264, 253.44295868,248.27852066, 244.37925819],[196.10874511, 223.82062745, 225.58968628, 237.20515686, 243.89742157,
 # 228.05128922, 220.34935539, 214.8253223,  255.08733885, 272.45633058,
 # 263.77183471, 243.11408264, 253.44295868,248.27852066, 244.37925819]])
	rf.fit(xall, yall)
	return rf

rf=earlywarning(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceseriesbangalore,
	[mandipriceseriesmumbai],[mandipriceseriesdelhi],[mandipriceseriesbangalore],'mandi price')


def detect_anomaly(df,date,rf):
	price_array=df[df['date']>date]['price'].tolist()
	final_array=[]
	for i in range(0,len(price_array)-43,7):
		print(i)
		l1=price_array[i:i+43]
		final_array.append(l1)
	final_array=np.array(final_array)
	print(final_array.shape)
	pred=rf.predict(final_array)
	print(pred)

df=pd.read_csv('/home/ictd/nishant/research/Low_Price/meeting/lasalgaon_price_arimax.csv',header=None,names=[
	'date','price'])

detect_anomaly(df,'2019-01-01',rf)