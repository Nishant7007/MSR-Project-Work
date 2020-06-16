import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from datetime import timedelta


from loadSeries import *

#CALCULATE RMSE ERROR
def rmse_error(actual,predicted,mean_divide=1):
    error = (actual - predicted)**2
    error = sum(error)/len(actual)
    error = math.sqrt(error)
    if  (mean_divide==1):
        error=error/actual.mean()
    return error

def avg_rmse(actual,predicted):
	rmse=[]
	i=0
	while(i+30<len(actual)):
		x = actual[i:i+30]
		y = predicted[i:i+30]
		rmse.append(rmse_error(x,y))
		#print(rmse_error(x,y))
		i = i+30
	rmse = np.array(rmse)
	#print(rmse)
	#rmse=np.mean(rmse)
	#print(".........avg rmse over 30 day period.......")
	# print(rmse)
	return rmse 

def rmse_error_calculator(actual,predicted):
    rmse=[]
    i=0
    while(i+30<len(actual)):
        x = actual[i:i+30]
        y = predicted[i:i+30]
        rmse.append(rmse_error(x,y))
        i = i+30
    rmse = np.array(rmse)
    #meanrmse=np.mean(rmse)
    return rmse

def mape_error(actual,predicted):
    actual.columns=['date','price']
    predicted.columns=['date','price']
    error=abs(actual['price']-predicted['price'])
    error/=actual['price']
    error=(sum(error)/len(actual))*100
    print(error)
    
#print(rmse_error_calculator(lasalgaon_original['price'],lasalgaon_mandi_arima['price']))
#print(rmse_error_calculator(lasalgaon_original['price'],lasalgaon_mandi_arimax1['price']))
#print(rmse_error_calculator(lasalgaon_original['price'],lasalgaon_mandi_arimax2['price']))
#print(rmse_error_calculator(lasalgaon_original['price'],lasalgaon_mandi_sarima['price']))
#print(rmse_error_calculator(lasalgaon_original['price'],lasalgaon_mandi_sarimax['price']))
#print(rmse_error_calculator(lasalgaon_original['price'],lasalgaon_mandi_arimax1['price']))

#print(rmse_error(bangalore_price_arima,bangalore_price_arimax))
#print(rmse_error(lasalgaon_original,lasalgaon_price_arima))
#print(rmse_error(azadpur_original,azadpur_price_arima))
#print(rmse_error(bangalore_original,bangalore_price_arimax))
#print(rmse_error(lasalgaon_original,lasalgaon_price_arimax))
#print(rmse_error(azadpur_original,azadpur_price_arimax))

def preprocess(df,start,end):
    #df.columns=['date','price']
    print(type(df))
    if df is None:
        return df
    df=df[df['date']>=start]
    df=df[df['date']<=end]
    df.index=pd.DatetimeIndex(df['date'])
    df=df[['price']]
    #print(df)
    return df    

def plot_series(df):
    if df is not None:
        plt.plot(df.rolling(7).mean())
    
def plot_all_series(start,end,titleName,original=None,arima=None,arimax1=None,arimax2=None,sarima=None,sarimax=None,lstm=None):
     #print(original) 
     original=preprocess(original,start,end)
     arima=preprocess(arima,start,end)
     arimax1=preprocess(arimax1,start,end)
     arimax2=preprocess(arimax2,start,end)
     sarima=preprocess(sarima,start,end)
     sarimax=preprocess(sarimax,start,end)
     lstm=preprocess(lstm,start,end)
     #print(original)
     plot_series(original['price'])
     plot_series(arima['price'])
     plot_series(arimax1['price'])
     plot_series(arimax2['price'])
     plot_series(sarima['price'])
     plot_series(sarimax['price'])
     plot_series(lstm['price'])
     plt.title(titleName+'('+str( start)+' to '+str( end)+')')
     plt.xlabel('Date')
     plt.ylabel('Price')
     plt.legend(['Original','Arima','Arimax','Arimax with sarimax','Sarima','Sarimax','LSTM'],fontsize='small')
     plt.show()   

#plot_all_series('2018-06-01','2018-12-31',titleName='Price Forecasting for Lasalgoan Mandi ',original=lasalgaon_mandi,arima=lasalgaon_mandi_arima,arimax1=lasalgaon_mandi_arimax1,arimax2=lasalgaon_mandi_arimax2,sarima=lasalgaon_mandi_sarima,sarimax=lasalgaon_mandi_sarimax,lstm=lasalgaon_mandi_lstm)
#plot_all_series('2018-06-01','2018-12-31',titleName='Price Forecasting for Azadpur Mandi ',original=azadpur_mandi,arima=azadpur_mandi_arima,arimax1=azadpur_mandi_arimax1,arimax2=azadpur_mandi_arimax2,sarima=azadpur_mandi_sarima,sarimax=azadpur_mandi_sarimax,lstm=azadpur_mandi_lstm)
#plot_all_series('2018-06-01','2018-12-31',titleName='Price Forecasting for Azadpur Mandi ',original=bangalore_mandi,arima=bangalore_mandi_arima,arimax1=bangalore_mandi_arimax1,arimax2=bangalore_mandi_arimax2,sarima=bangalore_mandi_sarima,sarimax=bangalore_mandi_sarimax,lstm=bangalore_mandi_lstm)

def rmse_box_plot(lucknow,arima,arimax,arimax_sarimax,sarima,sarimax,lstm,titleName):
    rmse_arima=avg_rmse(np.array(lucknow),np.array(arima))
    rmse_arimax1=avg_rmse(lucknow,arimax)
    rmse_arimax2=avg_rmse(lucknow,arimax_sarimax)
    rmse_sarima=avg_rmse(lucknow,sarima)
    rmse_sarimax=avg_rmse(lucknow,sarimax)
    rmse_lstm=avg_rmse(lucknow,lstm)
    labels=['ARIMA','ARIMAX1','ARIMAX2','SARIMA','SARIMAX','LSTM']
    box=plt.boxplot([rmse_arima,rmse_arimax1,rmse_arimax2,rmse_sarima,rmse_sarimax,rmse_lstm],patch_artist=True,labels=labels)
    colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'purple', 'red']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title(titleName)
    plt.ylabel('Normalized RMSE calculated monthly')
    plt.xlabel('MODEL')
    plt.show()
    
#rmse_box_plot(lasalgaon_mandi,lasalgaon_mandi_arima,lasalgaon_mandi_arimax1,lasalgaon_mandi_arimax2,lasalgaon_mandi_sarima,lasalgaon_mandi_sarimax,lasalgaon_mandi_arima,'Lasalgaon RMSE Box plots')

def rmse_error_monthly(actual,predicted,days):
	rmse=[]
	i=0
	while(i+days<len(actual)):
		x = actual[i:i+days]
		y = predicted[i:i+days]
		rmse.append(rmse_error(x,y))
		i = i+days
	rmse = np.array(rmse)
	return rmse 
def month_series(df,month):
	df['date'] = pd.to_datetime(df['date'])
	df['month'] = df['date'].dt.month
	mask = (df['month']==month)
	df = df.loc[mask]
	df = df['price']
	return np.array(df)


#def rmse_box_plot_all_month (df_lucknow,df_arima,df_arimax,df_sarima,df_sarimax,df_lstm,df_arimax_sarimax):
#	month=[1,2,3,4,5,6,7,8,9,10,11,12]
#	days =[31,28,31,30,31,30,31,31,30,31,30,31]
#	model =[df_arima,df_arimax,df_sarima,df_sarimax,df_lstm,df_arimax_sarimax]
#	model_name=["ARIMA","ARIMAX","Arimax using Sarimax","SARIMA","SARIMAX","LSTM"]
#    for i in range(6):
#        monthly_rmse =[]
#        for j in range(12):
##            lucknow = month_series(df_lucknow,month[j])
##            arima = month_series(model[i],month[j])
##            monthly_rmse.append(rmse_error_monthly(lucknow,arima,days[j]))
##        graph=["Jan","Feb","Mar","Apr","May","Jun","July","Aug","Sep","Oct","Nov","Dec"]
##        plt.boxplot(monthly_rmse,labels=graph,patch_artist=True)
##        plt.title(model_name[i]+' Lasalgaon Mandi monthly Box plot')
##        plt.xlabel('Month')
##        plt.ylabel('Normalized RMSE for month')
##        plt.show()


def rmse_box_plot_all_month1(df_lucknow,df_arima,df_arimax,df_sarima,df_sarimax,df_lstm,df_arimax_sarimax):
    month=[1,2,3,4,5,6,7,8,9,10,11,12]
    days =[31,28,31,30,31,30,31,31,30,31,30,31]
    model =[df_arima,df_arimax,df_sarima,df_sarimax,df_lstm,df_arimax_sarimax]
    #i=0
    model_name=["ARIMA","ARIMAX","Arimax using Sarimax","SARIMA","SARIMAX","LSTM"]
    for i in range(6):
        #i+=1
        monthly_rmse =[]
        for j in range(12):
            lucknow = month_series(df_lucknow,month[j])
            arima = month_series(model[i],month[j])
            monthly_rmse.append(rmse_error_monthly(lucknow,arima,days[j]))
        graph=["Jan","Feb","Mar","Apr","May","Jun","July","Aug","Sep","Oct","Nov","Dec"]
        plt.boxplot(monthly_rmse,labels=graph,patch_artist=True)
        plt.title(model_name[i]+'- Bangalore Mandi monthly Box plot')
        plt.xlabel('Month')
        plt.ylabel('NOrmalized RMSE')
        plt.show()
            

#rmse_box_plot_all_month1(lasalgaon_mandi,lasalgaon_mandi_arima,lasalgaon_mandi_arimax1,lasalgaon_mandi_arimax2,lasalgaon_mandi_sarima,lasalgaon_mandi_sarimax,lasalgaon_mandi_lstm)
#rmse_box_plot_all_month1(bangalore_mandi,bangalore_mandi_arima,bangalore_mandi_arimax1,bangalore_mandi_arimax2,bangalore_mandi_sarima,bangalore_mandi_sarimax,bangalore_mandi_lstm)

#import numpy as np
#import matplotlib.pyplot as plt
#
#x1 = 10*np.random.random(100)
#x2 = 10*np.random.exponential(0.5, 100)
#x3 = 10*np.random.normal(0, 0.4, 100)
#plt.boxplot ([x1, x2, x3])

#RETURN THE SCALED RESIDUAL SERIES
def residual_series(df1,df2,start,end):
    df1.columns=['date','price']
    df2.columns=['date','price']
    df1=df1[df1['date']>=start]
    df1=df1[df1['date']<=end]
    df2=df2[df2['date']>=start]
    df2=df2[df2['date']<=end]
    df3=pd.DataFrame(abs(df1.price-df2.price))
    df3.index=pd.DatetimeIndex(df1.date)
    df1.index=df3.index
    df3['norm']=df3['price']/df1['price']
    return df3


#PLOT THE SCALED RESIDUAL SERIES
def plot_residual_series(df1,df2,start,end,rolling_window,titleName,yName):
    df3=residual_series(df1,df2,start,end)
    #print(df3)
    if(rolling_window>1):
        plt.plot(df3['norm'].rolling(window=rolling_window).mean())
    else:
        plt.plot(df3['norm'])
    
    plt.title(titleName)
    plt.xlabel('Date')
    plt.ylabel(yName)
    plt.show()
 
#plot_residual_series(lasalgaon_original,lasalgaon_price_arima,'2009-01-01','2018-12-31',14,'Lasalgaon Mandi Price Arima Residual Series','Residual')
    
    
def plot_residual_anomaly_series(df1,df2,df3,start,end,title,yName):
    df4=residual_series(df1,df2,start,end)
    df3=df3[df3['reason']=='supply']
    df3['date']=pd.to_datetime(df3['start'])+timedelta(21)
    anomaly_list=df3['start'].tolist()
    df4['reason']='No'
    df4=df4.reset_index()
    for i in range(0,len(df4)):
        #print(str(df4.at[i,'date'])[:10])
        for j in anomaly_list:
            #print(str(df4.at[i,'date'])[:10],j)
            if(str(df4.at[i,'date'])[:10]==j):
                print(i)
                df4.at[i,'reason']='yes'
    #print(df4[df4.reason=='yes'])
    df4.index=pd.DatetimeIndex(df4.date)
    plt.plot(df4['norm'])
    #df4.to_csv('azadpur_anomaly_news.csv')
    plt.scatter(df4[df4.reason=='yes'].index,df4[df4.reason=='yes']['norm'],c='r')
    #plt.
    plt.show()
    
#plot_residual_series(azadpur_original,azadpur_arima,'2011-01-01','2012-12-31',14)

#plot_residual_series(azadpur_price,azadpur_arima,'2011-01-01','2012-12-31',14)
#plot_residual_anomaly_series(azadpur_price,azadpur_arima,azadpur_anomaly,'2015-01-01','2018-12-31')
#plot_residual_anomaly_series(bangalore_price,bangalore_arima,bangalore_anomaly,'2006-01-01','2018-12-31')    
#plot_residual_anomaly_series(bangalore_price,bangalore_arimax_sarimax,bangalore_anomaly,'2006-01-01','2018-12-31')
#print(bangalore_arima)
    
    
#plot_residual_anomaly_series(bangalore_price,bangalore_arimax_sarimax,bangalore_anomaly,'2011-01-01','2018-12-31')
    
    
#df=pd.read_csv('/home/ictd/nishant/research/low_price_Anomaly/Data/Anomaly/lasalgaon_anomaly_news.csv')
#df=df[df['date']>='2015-01-01']
#df=df[df['date']<='2018-12-31']
#plt.plot(df['norm'])
#plt.scatter(df[df.news=='yes'].index,df[df.news=='yes']['norm'],c='r')
#plt.title('lasalgaon 2015-2018')
#plt.show()
lasalgaon_mandi_arimax2.index=pd.to_datetime(lasalgaon_mandi_arimax2[0])
lasalgaon_mandi_arimax2=lasalgaon_mandi_arimax2[1] 

azadpur_mandi_arimax2.index=pd.to_datetime(azadpur_mandi_arimax2[0])
azadpur_mandi_arimax2=azadpur_mandi_arimax2[1]

bangalore_mandi_arimax2.index=pd.to_datetime(bangalore_mandi_arimax2[0])
bangalore_mandi_arimax2=bangalore_mandi_arimax2[1]

lasalgaon_arrival_arimax2.index=pd.to_datetime(lasalgaon_arrival_arimax2[0])
lasalgaon_arrival_arimax2=lasalgaon_arrival_arimax2[1] 

azadpur_arrival_arimax2.index=pd.to_datetime(azadpur_arrival_arimax2[0])
azadpur_arrival_arimax2=azadpur_arrival_arimax2[1]

bangalore_arrival_arimax2.index=pd.to_datetime(bangalore_arrival_arimax2[0])
bangalore_arrival_arimax2=bangalore_arrival_arimax2[1]

bangalore_retail_arimax2.index=pd.to_datetime(bangalore_retail_arimax2[0])
bangalore_retail_arimax2=bangalore_retail_arimax2[1] 

delhi_retail_arimax2.index=pd.to_datetime(delhi_retail_arimax2[0])
delhi_retail_arimax2=delhi_retail_arimax2[1]

mumbai_retail_arimax2.index=pd.to_datetime(mumbai_retail_arimax2[0])
mumbai_retail_arimax2=mumbai_retail_arimax2[1]
  
def monthly_rmse_anomaly_plot(actual,predicted,anomalies):
    rmse_list=rmse_error_calculator(actual,predicted)
    rmse_plot_list=[np.nan]*len(rmse_list)
    #anomaly[0]=pd.datetime(anomaly[0])
    anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[0]]
    anomalies[1] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[1]]
    anomalies[0]=pd.to_datetime(anomalies[0])
    rng = pd.date_range('2006-01-01','2018-12-31')
    df = pd.DataFrame({ 'Date': rng})
    for i in range(len(anomalies)):
        if (anomalies[2][i]!=' Normal_train'):
            mid_date=pd.to_datetime(anomalies[0][i]+timedelta(days=21))
            #print(mid_date==pd.to_datetime('2018-10-10'))
            index=((pd.to_datetime(mid_date)-pd.to_datetime('2006-01-01')).days ) // 30
            rmse_plot_list[index]=rmse_list[index]
    y_pos = np.arange(len(rmse_list[36:]))
    plt.scatter(y_pos,rmse_plot_list[36:],c='r')
    plt.plot(rmse_list[36:])
    plt.title('Monthly RMSE(Retail Price) and High Price anomaly - Bangalore')
    plt.ylabel('Normalized RMSE')
    plt.xlabel('30 Day Window starting from 2009')
    plt.legend(['RMSE','Anomaly'])
    plt.show()
monthly_rmse_anomaly_plot(bangalore_retail,bangalore_retail_arimax2,bangalore_high_anomaly)
    
    
