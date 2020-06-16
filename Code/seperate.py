import pandas as pd
import numpy as np

df=pd.read_csv('/home/ictd/nishant/research/low_price_Anomaly/Data/processed/karnataka.csv',header=None)
mandi_info=pd.read_csv('/home/ictd/nishant/research/low_price_Anomaly/Data/information/mandis.csv')

start='2006-01-01'
end='2019-06-30'

def get_mandi_code(df,name):
    #print(df.columns)
    mandicode=df[df['mandiname']==name]['mandicode']
    return mandicode
    
def get_arrival_series(df,mandi_info,name):
    #print(get_mandi_code(mandi_info,name))
    mandicode=int(get_mandi_code(mandi_info,name))
    print(mandicode)
    df[1]=df[1].interpolate(method='pchip')
    df=df[df[1]==mandicode]
    print(len(df))
    df=df[[0,2]]
    df.reset_index(inplace=True)
    df=df[[0,2]]
    df.index=df[0]
    df=df[[2]]
    idx = pd.date_range(start,end)
    df.index = pd.DatetimeIndex(df.index)
    df = df.reindex(idx, fill_value=0)
    df.columns=['arrival']
    df['arrival']=df['arrival'].replace(0,np.nan)
    df['arrival']=df['arrival'].interpolate(method='pchip')
    df[df<0]=0
    df.fillna(method='bfill',inplace=True)
	#print(df)
    return df
	
def get_price_series(df,mandi_info,name):
    mandicode=int(get_mandi_code(mandi_info,name))
    df=df[df[1]==mandicode]
    df=df[[0,7]]
    df.reset_index(inplace=True)
    df=df[[0,7]]
    df.index=df[0]
    df=df[[7]]
    idx = pd.date_range(start,end)
    df.index = pd.DatetimeIndex(df.index)
    df = df.reindex(idx, fill_value=0)
    df.columns=['price']
    df['price']=df['price'].replace(0,np.nan)
    df['price']=df['price'].interpolate(method='pchip')
    df[df<0]=0
    df.fillna(method='bfill',inplace=True)
	#print(df)
    return df
	
dfa=get_arrival_series(df,mandi_info,'Bangalore')
#print(dfa)
#dfa.to_csv('/home/ictd/nishant/research/low_price_Anomaly/Data/processed/keshopur_arrival.csv')
#
#dfb=get_price_series(df,mandi_info,'Keshopur')
#print(dfb)
#dfb.to_csv('/home/ictd/nishant/research/low_price_Anomaly/Data/processed/keshopur_price.csv')
    
#print(get_mandi_code(mandi_info,'Lasalgaon'))
