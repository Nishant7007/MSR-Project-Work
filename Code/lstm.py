import numpy as np
import pandas as pd
from keras.models import Sequential
import tensorflow
from keras.layers import LSTM,Dense
import matplotlib.pyplot as plt
from keras.models import load_model
import time
print('done')


# In[2]:

from common import get_mandi_price_series
#mandi_name='Lasalgaon'
price_series_main=get_mandi_price_series('Lasalgaon')
price_series_neighbour=get_mandi_price_series('Yeola')


# In[3]:


df_main=price_series_main
df_neighbour=price_series_neighbour
array_main = df_main.values.reshape(df_main.shape[0],1)
array_neighbour = df_neighbour.values.reshape(df_neighbour.shape[0],1)

def  minMaxScaler(array):
	mini=np.min(array)
	maxi=np.max(array)
	for i in range(array.shape[0]):
		array[i]=(array[i]-mini)/(maxi-mini)
	return array


#from sklearn.preprocessing import MinMaxScaler
#scl = MinMaxScaler()
array_main = minMaxScaler(array_main)
array_neighbour = minMaxScaler(array_neighbour)


# In[4]:


def process(train_main,train_neighbour):
    X,Y=[],[]
    for i in range(0,len(train_main)-390,30):
        x_main=train_main[i:i+365]
        x_neighbour=train_neighbour[i:i+365]
        x=np.hstack((x_main,x_neighbour))
        y=train_main[i+365:i+395]
        X.append(x)
        Y.append(y)
        #print('1')
        #print(len(X))
        #print(X[0].shape)
        #print(123)
    X,Y=np.array(X),np.array(Y)
    Y=np.reshape(Y,(Y.shape[0],Y.shape[1]))
    return X,Y


# In[ ]:


final=list(array_main[:365*3])
for date in range(365*3,4755,30):
    print(date)
    start=time.time()
    data_main=array_main[:date]
    data_neighbout=array_neighbour[:date]
    train_main=data_main[:-365]
    train_neighbour=data_neighbout[:-365]
    test_main=data_main[-365:]
    test_neighbour=data_neighbout[-365:]
    X_test=np.hstack((test_main,test_neighbour))
    #print(X_test.shape)
    X_train,Y_train=process(train_main,train_neighbour)
    NUM_NEURONS_FirstLayer = 1
    NUM_NEURONS_SecondLayer = 2
    EPOCHS = 1
    model = Sequential()
    model.add(LSTM(NUM_NEURONS_FirstLayer,input_shape=(365,2), return_sequences=False))
    #model.add(LSTM(NUM_NEURONS_SecondLayer,input_shape=(NUM_NEURONS_FirstLayer,1),return_sequences=False))
    model.add(Dense(30))
    #model.add(LSTM(NUM_NEURONS_FirstLayer,input_shape=(365,1)))
    #model.add(Dense(30))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(X_train,Y_train,epochs=EPOCHS,shuffle=True,batch_size=2, verbose=2)
    X_test=np.reshape(X_test,(1,X_test.shape[0],X_test.shape[1]))
    #print('shape is')
    #print(X_test.shape)
    Y_test = model.predict(X_test)
    #print(X_test)
    for p in range(Y_test.shape[1]):
        final.append(Y_test[0][p])
    #final.append(Y_train[0][p])
    #print(len(final))
    end=time.time()
    print(end-start)
dk=pd.DataFrame(final)
print(dk)
dk.to_csv('lstmforecasted.csv')




