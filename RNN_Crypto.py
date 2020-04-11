#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:05:47 2020

@author: clementgrattaroly
"""

# Importing the Keras libraries and packages


import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
import matplotlib .pyplot as plt
import math as m 

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
import tensorflow as tf

### Data importation ###

Data = pd.read_csv("BTC-USD_0.csv")

### Data Processing / Création du bon jeu de données indexé correctement ###

DataSet = Data [["Close"]]

DataSet = DataSet.dropna().reset_index(drop=True)

Base = DataSet[["Close"]]

Base_Bis = DataSet[["Close"]]

"""
BaseProcess = Base.iloc[1:len(Base)-1].values
DataSetProcess = DataSet.iloc[0:len(DataSet)-2,1:5].values
"""

### Création des MM ## 

Mm_10 = Base.rolling(window=10).mean().reset_index(drop=True)
Mm_50 = Base.rolling(window=50).mean().reset_index(drop=True)

Emw_10 = Base.ewm(span = 10, adjust=False , min_periods = 10).mean().reset_index(drop=True)
Emw_50 = Base.ewm(span = 50, adjust=False , min_periods = 50).mean().reset_index(drop=True)

Base_Numpy = Base_Bis.values
#Base_Numpy [1][0] = (Base_Numpy[0][0] + Base_Numpy[2][0])/2
Base_Numpy_Bis = Base.values

### Création de la volatilité

for i in range(1,len(Base_Numpy)):
    Base_Numpy[i][0] = m.log(Base_Numpy[i][0]/Base_Numpy_Bis[i-1][0])
    
Rendement = pd.DataFrame(data = Base_Numpy, columns =['Rendement daily'])

Vol_10 = Rendement.rolling(window=10).std().reset_index(drop=True)
Vol_50 = Rendement.rolling(window=50).std().reset_index(drop=True)

### Une valeur de close est manquante à la deuxieme ligne de Base ###


plt.plot(Emw_10 , color='blue')
#plt.plot(Emw_50, color ='red')
plt.plot(Base, color ='green')


NewDataSet = pd.concat((DataSet,Mm_10,Mm_50,Emw_50,Base),axis=1).dropna().reset_index(drop=True)


### Training & Validation Sets Creation ###

N_Train = int(0.3 * len(NewDataSet))

X = NewDataSet.iloc[0:len(NewDataSet)-1,0:4].dropna().reset_index(drop=True)
Y = NewDataSet.iloc[1:len(NewDataSet),4:5].dropna().reset_index(drop=True)


FinalDataSet = pd.concat((X,Y),axis=1)

Data_Train = FinalDataSet.iloc[:-N_Train].values

Data_Valid = FinalDataSet.iloc[-N_Train:].values

### Sequences creation ###

Length_Seq = 20

Data_Train_Seq = []
        
for col in range(Data_Train.shape[1]) : 
    Data_Seq = []
    for i in range(Length_Seq,len(Data_Train)):
        Data_Seq.append(Data_Train[i-Length_Seq:i,col])
    Data_Seq = np.array(Data_Seq)
    Data_Train_Seq.append(Data_Seq)
    
Data_Train_Seq = np.array(Data_Train_Seq)

Data_Train_Seq = np.swapaxes(np.swapaxes(Data_Train_Seq, 0, 1), 1, 2)

Data_Valid_Seq = []

for col in range(Data_Valid.shape[1]):
    Data_Seq=[]
    for i in range(Length_Seq,len(Data_Valid)):
        Data_Seq.append(Data_Valid[i-Length_Seq:i,col])
    Data_Seq = np.array(Data_Seq)
    Data_Valid_Seq.append(Data_Seq)

Data_Valid_Seq = np.array(Data_Valid_Seq)

Data_Valid_Seq = np.swapaxes(np.swapaxes(Data_Valid_Seq, 0, 1), 1, 2)

### Normalisation des données : 
## Ni = (Pi/P0) - 1 

for i in range(int(Data_Train_Seq.shape[0])):
    for col_i in range(Data_Train_Seq.shape[2]):
        Data_Train_Seq[i,:,col_i] = [ ((float(p)/float(Data_Train_Seq[i,0,col_i]))-1) for p in Data_Train_Seq[i,:,col_i]]

for i in range(int(Data_Valid_Seq.shape[0])):
    for col_i in range(Data_Valid_Seq.shape[2]):
        Data_Valid_Seq[i,:,col_i] = [ ((float(p)/float(Data_Valid_Seq[i,0,col_i]))-1) for p in Data_Valid_Seq[i,:,col_i]]

# Création de X_Train/ Y_Train / X_Valid Y_Valid

X_Train_Seq = Data_Train_Seq[:,:,0:4]
Y_Train = Data_Train_Seq[:,19:20,4:5].reshape(len(Data_Train_Seq),1)


X_Valid_Seq = Data_Valid_Seq[:,:,0:4]
Y_Valid = Data_Valid_Seq[:,19:20,4:5].reshape(len(Data_Valid_Seq),1)



### Création du modèle ###

NbFeatures = X_Train_Seq.shape[2]

Nb_Neurone = 50
DropOut = 0.2

Model = Sequential()

Model.add(LSTM(units=Nb_Neurone,return_sequences=True , input_shape=(Length_Seq,NbFeatures)))
Model.add(Dropout(DropOut))

Model.add(LSTM(units=Nb_Neurone, return_sequences=True ))
Model.add(Dropout(DropOut))

Model.add(LSTM(units=Nb_Neurone, return_sequences=True ))
Model.add(Dropout(DropOut))

Model.add(LSTM(units=Nb_Neurone))
Model.add(Dropout(DropOut))

Model.add(Dense(units=1))


Model.compile(optimizer='adam',loss='mse')


Model.fit(X_Train_Seq,Y_Train, batch_size = 25 , epochs = 30)



Y_Pred_Norm = Model.predict(X_Valid_Seq)

y=plt.plot(Y_Valid,color='green')+plt.plot(Y_Pred_Norm,color='red')


Model.reset_states()



        
    
    








































