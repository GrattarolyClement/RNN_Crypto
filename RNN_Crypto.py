#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:05:47 2020

@author: clementgrattaroly
"""

# Importing the Keras libraries and packages


import pandas as pd
import numpy as np
import math as m 
import os as os

import matplotlib .pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from DataLoader import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from Tech_analysis_Trading import *

### Création des datasets de training et d'évaluation 

def Get_Train_Val_Sets(NewDataSet,SplitCoef):
    
    N_Train = int(0.3 * len(NewDataSet))
    Nb_Features = NewDataSet.shape[1]-1
    
    ### On aligne les datas avec les targets souhaités (Train).
    
    X = NewDataSet.iloc[0:len(NewDataSet)-1,0:Nb_Features].dropna().reset_index(drop=True)
    Y = NewDataSet.iloc[1:len(NewDataSet),Nb_Features:5].dropna().reset_index(drop=True)
    
    FinalDataSet = pd.concat((X,Y),axis=1) # Axis = 1 : On cocnatene les colonnes
    
    Data_Train = FinalDataSet.iloc[:-N_Train].values

    Data_Valid = FinalDataSet.iloc[-N_Train:].values
    
    return Data_Train, Data_Valid
    
### Fonction permettant de créer des séquences de datas (ex : 20 lignes de data représente une séquence)

def Create_Seq(Length_Seq , Data): 
    
    Data_Seq = []
            
    for col in range(Data.shape[1]) : 
        
        Data_Row = []
        
        for i in range(Length_Seq,len(Data)):
            
            Data_Row.append(Data[i-Length_Seq:i,col])
            
        Data_Row = np.array(Data_Row)
        Data_Seq.append(Data_Row)
        
    Data_Seq = np.array(Data_Seq)
    Data_Seq = np.swapaxes(np.swapaxes(Data_Seq, 0, 1), 1, 2)
     
    return Data_Seq

### Fonction de normalisation. On normalise par séquence de la manière suivante : ## Ni = (Pi/P0) - 1 
### Le P0 (i.e. Base_Price_Norm) est choisi comme le premier prix Close de la séquence (Le plus vieux en date)

def Seq_Normalization(Data_Seq) : 
    
    for i in range(int(Data_Seq.shape[0])):
    
        Base_Price_Norm = Data_Seq[i,0,0] # On set notre prix de normalisation, 1er prix close 
        
        for col_i in range(Data_Seq.shape[2]):
            
            Data_Seq[i,:,col_i] = [ ((float(p)/float(Base_Price_Norm))-1) for p in Data_Seq[i,:,col_i]]
    
    return Data_Seq


### Fonction permettant de créer un modèle de réseau de neuronnes récurrents 
    
def Create_Model(Length_Seg, Nb_Features, Nb_couches_LSTM, Loss, Optimizer, Nb_Neurone ,DropOut ):
    
    # Initialisation du modèle
    
    Model = Sequential()
    
    # Création du premier layer hors boucle pour renseigner les dimensions des datas entrantes
    
    Model.add(LSTM(units=Nb_Neurone,return_sequences=True , input_shape=(Length_Seq,Nb_Features)))
    Model.add(Dropout(DropOut))
    
    
    for i in range(1, Nb_couches_LSTM):
        Model.add(LSTM(units=Nb_Neurone,return_sequences=True if i < Nb_couches_LSTM-1 else False ))
        Model.add(Dropout(DropOut))

    Model.add(Dense(units=1))
    
    Model.compile( optimizer = Optimizer, loss = Loss)
    
    return Model


### Préddictions et dénormalisation des données 
    
    
def Denormalized_Predictions(Model, X_Valid_Seq, Data_Denorm, Y_Valid_Denorm):
    
    Y_Pred = Model.predict(X_Valid_Seq)
    
    for i in range(Data_Valid_Seq.shape[0]):
        
        Base_Price_Norm = Data_Denorm[i,0,0] # On dénormalise la prédiction avec le prix : Base_Price_Norm (celui choisit au moment de la normalisation des dat)
        
        Y_Pred [i,:] = [( var  + 1 ) * Base_Price_Norm for var in Y_Pred[i,:]]


    #t = [ t for t in range(len(Y_Pred))]
    
    ### Rehaspe to plot predictions and real prices
    
    Y_Pred = Y_Pred.reshape(len(Y_Pred))
    
    Y_Valid_Denorm = Y_Valid_Denorm.reshape(len(Y_Valid_Denorm))
    
    
    plt.subplot(2, 1, 1)
    plt.plot(Y_Pred,color='green',label="Pred")
    plt.legend(loc='upper left')    
    
    
    plt.subplot(2, 1, 2)
    plt.plot(Y_Valid_Denorm,color='red',label="True")
    plt.legend(loc='upper left')  
    
    plt.show()
    
    return Y_Pred


### Fonction permettant de visualier l'erreur du training set et validation set 

def Eval_Training(Trained_Model):
    
    Train_Loss, Valid_Loss = Trained_Model.history['loss'], Trained_Model.history['val_loss']
   
    Index_Best_Valid = np.argmin(Valid_Loss)
    Min_Val_Error = np.min(Valid_Loss)
    
    
    plt.plot(Train_Loss,color = 'blue',label='Train_Loss')
    plt.plot(Valid_Loss,color='orange',label='Valid_Loss')
    plt.plot(Index_Best_Valid,Min_Val_Error,color='red',label='Best config',marker='+')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Error : Train VS Validation')
    plt.show()
    
    Train_Loss = np.array(Train_Loss)
    Valid_Loss = np.array(Valid_Loss)
    
        
    print("Best Valid Error : " + str(Valid_Loss.min()) + "     Nb Epochs : " + str(Index_Best_Valid + 1 ))
    print("Train error : " + str(Valid_Loss[Index_Best_Valid]))
    
### Function permettant de sauvergarder le modèle

def Save_Model(Model, Model_Name) : 
    
    print("Sauvergarde du fichier ? ")
    
    BoolSave = input()
    
    if BoolSave == "yes" or BoolSave == 'oui' or BoolSave == "Yes" or BoolSave == 'Oui' : 
    
    #os.makedirs('/Users/clementgrattaroly/Python/RNN_Crypto/Saved_Model')
        Model.save('/Users/clementgrattaroly/Python/RNN_Crypto/Saved_Model/'+Model_Name)
    

### Fonction permettant de charger les poids d'un ancien modèle

def Load_Weights(Model, FileName):
    
    

    Model.load_weights('/Users/clementgrattaroly/Python/RNN_Crypto/Saved_Model/'+FileName)
    
    return Model
  

### Data importation & DataSet Creation ###

Data = DataLoader()

Data.InitData('BTC-USD_0.csv')
Data.DataSetConstruction( Window1 = 10, Window2 = 50)

NewDataSet = Data.GetDataSet()


### Training & Validation Sets Creation ###


Data_Train, Data_Valid = Get_Train_Val_Sets(NewDataSet,SplitCoef=0.3)

### Sequences creation (Training/Validation Set) ###

# On set la longueur des séquences en définissant Length_Seq

Length_Seq = 50


Data_Train_Seq = Create_Seq( Length_Seq , Data_Train )

Data_Valid_Seq = Create_Seq( Length_Seq , Data_Valid )

# Création d'une copie du numpy array validation Set servant à la dénormalisation des prédictions

Data_Denorm = np.copy(Data_Valid_Seq)


### Normalisation des données 

Data_Train_Seq = Seq_Normalization(Data_Train_Seq)

Data_Valid_Seq = Seq_Normalization(Data_Valid_Seq)


# Création de X_Train/ Y_Train / X_Valid / Y_Valid (Norm/Denorm)

X_Train_Seq = Data_Train_Seq[:,:,0:4]
X_Valid_Seq = Data_Valid_Seq[:,:,0:4]


Y_Train = Data_Train_Seq[:,Length_Seq-1:Length_Seq,4:5].reshape(len(Data_Train_Seq),1)
Y_Valid = Data_Valid_Seq[:,Length_Seq-1:Length_Seq,4:5].reshape(len(Data_Valid_Seq),1)
Y_Valid_Denorm = Data_Denorm[:,Length_Seq-1:Length_Seq,4:5].reshape(len(Data_Denorm),1)

### Création du modèle ###

Nb_Features = X_Train_Seq.shape[2]

Nb_Neurone = 50
DropOut = 0.2
Nb_couches_LSTM = 4
Loss = 'mean_absolute_percentage_error'
Optimizer = 'adam'

Model = Create_Model(Length_Seq, Nb_Features, Nb_couches_LSTM, Loss, Optimizer , Nb_Neurone,DropOut)


### Entraînement du modèle 
Nb_Epochs = 200
Batch_Size = 25

Trained_Model = Model.fit(X_Train_Seq,Y_Train, batch_size = Batch_Size , epochs = Nb_Epochs, validation_data=(X_Valid_Seq,Y_Valid))


### Here we can load a former trained model from Save_Model directory



#Model = Load_Weights(Model,"Model_0")


### Prédictions et Dénormalisation des données


Y_Pred = Denormalized_Predictions(Model, X_Valid_Seq, Data_Denorm, Y_Valid_Denorm)


### A checker

PredVsPrice = Y_Pred[:] - Y_Valid_Denorm[:]
print("Écart moyen entre la prédiction et le marché " + str(PredVsPrice.mean()))


Eval_Training(Trained_Model)

Trend_Predict_Eval(Length_Seq,Data_Denorm,Y_Pred,0.01)

# Fonction de trade sur les futures 

Top_Var = 0.01
Bottom_Var = -0.01

TradingFutures(1000,Data_Denorm,Y_Pred,10,Top_Var,Bottom_Var,Length_Seq)

### Function to save the trained model in RNN_Crypto directory

Model_Name = 'Model ' + ' loss = ' + Loss + ' E = ' + str(Nb_Epochs) + ' B = ' + str(Batch_Size) + ' N = ' + str(Nb_Neurone) + ' D = ' + str(DropOut) + ' LSTM = ' + str(Nb_couches_LSTM)
Save_Model(Model, Model_Name)








        
    
    








































