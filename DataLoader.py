#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 07:31:57 2020

@author: clementgrattaroly
"""

import pandas as pd
import numpy as np 
import tensorflow as tf
import math as m
from alpha_vantage.cryptocurrencies import CryptoCurrencies


class DataLoader(object):
    
    def InitData(self,FileName):
        
        DataFrame = pd.read_csv("/Users/clementgrattaroly/Python/RNN_Crypto/DATA/"+FileName)
        self.Data = DataFrame
        self.Base = DataFrame[["Close"]]
        self.DataSet = None
    
    def Init_Crypto_Currency_API_Alpha_Vantage(self , API_Key, Crypto_Currency , Market):
    
        cc = CryptoCurrencies(key=API_Key,output_format='pandas')
        
        self.Data, self.MetaData = cc.get_digital_currency_daily(Crypto_Currency,Market)
        self.Base = self.data['4a. close (USD)']
        self.DataSet = None
        
        
        
        
    def GetBase(self):
        return self.Base
    
    def GetDataSet(self):
        return self.DataSet
        
    def DataSetConstruction(self,Window1, Window2):
        
        Mm_10 = self.Base.rolling(window=Window1).mean().reset_index(drop=True)
        Mm_50 = self.Base.rolling(window=Window2).mean().reset_index(drop=True)

        Emw_10 = self.Base.ewm(span = Window1, adjust=False , min_periods = Window1).mean().reset_index(drop=True)
        Emw_50 = self.Base.ewm(span = Window2, adjust=False , min_periods = Window2).mean().reset_index(drop=True)
        
        self.DataSet = pd.concat((self.Base,Mm_10,Mm_50,Emw_50,self.Base),axis=1).dropna().reset_index(drop=True)
        
        ### Création de la volatilité : On laisse de côté pour le moment
        
        """
        for i in range(1,len(Base_Numpy)):
            Base_Numpy[i][0] = m.log(Base_Numpy[i][0]/Base_Numpy_Bis[i-1][0])
            
        Rendement = pd.DataFrame(data = Base_Numpy, columns =['Rendement daily'])
        
        Vol_10 = Rendement.rolling(window=10).std().reset_index(drop=True)
        Vol_50 = Rendement.rolling(window=50).std().reset_index(drop=True)
        """     
    
data = DataLoader()  
