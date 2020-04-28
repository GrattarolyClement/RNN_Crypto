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
from alpha_vantage.techindicators import TechIndicators



class DataLoader(object):
    
    def InitData(self,FileName):
        
        DataFrame = pd.read_csv("/Users/clementgrattaroly/Python/RNN_Crypto/DATA/"+FileName)
        self.Data = DataFrame
        self.Base = DataFrame[["Close"]]
        self.DataSet = None
    
    def Init_Crypto_Currency_API_Alpha_Vantage(self , API_Key, Crypto_Currency , Market):
    
        cc = CryptoCurrencies(key=API_Key,output_format='pandas')
        
        self.Data, self.MetaData = cc.get_digital_currency_daily(Crypto_Currency,Market)
        self.Base = self.Data['4a. close (USD)'].to_frame().sort_index(axis = 0, ascending = True).reset_index().drop(columns='date')
        
        
        self.Data_BTC, self.MetaData_BTC = cc.get_digital_currency_daily('BTC','USD')
        self.Base_BTC = self.Data_BTC['4a. close (USD)'].to_frame().sort_index(axis = 0, ascending = True).reset_index().drop(columns='date')
        
        TI = TechIndicators( key=API_Key, output_format='pandas')
        
        self.Data_RSI, self.MetaData_RSI = TI.get_rsi(symbol=Crypto_Currency+Market,interval='daily', time_period=14,series_type='close')
        self.Base_RSI = self.Data_RSI.iloc[len(self.Data_RSI)-len(self.Base):,:].reset_index().drop(columns='date')
        
        
        self.Data_MACD, self.MetaData_MACD = TI.get_macd(symbol=Crypto_Currency+Market,interval='daily',series_type='close')
        self.Base_MACD = self.Data_MACD.iloc[len(self.Data_MACD)-len(self.Base):,2:3].reset_index().drop(columns='date') 
        
        self.DataSet = None
     
    def GetRSI(self) : 
        
        return self.Base_RSI
    
    def GetMACD(self):
        
        return self.Data_MACD
        
    def GetBase(self):
        
        return self.Base
    
    def GetDataSet(self):
        
        return self.DataSet
        
    def DataSetConstruction(self,Window1, Window2):
        
        Mm_10 = self.Base.rolling(window=Window1).mean()
        Mm_50 = self.Base.rolling(window=Window2).mean()

        Emw_10 = self.Base.ewm(span = Window1, adjust=False , min_periods = Window1).mean()
        Emw_50 = self.Base.ewm(span = Window2, adjust=False , min_periods = Window2).mean()
        
        if self.Base_RSI.empty : 
            
            self.DataSet = pd.concat((self.Base,Mm_10,Mm_50,Emw_50,self.Base),axis=1).dropna().reset_index(drop=True)

        else : 
            
            self.DataSet = pd.concat((self.Base,self.Base_BTC,self.Base_RSI,self.Base_MACD,Mm_50,self.Base),axis=1).dropna().reset_index(drop=True)
            
        
        ### Création de la volatilité : On laisse de côté pour le moment
        
        """
        for i in range(1,len(Base_Numpy)):
            Base_Numpy[i][0] = m.log(Base_Numpy[i][0]/Base_Numpy_Bis[i-1][0])
            
        Rendement = pd.DataFrame(data = Base_Numpy, columns =['Rendement daily'])
        
        Vol_10 = Rendement.rolling(window=10).std().reset_index(drop=True)
        Vol_50 = Rendement.rolling(window=50).std().reset_index(drop=True)
        """  


""" 
Test = DataLoader()
Test.Init_Crypto_Currency_API_Alpha_Vantage('PQWPDE41U7RFLYIE','ETH','USD')

y = Test.GetBase()

ma = Test.GetMACD()

Test.DataSetConstruction(10,20)

D = Test.GetDataSet()

print(Test.GetDataSet())

g = Test.GetRSI()

concat = pd.concat((y,g),axis=1)

eth_ma = y.rolling(10).mean()
"""
