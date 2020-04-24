#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:46:19 2020

@author: clementgrattaroly
"""
import numpy as np
import math as m
import matplotlib.pyplot as plt

def Trend_Predict_Eval(Length_Seq,Data_Valid_Denorm,Y_Pred_Denorm,Var_Bound):
    
    Trend_Predict_Eval = np.zeros((Y_Pred_Denorm.shape[0],2,2))
    
    # On définit une matrice 2x2 pour déterimner le pourcentage de réussite des prédictions sur les tendances
    # La matrice sera réalisée de la manière suivante : 
    #       Haut    Bas
    #  Haut
    
    #  Bas
    # Ex : L'algorithme placera la valeur 1 dans (Bas,Bas) si la prédiction était baissière et qu'elle disait vraie par rapport au vrai cours
    
    
    for i in range(Y_Pred_Denorm.shape[0]):
        
        
        Predict_Var = (Y_Pred_Denorm[i] - Data_Valid_Denorm[i][Length_Seq-1][0])/Data_Valid_Denorm[i][Length_Seq-1][0]
        Real_Var = (Data_Valid_Denorm[i][Length_Seq-1][Data_Valid_Denorm.shape[2]-1]-Data_Valid_Denorm[i][Length_Seq-1][0])/Data_Valid_Denorm[i][Length_Seq-1][0]
        
        if( m.fabs(Predict_Var) >= Var_Bound ):

            if(Predict_Var > 0 and Real_Var > 0):
                Trend_Predict_Eval[i,0,0] = 1
                
            elif(Predict_Var > 0 and Real_Var <= 0):
                Trend_Predict_Eval[i,0,1] = 1
                
            elif(Predict_Var <=0 and Real_Var <= 0):
                Trend_Predict_Eval[i,1,1] = 1
            
            else : 
                Trend_Predict_Eval[i,1,0] = 1
                
    Average_Matrix = np.mean(Trend_Predict_Eval,0)
    
    print(" \n Performance dans la prédiction des tendances : au seuil " + str(Var_Bound))
    print(" \n Hp Hv : " + str(Average_Matrix[0,0]*100)+ " %")
    print(" \n Hp Lv : " + str(Average_Matrix[0,1]*100)+ " %")
    print(" \n Lp Hv : " + str(Average_Matrix[1,0]*100)+ " %")
    print(" \n Lp Lv : " + str(Average_Matrix[1,1]*100)+ " %")
    
    
def TradingFutures(Capital,Data_Valid_Denorm,Y_Pred_Denorm,Leverage,Top_Var,Bottom_Var,Length_Seq):
    
    Nb_Periods = len(Y_Pred_Denorm)
    Gain = np.zeros(Nb_Periods)
    
    Bull_Trade = np.zeros(Nb_Periods)
    Bear_Trade = np.zeros(Nb_Periods)
    
    Bull_Gain = np.zeros(Nb_Periods)
    Bear_Gain = np.zeros(Nb_Periods)
    
    for t in range(Nb_Periods) : 
        
        Current_Price = Data_Valid_Denorm[t][Length_Seq-1][0]
        Next_Real_Price = Data_Valid_Denorm[t][Length_Seq-1][Data_Valid_Denorm.shape[2]-1]
        
        Pred_Var = (Y_Pred_Denorm[t] - Data_Valid_Denorm[t][Length_Seq-1][0])/Data_Valid_Denorm[t][Length_Seq-1][0]
        Real_Var = (Data_Valid_Denorm[t][Length_Seq-1][Data_Valid_Denorm.shape[2]-1]-Data_Valid_Denorm[t][Length_Seq-1][0])/Data_Valid_Denorm[t][Length_Seq-1][0]

        if (Pred_Var>Top_Var): 
            
            Bull_Trade[t] = 1
            
            Bull_Gain[t] = (Next_Real_Price - Current_Price) * Leverage
            
            Gain[t] = Bull_Gain[t]
        
        elif(Pred_Var<Bottom_Var): 
            
            Bear_Trade[t] = 1
            
            Bear_Gain[t] = -(Next_Real_Price - Current_Price) * Leverage
            
            Gain[t] = Bear_Gain[t]
       
    Gain_Over_Time = np.cumsum(Gain)
    
    Gain_Total = np.sum(Gain)
    
    print(Gain)
    
    plt.plot(Gain_Over_Time, color = 'purple' ,label = "Gain")
    plt.xlabel("Day")
    plt.ylabel("Gain €")
    plt.legend(loc='upper left')
    plt.show
    
        
    print("\n Le Gain est de : " + str(Gain_Total) + " €")
    
    Average_Bull_Gain = np.mean(Bull_Gain)
    
    Average_Bear_Gain = np.mean(Bear_Gain)
    
    Bull_Gain_Total = np.sum(Bull_Gain)
    
    Bear_Gain_Total = np.sum(Bear_Gain)
    
    Bull_Trade_Total = np.sum(Bull_Trade==1)
    
    Bear_Trade_Total = np.sum(Bear_Trade==1)
    
    #ain_Per_Bull_Trade = m.ceil(Bull_Gain_Total/Bull_Trade_Total)
    
    #Gain_Per_Bear_Trade = m.ceil(Bear_Gain_Total/Bear_Trade_Total)
    
    print("\n Gain à la hausse : " + str(Bull_Gain_Total) + " Gain à la baisse : "+ str(Bear_Gain_Total))
    
    print("\n Trade Long : " + str(Bull_Trade_Total) + " Trade short : " + str(Bear_Trade_Total))
    
    #print("\n Gain moyen par trade : Long = " + str(Gain_Per_Bull_Trade) + " Short = " + str(Gain_Per_Bear_Trade))
        
    
        
        
    
    
    
    

        
        
        
        
        
            
            

        
        
        
        
    