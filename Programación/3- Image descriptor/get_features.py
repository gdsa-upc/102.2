# -*- coding: utf-8 -*-
import os
import numpy as np
import cPickle as pk


def get_features(diropen,dirIDs,dirsave):
    #Obrim .txt d'entrada
    ID = open(dirIDs+'\\ImageIDs.txt', 'r')
    #Obrim .txt de sortida
    feat = open (dirsave+ '\\Features.txt', 'w')  
    #Declarem el dictionary
    dic1=dict()
    #Creació d'una llista buida
    lrand=[]
    for line in ID:
        #Creació d'un vector de mida 100 amb valors aleatoris més petits que 1
        lrand=np.random.rand(1,100)
        dic1[line.replace('\n','')]=lrand;
    #Tanquem el .txt d'entrada
    ID.close();
    #Exportem el dictionari a un fitxer de text
    pk.dump(dic1,feat);
    #Tanquem el .txt de sortida
    feat.close();



                

    
    
    
