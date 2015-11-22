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
    lrand=[]
    for line in ID:
        lrand=np.random.rand(1,100)
        dic1[line.replace('\n','')]=lrand;
    ID.close();
    #Exportem el dictionary a un fitxer de text
    pk.dump(dic1,feat);
    feat.close();



                

    
    
    