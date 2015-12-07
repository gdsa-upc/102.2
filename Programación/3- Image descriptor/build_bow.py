# -*- coding: cp1252 -*-
from sklearn import preprocessing

def build_bow(assignments,codebook,paraules):
    #Inicialitzem la llista bag of words amb tants zeros com assignments diferents hi hagi
    bow=[0]*paraules
    for x in range(0,len(bow)):
        for i in assignments:
            if i==x:
                bow[x]+=1
    #Normalitzem els valors amb la normalitzacio L2
    bow=preprocessing.normalize(bow)[0]
    return bow
