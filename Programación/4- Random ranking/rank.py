# -*- coding: utf-8 -*-
## Random_ranking
import os
import random
import numpy as np
import cPickle as pk

def randomranking(pathval,pathtrain,outpath):
    #Obrim els fitxers de característiques de validació i train
    fval= open(pathval+'\\Features.txt','r')
    ftrain= open(pathtrain+'\\Features.txt','r')
    #Declarem els dictionaries
    dval=dict()
    dtrain=dict()
    #Carreguem la informació als dictionaries
    dval=pk.load(fval)
    dtrain=pk.load(ftrain)
    #Per cada imatge de Validació generem un ranking aleatori
    for vkey in dval:
        frank = open(outpath+'\\'+vkey+'.txt','w')
        for tkey in dtrain:
            frank.write(random.choice(dtrain.keys())+"\n")
        frank.close()
    #Tanquem els fitxers de característiques de validació i train
    fval.close()
    ftrain.close()
    
    