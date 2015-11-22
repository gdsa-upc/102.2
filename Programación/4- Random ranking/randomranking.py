## Random_ranking
import os
import random
import numpy as np



def randomranking(pathval,pathtrain,outpath):
    
    feat= open(pathval,'r')
    feat_train= open(pathtrain,'r')
    #Ahora que tenemos abiertos los dos txt que nos interesan, debemos abrir un txt
    #para cada query y rellenarlo aleatoriamente con el valor de train txt
   
    for files in feat():
        ranking = open(outpath+'\\rank_val_test_'+files+'.txt','w')
    # Ahora ya tenemos todas los txt creados hay que rellenarlos de manera aleatoria
    for files in feat():
        ranking.write(random.choice(feat_train.keys())+"\\n")
        ranking.close()
    
    feat.close()
    feat_train.close()
    
    