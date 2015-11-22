# -*- coding: utf-8 -*-
import os
import random
import numpy as np


def getfeatures(path,IDs,path_out):
    #Obrim .txt de sortida
    feat = open (path_out+ '\\features.txt', 'w')   
    #Obrim .txt d'entrada
    ID = open(IDs, 'r')
    #path de lectura de les imatges
    for line in ID:
        aleat=np.random.rand(1,100)
        feat.write(line + '\t' + str(aleat).replace("\n"," ").replace("[["," ").replace("]]"," ") +'\n')
        # Cerramos el fichero.
    ID.close()
        #Tancar .txt de sortida
    feat.close()


                

    
    
    