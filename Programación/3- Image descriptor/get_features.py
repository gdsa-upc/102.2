# -*- coding: utf-8 -*-
import os
import random

path = os.path.dirname(os.path.abspath(__file__))

def getfeatures(test_val,path_out):
    #Obrim .txt de sortida
    feat = open (path + test_val + '.txt', 'w')   
    #Obrim .txt d'entrada
    ID = open(path + test_val +'.txt', 'r')
    
    for line in ID:
        #Assignem un valor aleatori entre 1 i 100 per a cada entrada
        feat = random.randrange(1,100)
        #Localitzar el salt de linea
        end = line.index("\n")
        #Omplim .txt de sortida
        feat.write(line[0:final] ... 
        #Tancar .txt de sortida
        feat.close()


                

    
    
    