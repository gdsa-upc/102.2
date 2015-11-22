# -*- coding: utf-8 -*-

import os 
import random 

def classify(feat,path_out,labels):
    #Obrim .txt amb les 'features'
    infeat = open(feat,'r') 
    #Obrim .txt amb les 'labels'
    inlabels = open(labels,'r')
    #Obrim un .txt de sortida on guardarem les classificacions aleatories
    outfile = open(path_out +'/classify.txt.', 'w');
    
    for line in infeat:
        #Opció aleatoria 
        rand = random.choice(open(inlabels).readlines())
        outfile.write(line + "\t" + rand + '\n') # Posem un salt de linea i l'image ID actual
    #Tancar .txt de sortida
    outfile.close()
    #Tancar .txt d'entrada
        
