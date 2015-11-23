# -*- coding: utf-8 -*-

import os 
import random 
import cPickle as pk

def classify(feat,path_out,labels):
    #Obrim .txt amb les 'labels' (12 línies, obviem la classe 'desconegut')
    in_l = open(labels,'r')
    #Obrim .txt amb les 'features'
    in_f = open(feat,'r') 
    #Obrim un .txt de sortida on guardarem les classificacions aleatories
    out = open(path_out +'\\classify.txt.', 'w');
    #Fem llegible el .txt de features (funció load)
    feat = pk.load(in_f)
    #Afegim primera linia al .txt de sortida
    out.write('ImageID' + '\t' + 'ClassID' + '\n')
    lista=[]
    for line in in_l:
        lista.append(line.replace('\n',''))
    for line1 in feat.keys():  #key es el camp que conté la ID de la imatge fins del .txt de 'features'
        #Escollim label aleatoria
        rand = random.choice(lista)
        #Escribim al .txt de sortida
        out.write(line1 + "\t" + rand + '\n')
        
    #Tancar .txt de sortida
    out.close()
    #Tancar .txt's d'entrada
    in_f.close()
    in_l.close()
        
