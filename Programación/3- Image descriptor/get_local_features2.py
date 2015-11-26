# -*- coding: utf-8 -*-
import os
import numpy as np
import cPickle as pk
import cv2
import pandas as pd


def get_local_features(params):
    #Obrim .txt d'entrada
    ID = pd.read_csv(os.path.join(params['root'],params['database'],'val','ImageIDs.txt'), sep='\t', header = 0)
    #Obrim .txt de sortida
    feat = open('C:\Users\Gerard\Documents\Universidad\TercerCurso\GDSA\Proyecto\Codigo\pruebalocalfeat.txt')  
    #Declarem el dictionary
    dic1=dict()
    orb=cv2.ORB()
        #Creació d'una llista buida
    
    for line in ID:
        #Ahora tenemos que calcular ORB para cada imagen y guardarlo!
        ##leemos
        img =  cv2.imread(os.path.join(params['root'],params['database'],'val','images',str(line) + '.jpg'))
        kp=orb.detect(img,None)
        kp, des= orb.compute(img,kp) #computamos
        #guardamos para cada valor de la imagen analizada sus descriptores
        dic1[line.replace('\n','')]=des
    
    #Tanquem el .txt d'entrada
    return dic1
    pk.dump(dic1,feat)
    #Exportem el dictionari a un fitxer de text
 
    #Tanquem el .txt de sortida
    feat.close();


