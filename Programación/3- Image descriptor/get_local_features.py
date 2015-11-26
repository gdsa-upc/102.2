# -*- coding: utf-8 -*-
import os
import numpy as np
import cPickle as pk
import cv2
import pandas as pd
from get_params import get_params

def get_local_features(params):
    #Obrim .txt d'entrada
    ID =open(os.path.join(params['root'],params['database'],'val','ImageIDs.txt'), 'r')
    #Obrim .txt de sortida
    feat = open('C:\Users\Gerard\Documents\Universidad\TercerCurso\GDSA\Proyecto\Codigo\pruebalocalfeat.txt')  
    #Declarem el dictionary
    dic1=dict()
    orb=cv2.ORB()
        #Recorrem cada Imatge 
    for line in ID:
        #Ahora tenemos que calcular ORB para cada imagen y guardarlo!
        ##leemos
        img =  cv2.imread(os.path.join(params['root'],params['database'],'val','images',str(line).replace('\n','') + '.jpg'))
        kp=orb.detect(img,None)
        kp, des= orb.compute(img,kp,params['descriptor_size']) #computamos
        #guardamos para cada valor de la imagen analizada sus descriptores
        dic1[line.replace('\n','')]=des
    return dic1
    ID.close()

    


