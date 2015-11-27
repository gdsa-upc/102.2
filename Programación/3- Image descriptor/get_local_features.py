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
    #Declarem el dictionary
    dic1=dict()
    orb=cv2.ORB()
    #ORB obté 500 features
        #Bucle per recorre cada imatge
    for line in ID:
        #Calcular ORB per a casa imatge
        img =  cv2.imread(os.path.join(params['root'],params['database'],'val','images',str(line).replace('\n','') + '.jpg'))
        kp=orb.detect(img,None)
        kp, des= orb.compute(img,kp,params['descriptor_size']) 
        #Guardem els valors al diccionari
        dic1[line.replace('\n','')]=des
    return dic1
    ID.close()


    
    


