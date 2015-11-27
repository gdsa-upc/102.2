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
    #ORB obtiene 500 features de serie 
        #Recorrem cada Imatge 
    for line in ID:
        #Ahora tenemos que calcular ORB para cada imagen y guardarlo!
        ##leemos
        img =  cv2.imread(os.path.join(params['root'],params['database'],'val','images',str(line).replace('\n','') + '.jpg'))
        kp=orb.detect(img,None)
        kp, des= orb.compute(img,kp,params['descriptor_size'])
        #guardamos para cada imagen analizada sus descriptores
        dic1[str(line.replace('\n',''))]=des
    return dic1
    ID.close()


    
    


