# -*- coding: utf-8 -*-
##train codebook

#import cv2
#import numpy as np
#from numpy import array
#import os 
#import sklearn.metrics
import scipy
from scipy.cluster.vq import kmeans, whiten

#Importar funció get_local_features
from get_local_features import get_local_features

def train_codebook(descriptors):
    #Normalitzem els valors dels descriptors
    des=whiten(descriptors)
    #Creem una llista buida
    codebook={}
    #Omplim els dos parametres de la llista (codebook i distortion) amb els valor que retorna la funcio kmeans.
    [codebook,distortion] = kmeans(des, k_or_guess=4.0)
   
    #Retornem solament el parametre codebook
    return codebook
    
    
