# -*- coding: utf-8 -*-
##train codebook

#import cv2
#import numpy as np
#from numpy import array
#import os 
#import sklearn.metrics
import scipy
from scipy.cluster.vq import kmeans, whiten
#from get_params import get_params

from get_local_features import get_local_features

def train_codebook(descriptores):
    des=whiten(descriptores)
    #Entrenamos el codebook, 4 es el numero de cluster que queremos 
    codebook={}
    codebook = kmeans(des, k_or_guess=4.0)
    return codebook
    
    
#Para probarlo
#params=get_params()
#train_codebook(get_local_features(params))
