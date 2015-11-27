# -*- coding: utf-8 -*-
##train codebook

import cv2
import numpy as np
from numpy import array
import os 
import sklearn.metrics
import scipy
from get_params import get_params
from scipy.cluster.vq import vq, kmeans, whiten

from get_local_features import get_local_features

def train_codebook(descriptores):
    des=whiten(descriptores)
    #Entrenamos el codebook, 4 es el numero de cluster que queremos 
    codebook={}
    codebook = scipy.cluster.vq.kmeans(des, 4, iter=20, thresh=1e-05, check_finite=True)
    return codebook
    
    
#Para probarlo
params=get_params()
train_codebook(get_local_features(params))
