# -*- coding: utf-8 -*-
##train codebook
from scipy.cluster.vq import kmeans
from sklearn import preprocessing

def train_codebook(params,descriptores):
    des=preprocessing.normalize(descriptores)
    #Entrenamos el codebook, 4 es el numero de cluster que queremos 
    [codebook,dist] = kmeans(des, k_or_guess=4.0)
    return codebook
#Para probarlo
#params=get_params()
#train_codebook(get_local_features(params))
