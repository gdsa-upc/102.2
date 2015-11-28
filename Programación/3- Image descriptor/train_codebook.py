# -*- coding: utf-8 -*-

from scipy.cluster.vq import kmeans
from sklearn import preprocessing

def train_codebook(params,descriptores,paraules):
    #Normalitzem els valors dels descriptors
    des=preprocessing.normalize(descriptores)[0]
    #Entrenem el KMeans, ajustem el número de clusters a 4
    [codebook,dist] = kmeans(des, k_or_guess=paraules)
    return codebook
