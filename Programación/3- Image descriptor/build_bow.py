import os
import numpy as np
import cPickle as pk
from sklearn import preprocessing

def build_bow(assignments,codebook, id_images):
    #Inicialitzem la llista bag of words amb tants zeros com assignments diferents hi hagi
    bow=[0]*len(set(assignments))
    for i in range(0,len(assignments)):
        bow[i]+=1
    bow=preprocessing.normalize(bow)