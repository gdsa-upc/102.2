import cv2
import numpy as np
from numpy import array
import os 
import sklearn.metrics
import scipy
from get_params import get_params
from scipy.cluster.vq import vq, kmeans, whiten

#Importem la funció train_codebook
from train_codebook import train_codebook


def get_assignments (descriptores,codebook):
    #Calculem les assignacions i omplim els parametres (code i dist)
    [code,dist]=vq(descriptores,codebook)
    
    #Retornem el vector amb les assignacions per cada descriptor
    return assignments
