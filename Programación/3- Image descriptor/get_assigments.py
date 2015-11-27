import cv2
import numpy as np
from numpy import array
import os 
import sklearn.metrics
import scipy
from get_params import get_params
from scipy.cluster.vq import vq, kmeans, whiten

from train_codebook import train_codebook

def get_assigments (codebook):
    