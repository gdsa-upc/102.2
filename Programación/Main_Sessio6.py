# -*- coding: cp1252 -*-
import numpy as np
from get_params import get_params
from build_database import build_database
from get_features import get_features
from train_classifier import train_classifier
from classify import classify
from eval_classification import eval_classification
from eval_classification import plot_confusion_matrix
#import warnings
#warnings.filterwarnings("ignore")

#Extraccio dels parametres
params=get_params()
#Creacio de la base de dades
#params['split']='train'
#build_database(params)
#params['split']='val'
#build_database(params)
#Extraccio de les caracteri­stiques
#get_features(params)
#Entrenem un model de classificacio
train_classifier(params)
#Classificacio
classify(params)
#Avaluacio de la classificacio
f1, precision, recall, accuracy,cm, labels = eval_classification(params)
print "Mesures:\n"    
print f1
print "-F1:", np.mean(f1)
print "-Precision:", np.mean(precision)
print "-Recall:", np.mean(recall)
print "-Accuracy:", accuracy
print "-Confusion matrix:\n", cm

plot_confusion_matrix(cm, labels,normalize = True)
