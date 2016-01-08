# -*- coding: cp1252 -*-
import numpy as np
from get_params import get_params
from build_database import build_database
from get_features_test import get_features_test
from rank_kaggle_test import rank_kaggle_test
from train_classifier_test import train_classifier_test
from classify_kaggle_test import classify_kaggle_test
import warnings
warnings.filterwarnings("ignore")

#Extraccio dels parametres
params=get_params()
#Creacio de la base de dades
params['split']='train'
build_database(params)
params['split']='test'
build_database(params)
params['split']='val'
build_database(params)
#Extraccio de les caracteristiques pel kaggle del examen
get_features_test(params)
#Calcul del ranking per les de test
rank_kaggle_test(params)
#Entrenem un model de classificacio
train_classifier_test(params)
#Classificacio per les de test
classify_kaggle_test(params)