# -*- coding: cp1252 -*-
import os
import numpy as np
import cPickle as pk
from get_params import get_params
from get_local_features import get_local_features
from train_codebook import train_codebook
from get_assignments import get_assignments
from build_bow import build_bow

#Extraccio dels parametres
params=get_params()
ID =open(os.path.join(params['root'],params['database'],'train','ImageIDs.txt'), 'r')
desc_train=get_local_features(params,os.path.join(params['root'],params['database'],'train','images',str(ID.readline()).replace('\n','') + '.jpg'))

#Extraccio de les características per a totes les imatges d'entrenament
for line in ID:
    x=get_local_features(params,os.path.join(params['root'],params['database'],'train','images',str(line).replace('\n','') + '.jpg'))
    #Concatenar les caracteristiques de cada imatge
    desc_train=np.concatenate((desc_train,x))
ID.close()

#Entrenament del KMeans només per les fotos d'entrenament
paraules=100
codebook=train_codebook(params,desc_train,paraules)
#Calculem les assignacions per les imatges d'entrenament
assignments=get_assignments(desc_train,codebook)
#Creació del diccionari
dictrain=dict()
ID =open(os.path.join(params['root'],params['database'],'train','ImageIDs.txt'), 'r')
dictrain[str(ID.readline()).replace('\n','')]=build_bow(assignments,codebook,paraules)

for line in ID:
    x=get_local_features(params,os.path.join(params['root'],params['database'],'train','images',str(line).replace('\n','') + '.jpg'))
    #Calculem les assignacions per les imatges d'entrenament
    assignments=get_assignments(x,codebook)
    #Creació del BoW per les imatges d'entrenament i emplenament del diccionari
    dictrain[str(line).replace('\n','')]=build_bow(assignments,codebook,paraules)
    print ('funcion build bow'+'\t')
    print build_bow(assignments,codebook,paraules)
    print ('\n')
    print dictrain[str(line).replace('\n','')]
ID.close()

#Creació del BoW per les imatges d'entrenament
bow_train = open (os.path.join(params['root'],params['database'],'train','Features.txt'), 'w')
pk.dump(dictrain,bow_train)
bow_train.close()

#Extraccio de les características per a totes les imatges de validacio
ID = open(os.path.join(params['root'],params['database'],'val','ImageIDs.txt'), 'r')
desc_val=get_local_features(params,os.path.join(params['root'],params['database'],'val','images',str(ID.readline()).replace('\n','') + '.jpg'))

#Calculem les assignacions per les imatges de validacio
assignments=get_assignments(desc_val,codebook)
#Creació del diccionari
dicval=dict()
dicval[str(ID.readline()).replace('\n','')]=build_bow(assignments,codebook,paraules)

for line in ID:
    x=get_local_features(params,os.path.join(params['root'],params['database'],'val','images',str(line).replace('\n','') + '.jpg'))
    #Calculem les assignacions per les imatges de validacio
    assignments=get_assignments(x,codebook)
    #Creació del BoW per les imatges de validacio i emplenament del diccionari
    dicval[str(line).replace('\n','')]=build_bow(assignments,codebook,paraules)
ID.close()

bow_val = open (os.path.join(params['root'],params['database'],'val','Features.txt'), 'w')
pk.dump(dicval,bow_val)
bow_val.close()

