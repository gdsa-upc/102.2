# -*- coding: cp1252 -*-
import os
import numpy as np
import cPickle as pk
from get_local_features import get_local_features
from train_codebook import train_codebook
from get_assignments import get_assignments
from build_bow import build_bow

def get_features(params):
    #Obrim el fitxer que conte les ID de les imatges d'entrenament
    ID=open(os.path.join(params['root'],params['database'],'train','ImageIDs.txt'), 'r')
    #Extraccio de les caracteristiques de la imatge de la primera linia del ImageIDs.txt
    nom=str(ID.readline()).replace('\n','')
    desc_train=get_local_features(params,os.path.join(params['root'],params['database'],'train','images',nom + '.jpg'))
    #Extraccio de les caracteristiques per a la resta de les imatges d'entrenament
    dictrain=dict()
    dictrain[nom]=desc_train
    for line in ID:
        nom=str(line).replace('\n','')
        x=get_local_features(params,os.path.join(params['root'],params['database'],'train','images',nom + '.jpg'))
        #Concatenar les caracteristiques de cada imatge en una numpy array
        desc_train=np.concatenate((desc_train,x))
        dictrain[nom]=x
    #Tanquem el fitxer
    ID.close()

    #Entrenament del KMeans nomes per a les imatges d'entrenament amb 1000 paraules
    paraules=1024
    codebook=train_codebook(params,desc_train,paraules)
    #Obrim el fitxer que conte les ID de les imatges d'entrenament
    ID=open(os.path.join(params['root'],params['database'],'train','ImageIDs.txt'), 'r')

    for line in ID:
        nom=str(line).replace('\n','')
        #Calculem les assignacions per les imatges d'entrenament
        assignments=get_assignments(dictrain[nom],codebook)
        #Creacio del BoW per les imatges d'entrenament i emplenament del diccionari
        dictrain[nom]=build_bow(assignments,codebook,paraules)
    #Tanquem el fitxer
    ID.close()

    #Guardem el diccionari amb el BoW de les imatges d'entrenament en l'arxiu "Features.txt"
    bow_train = open (os.path.join(params['root'],params['database'],'train','Features.txt'), 'w')
    pk.dump(dictrain,bow_train)
    bow_train.close()

    #Obrim el fitxer que conté les ID de les imatges de validacio
    ID = open(os.path.join(params['root'],params['database'],'val','ImageIDs.txt'), 'r')
    #Creacio del diccionari de les imatges de validacio
    dicval=dict()
    for line in ID:
        #Extraccio de les caracteristiques per a les imatges de validacio
        x=get_local_features(params,os.path.join(params['root'],params['database'],'val','images',str(line).replace('\n','') + '.jpg'))
        #Calculem les assignacions per les imatges de validacio
        assignments=get_assignments(x,codebook)
        #Creacio del BoW per les imatges de validacio i emplenament del diccionari
        dicval[str(line).replace('\n','')]=build_bow(assignments,codebook,paraules)
    #Tanquem el fitxer
    ID.close()

    #Guardem el diccionari amb el BoW de les imatges de validacio en l'arxiu "Features.txt"
    bow_val = open (os.path.join(params['root'],params['database'],'val','Features.txt'), 'w')
    pk.dump(dicval,bow_val)
    bow_val.close()



                

    
    
    
