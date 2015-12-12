# -*- coding: cp1252 -*-
import os
import numpy as np
import cPickle as pk
from get_local_features import get_local_features
from train_codebook import train_codebook
from get_assignments import get_assignments
from build_bow import build_bow
import warnings
warnings.filterwarnings("ignore")

def get_features(params):
    #Obrim el fitxer que conté les ID de les imatges d'entrenament
    ID=open(os.path.join(params['root'],params['database'],'train','ImageIDs.txt'), 'r')
    #Extraccio de les caracteri­stiques de la imatge de la primera linia del ImageIDs.txt
    desc_train,desc_sift=get_local_features(params,os.path.join(params['root'],params['database'],'train','images',str(ID.readline()).replace('\n','') + '.jpg'))
    #Extraccio de les caracteri­stiques per a la resta de les imatges d'entrenament
    i=0
    desc_train=np.concatenate((desc_train,desc_sift))
    for line in ID:
        x,y=get_local_features(params,os.path.join(params['root'],params['database'],'train','images',str(line).replace('\n','') + '.jpg'))
        #Concatenar les caracteristiques de cada imatge en una numpy array
        desc_train=np.concatenate((desc_train,x))
       
        desc_train=np.concatenate((desc_train,y))
        print i
        i=i+1
    #Tanquem el fitxer
    ID.close()

    #Entrenament del KMeans nomes per a les imatges d'entrenament amb 100 paraules
    paraules=1000
    codebook=train_codebook(params,desc_train,paraules)
    #Calculem les assignacions per les imatges d'entrenament
    assignments=get_assignments(desc_train,codebook)
    #Creacio del diccionari de les imatges d'entrenament
    dictrain=dict()
    #Obrim el fitxer que conté les ID de les imatges d'entrenament
    ID=open(os.path.join(params['root'],params['database'],'train','ImageIDs.txt'), 'r')

    for line in ID:
        x,y=get_local_features(params,os.path.join(params['root'],params['database'],'train','images',str(line).replace('\n','') + '.jpg'))
        #Calculem les assignacions per les imatges d'entrenament
        x=np.concatenate((x,y))
        #ojo aki   aaa
        assignments=get_assignments(x,codebook)
        #Creacio del BoW per les imatges d'entrenament i emplenament del diccionari
        dictrain[str(line).replace('\n','')]=build_bow(assignments,codebook,paraules)
    #Tanquem el fitxer
    ID.close()

    #Guardem el diccionari amb el BoW de les imatges d'entrenament en l'arxiu "Features.txt"
    bow_train = open (os.path.join(params['root'],params['database'],'train','Features.txt'), 'w')
    pk.dump(dictrain,bow_train)
    bow_train.close()

    #Obrim el fitxer que conté les ID de les imatges de validació
    ID = open(os.path.join(params['root'],params['database'],'val','ImageIDs.txt'), 'r')
    #Creacio del diccionari de les imatges de validacio
    dicval=dict()
    for line in ID:
        #Extraccio de les caracteri­stiques per a les imatges de validacio
        x,y=get_local_features(params,os.path.join(params['root'],params['database'],'val','images',str(line).replace('\n','') + '.jpg'))
        x=np.concatenate((x,y))
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



                

    
    
    
