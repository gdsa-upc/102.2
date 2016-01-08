# -*- coding: utf-8 -*-
## Ranking
import os
import cPickle as pk
from sklearn import metrics

def rank_kaggle_test(params):
    #Obrim els fitxers de característiques de validació i train
    ftest= open((os.path.join(params['root'],params['database'],'test','Features.txt')),'r')
    ftrain= open((os.path.join(params['root'],params['database'],'train','Features.txt')),'r')
    #Declarem els dictionaries
    dtest=dict()
    dtrain=dict()
    #Carreguem la informació als dictionaries
    dtest=pk.load(ftest)
    dtrain=pk.load(ftrain)
    #Generem una llista amb les keys de train
    lhtrain=list()
    lktrain=dtrain.keys()
    #Generem una llista amb els histogrames del diccionari train
    for tkey in lktrain:
        lhtrain.append(dtrain[tkey])
        
    #Declaració de variables 
    ldist=list()#Llista de distàncies entre l'histograma de test i els de train
    tup=()#Tupla on guardarem la ID de train i la distància al histograma de test
    ltup=list()#Llista de tuples 
    #Obrim el fitxer on escriurem el ranking per la imatge de test
    frank = open((os.path.join(params['root'],params['database'],'test','Resultats_rank.txt')),'w')
    #Per cada imatge de Validació generem un ranking 
    frank.write("Query,RetrievedDocuments\n")
    for vkey in dtest:
        #Calculem la distància entre l'histograma de validació i els de train
        ldist=metrics.pairwise.pairwise_distances(dtest[vkey],lhtrain,metric='euclidean',n_jobs=1)[0]
        #Creem una llista de tuples amb les IDs i les distancies de train
        ldist=list(ldist)
        for tkey in lktrain:
            tup=(tkey,ldist.pop(0))
            ltup.append(tup)
        #Ordenem la llista segons la distancia 
        ltup=sorted(ltup, key=lambda distancia: distancia[1])
        #Escrivim cada ID ordenada al fitxer rank
        frank.write(vkey+',')
        for tupla in ltup:
            frank.write(tupla[0]+' ')
        frank.write("\n")
        ltup=[]
    #Tanquem els fitxers de característiques de test i train
    ftest.close()
    ftrain.close()
    frank.close()