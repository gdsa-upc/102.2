# -*- coding: utf-8 -*-
## Classify
import os
import cPickle as pk
from sklearn import metrics

def classify(params):
    #Obrim el model entrenat
    train_model= open((os.path.join(params['root'],params['database'],'train','train_model.txt')),'r')
    #Obrim el fitxer de Bag of Word de validacio
    fval= open((os.path.join(params['root'],params['database'],'val','Features.txt')),'r')
    #Declarem els dictionaries
    dmodel=dict()
    dval=dict()
    #Carreguem la informacio als dictionaries
    dmodel=pk.load(train_model)
    dval=pk.load(fval)
    #Generem una llista amb les keys de train
    lktrain=dmodel.keys()
    #Generem una llista amb els histogrames del diccionari train
    lhtrain=list()
    for tkey in lktrain:
        lhtrain.append(dmodel[tkey])
    #Declaracio de variables 
    ldist=list()#Llista de distancies entre l'histograma de validació i els de train
    tup=()#Tupla on guardarem la ID de train i la distància al histograma de validacio 
    ltup=list()#Llista de tuples 
    #Obrim el fitxer on escriurem la classificacio per les imatges de validacio
    fclassi = open((os.path.join(params['root'],params['database'],'val','result','Classify.txt')),'w')
    fclassi.write('ImageID'+"\t"+'ClassID'+"\n")
    #Per cada imatge de Validacio generem una classificacio 
    for vkey in dval:
        #Calculem la distancia entre l'histograma de validació i els de train
        ldist=metrics.pairwise.pairwise_distances(dval[vkey],lhtrain,metric='euclidean',n_jobs=1)[0]
        #Creem una llista de tuples amb les IDs i les distancies de train
        ldist=list(ldist)
        for tkey in lktrain:
            tup=(tkey,ldist.pop(0))
            ltup.append(tup)
        #Ordenem la llista segons la distancia
        ltup=sorted(ltup, key=lambda distancia: distancia[1])
        annotation=open((os.path.join(params['root'],params['database'],'val','annotation.txt')),'r')
        for line in annotation:
            ImageID=line.split()[0]
            ClassID=line.split()[1]
            #Segons el .txt entrant, el primer paràmetre que trobem a cada linia és la ImageID i el segon la ClassID
            if ImageID==ltup[0][0]:
                fclassi.write(ltup[0][0]+"\t"+ClassID+"\n")
                break
        annotation.close()
        ltup=[]
    #Tanquem els fitxers de característiques de validació i train
    fval.close()
    fclassi.close()
