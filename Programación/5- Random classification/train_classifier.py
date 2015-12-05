# -*- coding: utf-8 -*-
import os
from sklearn import svm
import cPickle as pk
import numpy as np

def train_classifier(params):
    #Obrim el fitxer de les anotacions
    annotation = open(os.path.join(params['root'],params['database'],'train','annotation.txt'),'r')
    #Saltem la primera linia del fitxer
    annotation.readline()
    #Declarem la llista de les labels
    labels=[]
    #Bucle per obtenir les 'labels' del annotation.txt
    for line in annotation:
        #Afegim a la variable label totes les classes tretes del .txt de les annotacions
        #utilitzant la funcio split( ) i append( ) 
        labels.append(line.split()[1])
    annotation.close()
    #Declarem el diccionari amb les labels i el pes que tindran
    dic_labels=dict()
    #Omplim el diccionari amb les labels i els seus pesos
    for label in set(labels):
        if label is not 'desconegut':
            dic_labels[label]=1
        else:
           dic_labels[label]=0.17
    #Declaracio de la clf
    clf = svm.SVC(gamma=0.001, C=100, class_weight=dic_labels)
    #Obrim el fitxer Features.txt de train
    bow_train= open((os.path.join(params['root'],params['database'],'train','Features.txt')),'r')
    #Creem i omplim el diccionari de train
    ftrain=dict()
    ftrain=pk.load(bow_train)
    #Creem una llista amb les ImageIDs del diccionari
    lftrain=list()
    lftrain=ftrain.keys()
    #Creem una llista amb les "features" de les ImageIDs
    for tkey in lftrain:
        lftrain.append(ftrain[tkey])
    #Reobrim el fitxer de les anotacions
    annotation = open(os.path.join(params['root'],params['database'],'train','annotation.txt'),'r')
    #Saltem la primera linia del fitxer
    annotation.readline()
    #Per realitzar el pas seguent creem un diccionari amb les ImageIDs de train i el label
    dklab=dict()
    for line in annotation:
        #Afegim a la variable label totes les classes tretes del .txt de les annotacions
        #utilitzant la funcio split( ) i append( ) 
        dklab[line.split()[0]]=line.split()[1]
    annotation.close()
    
    #Creem una llista de 2 columnes, la primera amb els "labels" i la segona amb les "features"
    #Ej:  Classe2   Features[0]
    #     Classe4   Features[1]
    #     Classe2   Features[2]
    llf=list()
    for key in lftrain
        llf=
    
    
    
    #Entrenem el model SVM amb les dades d'entrenament
    x=clf.fit(llf,labels)
    
    #Guardem el model a memòria
    #.....