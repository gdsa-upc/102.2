# -*- coding: utf-8 -*-
import os
from sklearn import svm,grid_search
import cPickle as pk
import numpy as np

def train_classifier(params):
    #Obrim el fitxer de les anotacions
    annotation = open(os.path.join(params['root'],params['database'],'train','annotation.txt'),'r')
    #Saltem la primera linia del fitxer per no llegir-la
    annotation.readline()
    #Declarem la llista de les labels
    labels=[]
    #Bucle per obtenir les 'labels' del annotation.txt
    for line in annotation:
        #Afegim a la variable label totes les classes tretes del .txt de les annotacions
        #utilitzant la funcio split( ) i append( ) 
        labels.append(line.split()[1])
    #Tanquem el fitxer
    annotation.close()
    #Declarem el diccionari amb les labels i el pes que tindran
    dic_labels=dict()
    #Omplim el diccionari amb les labels i els seus pesos
    for label in set(labels):
        if label == 'desconegut':
            dic_labels[label]=0.167
        else:
           dic_labels[label]=1
    #Declaracio de la clf
    #clf = svm.SVC(C=1000, gamma=0.001, class_weight=dic_labels)
    svr = svm.SVC()
    parameters = {'C':[1, 10, 100, 1000, 10000],'kernel':('linear', 'rbf'), 'gamma':[0.001, 0.00001, 0.0000000001], 'class_weight':[dic_labels]}
    clf = grid_search.GridSearchCV(svr, parameters)
    #Obrim el fitxer Features.txt de train
    bow_train= open((os.path.join(params['root'],params['database'],'train','Features.txt')),'r')
    #Creem i omplim el diccionari de train
    dftrain=dict()
    dftrain=pk.load(bow_train)
    #Reobrim el fitxer de les anotacions
    annotation = open(os.path.join(params['root'],params['database'],'train','annotation.txt'),'r')
    #Saltem la primera linia del fitxer per no llegir-la
    annotation.readline()
    #Declaració del diccionari
    dklab=dict()
    for line in annotation:
        #Afegim a la variable label totes les classes tretes del .txt de les annotacions
        #utilitzant la funcio split( ) i append( ) 
        dklab[line.split()[0]]=line.split()[1]
    annotation.close()
    
    #Creem dues llistes on hi hagin les "features" i els "labels"
    lf=list()
    ll=list()
    for key in dklab.keys():
        ll.append(dklab[key])
        lf.append(dftrain[key])
        
    lf=np.array(lf)
    ll=np.array(ll)
    #Entrenem el model SVM amb les dades d'entrenament
    clf.fit(lf,ll) # Exemple: http://scikit-learn.org/stable/tutorial/basic/tutorial.html
    
    #Guardem el model a memoria. El pk.dumps guarda el model en un string en comptes de en un fitxer
    trained_model = open (os.path.join(params['root'],params['database'],'train','Trained_model.txt'), 'w')
    pk.dump(clf,trained_model)
    trained_model.close()
