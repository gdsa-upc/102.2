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
    #Declarem el diccionari amb les labels i el pes que tindran
    dic_labels=dict()
    #Omplim el diccionari amb les labels i els seus pesos
    for label in set(labels):
        if label is not 'desconegut':
            dic_labels[label]=1
        else:
           dic_labels[label]=0.17 
    clf = svm.SVC(gamma=0.001, C=100, class_weight=dic_labels)
    bow_train= open((os.path.join(params['root'],params['database'],'train','Features.txt')),'r')
    caracteristiques_train=dict()
    caracteristiques_train=pk.load(bow_train)
    diclist=[]
    for key,value in caracteristiques_train.iteritems():
        diclist.append(value)
    print clf.fit(diclist,labels)
