# -*- coding: utf-8 -*-
import os
from sklearn import svm,grid_search
import cPickle as pk
import numpy as np
from scipy.stats import itemfreq

def train_classifier_test(params):
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
    ##for label in set(labels):
    ##    if label == 'desconegut':
    ##         dic_labels[label]=0.167
    ##    else:
    ##       dic_labels[label]=1
    dic_labels=get_class_weights(labels)
    #Declaracio de la clf
    #clf = svm.SVC(C=1000, gamma=0.001, class_weight=dic_labels)
    clf = svm.SVC(class_weight=dic_labels)
    ##clf = tune_parameters(params,X,y,clf)
    ##parameters = {'C':[1, 10, 100, 1000, 10000],'kernel':('linear', 'rbf'), 'gamma':[0.001, 0.00001, 0.0000000001], 'class_weight':[dic_labels]}
    ##clf = grid_search.GridSearchCV(svr, parameters)
    #Obrim el fitxer Features.txt de train i validacio
    bow_train= open((os.path.join(params['root'],params['database'],'train','Features.txt')),'r')
    bow_val= open((os.path.join(params['root'],params['database'],'val','Features.txt')),'r')
    #Creem i omplim el diccionari de train i validacio
    dftrain=dict()
    dftrain=pk.load(bow_train)
    dfval=dict()
    dfval=pk.load(bow_val)
    #Reobrim el fitxer de les anotacions
    annotation = open(os.path.join(params['root'],params['database'],'train','annotation.txt'),'r')
    #Saltem la primera linia del fitxer per no llegir-la
    annotation.readline()
    #Declaració del diccionaris
    dklabt=dict()
    for line in annotation:
        #Afegim a la variable label totes les classes tretes del .txt de les annotacions
        #utilitzant la funcio split( ) i append( ) 
        dklabt[line.split()[0]]=line.split()[1]
    annotation.close()
    
    #Creem dues llistes on hi hagin les "features" i els "labels"
    lft=list()
    ll_t=list()
    for key in dklabt.keys():
        ll_t.append(dklabt[key])
        lft.append(dftrain[key])
    lft=np.array(lft)
    ll_t=np.array(ll_t)
    
    #Obrim el fitxer de les anotacions
    annotation = open(os.path.join(params['root'],params['database'],'val','annotation.txt'),'r')
    #Saltem la primera linia del fitxer per no llegir-la
    annotation.readline()
    #Declaració del diccionaris
    dklabv=dict()
    for line in annotation:
        #Afegim a la variable label totes les classes tretes del .txt de les annotacions
        dklabv[line.split()[0]]=line.split()[1]
    annotation.close()
    #Creem dues llistes on hi hagin les "features" i els "labels"
    lfv=list()
    ll_v=list()
    for key in dklabv.keys():
        ll_v.append(dklabv[key])
        lfv.append(dfval[key])
    lfv=np.array(lfv)
    ll_v=np.array(ll_v)
    
    lf=np.concatenate((lft,lfv))
    ll=np.concatenate((ll_t,ll_v))
    
    clf = tune_parameters(params,lf,ll,clf)
    #Entrenem el model SVM amb les dades d'entrenament
    clf.fit(lf,ll) # Exemple: http://scikit-learn.org/stable/tutorial/basic/tutorial.html
    
    #Guardem el model a memoria. El pk.dumps guarda el model en un string en comptes de en un fitxer
    trained_model = open (os.path.join(params['root'],params['database'],'train','Trained_model.txt'), 'w')
    pk.dump(clf,trained_model)
    trained_model.close()
    
def get_class_weights(labels):
    # Get number of times each class appears in the training data
    freq = itemfreq(labels)

    # Isolate values and convert to integer
    freq = np.array(freq[:,1]).astype(int)

    # Get individual class weights. If a class has more samples, its weight is smaller
    freq = float(len(labels))/ (13 * freq)

    # Get unique class names
    class_names = np.unique(labels)

    # Init dictionary
    class_weights = {}

    # Put class weights in dictionary
    for i in range(len(class_names)):

        class_weights[class_names[i]] = freq[i]

    return class_weights
    
def tune_parameters(params,X,y,clf):
    # Initialize the gridsearch object
    gs = grid_search.GridSearchCV(clf, params['svm_tune'],cv=5,scoring='f1_macro')
    
    # Fit data
    gs.fit(X,y)
    
    print "Chosen parameters", gs.best_params_
    print "Score during tuning:", gs.best_score_
    
    return gs.best_estimator_
