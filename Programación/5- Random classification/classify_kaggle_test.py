# -*- coding: utf-8 -*-
import os
import cPickle as pk
import numpy as np

def classify_kaggle_test(params):
    #Obrim el model entrenat
    train_model= open((os.path.join(params['root'],params['database'],'train','Trained_model.txt')),'r')
    #Obrim el fitxer de Bag of Word de test
    ftest= open((os.path.join(params['root'],params['database'],'test','Features.txt')),'r')
    #Declarem el diccionari
    dtest=dict()
    #Carreguem la informacio als dictionaries
    clf2=pk.load(train_model)
    dtest=pk.load(ftest)
    dickeys=[]
    diclist=[]
    for key,value in dtest.iteritems():
        #Afegim la key i value
        dickeys.append(key)
        diclist.append(value)
    diclist=np.array(diclist)
    prediction=clf2.predict(diclist)
    fclassi = open((os.path.join(params['root'],params['database'],'test','Resultats_classify.txt')),'w')
    fclassi.write("Id,Prediction\n")
    #Per cada imatge de test generem una classificacio 
    i=0
    for key in dickeys:
        fclassi.write(key+","+prediction[i]+"\n")
        i=i+1
    #Tanquem els fitxers de característiques de test i train
    ftest.close()
    fclassi.close()
    