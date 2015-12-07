# -*- coding: utf-8 -*-
import os
import cPickle as pk
import numpy as np

def classify(params):
    #Obrim el model entrenat
    train_model= open((os.path.join(params['root'],params['database'],'train','Trained_model.txt')),'r')
    #Obrim el fitxer de Bag of Word de validacio
    fval= open((os.path.join(params['root'],params['database'],'val','Features.txt')),'r')
    #Declarem el diccionari
    dval=dict()
    #Carreguem la informacio als dictionaries
    clf2=pk.load(train_model)
    dval=pk.load(fval)
    dickeys=[]
    diclist=[]
    for key,value in dval.iteritems():
        dickeys.append(key)
        diclist.append(value)
    diclist=np.array(diclist)
    prediction=clf2.predict(diclist)
    print (prediction)
    fclassi = open((os.path.join(params['root'],params['database'],'val','Classify.txt')),'w')
    fclassi.write('ImageID'+"\t"+'ClassID'+"\n")
    #Per cada imatge de Validacio generem una classificacio 
    i=0
    for key in dickeys:
        fclassi.write(key+"\t"+prediction[i]+"\n")
        i=i+1
    #Tanquem els fitxers de característiques de validació i train
    fval.close()
    fclassi.close()
    
    
