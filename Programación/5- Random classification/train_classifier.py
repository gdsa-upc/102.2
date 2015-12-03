import os
from sklearn import svm

def train_classifier(params,caracteristiques_train):
    clf=svm.SVC()
    #Obrim el fitxer de les anotacions
    annotation = open(os.path.join(params['root'],params['database'],'train','annotation.txt'),'r')
    labels=[]
    for line in annotation:
        labels.append(line.split()[0])
    labels=set(labels)
    clf.fit(caracteristiques_train,labels)