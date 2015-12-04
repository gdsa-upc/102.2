import os
from sklearn import svm

def train_classifier(params,caracteristiques_train):
    clf=svm.SVC()
    #Obrim el fitxer de les anotacions
    annotation = open(os.path.join(params['root'],params['database'],'train','annotation.txt'),'r')
    labels=[]
    for line in annotation:
        #Afegim a la variable label totes les classes tretes del .txt de les annotacions
        labels.append(line.split()[0])
    #Eliminem classes repetides amb la funcio set
    labels=set(labels)
    clf.fit(caracteristiques_train,labels)
