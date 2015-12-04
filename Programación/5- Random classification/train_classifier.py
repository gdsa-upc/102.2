import os
from sklearn import svm

def train_classifier(params,caracteristiques_train):
    #Creacio SVC
    clf = svm.SVC()
    #Obrim el fitxer de les anotacions
    annotation = open(os.path.join(params['root'],params['database'],'train','annotation.txt'),'r')
    #Creacio de la llista buida
    labels = []
    #Bucle per obtenir les 'labels' del annotation.txt
    for line in annotation:
        #Afegim a la variable label totes les classes tretes del .txt de les annotacions
        #utilitzant la funcio split( ) i append( ) 
        labels.append(line.split()[0])
    #Eliminem classes repetides amb la funcio set
    labels = set(labels)
    #Clasificacio entre les 'features' i les classes 'labels'
    clf.fit(caracteristiques_train,labels)
