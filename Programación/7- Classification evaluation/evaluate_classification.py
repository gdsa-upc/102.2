# -*- coding: utf-8 -*-
import os
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.metrics import recall_score,f1_score
from sklearn.metrics import confusion_matrix


def evaluate_classification(automatic_annot,annotation):
    automatic = open(automatic_annot,'r')
    #Obrim el fitxer de les nostres anotacions automàtiques
    gr_truth = open(annotation,'r')
    #Obrim el fitxer de la ground truth
    automatic.readline()
    gr_truth.readline()
    #Aquests readlines() son per saltar-nos els titols "ClassID i ImageID" 
    lista_pred=[]
    #Declarem la llista que portarà les ClassID de les nostres anotacions automàtiques
    lista_ground_truth=[]
    #Declarem la llista que portarà les ClassID de la ground truth
    for line in automatic:
        ImageID1=line.split()[0]
        ClassID1=line.split()[1]
        #Segons els .txt entrants, el primer paràmetre que trobem a cada linia és la ImageID i el segon la ClassID
        gr_truth = open(annotation,'r')
        for line1 in gr_truth:
            ImageID2=line1.split()[0]
            ClassID2=line1.split()[1]
            if ImageID1==ImageID2:
                lista_pred.append(ClassID1)
                lista_ground_truth.append(ClassID2)
                #Per a cada ImageID assignada automàticament, mirem en la gorund_truth quina es la seva ClassID real i omplim les llistes
    #Bucles per omplir les llistes d'anotacions automatiques i ground truth
    automatic.close()
    #Tanquem el fitxer de les nostres anotacions automàtiques
    gr_truth.close()
    #Tanquem el fitxer de la ground truth
    accuracy=accuracy_score(lista_ground_truth, lista_pred) 
    #Accuracy classification score.
    ruta_lab=os.path.dirname(os.path.abspath(__file__))+'\\labels.txt'
    #Ruta del fitxer labels.txt
    labels=open(ruta_lab,'r')
    #Obrim labels.txt
    labels_list=[]
    #Declarem la llista que portara les ClassID
    for line2 in labels:
        line2=line2.replace('\n','')
        labels_list.append(line2)
    #Bucle per omplir la llista de ClassID
    labels.close()
    (precision_cl,recall_cl,f1_score_cl,support)=precision_recall_fscore_support(lista_ground_truth, lista_pred, labels=labels_list) 
    #Compute precision, recall, F-measure and support for each class
    suma=0
    for i in precision_cl:
        suma += i
    average_precis=suma/len(precision_cl)
    #Compute average precision (AP) from prediction scores
    recall=recall_score(lista_ground_truth, lista_pred) 
    #Compute the recall
    f1=f1_score(lista_ground_truth, lista_pred) 
    #Compute the F1 score, also known as balanced F-score or F-measure
    confusion_mat=confusion_matrix(lista_ground_truth, lista_pred, labels_list) 
    #Compute confusion matrix to evaluate the accuracy of a classification
    print "Mesures:\n-Accuracy = %s" % accuracy
    print "\n-Precision per cada classe: %s" % precision_cl
    print "\n-Recall per cada classe: %s" % recall_cl
    print "\n-F1-score per cada classe: %s" % f1_score_cl
    print "\n-Averaged Precision: %s" % average_precis
    print "\n-Recall: %s" % recall
    print "\n-F1-score: %s" %f1
    print "\n-Confusion matrix: %s" % confusion_mat