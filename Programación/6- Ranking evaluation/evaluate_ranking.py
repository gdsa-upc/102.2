# -*- coding: utf-8 -*-
## Ranking evaluation
## Egara-View
import os
import random
import numpy as np
from sklearn.metrics import average_precision_score

def evaluate_ranking7(path,Gt_val_test):
    ##  EXPLICACIÓN:
    # Primero hay que recorrer todos los rankings generados, obtener a que clase pertenece la ID general de cada ranking
    #una vez tenemos la clase de la id, ver de todos los resultados del ranking correspondiente a esa imagen cuantos también
    #corresponden a esa clase, almacenarlo en un contador y calcular el average precision, con los aciertos para cada imagen 
    ## Hacer bucle que me recorra los 180 archivos del ranking
    ranking=open(path,'r')
    gr_truth=open(Gt_val_test,'r')
    dirs=os.listdir(path) #con esto podremos recorrer todos los archivos del directorio
    #Ahora recorro todos los ficheros de ranking 
    ##Creo las listas donde almacenaré datos
    lista_ground_truth=[]
    lista_aciertos=[]
    path.readline()
    Gt_val_test.readline()
    
    for file in dirs:
        #Ahora debo detectar a que clase pertenece la ID del ranking a analizar
        #saco la clase a la que pertenece esta fila comparandola con la GT
        (nombreFichero, extension) = os.path.splitext(file)
        var=nombreFichero
        #Ahora este nombre lo busco en la ground y saco la clase a la que pertenece
        for line in Gt_val_test:
            clase=line.split()[0]
            lista_ground_truth.append(line.split()[1])
            if var==clase:
                vardef=var
        
    for line1 in file:
            #una vez tengo la clase, comparo cada id del ranking con el elemento de la gt
          #y si tiene esa clase ago append en la lista de aciertos
            ImageID3=line1            
            for line2 in  Gt_val_test:
                if ImageID3==line2.split[0]:
                    if vardef==line2.split[1]:
                        lista_aciertos.append(ImageID3)
                        
            
            
    AP1=average_precision_score(lista_ground_truth,lista_aciertos) 
  ##  MAP=##Suma de 25 average precision(correspondientes a las imagenes x build)   
       
     
    
       
     
        
            
            



