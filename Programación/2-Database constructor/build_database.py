# -*- coding: utf-8 -*-
import os 

#Definim la funció build_database
def build_database(dirpath,dirsave):
    #Declarem la llista d'ImageIDs
    ImageIDs = []
    #Retorna una llista dels arxius al path introduit
    lista = os.walk(dirpath) 
    #Obrim el fitxer de la ground truth
    gr_truth = open(dirsave+'\\annotation.txt','r')    
    #Aquest readline() és per saltar-nos els titols "ClassID i ImageID" 
    gr_truth.readline()    
    #Declarem la llista que portarà les ID de la classe desconegut per poder descartar-les
    lista_ground_truth=[]
    for line in gr_truth:
        ImageID=line.split()[0]
        ClassID=line.split()[1]
        #Segons el .txt entrant, el primer paràmetre que trobem a cada linia és la ImageID i el segon la ClassID
        if ClassID=='desconegut':
            lista_ground_truth.append(ImageID)
            #Si la ClasseID és desconegut, posem la ImageID a la llista
    #Bucle per omplir la llista amb els nombres ID de les imatges sense la
    #extensió
    for root, dirs, files in lista:
        for fichero in files:
            (nombreFichero, extension) = os.path.splitext(fichero)
            if(nombreFichero in lista_ground_truth):
                continue
            else:
                ImageIDs.append(nombreFichero)
    #Tanquem la llista ground_truth
    gr_truth.close()
    #Guardem la longitud de ImageIDs en la variable i
    i=len(ImageIDs)
    #Creem el .txt que contindrà IDs de les imatges
    archi=open(dirsave+'\\ImageIDs.txt','w')
    #Bucle per omplir el .txt amb les IDs de les imatges
    for i in list(range(i)):
        if(i<len(ImageIDs)-1):
            archi.write(ImageIDs[i] + '\n')
        else:
            archi.write(ImageIDs[i])
    #Tanquem l'arxiu .txt 
    archi.close()
