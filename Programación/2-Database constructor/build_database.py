# -*- coding: utf-8 -*-
#Definim la funció build_database
def build_database(dirpath,dirsave):

    #Declarem la llista d'ImageIDs
    ImageIDs = []
    #os.walk()Retorna una llista directoris y fitxers al path introduit
    lstDir = os.walk(dirpath)   
 
    #Crea una llista amb els fitxers del directori i els inclou a la llista 
    for root, dirs, files in lstDir:
        for fichero in files:
            (nombreFichero, extension) = os.path.splitext(fichero)
            ImageIDs.append(nombreFichero)
    #Crea el fitxer .txt amb la ID de les imatges del directori
    archi=open(dirsave+'\\ImageIDs.txt','w')
    i=len(ImageIDs)
    for i in list(range(i)):
        if(i<len(ImageIDs)-1):
            archi.write(ImageIDs[i] + ',')
        else:
            archi.write(ImageIDs[i])
    archi.close()