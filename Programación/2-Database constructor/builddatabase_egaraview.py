# -*- coding: utf-8 -*-
import os 

def builddatabase_egaraview(dirpath,dirsave):
    ImageIDs = []
    lista = os.listdir(dirpath) 
    
    for file in lista:
        ImageIDs.append(file)
    
    i=len(ImageIDs)
    
    archi=open(dirsave+'\\ImageIDs.txt','w')
    
    for i in list(range(i)):
        if(i<len(ImageIDs)-1):
            archi.write(ImageIDs[i] + ',')
        else:
            archi.write(ImageIDs[i])
    archi.close()
        
    
    
    
    
