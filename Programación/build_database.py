def build_database(string dirtrain, string dirval, string dirsave):
import os
#Ruta Personal
ruta=os.path.dirname(os.path.abspath(__file__))
#Path de les Imatges
path = ruta + "\\TerrassaBuildings900\\train\\images"
#Declarem la llista d'ImageIDs
ImageIDs = []
#os.walk()Retorna una llista directoris y fitxers al path introduit
lstDir = os.walk(path)   
 
#Crea una llista amb els fitxers del directori i els inclou a la llista 
for root, dirs, files in lstDir:
    for fichero in files:
        (nombreFichero, extension) = os.path.splitext(fichero)
        ImageIDs.append(nombreFichero)
            

archi=open(ruta+'\\ImageIDs.txt','w')
i=len(ImageIDs)
for i in list(range(i)):
    if(i<len(ImageIDs)-1):
        archi.write(ImageIDs[i] + ',')
    else:
        archi.write(ImageIDs[i])
archi.close()