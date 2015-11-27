from scipy.cluster.vq import vq
from sklearn import preprocessing

def get_assignments (descriptores,codebook):
    #Normalitzem els valors
    des=preprocessing.normalize(descriptores)
    #Calculem les assignacions i omplim els parametres (code i dist)
    [code,dist]=vq(des,codebook)
    
    #Retornem el vector amb les assignacions per cada descriptor
    return code
