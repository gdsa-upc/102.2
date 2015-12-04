from sklearn.cluster import MiniBatchKMeans

def train_codebook(params,descriptores,paraules):
    km = MiniBatchKMeans(paraules)
    #Entrenem el KMeans, ajustem el número de clusters a 4
    km.fit(descriptores)
    return km
