def get_assignments (descriptores,codebook):
    #Calculem les assignacions
    assignments=codebook.predict(descriptores)
    
    #Retornem el vector amb les assignacions per cada descriptor
    return assignments
