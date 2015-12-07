def get_assignments (descriptores,codebook):
    
    #Calculem les assignacions mitjançant la funció predict( )
    assignments=codebook.predict(descriptores)
    #Retornem el vector amb les assignacions per cada descriptor
    return assignments
