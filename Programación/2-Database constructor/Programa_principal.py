import os
from builddatabase_egaraview import builddatabase_egaraview

ruta= os.path.dirname(os.path.abspath(__file__))+'\TerrassaBuildings900\\val\images'
savepath= os.path.dirname(os.path.abspath(__file__));

builddatabase_egaraview(ruta,savepath);