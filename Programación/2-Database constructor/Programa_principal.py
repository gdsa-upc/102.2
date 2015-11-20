import os
from build_database import build_database
ruta=os.path.dirname(os.path.abspath(__file__))
path1 = ruta + '\\TerrassaBuildings900\\train\\images'
path2 = ruta + '\\TerrassaBuildings900\\valid\\images'
savepath = ruta + '\\Build_database'
build_database(path1, savepath)
build_database(path2, savepath)