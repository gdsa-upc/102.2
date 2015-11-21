import os
from build_database import build_database

ruta1=os.path.dirname(os.path.abspath(__file__))+'\\TerrassaBuildings900\\val\\images'
ruta2=os.path.dirname(os.path.abspath(__file__))+'\\TerrassaBuildings900\\train\\images'
savepath1=os.path.dirname(os.path.abspath(__file__))+'\\TerrassaBuildings900\\val'
savepath2=os.path.dirname(os.path.abspath(__file__))+'\\TerrassaBuildings900\\train'

build_database(ruta1,savepath1);
build_database(ruta2,savepath2);