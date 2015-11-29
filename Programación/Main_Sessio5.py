from get_params import get_params
from build_database import build_database
from get_features import get_features
from rank import rank

#Extraccio dels parametres
params=get_params()
params['split']='val'
#Creació de la base de dades
build_database(params)
params['split']='train'
build_database(params)
#Extracció de les características
get_features(params)
#Calcul del ranquing
rank(params)


