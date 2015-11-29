from get_params import get_params
from build_database import build_database
from get_features import get_features
from rank import rank

#Extraccio dels parametres
params=get_params()
params['split']='val'
build_database(params)
params['split']='train'
build_database(params)
get_features(params)
rank(params)


