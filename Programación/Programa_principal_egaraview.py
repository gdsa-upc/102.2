import os
from build_database import build_database
from get_features import get_features
from rank import rank
from classify import classify
from evaluate_ranking import evaluate_ranking
from evaluate_classification import evaluate_classification

ruta1=os.path.dirname(os.path.abspath(__file__))+'\\TerrassaBuildings900\\val\\images'
ruta2=os.path.dirname(os.path.abspath(__file__))+'\\TerrassaBuildings900\\train\\images'
savepath1=os.path.dirname(os.path.abspath(__file__))+'\\TerrassaBuildings900\\val'
savepath2=os.path.dirname(os.path.abspath(__file__))+'\\TerrassaBuildings900\\train'

build_database(ruta1,savepath1);
build_database(ruta2,savepath2);

get_features(ruta1,savepath1,savepath1);
get_features(ruta2,savepath2,savepath2);

savepath_principal=os.path.dirname(os.path.abspath(__file__))
features_val=os.path.dirname(os.path.abspath(__file__))+'\\TerrassaBuildings900\\val'
features_train=os.path.dirname(os.path.abspath(__file__))+'\\TerrassaBuildings900\\train'
rank(features_val,features_train,savepath_principal);

feat=os.path.dirname(os.path.abspath(__file__))+'\\TerrassaBuildings900\\val\\Features.txt'
path_out=os.path.dirname(os.path.abspath(__file__))
labels=os.path.dirname(os.path.abspath(__file__))+'\\labels.txt'
classify(feat,path_out,labels)

path=os.path.dirname(os.path.abspath(__file__))
Gt_val_test=os.path.dirname(os.path.abspath(__file__))+'\\TerrassaBuildings900\\val\\annotation.txt'
evaluate_ranking(path,Gt_val_test)

automatic_annot=os.path.dirname(os.path.abspath(__file__))+'\\classify.txt'
annotation=os.path.dirname(os.path.abspath(__file__))+'\\TerrassaBuildings900\\val\\annotation.txt'
evaluate_classification(automatic_annot,annotation)

