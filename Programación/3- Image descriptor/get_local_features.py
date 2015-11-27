# -*- coding: utf-8 -*-
import cv2

def get_local_features(params,image):
    orb=cv2.ORB()
    #ORB obtiene 500 features de serie 
    #Ahora tenemos que calcular ORB para cada imagen y guardarlo!
    ##leemos
    #img = cv2.imread(os.path.join(params['root'],params['database'],'val','images',str(line).replace('\n','') + '.jpg'))
    img = cv2.imread(image)
    kp=orb.detect(img,None)
    kp,des= orb.compute(img,kp,params['descriptor_size'])
    #guardamos para cada imagen analizada sus descriptores
    return des


    
    


