# -*- coding: utf-8 -*-
import cv2

def get_local_features(params,image):
    orb=cv2.ORB()
    #ORB obté 500 características de serie
    #Calculem els ORB per a la imatge
    img = cv2.imread(image)
    #Guardem els Keypoints
    kp=orb.detect(img,None)
    #Guardem els Keypoints i els descriptors per la imatge
    kp,des= orb.compute(img,kp,params['descriptor_size'])
    #Retornem els descriptors de la imatge
    return des


    
    


