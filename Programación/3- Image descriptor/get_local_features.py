# -*- coding: utf-8 -*-
from RootSIFT import RootSIFT
import cv2
import numpy as np

def get_local_features(params,image):
    #orb=cv2.ORB(nfeatures=2000)
    #ORB obté 500 características de serie
    #Calculem els ORB per a la imatge
    img = cv2.imread(image)
    #Fem un resize de la imatge a la meitat de la original
    #img=cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    img = cv2.resize(img, (500, 500))
    #Si volem un tamany específic ho fem aixi: img = cv2.resize(img, (250, 250)) 
    #Guardem els Keypoints
    #kp=orb.detect(img,None)
    #Guardem els Keypoints i els descriptors per la imatge
    #kp,des= orb.compute(img,kp,params['descriptor_size'])
    #Retornem els descriptors de la imatge
    #return des
 
    # load the image we are going to extract descriptors from and convert
    # it to grayscale
    ###img = cv2.imread(image)
    ###img=cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # detect Difference of Gaussian keypoints in the image
    detector = cv2.FeatureDetector_create("SIFT")
    kp = detector.detect(gray)
 
    # extract normal SIFT descriptors
    extractor = cv2.DescriptorExtractor_create("SIFT")
    (kp, des) = extractor.compute(gray, kp)
    #print "SIFT: kps=%d, descriptors=%s " % (len(kps), descs1.shape)
 
    # extract RootSIFT descriptors
    ###surf = cv2.SURF()
    ###kp, des = surf.compute(gray,kp,params['descriptor_size'])
    rs = RootSIFT()
    kp,des= rs.compute(gray,kp,params['descriptor_size'])
    #print "RootSIFT: kps=%d, descriptors=%s " % (len(kp), des.shape)
    #descs=np.concatenate((descs1,descs2))
    return des