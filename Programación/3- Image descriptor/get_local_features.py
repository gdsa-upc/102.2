# -*- coding: utf-8 -*-
from RootSIFT import RootSIFT
import cv2
import numpy as np

def get_local_features(params,image):
    img = cv2.imread(image)
    #Fem un resize de la imatge
    img = resize_image(params,img) #Si volem un tamany específic ho fem aixi: img = cv2.resize(img, (250, 250)) 
    
    # load the image we are going to extract descriptors from and convert
    # it to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # detect Difference of Gaussian keypoints in the image
    detector = cv2.FeatureDetector_create("SIFT")
    kp = detector.detect(gray,None)
 
    # extract and compute RootSIFT descriptors
    extractor = RootSIFT()
    (kp, des) = extractor.compute(gray, kp)
    kp,des= extractor.compute(gray,kp,params['descriptor_size'])
    return des

def resize_image(params,im):
    # Get image dimensions
    height, width = im.shape[:2]

    # If the image width is smaller than the proposed small dimension, keep the original size !
    resize_dim = min(params['max_size'],width)

    # We don't want to lose aspect ratio:
    dim = (resize_dim, height * resize_dim/width)

    # Resize and return new image
    return cv2.resize(im,dim)
