# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

#Leemos la imagen de entrada a analizar
img=cv2.imread('teatre2.jpg',1)
#Habilitamos el módulo que nos permitirá utilizar el detector
orb=cv2.ORB()
#Ejecutamos la función, insertandole la imagen deseada y almacenamos los keyp.
kp=orb.detect(img,None)
#Los computamos y los ubicamos sobre la imagen
kp, des= orb.compute(img,kp)
#Dibujamos esos puntos en una imagen de salida llamada imagen 2 que será la suma
#de los puntos detectados más la imagen de entrada.
img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
#Mostramos por pantalla
cv2.namedWindow('image2')
cv2.imshow('image2',img2)
# Este algoritmo ha sido seleccionado por sus resultados computacionales, 
#analizado previamente en una comparativa entre detectores, actua similar a detectores
#del tipo SURF pero con la ventaja de que la patente es libre, por lo tanto,
#puede ser utilizado sin pagar por el en futuras aplicaciones.