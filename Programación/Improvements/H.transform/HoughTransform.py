import cv2
import numpy as np

img = cv2.imread('38-29833-926.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 1000
maxLineGap = 1
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('C:\Users\Gerard\Documents\Universidad\TercerCurso\GDSA\pruebas\houghlines5.jpg',img)