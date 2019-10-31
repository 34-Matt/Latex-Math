import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import urllib

path = r'Z:\caseyduncan\Casey Duncan\CSM Grad School Work\2019\Fall\CSCI 575B - Machine Learning\ML Project\Data\Data - Single Characters\C\c_1243.jpg'
img = cv2.imread(path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel = np.ones((2,2),np.uint8)
closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
#cv2.drawContours(img, [cnt], 0, (0,255,0), 2)
cnt = cnt.reshape((62,2))
left_tc = np.amin(cnt, axis=0)
right_bc = np.amax(cnt, axis=0)
min_x = left_tc[0]
max_x = right_bc[0]
min_y = left_tc[1]
max_y = right_bc[1]
cv2.rectangle(img,(min_x,min_y),(max_x,max_y),(0,0,255),1)

cv2.imshow('image', img)
cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)