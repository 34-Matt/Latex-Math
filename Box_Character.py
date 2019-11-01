import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import urllib

# Import Photo
#path = r'Z:\caseyduncan\Casey Duncan\CSM Grad School Work\2019\Fall\CSCI 575B - Machine Learning\ML Project\Data\Data - Equations\eqn_test.jpg'
path = r'Z:\caseyduncan\Casey Duncan\CSM Grad School Work\2019\Fall\CSCI 575B - Machine Learning\ML Project\Data\Data - Equations\eqn_test2.jpg'
img = cv2.imread(path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Convert to gray

# Threshold & Morpholigical Close
ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel = np.ones((2,2),np.uint8)
closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

# Find Characters (contours)
contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find position of Bounding Box around each Contours
chars_bb = []
for cnt in contours:
	cnt = cnt.reshape((cnt.shape[0],cnt.shape[2]))
	left_tc = np.amin(cnt, axis=0)
	right_bc = np.amax(cnt, axis=0)
	min_x = left_tc[0]
	max_x = right_bc[0]
	min_y = left_tc[1]
	max_y = right_bc[1]
	chars_bb.append([min_x,min_y,max_x,max_y])

for i in chars_bb:
	print(i)

# Delete near duplicate contours 
# (for example: sometimes the dot on an "i" is thought of as two contours)

# Find characters that are made of multiple contours (for example: "=" or "i")
chars_bb_new = []
for i in range(len(chars_bb)-1):
	cnt_i = chars_bb[i]
	chars_bb_new.append(cnt_i)
	i=0
	for j in range(i+1,len(chars_bb)):
		cnt_j = chars_bb[j]
		cent_i = cnt_i[0]+(cnt_i[2] - cnt_i[0])/2
		cent_j = cnt_j[0]+(cnt_j[2] - cnt_j[0])/2
		if cnt_j == cnt_i:
			pass
		elif abs(cent_i - cent_j) <= 20:
			min_x = min(cnt_j[0],cnt_i[0])
			min_y = min(cnt_j[1],cnt_i[1])
			max_x = max(cnt_j[2],cnt_i[2])
			max_y = max(cnt_j[3],cnt_i[3])
			vals_new = [min_x,min_y,max_x,max_y]
			chars_bb_new.append(vals_new)
			if i == 0:
				chars_bb_new.remove(cnt_i)
				i=i+1
			# Delete near duplicate contours 
			# (for example: sometimes the dot on an "i" is thought of as two contours)
			j=0
			for cnt_new in chars_bb_new:
				cent_new = cnt_new[0]+(cnt_new[2] - cnt_new[0])/2
				if cnt_j == cnt_new:
					pass
				elif (abs(cent_new - cent_j) <= 20) and (abs(cent_new - cent_j) > 0):
					if j == 0:
						chars_bb_new.remove(vals_new)
						j=j+1

# Delete duplicates
chars_bb = []
for i in chars_bb_new:
	if i not in chars_bb:
		chars_bb.append(i)

# Order Characters from left to right
print("Sorted Values:")
chars_bb.sort()
for i in chars_bb:
	print(i)

# Draw bounding box around character
cv2.imshow('image', img)
cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
for cnt in chars_bb:
	min_x = cnt[0]
	max_x = cnt[2]
	min_y = cnt[1]
	max_y = cnt[3]
	cv2.rectangle(img,(min_x,min_y),(max_x,max_y),(0,0,255),1)
	cv2.imshow('image', img)
	cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)