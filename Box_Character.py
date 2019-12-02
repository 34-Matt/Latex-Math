import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import urllib

def Box_Character(img):
	# Import Photo
	# path = r'Z:\caseyduncan\Casey Duncan\CSM Grad School Work\2019\Fall\CSCI 575B - Machine Learning\ML Project\Data\Data - Equations\eqn_test3.jpg'
	# img = cv2.imread(path)
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

	# Find characters that are made of multiple contours (for example: "=" or "i")
	chars_bb_new = chars_bb.copy()
	for i in range(len(chars_bb)-1):
		cnt_i = chars_bb[i]
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
	chars_bb_new2 = chars_bb_new.copy()
	for i in range(len(chars_bb_new)-1):
		cnt_i = chars_bb_new[i]
		for j in range(i+1,len(chars_bb_new)):
			cnt_j = chars_bb_new[j]
			cent_i = cnt_i[0]+(cnt_i[2] - cnt_i[0])/2
			cent_j = cnt_j[0]+(cnt_j[2] - cnt_j[0])/2
			area_i = (cnt_i[2] - cnt_i[0])*(cnt_i[3] - cnt_i[1])
			area_j = (cnt_j[2] - cnt_j[0])*(cnt_j[3] - cnt_j[1])
			if cnt_j == cnt_i:
				pass
			elif (abs(cent_i - cent_j) <= 20):
				if area_i > area_j:
					if cnt_j in chars_bb_new2:
						chars_bb_new2.remove(cnt_j)
				elif area_i < area_j:
					if cnt_i in chars_bb_new2:
						chars_bb_new2.remove(cnt_i)

	# Delete duplicates contours
	chars_bb = []
	for i in chars_bb_new2:
		if i not in chars_bb:
			chars_bb.append(i)

	# Order Characters from left to right
	chars_bb.sort()

	# Draw bounding box around character
	#cv2.imshow('image', img)
	#cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
	# for cnt in chars_bb:
	# 	min_x = cnt[0]
	# 	max_x = cnt[2]
	# 	min_y = cnt[1]
	# 	max_y = cnt[3]
	# 	cv2.rectangle(img,(min_x,min_y),(max_x,max_y),(0,0,255),1)
		# cv2.imshow('image', img)
		# cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

	# Save each character as its own image in a vector of images
	X_input = np.empty((0,45,45),dtype=np.float16)
	for cnt in chars_bb:
		size_x = 45
		size_y = 45
		pad = 3
		min_x = cnt[0]-pad
		max_x = cnt[2]+pad
		min_y = cnt[1]-pad
		max_y = cnt[3]+pad
		if (max_x - min_x) > size_x:
			size_x = max_x - min_x
		if (max_y - min_y) > size_y:
			size_y = max_y - min_y

		img_i = np.zeros((size_y,size_x), np.uint8)
		img_i[:,:] = 255

		start_x = int(0.5*(size_x-(max_x - min_x)))
		end_x = start_x + (max_x - min_x)
		start_y = int(0.5*(size_y-(max_y - min_y)))
		end_y = start_y + (max_y - min_y)

		img_i[start_y:end_y,start_x:end_x] = gray[min_y:max_y,min_x:max_x]
		img_i = cv2.resize(img_i,(45,45))

		X_input = np.append(X_input, [img_i],axis = 0) #might need to change

	#for x in X_input:
		#cv2.imshow('image', x)
		#cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

	return X_input
