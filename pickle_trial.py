import cv2
import pickle

with open('X_Y_Data.pickle', 'rb') as f:
	X, Y = pickle.load(f)

for i in range(0,len(X)):
	img = X[i]
	print(Y[i])
	cv2.imshow('image', img)
	cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)