import os
import numpy as np
import imageio
import csv
import sys
from sklearn.model_selection import train_test_split
import cv2
import pickle

def createDict(images_path):
	#images_path = './extracted_images/' 
	dirlist = os.listdir(images_path)

	single = []
	multiple = []

	for item in dirlist:
		item = item.lower() #make everything lowercase
		if len(item) == 1:
			single.append(item)
		else:
			multiple.append(item)

	multiple.sort() #alphabetical order

	#single_ascii = []

	#for item in single:
	#	single_ascii.append(ord(item)) #converts strings to ascii equivalent

	#single_ascii.sort() #ascii numerical order
	single.sort() #ascii numerical order

	dict = {}
	counter = 0

	for item in multiple:
		dict[item] = counter
		counter += 1
	for item in single:
		dict[item] = counter
		counter += 1

	#writing to an Excel file
	file = open("LabelDict.csv","w")
	w = csv.writer(file)

	for key, val in dict.items():
		w.writerow([key,val])

	file.close()

def loadDict_AB(file_name):
	dict = {}
	with open(file_name) as file:
		readCSV = csv.reader(file)
		for row in readCSV:
			if len(row) > 0:
				dict[row[0]] = int(row[1])
	return dict
    
def loadDict_BA(file_name):
	dict = {}
	with open(file_name) as file:
		readCSV = csv.reader(file)
		for row in readCSV:
			if len(row) > 0:
				dict[row[1]] = int(row[0])
	return dict

def loadDataset(file_name1,file_name2,rate = 0.2): #file_name1 location of all characters, file_name2 dict
	dict = loadDict(file_name2)
	ds1 = os.listdir(file_name1)
	file_count = sum([len(files) for r, d, files in os.walk(file_name1)])
	counter = 0
	X = np.empty((0,45,45),dtype=np.uint8)
	Y = np.empty((0,1),dtype=np.uint8) 
	for d in ds1:
		folder = os.path.join(file_name1,d)
		ds2 = os.listdir(folder)
		d = d.lower()
		for d2 in ds2:
			filei = os.path.join(folder,d2)
			image = cv2.imread(filei)
			image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # Convert to gray
			npi = np.asarray(image).reshape(45,45) #might need to change
			X = np.append(X, [npi],axis = 0) #might need to change
			Y = np.append(Y,dict[d])
			counter += 1
			output_string = f"Image File {counter} of {file_count}\n"
			sys.stdout.write(output_string)
			sys.stdout.flush()
	#x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = rate)	
	return X, Y

if __name__ == '__main__':
	path = 'C:/Users/cdunc/Documents/CSM Grad School Work/2019/Fall/CSCI 575B - Machine Learning/Group Project/Data/Single Characters/Removed Duplicates & Symbols'
	createDict(path)
	
	dict_name = 'LabelDict.csv'
	dict = loadDict(dict_name)
	#for key,val in dict.items():
	#	print("{} : {}".format(key,val))

	#x_train, x_test, y_train, y_test = loadDataset(path,dict_name,rate = 0.2)
	X, Y = loadDataset(path,dict_name,rate = 0.2)
	with open('X_Y_Data.pickle', 'wb') as f:
		pickle.dump([X, Y], f)