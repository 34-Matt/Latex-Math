import os
import numpy as np
import imageio
import csv
from sklearn.model_selection import train_test_split

def createDict(images_path):
	#images_path = './extracted_images/' 
	dirlist = os.listdir(images_path)

	single = []
	multiple = []

	for item in dirlist:
		item.lower() #make everything lowercase
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

def loadDict(file_name):
	dict = {}
	with open(file_name) as file:
		readCSV = csv.reader(file)
		for row in readCSV:
			dict[row[0]] = int(row[1])
	return dict

def loadDataset(file_name1,file_name2,rate = 0.2): #file_name1 location of all characters, file_name2 dict
	dict = loadDict(file_name2)
	ds1 = os.listdir(file_name1)
	counter = 0
	X = np.empty((0,45,45))
	for d in ds1:
		folder = os.path.join(file_name1,d)
		ds2 = os.listdir(folder)
		d.lower()
		for d2 in ds2:
			filei = os.path.join(folder,d2)
			image = imageio.imread(filei)
			npi = np.asarray(image).reshape(45,45) #might need to change
			X = np.append(X, npi,axis = 0) #might need to change 
			Y[counter] = dict[d]
			counter += 1
	x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = rate)	
	return x_train, x_test, y_train, y_test 

if __name__ == '__main__':
	createDict('./extracted_images/')
	dict = loadDict('LabelDict.csv')
	for key,val in dict.items():
		print("{} : {}".format(key,val))
