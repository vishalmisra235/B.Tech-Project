import sklearn
import os
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import svm, metrics
import cv2
import scipy.misc

categories=['NEUTROPHIL','EOSINOPHIL','MONOCYTE','LYMPHOCYTE','BASOPHIL']

#loading of image dataset
from tqdm import tqdm
def get_data(folder):

	X = []
	y = []
	z = []
	for wbc_type in os.listdir(folder):
		if not wbc_type.startswith('.'):
            		if wbc_type in ['NEUTROPHIL']:
                		label = 1
                		label2 = 1
            		elif wbc_type in ['EOSINOPHIL']:
                		label = 2
                		label2 = 1
            		elif wbc_type in ['MONOCYTE']:
                		label = 3  
                		label2 = 0
            		elif wbc_type in ['LYMPHOCYTE']:
                		label = 4 
                		label2 = 0
            		else:
                		label = 5
                		label2 = 0

		for image_filename in tqdm(os.listdir(folder + wbc_type)):
			img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
			if img_file is not None:
				img_file = scipy.misc.imresize(arr=img_file, size=(60, 80, 3))
				img_arr = np.asarray(img_file)
				X.append(img_arr)
				y.append(label)
				z.append(label2)
	X = np.asarray(X)
	y = np.asarray(y)
	z = np.asarray(z)
	return X,y,z

X_train, y_train, z_train = get_data('/home/razorback/BTP/metamorphic_testing/blood-cells-dataset/blood-cells/dataset2-master/dataset2-master/images/TRAIN/')
X_test, y_test, z_test = get_data('/home/razorback/BTP/metamorphic_testing/blood-cells-dataset/blood-cells/dataset2-master/dataset2-master/images/TEST/')

#dimensionality reduction of the image dataset
print(X_train.shape)
print(y_train.shape)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[3]*X_train.shape[2]*X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3])

print(len(X_train))
print(len(y_train))
print(X_train.shape)
print(y_train.shape)

print(len(X_test))
print(len(y_test))

#training SVM classifier on the dataset
classifier = svm.SVC(gamma=0.001)

print('Vishal')
classifier.fit(X_train, y_train)
print('Vishal')

y_pred = classifier.predict(X_test)

print("Classification report for classifier %s:\n%s\n"% (classifier, metrics.classification_report(y_test, y_pred)))

