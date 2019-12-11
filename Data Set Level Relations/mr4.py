'''
Metamorphic Relation 4:
Add 10% of images into each category of validation data set and then checking the classification accuracy
'''

#Importing model, training data set and validation data set from the CNN_classifier provided
from cnn_classifier import model, X_train, y_train, X_val, y_val
import os
import cv2
import scipy.misc
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPool2D, Input, Softmax, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical

#Importing dataset from the path
path = '/home/razorback/BTP/metamorphic_testing/meta_relations/data/TRAIN/'

num_classes = len(os.listdir(path))
data = []
labels = []
model_shape = model.layers[0].input_shape

#We randomly chose values from each class in the training data set and add it to the validation data set
for i in range(num_classes):
	new_path = path+str(i)
	images = next(os.walk(new_path))[2]
	num_images = len(images)
	count = 0

	for image in images:
		img = cv2.imread(new_path+'/'+image)
		if img is not None:
			if count<0.1*num_images:
				if type(model_shape) is list:
					img_file = scipy.misc.imresize(arr=img, size=model_shape[0][1:])
				else:
					img_file = scipy.misc.imresize(arr=img, size=model_shape[1:])
				img_arr = np.asarray(img_file)
				data.append(img_arr)
				labels.append(i)
			

Xnew_train=np.asarray(data)
ynew_train=np.array(labels)
ynew_train = to_categorical(ynew_train, num_classes)
Xnew_val = np.concatenate((Xnew_train,X_train), axis=0)
ynew_val = np.concatenate((ynew_train,y_train),axis=0)
epochs = 1

#Comparing accuracy of the model on two different datasets
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,
validation_data=(X_val, y_val))

history1 = model.fit(Xnew_train, ynew_train, batch_size=32, epochs=epochs,
validation_data=(Xnew_val, ynew_val))
