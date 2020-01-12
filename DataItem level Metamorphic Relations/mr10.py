'''
Metamorphic Relation 10:
Cropping the images and checking the accuracy 
'''

#Importing model, training data set and validation data set from the CNN_classifier provided
from cnn_classifier import model, X_train, y_train, X_val, y_val

import os
import cv2
import scipy.misc
import numpy as np
import json

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPool2D, Input, Softmax, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D, Cropping2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json

#Importing dataset from the path
path = '/home/razorback/BTP/metamorphic_testing/meta_relations/data/TRAIN/'

num_classes = len(os.listdir(path))
data = []
labels = []
model_shape = model.layers[0].input_shape
print(model_shape)

for i in range(num_classes):
	new_path = path+str(i)
	images = next(os.walk(new_path))[2]
	num_images = len(images)

	for image in images:
		img = cv2.imread(new_path+'/'+image)
		if img is not None:
			if type(model_shape) is list:
				img_file = scipy.misc.imresize(arr=img, size=model_shape[0][1:])
			else:
				img_file = scipy.misc.imresize(arr=img, size=model_shape[1:])
			img_arr = np.asarray(img_file)
			data.append(img_arr)
			labels.append(i)


Cells=np.asarray(data)
labels=np.array(labels)


(Xnew_train,Xnew_val)=Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]
Xnew_train = Xnew_train.astype('float32')/255 
Xnew_val = Xnew_val.astype('float32')/255
(ynew_train,ynew_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]

ynew_train = to_categorical(ynew_train, num_classes)
ynew_val = to_categorical(ynew_val, num_classes)

epochs = 1

#new standard pipeline model
model_new = Sequential([
	    Cropping2D(cropping=((2, 2), (4, 4)),input_shape=model_shape[1:]),
	    Conv2D(16, 3, padding='same', activation='relu'),
	    MaxPooling2D(),
	    Dropout(0.2),
	    Conv2D(32, 3, padding='same', activation='relu'),
	    MaxPooling2D(),
	    Conv2D(64, 3, padding='same', activation='relu'),
	    MaxPooling2D(),
	    Dropout(0.2),
	    Flatten(),
	    Dense(512, activation='relu'),
	    Dense(num_classes, activation='softmax')
	])

model_new.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model_new.fit(Xnew_train, ynew_train, batch_size=32, epochs=epochs, validation_data=(Xnew_val, ynew_val))
