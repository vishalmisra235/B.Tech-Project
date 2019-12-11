'''
Metamorphic Relation 5:
Creating a new category of class by duplicating any random class and comparing the accuracy 
'''

#Importing model, training data set and validation data set from the CNN_classifier provided
from cnn_classifier import model, X_train, y_train, X_val, y_val

import os
import cv2
import scipy.misc
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
from shutil import copyfile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPool2D, Input, Softmax, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json

#Importing dataset from the path
path = '/home/razorback/BTP/metamorphic_testing/meta_relations/data/TRAIN/'

num_classes = len(os.listdir(path))
data = []
labels = []
model_shape = model.layers[0].input_shape

#We randomly chose any class and create a new category and add it in our dataset
dir1 = random.randint(0,num_classes-1)
print(dir1)
path1 = path + str(dir1)
images1 = next(os.walk(path1))[2]
file_path = path + str(num_classes)+'/'+images1[0]
directory = os.path.dirname(file_path)
if not os.path.exists(directory):
	print(directory)
	os.mkdir(directory)

images1 = next(os.walk(path1))[2]
for image in images1:
	img = path1+'/'+image
	copyfile(img,path+str(num_classes)+'/'+image)

for i in range(num_classes+1):
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

ynew_train = to_categorical(ynew_train, num_classes+1)
ynew_val = to_categorical(ynew_val, num_classes+1)

epochs = 1

history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_val, y_val))

base_output = model.layers[-1].output
new_output = Dense(num_classes+1, activation="softmax")(base_output)
model2 = Model(inputs=model.inputs, outputs=new_output)

model2.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model2.summary()

history1 = model2.fit(Xnew_train, ynew_train, batch_size=32, epochs=epochs, validation_data=(Xnew_val, ynew_val))

