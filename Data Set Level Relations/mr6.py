'''
Metamorphic Relation 5:
Remove one category from the training dataset and then check the classification accuracy
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
from tensorflow.keras.models import model_from_json

#Importing dataset from the path
path = '/home/razorback/BTP/metamorphic_testing/meta_relations/data/TRAIN/'

num_classes = len(os.listdir(path))
data = []
labels = []
model_shape = model.layers[0].input_shape

#We manipulate the model by removing last layer from the model and adding our own layer to form a new model by mantaining the balance between the input and output tensors values.

for i in range(num_classes-1):
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

ynew_train = to_categorical(ynew_train, num_classes-1)
ynew_val = to_categorical(ynew_val, num_classes-1)

epochs = 1

#Comparing accuracy of the model on two different datasets
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.summary()

#Formulating new model from old model
base_output = model.layers[-1].output
new_output = Dense(num_classes-1, activation="softmax")(base_output)
model2 = Model(inputs=model.inputs, outputs=new_output)

model2.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

history1 = model2.fit(Xnew_train, ynew_train, batch_size=32, epochs=epochs, validation_data=(Xnew_val, ynew_val))

