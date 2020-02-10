#code referenced from kaggle kernel of gtsrb traffic sign dataset

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPool2D, Input, Softmax, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
import scipy.misc as sc
import cv2
from PIL import Image
from tqdm import tqdm

data=[]
labels=[]

height = 30
width = 30
channels = 3
classes = 43
n_inputs = height * width * channels

#Loading of dataset from the given path
for i in range(classes):
	path = r"C:\Users\Vishal\Documents\BTP\MR codes\traffic\Train"
	images = os.listdir(path+'\\'+str(i))

	for image in images:
		
                img = cv2.imread(path+'\\'+str(i)+'\\'+image)
                #print(img)
                if img is not None:
                        im = Image.fromarray(img)
                        img_arr = np.array(im.resize((height, width), Image.BICUBIC))
                        data.append(img_arr)
                        labels.append(i)
			

Cells=np.asarray(data)
print(Cells.shape)
labels=np.array(labels)

s=np.arange(Cells.shape[0])
np.random.seed(43)
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]

#Data Augmentation Procedure

(X_train,X_val)=Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]
X_train = X_train.astype('float32')/255 
X_val = X_val.astype('float32')/255
(y_train,y_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]

y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)

#Creation of Model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#Compilation of the model
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

#Validating your model against the validation data set
epochs = 1
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,
validation_data=(X_val, y_val))

		
