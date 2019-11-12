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

path = '/home/razorback/BTP/metamorphic_testing/meta_relations/data/train/'

num_classes = len(os.listdir(path))
data = []
labels = []

for i in range(num_classes):
	new_path = path+str(i)
	images = next(os.walk(new_path))[2]
	num_images = len(images)
	count = 0

	for image in images:
		img = cv2.imread(new_path+'/'+image)
		if img is not None and count< 0.9 * num_images:
			img_file = scipy.misc.imresize(arr=img, size=(30, 30, 3))
			img_arr = np.asarray(img_file)
			data.append(img_arr)
			labels.append(i)
			count += 1

Cells=np.asarray(data)
labels=np.array(labels)

s=np.arange(Cells.shape[0])
np.random.seed(43)
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]


(Xnew_train,Xnew_val)=Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]
Xnew_train = Xnew_train.astype('float32')/255 
Xnew_val = Xnew_val.astype('float32')/255
(ynew_train,ynew_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]

ynew_train = to_categorical(ynew_train, num_classes)
ynew_val = to_categorical(ynew_val, num_classes)

epochs = 1

history1 = model.fit(Xnew_train, ynew_train, batch_size=32, epochs=epochs,
validation_data=(Xnew_val, ynew_val))

history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,
validation_data=(X_val, y_val))


