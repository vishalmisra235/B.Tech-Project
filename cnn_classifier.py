import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPool2D, Input, Softmax, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import concatenate,AveragePooling2D
import os
import cv2
import scipy.misc as sc
import matplotlib.pyplot as plt


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
                img_file = sc.imresize(arr=img_file, size=(60, 80, 3))
                img_arr = np.asarray(img_file)
                X.append(img_arr)
                y.append(label)
                z.append(label2)
    X = np.asarray(X)
    y = np.asarray(y)
    z = np.asarray(z)
    return X,y,z


X_train, y_train, z_train = get_data(r'C:\Users\Vishal\Documents\BTP\dataset2-master\dataset2-master\images\TRAIN\\')
X_test, y_test, z_test = get_data(r'C:\Users\Vishal\Documents\BTP\dataset2-master\dataset2-master\images\TEST\\')
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from tensorflow.keras.utils import to_categorical
y_trainHot = to_categorical(y_train, num_classes = 5)
y_testHot = to_categorical(y_test, num_classes = 5)
z_trainHot = to_categorical(z_train, num_classes = 2)
z_testHot = to_categorical(z_test, num_classes = 2)
dict_characters = {1:'NEUTROPHIL',2:'EOSINOPHIL',3:'MONOCYTE',4:'LYMPHOCYTE'}
dict_characters2 = {0:'Mononuclear',1:'Polynuclear'}
print(dict_characters)
print(dict_characters2)
print("Train X Shape --> ",X_train.shape)
print("Train y Shape --> ",y_trainHot.shape)
print("Train z Shape --> ",z_trainHot.shape)
##
# Input Layer (-1, 60, 80, 3) All three channel RGB
# Output Layer 1 (-1, 5) Softmax
# Output Layer 2 (-1, 2) Softmax (Doesn't work as 2nd output backpropogation messes all the weights)
##


def keras_model():
    inp = Input(shape=(60,80,3))
    x = Conv2D(32, (11,11), padding="same",activation="relu")(inp)
    x = Conv2D(32, (7,7), padding="valid",activation="relu")(inp)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(32, (5, 5), padding="same",activation="relu")(x)
    x = Conv2D(32, (5, 5), padding="valid",activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding="same",activation="relu")(x)
    x = Conv2D(64, (3, 3), padding="valid",activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    print(x.shape)
    print('Vishal')
    x = Dense(1024,activation="relu")(x)
    x = Dropout(0.5)(x)
    y = Dense(5,activation="softmax")(x)
    #z = Dense(2,activation="softmax")(x)
    model = Model(inp, y)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model


model = keras_model()
model.summary()

history = model.fit(X_train,
         y_trainHot,
         epochs = 1,
         batch_size = 512,
         validation_data = (X_test,y_testHot),
         verbose = 1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
