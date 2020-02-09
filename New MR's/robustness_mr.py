#Importing model, training data set and validation data set from the CNN_classifier provided
from cnn_classifier import model, X_train, y_train, X_val, y_val

from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier

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

classifier = KerasClassifier(model=model, clip_values=(1, 100), use_logits=False)

classifier.fit(X_train, y_train, batch_size=64, nb_epochs=3)

predictions = classifier.predict(X_val)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_val, axis=1)) / len(y_val)
print('Accuracy on benign test examples: {}%'.format(accuracy * 100))

#Generate adversarial test examples
attack = FastGradientMethod(classifier=classifier, eps=0.2)
x_test_adv = attack.generate(x=X_val)

#Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_val, axis=1)) / len(y_val)
print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))




