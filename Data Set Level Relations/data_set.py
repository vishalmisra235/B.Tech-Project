import re
import os
import importlib.util

classifier_path = input()

assert os.path.exists(classifier_path), "Error! No file exists at, "+str(classifier_path)

dataset_path = input()

assert os.path.exists(dataset_path), "Error! No file exists at, "+str(dataset_path)



