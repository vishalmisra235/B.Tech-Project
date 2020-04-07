from data_set import classifier_path, dataset_path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from scipy.misc import imread
from glob import iglob

images = pd.DataFrame([])
no_classes = 43
for i in range(no_classes):
    path = dataset_path + '/' + str(i)


