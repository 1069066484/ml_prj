import tensorflow as tf
import os
from tensorflow.contrib.layers import flatten
import numpy as np
import tensorflow.contrib.slim as slim
from Ldata_helper import *
from Fer2013Dataset_tf import Fer2013Dataset


batch_size = 64


def batches(dataset):
    dataset = Fer2013Dataset()
    data_size = len(dataset.data)
    for i in range(0, data_size, batch_size):
        yield dataset.data[i:max(i+batch_size,data_size)], dataset.labels[i:max(i+batch_size,data_size)]



