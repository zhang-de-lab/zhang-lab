# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:28:14 2019
@author: luojiaxiu
"""

import scipy.io as sio
import os
import numpy as np
from keras.models import load_model
from keras.models import model_from_yaml
from keras import backend
backend.set_image_data_format('channels_first')

data1 = sio.loadmat('./t2flair_example.mat')
t2flair_X = (data1['t2flair_example'].astype('float32'))
t2flair_X = (t2flair_X.reshape(t2flair_X.shape[0], 1, 24, 24, 24))

data2 = sio.loadmat('./t1ce_example.mat')
t1ce_X = (data2['t1ce_example'].astype('float32'))
t1ce_X = (t2flair_X.reshape(t2flair_X.shape[0], 1, 24, 24, 24))

with open('./model1.yaml','r') as file1:
    model_yaml1 = file1.read()
model_t2flair = model_from_yaml(model_yaml1)  
model_t2flair.load_weights('./model1.yaml.h5')

with open('./model2.yaml','r') as file2:
    model_yaml2 = file2.read()
model_t1ce = model_from_yaml(model_yaml2) 
model_t1ce.load_weights('./model2.yaml.h5')

t2flair_result=model_t2flair.predict(t2flair_X)
t1ce_result=model_t1ce.predict(t1ce_X)

print('Predicting probability of t2flair example: ', t2flair_result[0][1])
print('Predicting probability of t1ce example: ', t1ce_result[0][1])