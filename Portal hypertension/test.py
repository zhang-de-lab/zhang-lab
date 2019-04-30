# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:22:23 2019

@author: 15129
"""

import numpy as np
import scipy.io as sio
from PIL import Image
from keras import backend
from keras.models import load_model

backend.set_image_data_format('channels_first')

def test():
    patch_liver = Image.open("liver.bmp")
    patch_spleen = Image.open("spleen.bmp")
    
    width,height = patch_liver.size
    patch_liver = patch_liver.getdata()
    patch_liver = np.matrix(patch_liver,dtype='float')/255.0
    patch_liver = np.reshape(patch_liver, (width,height))
    liver_input =np.zeros((1, 1, width, height))
    liver_input[0,0,:,:]=patch_liver

    patch_spleen = patch_spleen.getdata()
    patch_spleen = np.matrix(patch_spleen,dtype='float')/255.0
    patch_spleen = np.reshape(patch_spleen,(width, height))
    patch_spleen = patch_spleen.reshape((1, 1, width,height))
    spleen_input=np.zeros((1, 1, width, height))
    spleen_input[0,0,:,:]=patch_spleen

    model_l = load_model('F:/门脉高压/activate/3/sigmoid/l_cnn_best0.h5')
    model_s = load_model('F:/门脉高压/activate/3/sigmoid/s_cnn_best0.h5')
    
    predict_liver = model_l.predict(liver_input)
    predict_spleen = model_s.predict(spleen_input)
    
    print('The Probability of Portal Hypertension of liver : ', predict_liver[0][1])
    print('The Probability of Portal Hypertension of spleen : ', predict_spleen[0][1])

    
if __name__ == '__main__':
    test()