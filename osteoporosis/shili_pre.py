# coding=utf-8

import os, sys
import scipy.io as sio 
import numpy as np 
from keras.models import model_from_json

def loadmodels():

	modelC1 = model_from_json(open('models/modelYao.json').read())
	modelC1.load_weights('models/modelYaoC.h5')
	modelC2 = model_from_json(open('models/modelYao.json').read())
	modelC2.load_weights('models/modelYaoC-1.h5')

	modelZ1 = model_from_json(open('models/modelYao.json').read())
	modelZ1.load_weights('models/modelYaoZ.h5')
	modelZ2 = model_from_json(open('models/modelYao.json').read())
	modelZ2.load_weights('models/modelYaoZ-1.h5')

	return modelC1, modelC2, modelZ1, modelZ2


# modelC1, modelC2, modelZ1, modelZ2 = loadmodels()

def pre_one(pre_batch, model):
	out = model.predict_on_batch(pre_batch)
	return np.mean(out)

def pre_lumoral(path):
	data = sio.loadmat('mat/'+path)
	x1C = data['dataL1C'].astype('float32')
	x1C /= 255. 
	x1Z = data['dataL1Z'].astype('float32')
	x1Z /= 255.
	x2C = data['dataL2C'].astype('float32')
	x2C /= 255.
	x2Z = data['dataL2Z'].astype('float32')
	x2Z /= 255.
	x3C = data['dataL3C'].astype('float32')
	x3C /= 255.
	x3Z = data['dataL3Z'].astype('float32')
	x3Z /= 255.
	x4C = data['dataL4C'].astype('float32')
	x4C /= 255.
	x4Z = data['dataL4Z'].astype('float32')
	x4Z /= 255.
	y = data['label']
	modelC1, modelC2, modelZ1, modelZ2 = loadmodels()
	num11 = pre_one(x1C, modelC1)
	num12 = pre_one(x2C, modelC1)
	num13 = pre_one(x3C, modelC1)
	num14 = pre_one(x4C, modelC1)
	num21 = pre_one(x1Z, modelZ1)
	num22 = pre_one(x2Z, modelZ1)
	num23 = pre_one(x3Z, modelZ1)
	num24 = pre_one(x4Z, modelZ1)
	num = np.mean([num11, num21, num12, num22, num13, num23, num14, num24])
	if num>=0.5:
		print('The predicted results are less than -2.5, The corresponding label is 0')
		print('The probability that the predicted result is less than -2.5 is ', num, '  label is：', y[0][0], '(0 means that T value is less than - 2.5, 1 means that T value is greater than - 2.5 is less than - 1, and 2 means that T value is greater than - 1)')
		return
	if num<0.5:
		num11 = pre_one(x1C, modelC2)
		num12 = pre_one(x2C, modelC2)
		num13 = pre_one(x3C, modelC2)
		num14 = pre_one(x4C, modelC2)
		num21 = pre_one(x1Z, modelZ2)
		num22 = pre_one(x2Z, modelZ2)
		num23 = pre_one(x3Z, modelZ2)
		num24 = pre_one(x4Z, modelZ2)
		num2 = np.mean([num11, num21, num12, num22, num13, num23, num14, num24])
		if num2>=0.5:
			print('The predicted result is greater than - 1, and the corresponding label is 2.')
		else:
			print('The predicted results are between - 2.5 and - 1, and the corresponding label is 1.')
		print('The probability of the predicted results being greater than - 2.5 and greater than - 1 is respectively ', num, ',' , num2, '  label is', y[0][0], '(0 means that T value is less than - 2.5, 1 means that T value is greater than - 2.5 is less than - 1, and 2 means that T value is greater than - 1)')
		return
