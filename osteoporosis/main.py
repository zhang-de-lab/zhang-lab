# coding=utf-8

import os, sys
import numpy as np 

from shili_pre import pre_femoral, pre_lumoral

if __name__ == '__main__':
	path1 = 'example1_for_femoral.mat'
	pre_femoral(path1)
	path2 = 'example1_for_lumoral.mat'
	pre_lumoral(path2)