# -*-coding:utf-8-*-
#!/usr/bin/env python


'''
__author__ = "kimyoonyoung"
'''
import sys

img_cls_path = './models/research/slim'
sys.path.append(img_cls_path)

from datasets import convert_nuclei
import sklearn
#import pandas as pd
from ast import literal_eval

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedShuffleSplit
import time
from collections import Counter
import sys
import tensorflow as tf
import os

slim = tf.contrib.slim



# tf.app.flags.DEFINE_string(
#     'trainval_list', None , 'use pre-defined train and valiation list. \n'
#     '(--trainval_list:/path/to/train_val file) \n'
# )


FLAGS = tf.app.flags.FLAGS
#cross-validation?
train_data_ratio = 0.9

#make_model = True  # if false, ma
trainval_path = './data/1-nuclei/'
train_list = trainval_path + 'train_w32_1.txt'
val_list = trainval_path + 'test_w32_1.txt'

SEED = 30 # reproduce same train and valation datasets



print('train imag data is preparing for image classifcation... ')



if train_list and val_list:
     with open(train_list, "r") as trf, open(val_list, "r") as valf:
        train_list = trf.readlines()
        val_list = valf.readlines()
else:
    # sample test data as training_ratio per category to reduce bias
    split = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_data_ratio, random_state=SEED)
    for train_index, val_index in split.split(data, data['cate']):
        train_data = data.loc[train_index]
        val_data = data.loc[val_index]


# convert data format to tf.slim format
convert_nuclei.run(trainval_path + 'images/', train_list, val_list)


