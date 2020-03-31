from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import layers, models
from keras.models import Sequential
from keras import layers
import os
import sys
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import random
from keras import optimizers
from keras.layers import SimpleRNN, Dense
from keras.layers import Bidirectional
import tensorflow as tf
from numpy import argmax

def load_labels(dirname):
    label = {}
    count = 1
    listfile=os.listdir(dirname)
    for l in listfile:
        if "_" in l:
            continue
        label[l] = count
        count += 1
    return label

x_test,y_test=load_data("/Users/anna/SLR/Seperate/testinput/")
new_model = tf.keras.models.load_model('simpeRNN.h5')
new_model.summary()

#모델평가
#loss, acc = new_model.evaluate(x_test,y_test, verbose=2)
#print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

#print(new_model.predict(x_test).shape)


#모델 사용

xhat = x_test
yhat = new_model.predict(xhat)
print('## yhat ##')
#labels=load_label(dirname) 
a=xhat.shape[0]
for i in range(a):
    print('True: '+str(argmax(y_test[i]))+', Predict: '+str(yhat[i]))
    
