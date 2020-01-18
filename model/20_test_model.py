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

def load_data(dirname):
    listfile=os.listdir(dirname)
    X = []
    Y = []
    for file in listfile:
        if "_" in file:
            continue
        wordname=file
        textlist=os.listdir(dirname+wordname)
        for text in textlist:
            if "DS_" in text:
                continue
            textname=dirname+wordname+"/"+text
            numbers=[]
            #print(textname)
            with open(textname, mode = 'r') as t:
                numbers = [float(num) for num in t.read().split()]
                #print(len(numbers[0]))
                for i in range(len(numbers),12600):
                    numbers.extend([0.000]) #300 frame 고정
            #numbers=np.array(numbers)
            #print(numbers[0])
            #numbers=np.array(numbers)
            #print(numbers)
            row=42*8#앞의 8프레임 제거
            landmark_frame=[]
            for i in range(0,100):#뒤의 142프레임제거==> 총 150프레임으로 고정
                #print(numbers[row*42:(row*42)+41])
                landmark_frame.extend(numbers[row:row+42])
                row += 42
            landmark_frame=np.array(landmark_frame)
            landmark_frame=list(landmark_frame.reshape(-1,42))#2차원으로 변환(260*42)
            #print(landmark_frame.shape)
            X.append(np.array(landmark_frame))
            Y.append(wordname)
    X=np.array(X)
    Y=np.array(Y)
    tmp = [[x,y] for x, y in zip(X,Y)]
    random.shuffle(tmp)
    X = [n[0] for n in tmp]
    Y = [n[1] for n in tmp]
    #print(Y)
    #print(X.shape)
    #t = Tokenizer()
    #t.fit_on_texts(Y)
    #encoded=t.texts_to_sequences(Y)
    text="Apple Bird Blue Cents Child Cow Drink Green Hello Like Metoo No Orange Pig Sorry Thankyou Where Who Yes You"

    t = Tokenizer()
    t.fit_on_texts([text])
    print(t.word_index) 
    #one_hot=to_categorical(encoded)
    encoded=t.texts_to_sequences([Y])[0]
    print(encoded)
    one_hot = to_categorical(encoded)


    (x_train, y_train) = X, one_hot
    #print(x_train[0])
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    return x_train,y_train


#prediction
def load_label():
    label = {}
    count = 1
    listfile=['Apple','Bird','Blue','Cents','Child','Cow','Drink','Green','Hello','Like','Metoo','No',
              'Orange','Pig','Sorry','Thankyou','Where','Who','Yes','You']
    for l in listfile:
        if "_" in l:
            continue
        label[l] = count
        count += 1
    return label



x_test,y_test=load_data("/Users/jongwook/Desktop/testdata/")
new_model = tf.keras.models.load_model('simpleRNN.h5')
new_model.summary()

labels=load_label()

#모델 사용

xhat = x_test
#print()
#xhat=xhat[55:56]
yhat = new_model.predict(xhat)
print('## yhat ##')

predictions = np.array([np.argmax(pred) for pred in yhat])
Y=np.array([np.argmax(i) for i in y_test])
print(Y)
rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
#print(rev_labels[predictions[0]])

#print(rev_labels)
#print(predictions)
with open("result.txt", "w") as f:
    f.write("gold, pred\n")
    for a, b in zip(Y, predictions):
        f.write("%s %s\n" % (rev_labels[a], rev_labels[b]))

acc = 100 * np.sum(predictions == Y) / len(Y)

print("Accuracy: ", acc)
