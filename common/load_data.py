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

def load_data(dirname):
    listfile=os.listdir(dirname)
    X = []
    Y = []
    for file in listfile:
        if "_" in file: #text파일이 들어있는 directory만 설정
            continue
        wordname=file
        if not(os.path.isdir(dirname+file)): #ignore .DS_Store
            continue
        textlist=os.listdir(dirname+wordname) #Word의 하위 파일들의 리스트(text파일들)
        for text in textlist:
            textname=dirname+wordname+"/"+text
            numbers=[]
            #print(textname)
            with open(textname, mode = 'r') as t:
                numbers = [float(num) for num in t.read().split()]#텍스트파일 읽기
                #print(len(numbers[0]))
                for i in range(len(numbers),10920): #단어의 동영상을 260 frame을 최대라고 가정하여 0으로 채우기
                    numbers.extend([0.000]) #260 frame 고정
            #numbers=np.array(numbers)
            #print(numbers[0])
            #numbers=np.array(numbers)
            #print(numbers)
            row=0
            landmark_frame=[]
            for i in range(0,len(numbers)): #42개 씩 끊어서 읽기
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
    
    t = Tokenizer() #Y를 원핫 벡터로 바꾸기
    t.fit_on_texts(Y)
    encoded=t.texts_to_sequences(Y)
    one_hot=to_categorical(encoded)
    
    (x_train, y_train) = X, one_hot
    #print(x_train[0])
    return x_train,y_train #학습데이터 

    

