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
    

def load_data(dirname):
    listfile=os.listdir(dirname)
    X = []
    Y = []
    XT = []
    YT = []
    for file in listfile:
        if "_" in file:
            continue
        wordname=file
        textlist=os.listdir(dirname+wordname)
        a=len(textlist)
        #print(a)
        b=a//3
        #print(b)
        k=0
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
            row=42*8#앞의 8프레임 제거
            landmark_frame=[]
            for i in range(0,100):#뒤의 142프레임제거==> 총 150프레임으로 고정
                landmark_frame.extend(numbers[row:row+42])
                row += 42
            landmark_frame=np.array(landmark_frame)
            landmark_frame=list(landmark_frame.reshape(-1,42))#2차원으로 변환(260*42)
            if (k%3==2):
                XT.append(np.array(landmark_frame))
                YT.append(wordname)
            else:
                X.append(np.array(landmark_frame))
                Y.append(wordname)
            k+=1
            
    X=np.array(X)
    Y=np.array(Y)
    #print(YT)
    XT=np.array(XT)
    YT=np.array(YT)
    #print(X.shape)
    
    tmp = [[x,y] for x, y in zip(X, Y)]
    random.shuffle(tmp)
    random.shuffle(tmp)

    tmp1 = [[xt,yt] for xt, yt in zip(XT, YT)]
    random.shuffle(tmp1)
    
    X = [n[0] for n in tmp]
    Y = [n[1] for n in tmp]
    XT = [n[0] for n in tmp1]
    YT = [n[1] for n in tmp1]
    '''
    t = Tokenizer()
    t.fit_on_texts(Y)
    encoded=t.texts_to_sequences(Y)
    one_hot=to_categorical(encoded)
    '''
    text="Apple Bird Blue Cents Child Cow Drink Green Hello Like Metoo No Orange Pig Sorry Thankyou Where Who Yes You"
    t = Tokenizer()
    t.fit_on_texts([text])
    encoded=t.texts_to_sequences([Y])[0]
    encoded2=t.texts_to_sequences([YT])[0]
    one_hot = to_categorical(encoded)
    one_hot2=to_categorical(encoded2)
    
    (x_train, y_train) = X, one_hot
    #print(x_train[0])
    (x_test,y_test)=XT,one_hot2
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_test=np.array(x_test)
    y_test=np.array(y_test)
    #return x_train[0:2*a],y_train[0:2*a],x_train[2*a:-1],y_train[2*a:-1]
    return x_train,y_train,x_test,y_test

def simple_rnn():
    model = Sequential()
    model.add(SimpleRNN(units=64, input_shape=(200, 42)))
    model.add(Dense(64, activation="softmax")) #softmax, linear 어떤걸 기준으로 하지
    model.add(Dense(128, activation="linear")) #softmax, linear 어떤걸 기준으로 하지
    model.add(Dense(21))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model
    

def rnn_lstm():
    model = Sequential()
    model.add(layers.LSTM(64,return_sequences=True,input_shape=(100,42)))  # returns a sequence of vectors of dimension 32
    model.add(layers.LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(layers.LSTM(32))  # return a single vector of dimension 32
    model.add(layers.Dense(4, activation='softmax'))    
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model

def bidirectional_lstm():
    model = Sequential()
    model.add(Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(100, 42)))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
    return model

def build_model():
    model = Sequential()
    model.add(layers.LSTM(64, return_sequences=True,
                   input_shape=(100, 42)))  # returns a sequence of vectors of dimension 32
    model.add(layers.LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(layers.LSTM(32))  # return a single vector of dimension 32
    model.add(layers.Dense(21, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def main(dirname):
    x_train,y_train,x_test,y_test=load_data(dirname)
    num_val_samples=(x_train.shape[0])//5
    #print(num_val_samples)
    #num_epochs=5
    #all_scores=[]
    model=build_model()
    '''
    for i in range(5):#5개의 분할로 시행 # k-겹 교차 검증 
        print('처리중인 폴드 #',i)
        val_data=x_train[i*num_val_samples:(i+1)*num_val_samples]
        val_targets=y_train[i*num_val_samples:(i+1)*num_val_samples]
        partial_train_data=np.concatenate([x_train[:i*num_val_samples],
                                          x_train[(i+1)*num_val_samples:]],
                                         axis=0)
        partial_train_targets=np.concatenate([y_train[:i*num_val_samples],
                                             y_train[(i+1)*num_val_samples:]],
                                            axis=0)
        #labels=load_label(dirname)
    '''
    print('Training stage')
    print('==============')
    history=model.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_test,y_test))
    score, acc = model.evaluate(x_test,y_test,batch_size=32,verbose=0)
    print('Test performance: accuracy={0}, loss={1}'.format(acc, score))
    model.save('simpleRNN.h5')

        #score, acc = model.evaluate(x_test,y_test,batch_size=32)
        #print('Test performance: accuracy={0}, loss={1}'.format(acc, score))
    
if __name__=='__main__':
    main("/Users/jongwook/Desktop/traindata/")
