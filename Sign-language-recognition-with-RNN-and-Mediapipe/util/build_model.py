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
import argparse

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
            #print(numbers)
            row=42*8#앞의 8프레임 제거
            landmark_frame=[]
            for i in range(0,100):#총 100프레임으로 고정
                #print(numbers[row*42:(row*42)+41])
                landmark_frame.extend(numbers[row:row+42])
                row += 42
            landmark_frame=np.array(landmark_frame)
            landmark_frame=landmark_frame.reshape(-1,42)#2차원으로 변환(260*42)
            #print(landmark_frame.shape)
            X.append(np.array(landmark_frame))
            Y.append(wordname)
    X=np.array(X)
    Y=np.array(Y)
    #tmp = [[x,y] for x, y in zip(X,Y)]
    #random.shuffle(tmp)
    #X = [n[0] for n in tmp]
    #Y = [n[1] for n in tmp]
    #print(Y)
    #print(X.shape)
    #t = Tokenizer()
    #t.fit_on_texts(Y)
    #encoded=t.texts_to_sequences(Y)

    x_train = X
    #print(x_train[0])
    x_train=np.array(x_train)
    return x_train


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
    
def main(input_data_path,output_data_path):
    comp='bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_cpu'
    #명령어 컴파일
    cmd='GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_cpu \
    --calculator_graph_config_file=mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live.pbtxt'
    #미디어 파이프 명령어 저장
    listfile=os.listdir(input_data_path)
    output_dir=""
    filel=[]
    for file in listfile:
        if ".DS_" in file:
            continue
        word=file+'/'
        fullfilename=os.listdir(input_data_path+word)
        # 하위디렉토리의 모든 비디오들의 이름을 저장
        if not(os.path.isdir(output_data_path+"_"+word)):
            os.mkdir(output_data_path+"_"+word)
        if not(os.path.isdir(output_data_path+word)):
            os.mkdir(output_data_path+word)
        os.system(comp)
        outputfilelist=os.listdir(output_data_path+'_'+word)
        for mp4list in fullfilename:
            if ".DS_Store" in mp4list:
                continue
            filel.append(mp4list)
            inputfilen='   --input_video_path='+input_data_path+word+mp4list
            outputfilen='   --output_video_path='+output_data_path+'_'+word+mp4list
            cmdret=cmd+inputfilen+outputfilen
            os.system(cmdret)

    #mediapipe동작 작동 종료:
    output_dir=output_data_path
    x_test=load_data(output_dir)
    new_model = tf.keras.models.load_model('simpleRNN1.h5')
    #new_model.summary()

    labels=load_label()

    #모델 사용

    xhat = x_test
    #print()
    #xhat=xhat[55:56]
    yhat = new_model.predict(xhat)
    #print('## yhat ##')

    predictions = np.array([np.argmax(pred) for pred in yhat])
    rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
    s=0
    filel=np.array(filel)
    print(filel)
    for i in predictions:
        if s==4:
            continue
        txtpath="/Users/jongwook/Desktop/word"+str(s)+".txt"
        with open(txtpath, "w") as f:
            f.write(filel[s])
            f.write(" ")
            f.write(rev_labels[i])
        s+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='operating Mediapipe')
    parser.add_argument("--input_data_path",help=" ")
    parser.add_argument("--output_data_path",help=" ")
    args=parser.parse_args()
    input_data_path=args.input_data_path
    output_data_path=args.output_data_path
    #print(input_data_path)
    main(input_data_path,output_data_path)
