from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import layers, models
from keras.models import Sequential
from keras import layers

import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


class Data:
    def __init__(self):
        
        X = []#학습데이터
        Y = []#학습데이터
        XT= []#평가데이터
        YT= []#평가데이터
        maxlength=0
        with open("/Users/jongwook/Desktop/output.pkl", 'rb') as fin:
            frames = pickle.load(fin)
            for i, frame in enumerate(frames):
                features = frame[0]
                maxlength=len(features)
                word = frame[1]
                if i%3 != 0:
                    X.append(np.array(features))
                    Y.append(word)
                else:
                    XT.append(np.array(features))
                    YT.append(word)
        X = np.array(X)
        Y = np.array(Y)
        XT= np.array(XT)
        YT=np.array(YT)
        
        t = Tokenizer()
        t.fit_on_texts(Y)
    
        encoded=t.texts_to_sequences(Y)
        one_hot=to_categorical(encoded)
        t1 = Tokenizer()
        t1.fit_on_texts(YT)

        encoded1=t1.texts_to_sequences(YT)
        one_hot1=to_categorical(encoded1)
     
        (x_train, y_train) = X, one_hot
        (x_test, y_test) = XT, one_hot1
        
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.maxlen=maxlength
        
class RNN_LSTM(models.Model):
    def __init__(self,maxlen):
        x = layers.Input((maxlen,))
        h = layers.Embedding(maxlen, 256)(x)
        h = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(h)
        y = layers.Dense(18, activation='softmax')(h)
        super().__init__(x, y)

        # try using different optimizers and different optimizer configs
        self.compile(loss='binary_crossentropy',
                     optimizer='adam', metrics=['accuracy'])


class Machine:
    def __init__(self):
        self.data = Data()
        self.model = RNN_LSTM(self.data.maxlen)

    def run(self, epochs=3, batch_size=32):
        data = self.data
        model = self.model
        print('Training stage')
        print('==============')
        history=model.fit(data.x_train, data.y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(data.x_test, data.y_test))

        score, acc = model.evaluate(data.x_test, data.y_test,
                                    batch_size=batch_size)
        print('Test performance: accuracy={0}, loss={1}'.format(acc, score))

def main():
    m = Machine()]
    m.run()


if __name__ == '__main__':
    main()
