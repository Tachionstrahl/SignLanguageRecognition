from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers



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
                   input_shape=(100, 86)))  # returns a sequence of vectors of dimension 32
    model.add(layers.LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(layers.LSTM(32))  # return a single vector of dimension 32
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model