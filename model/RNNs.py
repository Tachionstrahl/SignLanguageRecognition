#Simple RNN, LSTM, GRU 비교  
def simple_rnn():
    model_RNN = Sequential()
    model_RNN.add(SimpleRNN(units=64, input_shape=(260, 42)))
    model_RNN.add(Dense(10, activation="relu")) #softmax, linear 어떤걸 기준으로 하지
    model_RNN.add(Dense(17))
    model_RNN.compile(loss='mse', optimizer='adam')
    
    return model_RNN
    
def rnn_lstm():
    model_LSTM = Sequential()
    model_LSTM.add(layers.LSTM(64,return_sequences=True,input_shape=(260,42))) #time_steps(network에 사용할 단위):260 픽스, features:42
    model_LSTM.add(layers.LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model_LSTM.add(layers.LSTM(32))  # return a single vector of dimension 32
    model_LSTM.add(layers.Dense(9, activation='softmax')) #단어수:9->17  
    model_LSTM.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    
    return model_LSTM

def bidirectional_lstm():
    Bidirectional_LSTM.add(Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(260, 42)))
    Bidirectional_LSTM.add(layers.Bidirectional(layers.LSTM(32)))
    Bidirectional_LSTM.add(layers.Dense(17, activation='softmax'))
    Bidirectional_LSTM.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    return Bidirectional_LSTM
