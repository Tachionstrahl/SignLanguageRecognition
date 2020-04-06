# Imports
import os
import warnings
import tools
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.layers import Bidirectional
from matplotlib import pyplot
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters



# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Root CSV files directory
dirname = "./data/"

# Constant frame count.
frames = 100

listfile = os.listdir(dirname)
data = []
for wordname in listfile:
    if wordname == ".DS_Store":
        continue
    for csv in os.listdir(dirname + wordname):
        filepath = os.path.join(dirname, wordname, csv)
        content = pd.read_csv(filepath, sep=';')
        content = content.reindex(list(range(0, frames)), fill_value=0.0)
        content.fillna(0.0, inplace = True) 
        data.append((wordname, content))

features = [n[1] for n in data]
features = [f.to_numpy() for f in features]
labels = [n[0] for n in data]
x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.40, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.50, random_state=42)

tokenizer = tools.tokenize(dirname)
encoded_train=tokenizer.texts_to_sequences([y_train])[0]
encoded_val=tokenizer.texts_to_sequences([y_val])[0]
encoded_test=tokenizer.texts_to_sequences([y_test])[0]
y_train = to_categorical(encoded_train)
y_val = to_categorical(encoded_val)
y_test = to_categorical(encoded_test)

x_train=np.array(x_train)
y_train=np.array(y_train)
x_val=np.array(x_val)
y_val=np.array(y_val)
x_test=np.array(x_test)
y_test=np.array(y_test)

LOG_DIR = "Optimization_"f"{int(time.time())}"

def build_model(hp):
    model = Sequential()
    
    model.add(layers.LSTM(hp.Int("LSTM_input", min_value =32, max_value=256,step=32, default=32), 
                            return_sequences=True,
                            input_shape=(x_train.shape[1], x_train.shape[2])))
    
    for i in range(hp.Int("n_layers" , 1, 4)):    
        model.add(layers.LSTM(hp.Int(f"LSTM_{i}_units", min_value =32, max_value=256,step=32, default=32),
                                return_sequences=True))
    
    model.add(layers.LSTM(hp.Int(f"LSTM_End", min_value =32, max_value=256,step=32, default=32)))
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    model.summary()
    return model


'''
tuner  = RandomSearch(
    build_model,
    objective = "val_accuracy",
    max_trials = 20,
    executions_per_trial = 1,
    directory = LOG_DIR,
    project_name='SignLagnuageModelOptimization'
    )

tuner.search_space_summary()

tuner.search(x=x_train, 
                y= y_train,
            epochs=80,
            batch_size=32,
            validation_data=(x_val,y_val))

print(tuner.get_best_hyperparameters()[0].values)
print(tuner.results_summary())
#model= build_model()
#history=model.fit(x_train,y_train,epochs=80,validation_data=(x_val,y_val),shuffle=False
'''

tuner  = Hyperband(
    build_model,
    objective = "val_accuracy",
    max_epochs=80,    
    directory = LOG_DIR,
    project_name='SignLagnuageModelOptimization'
    )

tuner.search(x=x_train, 
            y= y_train,
            batch_size=32,
            validation_data=(x_val,y_val))


#print(tuner.get_best_hyperparameters()[0].values)
#print(tuner.results_summary())

#history=tuner.get_best_hyperparameters()[0]


history=tuner.get_best_models()[0]
history.summary()
model.save("sign_lang_recognition_tuned.h5")