# Imports
import os
import warnings
import tools
import pandas as pd
import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
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
from sklearn.metrics import confusion_matrix

# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# wandb init
wandb.init(project="SLR")
# Root CSV files directory
dirname = "./data/absolute/2D/"  

# Constant frame count.
frames = 100

#Preparation Stage - Load data and normalize
listfile = os.listdir(dirname)
data = []
for wordname in listfile:
    if wordname == ".DS_Store":
        continue
    for csv in os.listdir(dirname + wordname):
        filepath = os.path.join(dirname, wordname, csv)
        content = pd.read_csv(filepath, sep=';')
        content = content.reindex(list(range(0, frames)), fill_value=0.5)
        content.fillna(0.5, inplace = True) 
        data.append((wordname, content))
        
#Split data 60-20-20

features = [n[1] for n in data]
features = [f.to_numpy() for f in features]
labels = [n[0] for n in data]
x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.40, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.50, random_state=42)

#Tokenize (One Hot)
tokenizer = tools.tokenize(dirname)
print('Tokens:')
print(tokenizer.word_index)
print('')
with open('tokens_json.txt', 'w') as outfile:
    outfile.write(tokenizer.to_json())

encoded_train=tokenizer.texts_to_sequences([y_train])[0]
encoded_val=tokenizer.texts_to_sequences([y_val])[0]
encoded_test=tokenizer.texts_to_sequences([y_test])[0]

y_train = to_categorical(encoded_train)
y_val = to_categorical(encoded_val)
y_test = to_categorical(encoded_test)

# Making numpy arrays
x_train=np.array(x_train)
y_train=np.array(y_train)
x_val=np.array(x_val)
y_val=np.array(y_val)
x_test=np.array(x_test)
y_test=np.array(y_test)

# GPU-KRAM
physical_devices = tf.config.list_physical_devices('GPU') 
print("Num GPUs:", len(physical_devices)) 

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

nodesizes = [wandb.config.node_size2, wandb.config.node_size3, wandb.config.node_size4]

for i in range(0,wandb.config.num_layers):    #number of layers ramdom between 1 an 3
    print(nodesizes[i])

# Model
model = Sequential()

model.add(layers.LSTM(wandb.config.node_size1, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))

for i in range(0,wandb.config.num_layers):    #number of layers ramdom between 1 an 3
    model.add(layers.LSTM(nodesizes[i],return_sequences=True))

model.add(layers.LSTM(wandb.config.node_size5))  

model.add(layers.Dense(12, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=wandb.config.optimizer,
              metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
model.summary()


history=model.fit(x_train,y_train,epochs=wandb.config.epochs ,batch_size=wandb.config.batch_size, validation_data=(x_val,y_val),shuffle=False,verbose=2, callbacks=[WandbCallback()])
#history=model.fit(x_train,y_train,epochs=30 ,batch_size=32, validation_data=(x_val,y_val),shuffle=False,verbose=2)