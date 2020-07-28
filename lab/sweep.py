# Imports
import os
import warnings
import tools
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
import sklearn
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.layers import Bidirectional
from matplotlib import pyplot
# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# wandb init
wandb.init(project="SLR")
# Root CSV files directory
dirname = "./data/absolute/2D/"  

# Load data and print summary, if desired
x_train, x_val, x_test, y_train, y_val, y_test, labels = tools.load_from(dirname, verbose=False) 

#load tokens
with open('tokens_json.txt', 'r') as outfile:
    json_ex = outfile.read()
    

tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_ex)
token_labels = {y:x for x,y in tokenizer.word_index.items()}

# GPU-initialization
physical_devices = tf.config.list_physical_devices('GPU') 
print("Num GPUs:", len(physical_devices)) 

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# Model
nodesizes = [wandb.config.node_size2, wandb.config.node_size3, wandb.config.node_size4]

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

wandb.config.optimizer_config = model.optimizer.get_config()

history=model.fit(x_train,y_train,epochs=wandb.config.epochs ,batch_size=wandb.config.batch_size, validation_data=(x_val,y_val),shuffle=False,verbose=2, callbacks=[WandbCallback()])

y_eval = model.evaluate(x_test, y_test, verbose=2)

wandb.config.update({'test_loss': y_eval[0],'test_accuracy': y_eval[1], 'test_precision': y_eval[2], 'test_recall': y_eval[3]})

#wandb.log({'test_loss': y_eval[0],'test_accuracy': y_eval[1], 'test_precision': y_eval[2], 'test_recall': y_eval[3]})



#Confusion Matrix
y_pred = model.predict(x_test)

y_pred_integer = np.argmax(y_pred, axis=1)
y_test_integer = np.argmax(y_test, axis=1)

y_pred_name = ([token_labels[p] for p in y_pred_integer])
y_test_name = ([token_labels[p] for p in y_test_integer])

wandb.sklearn.plot_confusion_matrix(y_test_name, y_pred_name)