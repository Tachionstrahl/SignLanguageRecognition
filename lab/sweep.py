# Imports
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
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
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from matplotlib import pyplot
from data_repository import DataRepository
import sys
import tensorflow.keras as K

np.set_printoptions(threshold=sys.maxsize)

# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# wandb init
wandb.init()
# Root CSV files directory
dirname = wandb.config.path 

# Load data and print summary, if desired
repo = DataRepository(dirname)
x_train, x_val, x_test, y_train, y_val, y_test, labels = repo.getForTraining()
num_classes = repo.numClasses
wandb.config.update({'Size_Training_Set': len(x_train),'Size_Validation_Set': len(x_val), 'Size_Test_Set': len(x_test)})

#load tokens
tokens = os.listdir(dirname)
tokens = sorted(tokens, key=str.casefold) 
token_labels = {i:tokens[i] for i in range(0, len(tokens))}

# GPU-initialization
physical_devices = tf.config.list_physical_devices('GPU') 
print("Num GPUs:", len(physical_devices)) 

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# Model
dropout = wandb.config.dropout
nodesizes = [wandb.config.node_size2, wandb.config.node_size3, wandb.config.node_size4]

model = Sequential()
model.add(Bidirectional(layers.LSTM(wandb.config.node_size1, return_sequences=True), input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(layers.Dropout(rate=dropout))  

for i in range(0,wandb.config.num_layers):    #number of layers ramdom between 1 an 3
    model.add(Bidirectional(layers.LSTM(nodesizes[i],return_sequences=True)))
    model.add(layers.Dropout(rate=dropout))  

model.add(Bidirectional(layers.LSTM(wandb.config.node_size5)))
model.add(layers.Dropout(rate=dropout))

model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=wandb.config.optimizer,
              metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
model.summary()

wandb.config.optimizer_config = model.optimizer.get_config()

history=model.fit(x_train,y_train,
epochs=wandb.config.epochs,
batch_size=wandb.config.batch_size,
validation_data=(x_val,y_val),
shuffle=False,
verbose=2, 
callbacks=[WandbCallback()])

model_best_path = os.path.join(wandb.run.dir, "model-best.h5")
#Test accuracy

best_model= tf.keras.models.load_model(filepath=model_best_path)
y_eval = best_model.evaluate(x_test, y_test, verbose=2)
wandb.config.update({'test_loss': y_eval[0],'test_accuracy': y_eval[1], 'test_precision': y_eval[2], 'test_recall': y_eval[3]})


#Confusion Matrix
y_pred = best_model.predict(x_test)

y_pred_integer = np.argmax(y_pred, axis=1)
y_test_integer = np.argmax(y_test, axis=1)

y_pred_name = ([token_labels[p] for p in y_pred_integer])
y_test_name = ([token_labels[p] for p in y_test_integer])

wandb.sklearn.plot_confusion_matrix(y_test_name, y_pred_name)


#Convert to TFLite


tflite_converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
# Needed for some ops.
tflite_converter.experimental_new_converter = True
# tflite_converter.allow_custom_ops = True
tflite_model = tflite_converter.convert()
open(os.path.join(wandb.run.dir, "model-best.tflite"), "wb").write(tflite_model)
os.remove(model_best_path)
