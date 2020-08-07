# Imports
import os
import warnings
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
import sklearn
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from matplotlib import pyplot
from data_repository import DataRepository
import sys

np.set_printoptions(threshold=sys.maxsize)

# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# GPU-initialization
physical_devices = tf.config.list_physical_devices('GPU') 
print("Num GPUs:", len(physical_devices)) 

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# wandb init
wandb.init()
# Root CSV files directory
dirname = wandb.config.path 
unseen_dirname = './data/absolute/2D_unknown/'
# Load data and print summary, if desired
repo = DataRepository(dirname)
X_train, X_val, X_test, y_train, y_val, y_test, labels = repo.getForTraining()
X_unseen = repo.getUnseenX(unseen_dirname)
num_classes = repo.numClasses
print('num_classes', num_classes)
wandb.config.update({'Size_Training_Set': len(X_train),'Size_Validation_Set': len(X_val), 'Size_Test_Set': len(X_test)})

#load tokens
tokens = os.listdir(dirname)
tokens = sorted(tokens, key=str.casefold) 
token_labels = {i:tokens[i] for i in range(0, len(tokens))}
print(token_labels)    
# Model
dropout = 0.2
nodesizes = [wandb.config.node_size2, wandb.config.node_size3, wandb.config.node_size4]

inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))

lstm = Bidirectional(layers.LSTM(wandb.config.node_size1, return_sequences=True))(inputs)
lstm = layers.Dropout(rate=dropout)(lstm)  

for i in range(0,wandb.config.num_layers):    #number of layers ramdom between 1 an 3
    lstm = Bidirectional(layers.LSTM(nodesizes[i],return_sequences=True))(lstm)
    lstm = layers.Dropout(rate=dropout)(lstm)

lstm = Bidirectional(layers.LSTM(wandb.config.node_size5))(lstm)
lstm = layers.Dropout(rate=dropout)(lstm)

class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(lstm)

reject_output = layers.Dense(num_classes, activation='sigmoid', name='reject_output')(lstm)

model = keras.models.Model(inputs=inputs, outputs=[class_output, reject_output])
# Plot the model graph
#keras.utils.plot_model(model, os.path.join(wandb.run.dir, 'nn_graph.png'), show_shapes=True)

model.compile(loss={
    'class_output': 'categorical_crossentropy', 
    'reject_output': 'binary_crossentropy'
    },
    optimizer=wandb.config.optimizer,
    metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

model.summary()

wandb.config.optimizer_config = model.optimizer.get_config()

history = model.fit(
    X_train, [y_train, y_train], 
    epochs=wandb.config.epochs, 
    batch_size=wandb.config.batch_size, 
    validation_data=(X_val,[y_val, y_val]), 
    shuffle=False,
    verbose=2, 
    callbacks=[WandbCallback()])


#Test accuracy

y_eval = model.evaluate(X_test, [y_test, y_test], verbose=2)

wandb.config.update({'test_loss': y_eval[1],'test_accuracy': y_eval[3], 'test_precision': y_eval[4], 'test_recall': y_eval[5]})


#Confusion Matrix
y_pred = model.predict(X_test)[0]

y_pred_integer = np.argmax(y_pred, axis=1)
y_test_integer = np.argmax(y_test, axis=1)

y_pred_name = ([token_labels[p] for p in y_pred_integer])
y_test_name = ([token_labels[p] for p in y_test_integer])

wandb.sklearn.plot_confusion_matrix(y_test_name, y_pred_name)

# Reject prediction

# Decode one_hot
y_train_dec = [np.argmax(encoded) for encoded in y_train]
y_train_dec = np.array(y_train_dec)
# predict on training examples for calculate standard deviation
seen_train_X_pred = np.array(model.predict(X_train)[1])
# Fit gaussian model
from scipy.stats import norm as dist_model
def fit(prob_pos_X):
    prob_pos = [p for p in prob_pos_X]+[2-p for p in prob_pos_X]
    pos_mu, pos_std = dist_model.fit(prob_pos)
    return pos_mu, pos_std
mu_stds = []
for i in range(num_classes):
    pos_mu, pos_std = fit(seen_train_X_pred[y_train_dec==i,i])
    mu_stds.append([pos_mu, pos_std])

# Predict on test examples
test_X_pred = model.predict(np.concatenate([X_test, X_unseen], axis=0))[1]
test_y_gt = np.concatenate([[np.argmax(encoded) for encoded in y_test], [num_classes for _ in X_unseen]], axis=0)

# get reject prediction based on threshold
test_y_pred = []
scale = 1.
for p in test_X_pred:
    max_class = np.argmax(p)
    max_value = np.max(p)
    threshold = max(0.5, 1. - scale * mu_stds[max_class][1])
    if max_value > threshold:
        test_y_pred.append(max_class)
    else:
        test_y_pred.append(num_classes)

precision, recall, fscore, _ = precision_recall_fscore_support(test_y_gt, test_y_pred)
wandb.run.summary["macro_f_score"] = np.mean(fscore)
with open(os.path.join(wandb.run.dir, 'mu_stds.txt'), 'w') as f:
    for item in mu_stds:
        f.write("%s\n" % item[1])
#Convert to TFLite
print(wandb.run.dir)

new_model= tf.keras.models.load_model(filepath=os.path.join(wandb.run.dir, "model-best.h5"))

tflite_converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
# Needed for some ops.
tflite_converter.experimental_new_converter = True
# tflite_converter.allow_custom_ops = True

tflite_model = tflite_converter.convert()

open(os.path.join(wandb.run.dir, "model-best.tflite"), "wb").write(tflite_model)