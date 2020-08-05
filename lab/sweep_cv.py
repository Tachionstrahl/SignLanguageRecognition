# Imports
import os
import warnings
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
from data_repository import DataRepository
from sklearn.model_selection import KFold, StratifiedKFold
import sys
import tensorflow.keras as K

np.set_printoptions(threshold=sys.maxsize)

# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# wandb init
wandb.init(reinit=True)
# Root CSV files directory
dirname = wandb.config.path 

# Load data and print summary, if desired
repo = DataRepository(dirname)

x, y = repo.getDataAndLabels()


wandb.config.update({'Size_Training_Set': len(x)})

#load tokens
with open('tokens_json.txt', 'r') as outfile:
    json_ex = outfile.read()
    

tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_ex)
token_labels = {y:x for x,y in tokenizer.word_index.items()}

y_integer = np.argmax(y, axis=1)
y_name= ([token_labels[p] for p in y_integer])

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
dropout = wandb.config.dropout
#lr = wandb.config.lr
nodesizes = [wandb.config.node_size2, wandb.config.node_size3, wandb.config.node_size4]

#group_id=wandb.util.generate_id()

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
cvscores = []
for train, test in skfold.split(x, y_name):
    print("size_x_train: ", len(x[train]))
    print("size_y_train: ", len(y[train]))
    print("size_x_test: ", len(x[test]))
    print("size_y_test: ", len(y[test]))

    model = Sequential()

    model.add(Bidirectional(layers.LSTM(wandb.config.node_size1, return_sequences=True), input_shape=(x.shape[1], x.shape[2])))
    model.add(layers.Dropout(rate=dropout))  

    for i in range(0,wandb.config.num_layers):    #number of layers ramdom between 1 an 3
        model.add(Bidirectional(layers.LSTM(nodesizes[i],return_sequences=True)))
        model.add(layers.Dropout(rate=dropout))  

    model.add(Bidirectional(layers.LSTM(wandb.config.node_size5)))
    model.add(layers.Dropout(rate=dropout))

    model.add(layers.Dense(12, activation='softmax'))

    
    wb_optimizer = wandb.config.lr

    model.compile(loss='categorical_crossentropy',
                optimizer= wandb.config.optimizer,
                metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    model.summary()

    wandb.config.optimizer_config = model.optimizer.get_config()

    history=model.fit(x[train],y[train],epochs=wandb.config.epochs ,batch_size=wandb.config.batch_size, validation_data=(x[test],y[test]),shuffle=False,verbose=2, callbacks=[WandbCallback()])
    #history=model.fit(x[train],y[train],epochs=170,validation_data=(x[test],y[test]),verbose=2)
    #Schreiben der Scores
    scores = model.evaluate(x[test], y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    print("Scores: ", scores)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    wandb.join()


#Test accuracy
y_eval = model.evaluate(x, y, verbose=2)

wandb.config.update({'test_loss': y_eval[0],'test_accuracy': y_eval[1], 'test_precision': y_eval[2], 'test_recall': y_eval[3]})


#Confusion Matrix
y_pred = model.predict(x)

y_pred_integer = np.argmax(y_pred, axis=1)
y_test_integer = np.argmax(y, axis=1)

y_pred_name = ([token_labels[p] for p in y_pred_integer])
y_test_name = ([token_labels[p] for p in y_test_integer])

wandb.sklearn.plot_confusion_matrix(y_test_name, y_pred_name)


#Convert to TFLite
#print(wandb.run.dir)

#new_model= tf.keras.models.load_model(filepath=os.path.join(wandb.run.dir, "model-best.h5"))

#tflite_converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
# Needed for some ops.
#tflite_converter.experimental_new_converter = True
# tflite_converter.allow_custom_ops = True

#tflite_model = tflite_converter.convert()

#open(os.path.join(wandb.run.dir, "model-best.tflite"), "wb").write(tflite_model)