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
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.layers import Bidirectional
from matplotlib import pyplot
from data_repository import DataRepository
# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# wandb init
wandb.init()
# Root CSV files directory
dirname = wandb.config.path 

# Load data and print summary, if desired
repo = DataRepository(dirname)
x_train, x_val, x_test, y_train, y_val, y_test, labels = repo.getForTraining()
num_classes = 12
x_train = np.concatenate((x_train, x_val))
y_train = np.concatenate((y_train, y_val))
wandb.config.update({'Size_Training_Set': len(x_train),'Size_Validation_Set': len(x_val), 'Size_Test_Set': len(x_test)})

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


# Sequential Model
nodesizes = [wandb.config.node_size2, wandb.config.node_size3, wandb.config.node_size4]

# model = Sequential()

# Layers
# model.add(layers.LSTM(wandb.config.node_size1, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))

# for i in range(0,wandb.config.num_layers):    #number of layers ramdom between 1 an 3
#     model.add(layers.LSTM(nodesizes[i],return_sequences=True))

# model.add(layers.LSTM(wandb.config.node_size5))
# model.add(layers.Dense(num_classes, activation='softmax'))

# Compile
# model.compile(loss='categorical_crossentropy',
#               optimizer=wandb.config.optimizer,
#               metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
# model.summary()

# Functional Model
inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))

embedding = layers.LSTM(wandb.config.node_size1, return_sequences=True)(inputs)
for i in range(0, wandb.config.num_layers):    #number of layers ramdom between 1 an 3
    embedding = layers.LSTM(nodesizes[i],return_sequences=True)(embedding)
embedding = layers.LSTM(wandb.config.node_size5)(embedding)

reject_output = layers.Dense(1, activation="sigmoid", name="reject_output")(embedding)
class_output = layers.Dense(num_classes, activation='softmax', name="class_output")(embedding)

model = keras.models.Model(inputs=inputs, outputs=[reject_output, class_output])
keras.utils.plot_model(model, "model.png", show_shapes=True)
model.compile(
    loss={
        'class_output': 'categorical_crossentropy',
        'reject_output': 'binary_crossentropy'
        },
    optimizer=wandb.config.optimizer,
    metrics={'class_output': [
                    'accuracy',
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()
                ]},
    loss_weights={'reject_output': 1.0, 'class_output': 2.0})
model.summary()

wandb.config.optimizer_config = model.optimizer.get_config()

y_train_reject = [False for _ in y_train]
y_train_reject = to_categorical(y_train_reject)
y_val_reject = [False for _ in y_val]
y_val_reject = to_categorical(y_val_reject)
# history=model.fit(x_train,y_train,epochs=wandb.config.epochs ,batch_size=wandb.config.batch_size, validation_data=(x_val,y_val),shuffle=False,verbose=2, callbacks=[WandbCallback()])
history=model.fit(
    x_train,
    {'class_output': y_train, 'reject_output': y_train_reject},
    epochs=wandb.config.epochs,
    batch_size=wandb.config.batch_size,
    validation_data=(x_val, {'class_output': y_val, 'reject_output': y_val_reject}),
    shuffle=False,
    verbose=2, 
    callbacks=[WandbCallback()])
y_test_reject = [False for _ in y_test]
y_test_reject = to_categorical(y_test_reject)
y_eval = model.evaluate(x_test, {'class_output': y_test, 'reject_output': y_test_reject}, verbose=2)

wandb.config.update({'test_loss': y_eval[2],'test_accuracy': y_eval[3], 'test_precision': y_eval[4], 'test_recall': y_eval[5]})


#Confusion Matrix
y_pred = model.predict(x_test)[1]

y_pred_integer = np.argmax(y_pred, axis=1)
y_test_integer = np.argmax(y_test, axis=1)

y_pred_name = ([token_labels[p] for p in y_pred_integer])
y_test_name = ([token_labels[p] for p in y_test_integer])

wandb.sklearn.plot_confusion_matrix(y_test_name, y_pred_name)