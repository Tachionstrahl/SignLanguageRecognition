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
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.model_selection import KFold, StratifiedKFold
from matplotlib import pyplot
from data_repository import DataRepository
import sys
import tensorflow.keras as K

np.set_printoptions(threshold=sys.maxsize)
init = wandb.init(project="slr", reinit=True)
# GPU-initialization
physical_devices = tf.config.list_physical_devices('GPU') 
print("Num GPUs:", len(physical_devices)) 

gpu_config = ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.3
gpu_config.gpu_options.allow_growth = True
session = InteractiveSession(config=gpu_config)
config = wandb.config
sweep_id = init.sweep_id
# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    cvscores = []
    group_id = wandb.util.generate_id()
    # Load data and print summary, if desired
    dirname = config.path
    repo = DataRepository(dirname)
    x, y = repo.getDataAndLabels()
    #load tokens
    tokens = os.listdir(dirname)
    tokens = sorted(tokens, key=str.casefold) 
    token_labels = {i:tokens[i] for i in range(0, len(tokens))}
    y_integer = np.argmax(y, axis=1)
    y_name = ([token_labels[p] for p in y_integer])
     
    num_classes = repo.numClasses
    i = 0
    for train, test in skfold.split(x, y_name):
        i=i+1
        # wandb init
        run = wandb.init(group=group_id, reinit=True, name=group_id+"#"+str(i))
        print(sweep_id)
        run.sweep_id = sweep_id
        print("Run sweep id", run.sweep_id)
        # Root CSV files directory
        print("Sweep URL:", run.get_sweep_url())
        
        config.update({'Size_Training_Set': len(train), 'Size_Test_Set': len(test)})
        
        # Model
        dropout = config.dropout
        nodesizes = [config.node_size2, config.node_size3, config.node_size4]

        model = Sequential()
        model.add(Bidirectional(layers.LSTM(config.node_size1, return_sequences=True), input_shape=(x.shape[1], x.shape[2])))
        model.add(layers.Dropout(rate=dropout))  

        for i in range(0,config.num_layers):    #number of layers ramdom between 1 an 3
            model.add(Bidirectional(layers.LSTM(nodesizes[i],return_sequences=True)))
            model.add(layers.Dropout(rate=dropout))  

        model.add(Bidirectional(layers.LSTM(config.node_size5)))
        model.add(layers.Dropout(rate=dropout))

        model.add(layers.Dense(num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                    optimizer=config.optimizer,
                    metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
        model.summary()

        config.optimizer_config = model.optimizer.get_config()

        history=model.fit(x[train],y[train],epochs=10 ,batch_size=config.batch_size, validation_data=(x[test],y[test]),shuffle=False,verbose=2, callbacks=[WandbCallback()])
        scores = model.evaluate(x[test], y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        print("Scores: ", scores)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        wandb.join()

        # #Test accuracy
        # y_eval = model.evaluate(x_test, y_test, verbose=2)

        # config.update({'test_loss': y_eval[0],'test_accuracy': y_eval[1], 'test_precision': y_eval[2], 'test_recall': y_eval[3]})


        # #Confusion Matrix
        # y_pred = model.predict(x_test)

        # y_pred_integer = np.argmax(y_pred, axis=1)
        # y_test_integer = np.argmax(y_test, axis=1)

        # y_pred_name = ([token_labels[p] for p in y_pred_integer])
        # y_test_name = ([token_labels[p] for p in y_test_integer])

        # wandb.sklearn.plot_confusion_matrix(y_test_name, y_pred_name)


        # #Convert to TFLite
        # print(wandb.run.dir)

        # new_model= tf.keras.models.load_model(filepath=os.path.join(wandb.run.dir, "model-best.h5"))

        # tflite_converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
        # # Needed for some ops.
        # tflite_converter.experimental_new_converter = True
        # # tflite_converter.allow_custom_ops = True

        # tflite_model = tflite_converter.convert()

        # open(os.path.join(wandb.run.dir, "model-best.tflite"), "wb").write(tflite_model)

if __name__ == "__main__":
    main()