#!/usr/bin/env python

import wandb
import os
import multiprocessing
import collections
import random
import warnings
import tensorflow as tf
import sklearn
import numpy as np
import sys
import tensorflow.keras as K

from wandb.keras import WandbCallback
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
from sklearn.model_selection import KFold, StratifiedKFold



Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "sweep_id", "sweep_run_name", "config","train","test","x","y","num_classes")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("val_accuracy"))


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def training(sweep_q, worker_q):
    # GPU-initialization
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices)) 

    gpu_config = ConfigProto()
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.3
    gpu_config.gpu_options.allow_growth = True
    session = InteractiveSession(config=gpu_config)

    reset_wandb_env()
    worker_data = worker_q.get()
    run_name = "{}-{}".format(worker_data.sweep_run_name, worker_data.num)
    config = worker_data.config
    train=worker_data.train
    test=worker_data.test
    num_classes=worker_data.num_classes
    x=worker_data.x
    y=worker_data.y
    run = wandb.init(
        group=worker_data.sweep_id,
        job_type=worker_data.sweep_run_name,
        name=run_name,
        config=config,
    )
  ##########################################################  
    run.config.update({'Size_Training_Set': len(train), 'Size_Test_Set': len(test)})
        
    # Model
    dropout = run.config.dropout
    nodesizes = [run.config.node_size2, run.config.node_size3, run.config.node_size4]

    model = Sequential()
    model.add(Bidirectional(layers.LSTM(run.config.node_size1, return_sequences=True), input_shape=(x.shape[1], x.shape[2])))
    model.add(layers.Dropout(rate=dropout))  

    for i in range(0,run.config.num_layers):    #number of layers ramdom between 1 an 3
        model.add(Bidirectional(layers.LSTM(nodesizes[i],return_sequences=True)))
        model.add(layers.Dropout(rate=dropout))  

    model.add(Bidirectional(layers.LSTM(run.config.node_size5)))
    model.add(layers.Dropout(rate=dropout))

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer=run.config.optimizer,
                metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    model.summary()

    run.config.optimizer_config = model.optimizer.get_config()

    history=model.fit(x[train],y[train],epochs=10 ,batch_size=run.config.batch_size, validation_data=(x[test],y[test]),shuffle=False,verbose=2, callbacks=[WandbCallback()])
    scores = model.evaluate(x[test], y[test], verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #cvscores.append(scores[1] * 100)
    #print("Scores: ", scores)
    #print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))  


    ############################################################
    run.log(dict(val_accuracy=scores[1]))
    wandb.join()
    sweep_q.put(WorkerDoneData(val_accuracy=scores[1]))


def main():
    num_folds = 5

    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    sweep_q = multiprocessing.Queue()
    workers = []
    print("WORKERS!!!")
    for num in range(num_folds):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=training, kwargs=dict(sweep_q=sweep_q, worker_q=q)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))

    print("INIT WANDB!!!!!")
    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"

    print("FINISH INIT!!!!!")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    np.set_printoptions(threshold=sys.maxsize)    
    
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    # Load data and print summary, if desired
    dirname = sweep_run.config.path
    print("Ordnername: ", dirname)
    repo = DataRepository(dirname)
    x, y = repo.getDataAndLabels()
    print("Länge X: ", len(x), " Länge Y: ", len(y))
    
    #load tokens
    tokens = os.listdir(dirname)
    tokens = sorted(tokens, key=str.casefold) 
    token_labels = {i:tokens[i] for i in range(0, len(tokens))}

    y_integer = np.argmax(y, axis=1)
    y_name = ([token_labels[p] for p in y_integer])
     
    num_classes = repo.numClasses
    print("Anzahl Wörter : ", num_classes)
    metrics = []
    num=0
    for train, test in skfold.split(x, y_name):
        worker = workers[num]
        # start worker
        worker.queue.put(
            WorkerInitData(
                sweep_id=sweep_id,
                num=num,
                sweep_run_name=sweep_run_name,
                config=dict(sweep_run.config),
                train=train,
                test=test,
                x=x,
                y=y,
                num_classes=num_classes
            )
        )
        # get metric from worker
        result = sweep_q.get()
        # wait for worker to finish
        worker.process.join()
        # log metric to sweep_run
        metrics.append(result.val_accuracy)
        num=num+1

    sweep_run.log(dict(val_accuracy=sum(metrics) / len(metrics)))
    wandb.join()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)


if __name__ == "__main__":
    main()
