import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import wandb
import sys
import multiprocessing
import collections
import random
import warnings
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from wandb.keras import WandbCallback
from data_repository import DataRepository
from sklearn.model_selection import StratifiedKFold

Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "sweep_id", "sweep_run_name", "sweep_name","config","train","test","x","y","num_classes","token_labels")
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
        group=worker_data.sweep_name,
        job_type=worker_data.sweep_run_name,
        name=run_name,
        config=config,
    )
    wandb.config.update({'hostname':os.uname()[1]})    
    # Model
    dropout = run.config.dropout
    nodesizes = [run.config.node_size2, run.config.node_size3, run.config.node_size4]

    model = Sequential()
    model.add(Bidirectional(LSTM(run.config.node_size1, return_sequences=True), input_shape=(x.shape[1], x.shape[2])))
    model.add(Dropout(rate=dropout))  

    for i in range(0,run.config.num_layers):    #number of layers ramdom between 1 an 3
        model.add(Bidirectional(LSTM(nodesizes[i],return_sequences=True)))
        model.add(Dropout(rate=dropout))  

    model.add(Bidirectional(LSTM(run.config.node_size5)))
    model.add(Dropout(rate=dropout))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer=run.config.optimizer,
                metrics=['accuracy',Precision(),Recall()])
    model.summary()

    model.fit(x[train],y[train],
    epochs=run.config.epochs,
    batch_size=run.config.batch_size,
    validation_data=(x[test],y[test]),
    shuffle=False,verbose=2, 
    callbacks=[WandbCallback()])

    #Test accuracy
    model_best_path = os.path.join(run.dir, "model-best.h5")
    best_model= load_model(filepath=model_best_path)
    y_eval = best_model.evaluate(x[test],y[test], verbose=0)

    #Confusion Matrix
    y_pred = best_model.predict(x[test])

    y_pred_integer = np.argmax(y_pred, axis=1)
    y_test_integer = np.argmax(y[test], axis=1)

    y_pred_name = ([worker_data.token_labels[p] for p in y_pred_integer])
    y_test_name = ([worker_data.token_labels[p] for p in y_test_integer])

    wandb.sklearn.plot_confusion_matrix(y_test_name, y_pred_name)

    #Convert to TFLite
    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    tflite_converter.experimental_new_converter = True
    tflite_model = tflite_converter.convert()
    open(os.path.join(wandb.run.dir, "model-best.tflite"), "wb").write(tflite_model)
        
    #Finish Run
    run.log(dict(val_accuracy=y_eval[1]))
    wandb.join()
    sweep_q.put(WorkerDoneData(val_accuracy=y_eval[1]))

def main():
    num_folds = 5

    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    sweep_q = multiprocessing.Queue()
    workers = []  
    for num in range(num_folds):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=training, kwargs=dict(sweep_q=sweep_q, worker_q=q)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))
    
    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_name = sweep_run.config.sweep_name
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_name)
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown"
    artifact =  sweep_run.use_artifact(sweep_run.config.artifact, type='dataset')
    artifact_dir = artifact.download()
    dirname= artifact_dir + '\\'
    dirname= dirname.replace('\\','/') 
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    np.set_printoptions(threshold=sys.maxsize)    
    
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    # Load data and print summary, if desired
    repo = DataRepository(dirname)
    x, y = repo.getDataAndLabels()
    
    #load tokens
    tokens = os.listdir(dirname)
    tokens = sorted(tokens, key=str.casefold) 
    token_labels = {i:tokens[i] for i in range(0, len(tokens))}

    y_integer = np.argmax(y, axis=1)
    y_name = ([token_labels[p] for p in y_integer])
     
    num_classes = repo.numClasses
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
                sweep_name=sweep_name,
                config=dict(sweep_run.config),
                train=train,
                test=test,
                x=x,
                y=y,
                num_classes=num_classes,
                token_labels=token_labels
            )
        )
        # get metric from worker
        result = sweep_q.get()
        # wait for worker to finish
        worker.process.join()
        # log metric to sweep_run
        metrics.append(result.val_accuracy)
        num=num+1

    wandb.config.update({'hostname':os.uname()[1]})
    sweep_run.log(dict(val_accuracy=sum(metrics) / len(metrics)))
    wandb.join()

if __name__ == "__main__":
    main()