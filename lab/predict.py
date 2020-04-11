import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

import tools

def splitList(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

# Root CSV files directory
filePath = os.path.dirname(__file__)
dirname = filePath+"\\prediction_data\\"
#dirname = "./data/"

# Frame count
frames = 100

model = tf.keras.models.load_model(filePath+'\\sign_lang_recognition.h5')
model.summary()

listfile = os.listdir(dirname)
data = []
for wordname in listfile:
    if wordname == ".DS_Store":
        continue
    for csv in os.listdir(dirname + wordname):
        filepath = os.path.join(dirname, wordname, csv)
        content = pd.read_csv(filepath, sep=';')
        #content = content.reindex(list(range(0, frames)), fill_value=0.0)
        content.fillna(0.0, inplace = True) 
        data.append((wordname, content))
        del content
features = [n[1] for n in data]
features = [f.to_numpy() for f in features]
labels = [n[0] for n in data]

#Wörter wieder loopen
for f in features:
    print(f.shape)
    #Split in gleich große Arrays mit x Frames
    splitSize = int(f.shape[0] / 20)
    splits = np.array_split(f, splitSize)

    for split in splits:
        #Split wieder mit gewünschten Frames auffüllen 
        split = split.reindex(list(range(0, frames)), fill_value=0.0)
        split.fillna(0.0, inplace = True) 
        #Numpy Array erstellen für Prediction 
        x_train=np.array(split, dtype="float32")
        #Prediction mit Model laufen lassen
        y_pred = model.predict(x_train)
        predictions = np.array([np.argmax(pred) for pred in y_pred])
        print(predictions)