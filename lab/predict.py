import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

import tools

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
        print(content)
        content = content.reindex(list(range(0, frames)), fill_value=0.0)
        content.fillna(0.0, inplace = True) 
        data.append((wordname, content))
        del content
features = [n[1] for n in data]
features = [f.to_numpy() for f in features]
labels = [n[0] for n in data]
x_train=np.array(features, dtype="float32")

y_pred = model.predict(x_train)
predictions = np.array([np.argmax(pred) for pred in y_pred])
print(predictions)