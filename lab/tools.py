import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

def load(dirname):
    listfile = os.listdir(dirname)
    contents = []
    for wordname in listfile:
        if wordname == ".DS_Store":
            continue
        for csv in os.listdir(dirname + wordname):
            filepath = os.path.join(dirname, wordname, csv)
            content = pd.read_csv(filepath, sep=';')
            content.fillna(0.0)
            contents.append((wordname, content))
    return contents

def tokenize(dirname):
    words = [word for word in os.listdir(dirname)]
    text = " ".join(words)
    t = Tokenizer()
    t.fit_on_texts([text])
    return t