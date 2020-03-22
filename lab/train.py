import os
import pandas as pd

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