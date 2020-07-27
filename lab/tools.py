import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
# Constant frame count.
frames = 100
# Default value for empty cells
default = .5


def load_from(dirname, verbose = False, val_size=0.2, test_size=0.2):
    #Preparation Stage - Load data and normalize
    listfile = os.listdir(dirname)
    listfile= sorted(listfile, key=str.casefold) 
    data = []
    for wordname in listfile:
        if wordname == ".DS_Store":
            continue
        for csv in os.listdir(dirname + wordname):
            filepath = os.path.join(dirname, wordname, csv)
            content = pd.read_csv(filepath, sep=';')
            content = content.reindex(list(range(0, frames)), fill_value=default)
            content.fillna(default, inplace = True) 
            data.append((wordname, content))
            
    #Split data 60-20-20

    features = [n[1] for n in data]
    features = [f.to_numpy() for f in features]
    labels = [n[0] for n in data]
    split = val_size + test_size
    x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=split, random_state=42, stratify=labels)
    split = test_size / split
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=split, random_state=42, stratify=y_val)
    if verbose:
        summary(y_train, y_val, y_test, labels)
    #Tokenize (One Hot)
    tokenizer = tokenize(listfile)
    if verbose:
        print('Tokens:')
        print(tokenizer.word_index)
    with open('tokens_json.txt', 'w') as outfile:
        outfile.write(tokenizer.to_json())
    encoded_train=tokenizer.texts_to_sequences([y_train])[0]
    encoded_val=tokenizer.texts_to_sequences([y_val])[0]
    encoded_test=tokenizer.texts_to_sequences([y_test])[0]

    y_train = to_categorical(encoded_train)
    y_val = to_categorical(encoded_val)
    y_test = to_categorical(encoded_test)
    # Making numpy arrays
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_val=np.array(x_val)
    y_val=np.array(y_val)
    x_test=np.array(x_test)
    y_test=np.array(y_test)
    return x_train, x_val, x_test, y_train, y_val, y_test, labels

def printCountDataSets(dataset):
    wortCounter = []
    #Liste mit einmaligen Labels erstellen
    labels = sorted(set(dataset), key=dataset.index)
    #Liste nochmal Alphabetisch sortieren
    labels = sorted(labels)
    for label in labels:
        wortCounter.append(0)
    for row in dataset:
        for i in range(len(labels)):
            if str(labels[i]).startswith(row):
                wortCounter[i] += 1
    for i in range(len(labels)):
        print(labels[i], ': ', wortCounter[i], end =";  ")
    print(' ')

def summary(y_train, y_val, y_test, labels):
    #Enumerate

    print('Amount Datasets by word total:')
    printCountDataSets(labels)
    print('Amount Datasets by word training:')
    printCountDataSets(y_train)
    print('Amount Datasets by word validiation:')
    printCountDataSets(y_val)
    print('Amount Datasets by word test:')
    printCountDataSets(y_test)

    # Display data distribution
    print('Distribution of data:')
    print("Amount total:", len(labels))
    print("Amount training:", len(y_train))
    print("Amount validiation:", len(y_val))
    print("Amount test:", len(y_test))      

def tokenize(words):
    text = " ".join(words)
    t = Tokenizer()
    t.fit_on_texts([text])
    return t
