import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
# Constant frame count.
frames = 100
# Default value for empty cells
default = 0.0
json_token_filename = 'tokens_json.txt'

rejected_class_label ="unknown"

class DataRepository():

    def __init__(self, dirname: str, verbose = False):
        self.__dirname = dirname
        self.__verbose = verbose
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.features = None
        self.labels = None
        self.dataPerWord = []
        self.tokenizer = None
        self.numClasses = 0
        self.__loadData()
    

    def __loadData(self):
        listfile = os.listdir(self.__dirname)
        listfile= sorted(listfile, key=str.casefold) 
        self.numClasses = len(listfile)
        print(listfile)
        for word in listfile:
            if word == ".DS_Store":
                continue
            for csv in os.listdir(self.__dirname + word):
                filepath = os.path.join(self.__dirname, word, csv)
                content = pd.read_csv(filepath, sep=';')
                content = content.reindex(list(range(0, frames)), fill_value=default)
                content.fillna(default, inplace = True) 
                self.dataPerWord.append((word, content))
        features = [n[1] for n in self.dataPerWord]
        self.features = [f.to_numpy() for f in features]
        self.labels = [n[0] for n in self.dataPerWord]
       
        #Tokenize (One Hot)
        self.tokenizer = self.getFittedTokenizer(listfile)
        if self.__verbose:
            print('Tokens:')
            print(self.tokenizer.word_index)
        with open(json_token_filename, 'w') as outfile:
            outfile.write(self.tokenizer.to_json())

    def getDataAndLabels(self):
        features = [n[1] for n in self.dataPerWord]
        x = [f.to_numpy() for f in features]
        x = np.array(x)
        encoded_y = self.tokenizer.texts_to_sequences([self.labels])[0]
        y = to_categorical(encoded_y)
        return x, y
    
    def getForTraining(self):
        x_train, x_val, y_train, y_val = train_test_split(self.features, self.labels, test_size=0.40, random_state=42, stratify=self.labels)
        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.50, random_state=42, stratify=y_val)
        if self.__verbose:
            self.__summary()
        encoded_train=self.tokenizer.texts_to_sequences([y_train])[0]
        encoded_val=self.tokenizer.texts_to_sequences([y_val])[0]
        encoded_test=self.tokenizer.texts_to_sequences([y_test])[0]
        y_train = to_categorical(encoded_train)
        y_val = to_categorical(encoded_val)
        y_test = to_categorical(encoded_test)
        # Making numpy arrays
        self.x_train=np.array(x_train)
        self.y_train=np.array(y_train)
        self.x_val=np.array(x_val)
        self.y_val=np.array(y_val)
        self.x_test=np.array(x_test)
        self.y_test=np.array(y_test)
        
        return self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test, self.labels
    
    def getForTrainingWithRejected(self):
        x_train, x_val, y_train, y_val = train_test_split(self.features, self.labels, test_size=0.40, random_state=42, stratify=self.labels)
        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.50, random_state=42, stratify=y_val)
        y_train_rej = self.__y_to_reject(y_train)
        y_val_rej = self.__y_to_reject(y_val)
        y_test_rej = self.__y_to_reject(y_test)
        # y_train_rej = to_categorical(y_train_rej)
        # y_val_rej = to_categorical(y_val_rej)
        # y_test_rej = to_categorical(y_test_rej)
        encoded_train=self.tokenizer.texts_to_sequences([y_train])[0]
        encoded_val=self.tokenizer.texts_to_sequences([y_val])[0]
        encoded_test=self.tokenizer.texts_to_sequences([y_test])[0]
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
        return x_train, x_val, x_test, y_train, y_val, y_test, y_train_rej, y_val_rej, y_test_rej, self.labels

    def __y_to_reject(self, items):
        return [(0b1 if item == rejected_class_label else 0b0) for item in items]
            

    def __printCountDataSets(self, dataset):
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

    def __summary(self):
    #Enumerate
        print('Amount Datasets by word total:')
        self.__printCountDataSets(self.labels)
        print('Amount Datasets by word training:')
        print(self.y_train)
        self.__printCountDataSets(self.y_train[0])
        print('Amount Datasets by word validiation:')
        self.__printCountDataSets(self.y_val)
        print('Amount Datasets by word test:')
        self.__printCountDataSets(self.y_test)

        # Display data distribution
        print('Distribution of data:')
        print("Amount total:", len(self.labels))
        print("Amount training:", len(self.y_train))
        print("Amount validiation:", len(self.y_val))
        print("Amount test:", len(self.y_test))    
    
    def getFittedTokenizer(self, words):
        text = " ".join(words)
        t = Tokenizer()
        t.fit_on_texts([text])
        return t

if __name__ == "__main__":
    repo = DataRepository("./data/absolute/2D/")
    repo.getForTrainingWithRejected()