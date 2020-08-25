import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
# Constant frame count.
frames = 100
# Default value for empty cells
default = 0.0

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
        self.dataPerWord = []
        self.numClasses = 0
        self.features, self.labels = self.__loadData(dirname, updateClasses=True)
    

    def __loadData(self, dirname, updateClasses=False):
        self.listfile = os.listdir(dirname)
        self.listfile = sorted(self.listfile, key=str.casefold) 
        if updateClasses:
            self.numClasses = len(self.listfile)
        print(self.listfile)
        for word in self.listfile:
            if word == ".DS_Store":
                continue
            for csv in os.listdir(dirname + word):
                filepath = os.path.join(dirname, word, csv)
                content = pd.read_csv(filepath, sep=';')
                content = content.reindex(list(range(0, frames)), fill_value=default)
                content.fillna(default, inplace = True) 
                self.dataPerWord.append((word, content))

        features = [n[1] for n in self.dataPerWord]
        features = [f.to_numpy() for f in features]
        labels = [n[0] for n in self.dataPerWord]
        return features, labels

    def getDataAndLabels(self):
        features = [n[1] for n in self.dataPerWord]
        x = [f.to_numpy() for f in features]
        lower_words = [x.lower() for x in self.listfile]  
        
        y = [label.lower() for label in self.labels]
        encoder = LabelBinarizer()
        encoder.fit(lower_words)
        y = encoder.transform(y)
        return np.array(x), np.array(y)
    
    def getForTraining(self):
        x_train, x_val, y_train, y_val = train_test_split(self.features, self.labels, test_size=0.40, random_state=42, stratify=self.labels)
        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.50, random_state=42, stratify=y_val)
        if self.__verbose:
            self.__summary()
        lower_words = [x.lower() for x in self.listfile]

        y_train = [y.lower() for y in y_train]
        y_val = [y.lower() for y in y_val]
        y_test = [y.lower() for y in y_test]

        encoder = LabelBinarizer()
        test = encoder.fit_transform(lower_words)

        y_train = encoder.transform(y_train)
        y_val = encoder.transform(y_val)
        y_test = encoder.transform(y_test)
        # Making numpy arrays
        self.x_train=np.array(x_train)
        self.y_train=np.array(y_train)
        self.x_val=np.array(x_val)
        self.y_val=np.array(y_val)
        self.x_test=np.array(x_test)
        self.y_test=np.array(y_test)
        return self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test, self.labels
    
    def getUnseenX(self, dirname):
        X, _ = self.__loadData(dirname)
        return np.array(X)

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

if __name__ == "__main__":
    repo = DataRepository("lab/data/absolute/2D/")
    repo.getForTraining()