import numpy as np
from sklearn import feature_selection
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
import config as cfg
import random

import warnings
warnings.simplefilter("ignore")


class resistance_prediction_CNN(object):
    def __init__(self):

        self.batchsize = 3
        self.kmers_by_strains = 1000
        self.onehotX = preprocessing.OneHotEncoder(sparse=False)
        self.onehotX.fit(np.array(['A', 'T', 'G', 'C']).reshape(-1, 1))
        self.onehoty = preprocessing.LabelEncoder()
        self.onehoty.fit(np.array(['susceptible', 'resistant']).reshape((-1, 1)))

    def CHI2_feature_selection(self):
        with open(cfg.pathtoxp + cfg.xp_name + '/kmers_DF.pkl', 'rb') as f:
            self.dataframe = pickle.load(f)

        le = preprocessing.LabelEncoder()
        self.dataframe['label'] = le.fit_transform(self.dataframe['label'])
        self.y = self.dataframe['label']
        self.X = self.dataframe.drop(['label'], axis=1)
        self.X = self.dataframe.drop(['strain'], axis=1)
        self.columns = self.X.columns
        self.chi2_selector = feature_selection.SelectKBest(feature_selection.chi2, k=1000)
        self.chi2_selector.fit(self.X, self.y)
        mask = self.chi2_selector.get_support()
        self.chosen_kmers = self.columns[mask]

        return

    def create_batchs(self):
        with open(cfg.pathtoxp + cfg.xp_name + '/kmerdicts.pkl', 'rb') as f:
            self.ls_dict = pickle.load(f)

        self.y = []
        self.kmers = []
        for dict in self.ls_dict:
            # self.y.extend([dict['label']]*self.batchsize)
            dict.pop('strain', None)
            ls_kmers = random.sample(dict.keys(), self.kmers_by_strains)
            for X, y in zip(ls_kmers, [dict['label']] * self.kmers_by_strains):
                X = list(X)
                self.kmers.append(self.onehotX.transform(np.array(X).reshape(-1, 1)).transpose())
                self.y.append(self.onehoty.transform(np.array(y).reshape(-1, 1)))
        # self.y=np.asarray(self.y)
        self.dataset = tf.data.Dataset.from_tensors((self.kmers, self.y))

    def create_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv1D(90, 2, activation='relu'))
        self.model.add(layers.MaxPooling1D(2))
        # self.model.add(layers.Conv2D(64, (4, 3), activation='relu'))
        # self.model.add(layers.MaxPooling2D((1, 2)))
        # self.model.add(layers.Conv2D(64, (4, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(20))
        self.model.add(layers.Dense(1, activation='sigmoid'))

    def train_model(self):
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        self.history = self.model.fit(self.dataset, epochs=5)

    # self.history = self.model.fit(self.kmers, self.y, batch_size=3, epochs=5)

    def evaluate_model(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=2)


# CNN = resistance_prediction_CNN()
# CNN.CHI2_feature_selection()
# CNN.create_batchs()
# CNN.create_model()
# CNN.train_model()


class RelevantKmerExtraction(object):
    def __init__(self):
        with open(cfg.pathtoxp + '/' + cfg.xp_name + '/CVresults.pkl', 'rb') as f:
            [self.CVclf, self.columns] = pickle.load(f)

        self.feature_importance = self.CVclf.best_estimator_.feature_importances

    def get_relevant_kmers_list(self):
        relevantindex = np.nonzero(self.feature_importance)
        self.relevantKmers = self.columns[relevantindex]

    def intersection_distance(ls1, ls2):
        """
        Compute the intersection ratio between 2 sets of kmers
        -> distance between two P(resistance|kmers) distributions estimated by different random forest classifiers
        """
        ls_inter = list(set(ls1) & set(ls2))
        dist = len(ls_inter) / ((len(ls1) + len(ls2)) / 2)
        return dist