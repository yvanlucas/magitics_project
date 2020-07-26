import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

import config as cfg


# random.seed(42)


class ResistancePredictionKMers:
    def __init__(self, dataframe, clf, param_grid):
        self._check_clf(clf)

        self.dataframe = dataframe
        self.clf = clf
        self.param_grid = param_grid

        self.le = preprocessing.LabelEncoder()
        self.cv_clf = model_selection.GridSearchCV(
            estimator=self.clf,
            param_grid=self.param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = self.preprocess(self.dataframe)

        self.y_predict = None
        self.score = None
        self.pvt = None

    def preprocess(self, df):
        to_drop = ["label", "strain"]
        X = df.drop(to_drop, axis=1)
        y = self.le.fit_transform(df["label"].values)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4)
        return X_train, X_test, y_train, y_test

    def fit(self, X_train, y_train):
        self.cv_clf.fit(X_train, y_train)

    def predict(self, X_test):
        self.y_predict = self.cv_clf.predict_proba(X_test)

    def eval(self, y_test, y_pred, X_test):
        # ROC AUC value
        self.score = metrics.roc_auc_score(y_test, y_pred[:, 1])
        print('*** ROC AUC = ***')
        print(self.score)
        # Heatmap for GridSearchCV
        ls_params=list(self.param_grid.keys())
        self.pvt = pd.pivot_table(pd.DataFrame(self.cv_clf.cv_results_),values='mean_test_score', index='param_'+
                                        ls_params[0], columns='param_'+ls_params[1]) #Only works if 2 params in grid

        ax = sns.heatmap(self.pvt)
        ax.set(ylabel=ls_params[0], xlabel=ls_params[1])
        ax.figure.savefig(os.path.join(cfg.pathtoxp, cfg.xp_name, f"{cfg.model}_gridCV_heatmap.png"))

        # Boosting learning curve
        if cfg.model == 'gradient':
            test_score = np.zeros((self.cv_clf.best_params_['n_estimators'],), dtype=np.float64)
            for i, y_pred in enumerate(self.cv_clf.best_estimator_.staged_predict(X_test)):
                test_score[i] = self.cv_clf.best_estimator_.loss_(y_test, y_pred)
            sns.set()
            fig = plt.figure(figsize=(6, 6))
            plt.subplot(1, 1, 1)
            plt.title(cfg.xp_name + '  //  ROC AUC = ' + str(self.score))
            plt.plot(np.arange(self.cv_clf.best_params_['n_estimators']) + 1, self.cv_clf.best_estimator_.train_score_, 'b-',
                     label='Training Set Deviance')
            plt.plot(np.arange(self.cv_clf.best_params_['n_estimators']) + 1, test_score, 'r-',
                     label='Test Set Deviance')
            plt.legend(loc='upper right')
            plt.xlabel('Boosting Iterations')
            plt.ylabel('Deviance')
            fig.tight_layout()
            plt.savefig(os.path.join(cfg.pathtoxp, cfg.xp_name, f"{cfg.model}boosting_learning_curve.png"))

    def write_report(self):
        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, f"{cfg.model}_report.txt"), 'w') as txt:
            txt.write(cfg.xp_name + '  //  ROC AUC = ' + str(self.score)+'\n')
            txt.write('\n')
            txt.write('Len_kmers = ' + str(cfg.len_kmers)+'\n')
            txt.write('Min_abundance = ' + str(cfg.min_abundance)+'\n')
            txt.write('Model = ' + str(self.clf)+'\n')
            txt.write('Param_grid = ' + str(self.param_grid)+'\n')
            txt.write('\n')
            txt.write('Relevant kmers : \n')
            if cfg.model == 'rf' or cfg.model == 'gradient':
                featimp = self.cv_clf.best_estimator_.feature_importances_
                kmers = self.columns[np.nonzero(featimp)]
                for kmer in kmers:
                    txt.write(str(kmer)+'\n')

    def dump_eval(self, X_test, y_test, y_predict):
        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, f"{cfg.model}_CVresults.pkl"), "wb") as f:
            pickle.dump({"classifier": self.cv_clf,
                         "features": X_test.columns,
                         "y_pred": y_predict,
                         "y_true": y_test}, f)

    def run(self, evaluate=True):
        self.preprocess(self.dataframe)
        self.fit(self.X_train, self.y_train)
        if evaluate:
            self.predict(self.X_test)
            self.eval(self.y_test, self.y_predict, self.X_test)
            self.write_report()
            self.dump_eval(self.X_test, self.y_train, self.y_predict)

    def _check_clf(self, clf):
        if not hasattr(clf, "fit") or not hasattr(clf, "predict"):
            raise ValueError("'clf' must implement a 'fit' and a 'predict' method.")
