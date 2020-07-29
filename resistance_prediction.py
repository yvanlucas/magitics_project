from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn import feature_selection
import pyscm
import config as cfg
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd


# import fastparquet
# import pyarrow.parquet as pq
# random.seed(42)


class ResistancePredictionkmers(object):
    def __init__(self, dataframe=None, classifier=None, param_grid=None):

        self.chi2_selector = feature_selection.SelectKBest(feature_selection.chi2, k=1000000)
        self.dataframe = dataframe
        if self.dataframe == None:
            # table=pq.read_table(os.path.join(cfg.pathtoxp, cfg.xp_name, 'kmers_DF.parquet'))
            # self.dataframe=table.to_pandas()
            with open(os.path.join(cfg.pathtoxp, cfg.xp_name, 'kmers_DF.pkl'), 'rb') as f:
                self.dataframe = pickle.load(f)
        print('0')
        self.le = preprocessing.LabelEncoder()
        self.clf = classifier
        self.param_grid = param_grid

        self.cv_clf = model_selection.GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=3,
                                                   scoring='accuracy', n_jobs=-1)

    def preprocess(self, df):
        to_drop = ["label", "strain"]
        X = df.drop(to_drop, axis=1)
        y = self.le.fit_transform(df["label"].values)
        self.columns = X.columns
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4)
        return X_train, X_test, y_train, y_test

    def chi2_feature_selection(self, X_train, X_test, y_train):

        X_train = self.chi2_selector.fit_transform(X_train, y_train)
        X_test = self.chi2_selector.transform(X_test)
        return X_train, X_test

    def _check_clf(self, clf):
        if not hasattr(clf, "fit") or not hasattr(clf, "predict"):
            raise ValueError("'clf' must implement a 'fit' and a 'predict' method.")

    def fit(self, X_train, y_train):
        self.cv_clf.fit(X_train, y_train)

    def predict(self, X_test):
        y_predict = self.cv_clf.predict_proba(X_test)
        return y_predict

    def eval(self, y_test, y_pred, X_test):
        # Create eval dir
        mkdircmd = 'mkdir %s' % (os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id))
        os.system(mkdircmd)

        # metrics values
        self.score = {}
        self.score['ROC_AUC'] = metrics.roc_auc_score(y_test, y_pred[:, 1])
        self.score['Accuracy'] = metrics.accuracy_score(y_test, y_pred[:, 1].round(), normalize=False)
        self.score['MAE'] = metrics.mean_absolute_error(y_test, y_pred[:, 1])
        self.score['MSE'] = metrics.mean_squared_error(y_test, y_pred[:, 1])
        # self.acc = metrics.accuracy_score(y_test, y_pred[:,1])
        print('*** ROC AUC = ***')
        print(self.score['ROC_AUC'])
        # Heatmap for GridSearchCV
        ls_params = list(self.param_grid.keys())
        self.pvt = pd.pivot_table(pd.DataFrame(self.cv_clf.cv_results_), values='mean_test_score', index='param_' +
                                                                                                         ls_params[0],
                                  columns='param_' + ls_params[1])  # Only works if 2 params in grid

        ax = sns.heatmap(self.pvt)
        ax.set(ylabel=ls_params[0], xlabel=ls_params[1])
        ax.figure.savefig(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_gridCV_heatmap.png"))

        # Boosting learning curve
        if cfg.model == 'gradient':
            test_score = np.zeros((self.cv_clf.best_params_['n_estimators'],), dtype=np.float64)
            for i, y_pred in enumerate(self.cv_clf.best_estimator_.staged_predict(X_test)):
                test_score[i] = self.cv_clf.best_estimator_.loss_(y_test, y_pred)
            sns.set()
            fig = plt.figure(figsize=(6, 6))
            plt.subplot(1, 1, 1)
            plt.title(cfg.xp_name + '  //  ROC AUC = ' + str(self.score))
            plt.plot(np.arange(self.cv_clf.best_params_['n_estimators']) + 1, self.cv_clf.best_estimator_.train_score_,
                     'b-',
                     label='Training Set Deviance')
            plt.plot(np.arange(self.cv_clf.best_params_['n_estimators']) + 1, test_score, 'r-',
                     label='Test Set Deviance')
            plt.legend(loc='upper right')
            plt.xlabel('Boosting Iterations')
            plt.ylabel('Deviance')
            fig.tight_layout()
            plt.savefig(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}boosting_learning_curve.png"))

    def write_report(self):
        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_report.txt"), 'w') as txt:
            txt.write(cfg.xp_name + '\n')
            txt.write('\n')
            txt.write(str(self.score)+'\n')
            txt.write('Len_kmers = ' + str(cfg.len_kmers) + '\n')
            txt.write('Min_abundance = ' + str(cfg.min_abundance) + '\n')
            txt.write('Model = ' + str(self.clf) + '\n')
            txt.write('Param_grid = ' + str(self.param_grid) + '\n')
            txt.write('\n')
            txt.write('Relevant kmers : \n')
            if cfg.model == 'rf' or cfg.model == 'gradient':
                featimp = self.cv_clf.best_estimator_.feature_importances_
                kmers = self.columns[np.nonzero(featimp)]
                for kmer in kmers:
                    txt.write(str(kmer) + '\n')

    def dump_eval(self, y_test, y_predict):
        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_CVresults.pkl"), "wb") as f:
            pickle.dump({"classifier": self.cv_clf,
                         "features": self.columns,
                         "y_pred": y_predict,
                         "y_true": y_test}, f, protocol=4)

    def run(self, evaluate=True):
        X_train, X_test, y_train, y_test = self.preprocess(self.dataframe)
        X_train, X_test = self.chi2_feature_selection(X_train, X_test, y_train)
        self._check_clf(self.cv_clf)
        self.fit(X_train, y_train)
        if evaluate:
            y_predict = self.predict(X_test)
            self.eval(y_test, y_predict, X_test)
            self.write_report()
            self.dump_eval(y_train, y_predict)

#
# if cfg.model == 'rf':
#     clf = ensemble.RandomForestClassifier()
#     param_grid = cfg.rf_grid
# elif cfg.model == 'SCM':
#     clf = pyscm.SetCoveringMachineClassifier()
#     param_grid = cfg.SCM_grid
# elif cfg.model == 'gradient':
#     clf = ensemble.GradientBoostingClassifier(max_depth=4, max_features=None)
#     param_grid = cfg.gradient_grid
#
# expe = ResistancePredictionkmers(classifier=clf, param_grid=param_grid)
# expe.run()
