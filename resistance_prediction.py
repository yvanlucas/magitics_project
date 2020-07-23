import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

import config as cfg


class ResistancePredictionKmers:
    def __init__(self, dataframe, classifier, param_grid):
        self.dataframe = dataframe
        self.param_grid = param_grid
        self.le = preprocessing.LabelEncoder()
        self.clf = classifier
        self.cv_clf = model_selection.GridSearchCV(
            estimator=self.clf, param_grid=self.param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        self.X, self.y = self.preprocess(self.dataframe)

    def preprocess(self, df):
        to_drop = [
            "target",
            "strain",
        ]
        X = df.drop(to_drop, axis=1)
        y = self.le.fit_transform(df["label"])
        return X, y

    def fit(self, X, y):
        self.cv_clf.fit(X, y)

    def test_model(self, X, y):
        y_predict = self.cv_clf.predict_proba(X)
        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, f"{cfg.model}_CVresults.pkl"), "wb") as f:
            pickle.dump(
                {
                    "classifier": self.cv_clf,
                    "features": self.X.columns,
                    "y_pred": y_predict,
                    "y_true": y
                },
                f
            )

    def roc_auc_evaluation(self, y_true, y_pred):
        # TODO export this function to an evaluation submodule.
        score = metrics.roc_auc_score(y_true, y_pred[:, 1])
        print('*** ROC AUC on test set is: ***')
        print(score)
        # TODO code pour heatmap
        # pvt = pd.pivot_table(pd.DataFrame(grid.cv_results_),
        #                     values='mean_test_score', index='param_alpha', columns='param_l1_ratio')
        # ax = sns.heatmap(pvt)
        return score

    def boosting_learning_curve(self):
        # TODO export this function to an evaluation toolbox.
        #  Renaming it as "plot_boosting_learning_curve"
        test_score = np.zeros(self.param_grid['n_estimators'][0], dtype=np.float64)
        for i, y_pred in enumerate(self.cv_clf.best_estimator_.staged_predict(self.X_test)):
            # TODO plot test_score pour voir si ca diminue doucement. Si non: kmers non informatifs
            #  seuls mais informatifs ensemble
            test_score[i] = self.cv_clf.best_estimator_.loss_(self.y_test, y_pred)

        print(test_score)
        fig = plt.figure(figsize=(6, 6))
        plt.subplot(1, 1, 1)
        plt.title(cfg.xp_name + '  //  ROC AUC = ' + str(self.score))
        plt.plot(np.arange(self.param_grid['n_estimators'][0]) + 1,
                 self.CVclf.best_estimator_.train_score_, 'b-',
                 label='Training Set Deviance')
        plt.plot(np.arange(self.param_grid['n_estimators'][0]) + 1, test_score, 'r-',
                 label='Test Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')
        fig.tight_layout()
        plt.savefig(cfg.pathtoxp + '/' + cfg.xp_name + '/boosting_learning_curve2.png')
