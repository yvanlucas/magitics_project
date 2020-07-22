from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
import pyscm
import config as cfg
import pickle
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#random.seed(42)


class Resistance_Prediction_kmers(object):
    def __init__(self):
        with open(cfg.pathtoxp + cfg.xp_name + '/kmers_DF.pkl', 'rb') as f:
            self.dataframe = pickle.load(f)

        if cfg.model == 'rf':  # TODO use boosting to predict (adaboost)
            self.clf = ensemble.RandomForestClassifier()
            self.param_grid = cfg.rf_grid
        elif cfg.model == 'SCM':
            self.clf = pyscm.SetCoveringMachineClassifier()
            self.param_grid = cfg.SCM_grid
        elif cfg.model == 'gradient':
            self.clf = ensemble.GradientBoostingClassifier(max_depth=1, max_features=None)
            self.param_grid = cfg.gradient_grid

    def preprocessing_dataframe(self):
        le = preprocessing.LabelEncoder()
        self.dataframe['label'] = le.fit_transform(self.dataframe['label'])
        self.y = self.dataframe['label']
        self.X = self.dataframe.drop(['label'], axis=1)
        self.X = self.dataframe.drop(['strain'], axis=1)
        self.columns = self.X.columns
        # self.dataframe = 0  # clear memory

    def split_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.X, self.y,
                                                                                                test_size=0.4,
                                                                                                random_state=42)

    def train_model(self):
        self.CVclf = model_selection.GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=3,
                                                  scoring='accuracy', n_jobs=-1)

        self.CVclf.fit(self.X_train, self.y_train)

    def test_model(self):
        self.y_predict = self.CVclf.predict_proba(self.X_test)
        with open(cfg.pathtoxp + cfg.xp_name + '/'+cfg.model+'CVresults.pkl', 'wb') as f:
            pickle.dump([self.CVclf, self.columns, self.y_predict, self.y_test], f)

    def ROC_AUC_evaluation(self):
        self.score = metrics.roc_auc_score(self.y_test, self.y_predict[:, 1])
        print('*** ROC AUC on test set is: ***')
        print(self.score)
        #TODO code pour heatmap
        #pvt = pd.pivot_table(pd.DataFrame(grid.cv_results_),
        #                     values='mean_test_score', index='param_alpha', columns='param_l1_ratio')
        #ax = sns.heatmap(pvt)

    def boosting_learning_curve(self):
        test_score = np.zeros((self.param_grid['n_estimators'][0],), dtype=np.float64)
        for i, y_pred in enumerate(self.CVclf.best_estimator_.staged_predict(self.X_test)):
            #TODO plot test_score pour voir si ca diminue doucement. Si non: kmers non informatifds seuls mais informatifs ensemble
            test_score[i] = self.CVclf.best_estimator_.loss_(self.y_test, y_pred)


        print(test_score)
        fig = plt.figure(figsize=(6, 6))
        plt.subplot(1, 1, 1)
        plt.title(cfg.xp_name+'  //  ROC AUC = '+str(self.score))
        plt.plot(np.arange(self.param_grid['n_estimators'][0]) + 1, self.CVclf.best_estimator_.train_score_, 'b-',
                 label='Training Set Deviance')
        plt.plot(np.arange(self.param_grid['n_estimators'][0]) + 1, test_score, 'r-',
                 label='Test Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')
        fig.tight_layout()
        plt.savefig(cfg.pathtoxp+'/'+cfg.xp_name+'/boosting_learning_curve2.png')



model = Resistance_Prediction_kmers()
model.preprocessing_dataframe()
model.split_train_test()
model.train_model()
model.test_model()
model.ROC_AUC_evaluation()
model.boosting_learning_curve()
