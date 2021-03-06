import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyscm
import scipy.sparse as sp
import seaborn as sns
from sklearn import (ensemble, tree, feature_selection, metrics, model_selection, preprocessing)
import config as cfg


class Train_kmer_clf(object):
    """
    Train in batch version and optionnally test in batch settings
    Also optimize the treshold for computing accuracies
    """
    def __init__(self):
        if cfg.model == "rf":
            self.clf = ensemble.RandomForestClassifier()
            self.param_grid = cfg.rf_grid
        elif cfg.model == "SCM":
            self.clf = pyscm.SetCoveringMachineClassifier()
            self.param_grid = cfg.SCM_grid
        elif cfg.model == "gradient":
            self.clf = ensemble.GradientBoostingClassifier(max_depth=4, max_features=None)
            self.param_grid = cfg.gradient_grid
        elif cfg.model == 'Ada':
            self.clf = ensemble.AdaBoostClassifier()
            self.param_grid = cfg.ada_grid

        if cfg.dtype == 'sparse':
            with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, "kmers_mats.pkl"), "rb") as f:
                [self.mat, self.labels, self.strain_to_index, self.kmer_to_index] = pickle.load(f)
            self.testratio = 0.0
        elif cfg.dtype == 'df':
            with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, "kmers_DF.pkl"), "rb") as f:
                [self.mat, self.labels] = pickle.load(f)
            self.kmer_to_index = self.mat.columns
            self.testratio = 0.3

    def preprocess_y(self):
        """
        Transform y into a binary vector
        """
        # to_drop = ["label", "strain"]
        # X = self.mat.drop(to_drop, axis=1)
        # self.columns = X.columns
        le = preprocessing.LabelEncoder()
        self.y = le.fit_transform(self.labels)

    def split_train_test(self):
        """
        Split the data matrix and target vector into train and test matrices and vector
        :return: X_train, X_test, y_train, y_test
        """
        if self.testratio > 0:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(self.mat, self.y,
                                                                                test_size=self.testratio)
        else:
            X_train = self.mat
            y_train = self.y
            X_test = None
            y_test = None
        del self.mat, self.y
        return X_train, X_test, y_train, y_test

    def chi2_feature_selection(self, X_train, X_test, y_train):
        """
        Refactor X_train and X_test only keeping features that are correlated with the target y_train
        :param X_train: train numpy array
        :param X_test: test numpy array
        :param y_train: train target variable vector
        :return: refined X_train and X_test
        """
        chi2_selector = feature_selection.SelectKBest(feature_selection.chi2, k=1000000)
        X_train = chi2_selector.fit_transform(X_train, y_train)
        X_test = chi2_selector.transform(X_test)
        return X_train, X_test

    def fit(self, X_train, y_train):
        """
        Fit the chosen classifier using gridsearch cross validation
        :param X_train:
        :param y_train:
        """
        self.cv_clf = model_selection.GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=2,
                                                   scoring="accuracy", n_jobs=1)
        self.cv_clf.fit(X_train, y_train)
        self.y_pred = self.cv_clf.predict_proba(X_train)
        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f'{cfg.model}_CVresults.pkl'), 'wb') as f:
            pickle.dump({"classifier": self.cv_clf, "features": self.kmer_to_index}, f, protocol=4)

    def predict(self, X_test):
        """
        Batch predict of X_test labels
        :param X_test:
        :return: y_predict
        """
        y_predict = self.cv_clf.predict_proba(X_test)
        return y_predict



    def get_accuracy_treshold(self, X_train, y_train):
        """
        Calculate the treshold to obtain the best accuracy on the train set
        :param X_train:
        :param y_train:
        :return: treshold value
        """
        train_predict = self.cv_clf.best_estimator_.predict_proba(X_train)[:, -1]
        accuracies = []
        nsteps = 100
        for i in range(nsteps):
            tres = i / nsteps
            tresd_predict = []
            for pred in train_predict:
                if pred > tres:
                    tresd_predict.append(1)
                else:
                    tresd_predict.append(0)
            accuracies.append(metrics.accuracy_score(y_train, tresd_predict))
        ind = accuracies.index(max(accuracies))
        treshold = float(ind) / float(nsteps)
        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_tres_value.txt"), "w") as f:
            f.write(str(treshold))
        return treshold

    def adapted_accuracy(self, y_test, y_pred, treshold):
        """
        Predict accuracy with respect to a calculated treshold (inherited from Test_streaming class)
        :param y_test:
        :param y_pred:
        :return:
        """
        return Test_streaming.adapted_accuracy(self, y_test, y_pred, treshold)

    def evaluate_and_write_report(self, y_test, y_pred, treshold):
        """
        Predict scores and write report
        :param y_test:
        :param y_pred:
        """
        Test_streaming.evaluate_and_write_report(self, y_test, y_pred, treshold)

    def plot_CV_heatmap(self):
        """
        plot gridsearchCV heatmap
        """
        ls_params = list(self.param_grid.keys())
        self.pvt = pd.pivot_table(pd.DataFrame(self.cv_clf.cv_results_), values="mean_test_score",
                                  index="param_" + ls_params[0], columns="param_" + ls_params[1])
        ax = sns.heatmap(self.pvt)
        ax.set(ylabel=ls_params[0], xlabel=ls_params[1])
        ax.figure.savefig(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_gridCV_heatmap.png"))

    def plot_boosting_learning_curve(self, X_test, y_test):
        """
        Plot boosting learning curve (test/train deviance = f (n_estimator) )
        :param X_test:
        :param y_test:
        """
        if cfg.model == "gradient":
            test_score = np.zeros((self.cv_clf.best_params_["n_estimators"],), dtype=np.float64)
            for i, y_pred in enumerate(self.cv_clf.best_estimator_.staged_predict(X_test)):
                test_score[i] = self.cv_clf.best_estimator_.loss_(y_test, y_pred)
            sns.set()
            fig = plt.figure(figsize=(6, 6))
            plt.subplot(1, 1, 1)
            plt.title(cfg.xp_name + "  //  ROC AUC = " + str(self.score))
            plt.plot(np.arange(self.cv_clf.best_params_["n_estimators"]) + 1, self.cv_clf.best_estimator_.train_score_,
                     "b-", label="Training Set Deviance")
            plt.plot(np.arange(self.cv_clf.best_params_["n_estimators"]) + 1, test_score, "r-",
                     label="Test Set Deviance")
            plt.legend(loc="upper right")
            plt.xlabel("Boosting Iterations")
            plt.ylabel("Deviance")
            fig.tight_layout()
            plt.savefig(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}boosting_learning_curve.png"))

    def run(self):
        """
        Run method to wrap the class methods, train and optionnaly test the model in batch settings
        """
        self.preprocess_y()
        X_train, X_test, y_train, y_test = self.split_train_test()
        #        X_train, X_test = self.chi2_feature_selection(X_train, X_test, y_train)
        self.fit(X_train, y_train)
        tres = self.get_accuracy_treshold(X_train, y_train)
        if cfg.dtype == 'df':
            y_predict = self.predict(X_test)
            self.evaluate_and_write_report(y_test, y_predict, tres)
            self.plot_CV_heatmap()
            self.plot_boosting_learning_curve(X_test, y_test)

            with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_CVresults.pkl"), "wb") as f:
                pickle.dump(
                    {"classifier": self.cv_clf, "features": self.kmer_to_index, "y_pred": y_predict, "y_true": y_test},
                    f,
                    protocol=4)



class Test_streaming(object):
    """
    Test in stream settings
    Also optionally prune the classifier discarding trees that use
    """
    def __init__(self, kmer_to_index=None, clf=None, batchsize=10):
        self.batchsize = 1

        self.testdir = os.path.join(cfg.pathtodata, cfg.testdir)
        self.kmer_to_index = kmer_to_index
        try:
            self.clf = clf.best_estimator_
        except Exception as e:
            print(e)
            self.clf = clf
        self.pathtotemp = os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, "test-temp")
        self.pathtosave = os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, "test-output")

        if not (os.path.isdir(self.pathtotemp) and os.path.isdir(self.pathtosave)):
            mkdirCmd = "mkdir %s" % (self.pathtotemp)
            os.system(mkdirCmd)
            mkdirCmd = "mkdir %s" % (self.pathtosave)
            os.system(mkdirCmd)

    def create_sparse_coos(self, cols, rows, datas, col, row, data):
        cols.extend(col)
        rows.extend(row)
        datas.extend(data)
        return cols, rows, datas

    def prune_boosting(self):
        import difflib

        # Select index of redundant kmers with lower importances
        ls_index = []
        featimp = self.clf.feature_importances_
        kmers = [list(self.kmer_to_index.keys())[i] for i in np.nonzero(featimp)[0]]
        imps = [featimp[i] for i in np.nonzero(featimp)[0]]
        index = [i for i in np.nonzero(featimp)[0]]
        for kmer1, imp1, ind1 in zip(kmers, imps, index):
            for kmer2, imp2, ind2 in zip(kmers, imps, index):
                similarity = difflib.SequenceMatcher(None, kmer1, kmer2).ratio()
                if similarity > cfg.pruning_tresh and kmer1 != kmer2:
                    if imp1 > imp2:
                        ls_index.append(ind2)
                    elif imp2 > imp1:
                        ls_index.append(ind1)

        return list(set(ls_index))  # list of redundant kmer indexes

    def predict_pruned(self, X_test,
                       ls_index):  # here ls_index is assumed to be a list of trees to not take into account -> issue
        cumpred = np.array([x for x in self.clf.staged_decision_function(X_test)])[:, :, 0]
        preds_out = cumpred[-1, :]

        for i in ls_index:  # i can't be 0 but who would prune first tree of boosting
            preds_out = preds_out - (cumpred[i - 1, :] - cumpred[i, :])
        return preds_out

    def populate_sparse_matrix(self, cols, rows, datas, batch):
        X_test = sp.csr_matrix((datas, (rows, cols)), shape=(batch, len(self.kmer_to_index)), dtype=np.int16)
        return X_test

    def append_prediction(self, X_test, y_preds, y_pruned, y_test, ls_index, y):
        y_preds.extend(self.clf.predict_proba(X_test))
        # y_pruned.extend(self.predict_pruned(X_test, ls_index))
        y_test.append(y)

        return y_preds, y_pruned, y_test

    def parse_and_map_kmers(self, fastaname, batchnumber):
        self.parse_kmers_dsk(fastaname)
        y = fastaname[:5]
        kmer_count = self.get_kmer_counts(fastaname)
        cols, rows, datas = self.map_data_to_coords(kmer_count, batchnumber)
        return cols, rows, datas, y

    def get_kmer_counts(self, fastaname):
        kmer_count = {}
        with open(os.path.join(self.pathtosave, fastaname), "r") as fasta:
            lines = fasta.readlines()
            for line in lines:
                try:
                    [ID, count] = line.split(" ")
                    kmer_count[str(ID)] = int(count)
                except:
                    print('line = {0}'.format(line))
        return kmer_count

    def map_data_to_coords(self, kmer_count, batchnumber):
        rows = []
        data = []
        columns = []

        for kmer in kmer_count:
            try:
                columns.append(self.kmer_to_index[kmer])
                rows.append(batchnumber)
                if cfg.kmer_count == 1:
                    data.append(kmer_count[kmer])
                else:
                    data.append(1)
            except:
                self.missing_kmers.append(kmer)

        return columns, rows, data

    def parse_kmers_dsk(self, fastaname):
        kmerCmd = "dsk -file %s -out %s -kmer-size %d -abundance-min 1 -verbose 0" % (
            os.path.join(self.testdir, fastaname), os.path.join(self.pathtotemp, fastaname), cfg.len_kmers)
        os.system(kmerCmd)
        outputCmd = "dsk2ascii -file %s -out  %s" % (
            os.path.join(self.pathtotemp, fastaname), os.path.join(self.pathtosave, fastaname))
        os.system(outputCmd)

    def adapted_accuracy(self, y_test, y_preds, tres):

        y_preds_adapted = []
        for pred in y_preds[:, -1]:
            if pred > float(tres):
                y_preds_adapted.append(1.0)
            else:
                y_preds_adapted.append(0.0)
        score = metrics.accuracy_score(y_test, y_preds_adapted)
        print(score)
        return score

    def evaluate_and_write_report(self, y_preds, y_test, tres=None, pruned=False):
        le = preprocessing.LabelEncoder()
        y_test = le.fit_transform(y_test)

        self.score = {}
        self.score["ROC_AUC"] = metrics.roc_auc_score(y_test, y_preds[:, -1])
        if tres==None:
            with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_tres_value.txt"), "r") as f:
                tres = f.readlines()[0]
        self.score["Accuracy"] = self.adapted_accuracy(y_test, y_preds, tres)
        self.score["Accuracy"] = metrics.accuracy_score(y_test, y_preds[:, -1].round())
        self.score["MAE"] = metrics.mean_absolute_error(y_test, y_preds[:, -1])
        self.score["MSE"] = metrics.mean_squared_error(y_test, y_preds[:, -1])
        print("*** ROC AUC = ***")
        print(self.score["ROC_AUC"])

        self.write_report(pruned)

    def write_report(self, pruned=False):
        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_report.txt"), "a") as txt:
            if pruned:
                txt.write('PRUNED' + "\n\n")
            txt.write(cfg.xp_name + "/" + cfg.id + "\n\n")
            txt.write(str(self.score) + "\n")
            txt.write("Len_kmers = " + str(cfg.len_kmers) + "\n")
            txt.write("Model = " + str(self.clf) + "\n")
            # txt.write("Best_params = "+str(self.clf.best_params_)+"\n")
            # txt.write("Param_grid = " + str(self.param_grid) + "\n")
            # txt.write("best params = " + str(self.clf.best_params_)+'\n')
            txt.write("\n Relevant kmers : \n")
            if cfg.model == "rf" or cfg.model == "gradient":
                featimp = self.clf.feature_importances_
                kmers = [list(self.kmer_to_index.keys())[i] for i in np.nonzero(featimp)[0]]
                for kmer in kmers:
                    txt.write(str(kmer) + "\n")

    def clean_temp_directories(self):
        cleankmertempcmd = "rm -rf %s" % (self.pathtotemp)
        os.system(cleankmertempcmd)
        cleantempcmd = "rm -rf %s" % (self.pathtosave)
        os.system(cleantempcmd)

    def run(self):
        self.missing_kmers = []
        files = [file for file in os.listdir(self.testdir)]
        remaining = len(files)
        fileindex = 0
        y_test = []
        y_preds = []
        y_pruned = []

        ls_index = self.prune_boosting()
        while remaining > 0:
            batchiter = 0
            batch = min(remaining, self.batchsize)
            for file in files[fileindex: fileindex + batch]:
                cols = []
                rows = []
                datas = []
                col, row, data, y = self.parse_and_map_kmers(file, batchiter)
                cols, rows, datas = self.create_sparse_coos(cols, rows, datas, col, row, data)
                y = file[:5]
                batchiter += 1
                remaining -= 1

                X_test = self.populate_sparse_matrix(cols, rows, datas, batchiter)
                try:
                    y_preds, y_pruned, y_test = self.append_prediction(X_test, y_preds, y_pruned, y_test, ls_index, y)
                except Exception as e:
                    print('exception')
                    print(e)
                print(y_test)
                print(y_preds)
            fileindex += batch

        y_preds = np.vstack(y_preds)
        self.evaluate_and_write_report(y_preds, y_test)
        self.y_preds = y_preds
        self.y_test = y_test
        self.X_test = X_test
        # self.evaluate_and_dump(y_pruned, y_test, pruned=True)
        print(ls_index)
        # self.write_report()
        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_CVresults.pkl"), "wb") as f:
            pickle.dump(
                {"classifier": self.clf, "features": self.kmer_to_index, "y_pred": y_preds, "y_pruned": y_pruned,
                 "y_true": y_test, "score": self.score},
                f, protocol=4)
        self.clean_temp_directories()

    def dump_eval(self, y_test, y_predict):
        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_CVresults.pkl"), "wb") as f:
            pickle.dump(
                {"classifier": self.cv_clf, "features": self.kmer_to_index, "y_pred": y_predict, "y_true": y_test}, f,
                protocol=4)

# train=Train_kmer_clf()
# train.run()
#
#
# #with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f'{cfg.model}_CVresults.pkl'), 'rb') as f:
# #    dic=pickle.load(f)
# #test=Test_streaming(batchsize=1, kmer_to_index=dic['features'], clf=dic['classifier'])
# test = Test_streaming(batchsize=1, kmer_to_index=train.kmer_to_index, clf=train.cv_clf)
# test.run()
