import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyscm
import scipy.sparse as sp
import seaborn as sns
from sklearn import (ensemble, feature_selection, metrics, model_selection, preprocessing)
import config as cfg


class Train_kmer_clf(object):
    def __init__(self, dataframe=None, clf=None, param_grid=None):

        if cfg.model == "rf":
            clf = ensemble.RandomForestClassifier()
            param_grid = cfg.rf_grid
        elif cfg.model == "SCM":
            clf = pyscm.SetCoveringMachineClassifier()
            param_grid = cfg.SCM_grid
        elif cfg.model == "gradient":
            clf = ensemble.GradientBoostingClassifier(max_depth=4, max_features=None)
            param_grid = cfg.gradient_grid

        self.mat = dataframe
        self.clf = clf
        self.param_grid = param_grid
        # if self.dataframe == None:
        #     table=pq.read_table(os.path.join(cfg.pathtoxp, cfg.xp_name, 'kmers_DF.parquet'))
        #     self.dataframe=table.to_pandas()
        #     self.dataframe.transpose()

        with open(os.path.join(cfg.pathtoxp, cfg.xp_name,cfg.id , "kmers_mats.pkl"), "rb") as f:
            [self.mat, self.labels, self.strain_to_index, self.kmer_to_index] = pickle.load(f)


    def preprocess(self):
        # to_drop = ["label", "strain"]
        # X = self.mat.drop(to_drop, axis=1)
        # self.columns = X.columns
        self.le = preprocessing.LabelEncoder()
        self.y = self.le.fit_transform(self.labels)

    def split_train_test(self, testratio=0.2):
        if testratio > 0:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(self.mat, self.y, test_size=testratio)
        else:
            X_train = self.mat
            y_train = self.y
            X_test = None
            y_test = None
        del self.mat, self.y
        return X_train, X_test, y_train, y_test

    def chi2_feature_selection(self, X_train, X_test, y_train):
        self.chi2_selector = feature_selection.SelectKBest(feature_selection.chi2, k=1000000)
        X_train = self.chi2_selector.fit_transform(X_train, y_train)
        X_test = self.chi2_selector.transform(X_test)
        return X_train, X_test

    def fit(self, X_train, y_train):
        self.cv_clf = model_selection.GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=2,
                                                   scoring="accuracy", n_jobs=4)
        self.cv_clf.fit(X_train, y_train)

        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f'{cfg.model}_CVresults.pkl'), 'wb') as f:
            pickle.dump({"classifier": self.cv_clf, "features": self.kmer_to_index}, f, protocol=4)


    def predict(self, X_test):
        y_predict = self.cv_clf.predict_proba(X_test)
        return y_predict

    def prediction_scores(self, y_test, y_pred):

        self.score = {}
        self.score["ROC_AUC"] = metrics.roc_auc_score(y_test, y_pred[:, 1])
        self.score["Accuracy"] = metrics.accuracy_score(y_test, y_pred[:, 1].round())
        self.score["MAE"] = metrics.mean_absolute_error(y_test, y_pred[:, 1])
        self.score["MSE"] = metrics.mean_squared_error(y_test, y_pred[:, 1])
        print("*** ROC AUC = ***")
        print(self.score["ROC_AUC"])

    def plot_CV_heatmap(self):
        # Heatmap for GridSearchCV
        ls_params = list(self.param_grid.keys())
        self.pvt = pd.pivot_table(pd.DataFrame(self.cv_clf.cv_results_), values="mean_test_score",
                                  index="param_" + ls_params[0], columns="param_" + ls_params[1])
        ax = sns.heatmap(self.pvt)
        ax.set(ylabel=ls_params[0], xlabel=ls_params[1])
        ax.figure.savefig(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_gridCV_heatmap.png"))

    def plot_boosting_learning_curve(self, X_test, y_test):
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

    def write_report(self):
        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_report.txt"), "w") as txt:
            txt.write(cfg.xp_name + "\n\n")
            txt.write(str(self.score) + "\n")
            txt.write("Len_kmers = " + str(cfg.len_kmers) + "\n")
            txt.write("Model = " + str(self.clf) + "\n")
            txt.write("Param_grid = " + str(self.param_grid) + "\n")
            txt.write("\n Relevant kmers : \n")
            if cfg.model == "rf" or cfg.model == "gradient":
                featimp = self.cv_clf.best_estimator_.feature_importances_
                kmers = [list(self.kmer_to_index.keys())[i] for i in np.nonzero(featimp)[0]]
                for kmer in kmers:
                    txt.write(str(kmer) + "\n")

    def dump_eval(self, y_test, y_predict):
        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_CVresults.pkl"), "wb") as f:
            pickle.dump(
                {"classifier": self.cv_clf, "features": self.kmer_to_index, "y_pred": y_predict, "y_true": y_test}, f,
                protocol=4)

    def run(self, evaluate=True):
        self.preprocess()
        X_train, X_test, y_train, y_test = self.split_train_test(testratio=0.0)
#        X_train, X_test = self.chi2_feature_selection(X_train, X_test, y_train)

        self.fit(X_train, y_train)

        if evaluate:
            y_predict = self.predict(X_test)

            self.prediction_scores(y_test, y_predict)
            self.plot_CV_heatmap()
            self.plot_boosting_learning_curve(X_test, y_test)

            self.write_report()
            self.dump_eval(y_train, y_predict)


class Test_streaming(object):
    def __init__(self, kmer_to_index=None, clf=None, batchsize=10):
        self.batchsize = 1
        self.testdir = os.path.join(cfg.pathtodata, cfg.testdir)
        self.kmer_to_index = kmer_to_index
        self.clf = clf.best_estimator_
        self.pathtotemp = os.path.join(cfg.pathtoxp,cfg.xp_name, "test-temp")
        self.pathtosave = os.path.join(cfg.pathtoxp, cfg.xp_name,"test-output")
        if not (os.path.isdir(self.pathtotemp) and os.path.isdir(self.pathtosave)):
            mkdirCmd = "mkdir %s" % (self.pathtotemp)
            os.system(mkdirCmd)
            mkdirCmd = "mkdir %s" % (self.pathtosave)
            os.system(mkdirCmd)

    def create_sparse_coos(self, cols, rows, datas, y_test, col, row, data, y):
        cols.extend(col)
        rows.extend(row)
        datas.extend(data)
        y_test.append(y)

        return cols, rows, datas, y_test

    def prune_boosting(self):
        import difflib

        # Select index of redundant kmers with lower importances
        ls_index=[]
        featimp = self.clf.feature_importances_
        kmers = [list(self.kmer_to_index.keys())[i] for i in np.nonzero(featimp)[0]]
        imps=[featimp[i] for i in np.nonzero(featimp)[0]]
        index=[i for i in np.nonzero(featimp)[0]]
        for kmer1, imp1,ind1 in zip(kmers, imps, index):
            for kmer2, imp2, ind2 in zip(kmers, imps, index):
                similarity=difflib.SequenceMatcher(None, kmer1, kmer2).ratio()
                if similarity >cfg.pruning_tresh and kmer1!= kmer2:
                    if imp1> imp2:
                        ls_index.append(ind2)
                    elif imp2>imp1:
                        ls_index.append(ind1)

        return list(set(ls_index))

    def predict_pruned(self, X_test, ls_index):
        cumpred=np.array([x for x in self.clf.staged_decision_function(X_test)])[:,:,0]
        preds_out = cumpred[-1, :]
        for i in ls_index:  # i can't be 0 but who would prune first tree of boosting
            preds_out = preds_out - (cumpred[i - 1, :] - cumpred[i, :])
        return preds_out

    def populate_sparse_matrix_and_append_prediction(self, cols, rows, datas, y_preds, y_pruned, batch, ls_index):
        X_test = sp.csr_matrix((datas, (rows, cols)), shape=(batch, len(self.kmer_to_index)), dtype=np.int8)
        y_preds.extend(self.clf.predict_proba(X_test)[:, 1])
        y_pruned.extend(self.predict_pruned(X_test, ls_index))
        return y_preds, y_pruned


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
                    print("line = " + line)
        return kmer_count

    def map_data_to_coords(self, kmer_count, batchnumber):
        rows = []
        data = []
        columns = []

        for kmer in kmer_count:
            try:
                columns.append(self.kmer_to_index[kmer])
                rows.append(batchnumber)
                data.append(kmer_count[kmer])
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

    def evaluate_and_dump(self, y_preds, y_test, pruned=False):
        le = preprocessing.LabelEncoder()
        y_test = le.fit_transform(y_test)

        y_preds = np.vstack(y_preds)
        self.score = {}
        self.score["ROC_AUC"] = metrics.roc_auc_score(y_test, y_preds)
        self.score["Accuracy"] = metrics.accuracy_score(y_test, y_preds.round())
        self.score["MAE"] = metrics.mean_absolute_error(y_test, y_preds)
        self.score["MSE"] = metrics.mean_squared_error(y_test, y_preds)
        print("*** ROC AUC = ***")
        print(self.score["ROC_AUC"])

        self.write_report(pruned)


    def write_report(self, pruned=False):
        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_report.txt"), "a") as txt:
            if pruned:
                txt.write('PRUNED' + "\n\n")
            txt.write(cfg.xp_name +"/" +cfg.id + "\n\n")
            txt.write(str(self.score) + "\n")
            txt.write("Len_kmers = " + str(cfg.len_kmers) + "\n")
            txt.write("Model = " + str(self.clf) + "\n")
            #txt.write("Best_params = "+str(self.clf.best_params_)+"\n")
            #txt.write("Param_grid = " + str(self.param_grid) + "\n")
            #txt.write("best params = " + str(self.clf.best_params_)+'\n')
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
        y_pruned =[]

        ls_index=self.prune_boosting()
        while remaining > 0:
            batchiter = 0
            batch = min(remaining, self.batchsize)

            datas = []
            print(batch)
            for file in files[fileindex: fileindex + batch]:
                try:
                    cols = []
                    rows = []
                    col, row, data, y = self.parse_and_map_kmers(file, batchiter)
                    cols, rows, datas, y_test = self.create_sparse_coos(cols, rows, datas, y_test, col, row, data, y)
                    batchiter += 1
                    remaining -= 1
                    y_preds, y_pruned = self.populate_sparse_matrix_and_append_prediction(cols, rows, datas, y_preds,
                                                                                          y_pruned, batchiter, ls_index)
                except:
                    print('issue with testing file: '+file)
                    remaining -=1
                print(np.shape(y_test))
                print(np.shape(y_preds))

            fileindex += batch


        self.evaluate_and_dump(y_preds, y_test)
        self.evaluate_and_dump(y_pruned, y_test, pruned=True)
        #self.write_report()
        with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_CVresults.pkl"), "wb") as f:
            pickle.dump({"classifier": self.clf, "features": self.kmer_to_index, "y_pred": y_preds, "y_pruned":y_pruned,"y_true": y_test, "score":self.score},
                        f, protocol=4)
        self.clean_temp_directories()

# if cfg.model == "rf":
#     clf = ensemble.RandomForestClassifier()
#     param_grid = cfg.rf_grid
# elif cfg.model == "SCM":
#     clf = pyscm.SetCoveringMachineClassifier()
#     param_grid = cfg.SCM_grid
# elif cfg.model == "gradient":
#     clf = ensemble.GradientBoostingClassifier(max_depth=4, max_features=None)
#     param_grid = cfg.gradient_grid
# expe = ResistancePredictionkmers(classifier=clf, param_grid=param_grid)
# expe.run(evaluate=False)
#
# with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f"{cfg.model}_CVresults.pkl"), "rb") as f:
#    dic = pickle.load(f)
#    clf = dic['classifier']
#    kmer_to_index = dic['features']
#    print('loaded')
#
# test = TestStreamingBatch(clf=clf, kmer_to_index=kmer_to_index)
# test.run()
