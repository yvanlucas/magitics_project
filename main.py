import pyscm
from sklearn import ensemble

import config as cfg
import data
import learning


def create_trainDB():
    datas = data.Kmercount_to_matrix()
    datas.run()
    print('***Dataframe created***')



def train_test_model():
    if cfg.model == "rf":
        clf = ensemble.RandomForestClassifier()
        param_grid = cfg.rf_grid
    elif cfg.model == "SCM":
        clf = pyscm.SetCoveringMachineClassifier()
        param_grid = cfg.SCM_grid
    elif cfg.model == "gradient":
        clf = ensemble.GradientBoostingClassifier(max_depth=4, max_features=None)
        param_grid = cfg.gradient_grid

    train=learning.Train_kmer_clf()
    train.run(evaluate=False)

    test=learning.Test_streaming(batchsize=10, kmer_to_index=train.kmer_to_index, clf=train.cv_clf)
    test.run()


if __name__ == "__main__":
    create_trainDB()
    train_test_model()

