import Create_kmersDB as kmer
import resistance_prediction as pred
import config as cfg
import pyscm
from sklearn import ensemble

def create_DF():
    kmersDB = kmer.KmersCounts2Dataframe()
    kmersDB.iteratefastas()
    kmersDB.create_dataframe()  # Dataframe stored as a pickle
    print('***Dataframe created***')


def train_test_model():
    if cfg.model == 'rf':
        clf = ensemble.RandomForestClassifier()
        param_grid = cfg.rf_grid
    elif cfg.model == 'SCM':
        clf = pyscm.SetCoveringMachineClassifier()
        param_grid = cfg.SCM_grid
    elif cfg.model == 'gradient':
        clf = ensemble.GradientBoostingClassifier(max_depth=4, max_features=None)
        param_grid = cfg.gradient_grid

    expe = pred.ResistancePredictionkmers(classifier=clf, param_grid=param_grid)
    expe.run()



if __name__ == '__main__':
    create_DF()
    train_test_model()




