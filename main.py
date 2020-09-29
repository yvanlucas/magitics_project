import pyscm
from sklearn import ensemble
import os
import pickle
import config as cfg
import data
import learning
import memory_profiler


def create_sparseDB():
    datas = data.Kmercount_to_matrix()
    datas.run()
    print('***Sparse matrix created***')

def create_genelimitsDB():
    datas=data.plfam_to_matrix()
    datas.run()
    print('***Dataframe created***')


def train_test_model_stream():
    train=learning.Train_kmer_clf()
    train.run()
    #with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f'{cfg.model}_CVresults.pkl'), 'rb') as f:
    #    dic=pickle.load(f)
    #test=learning.Test_streaming(batchsize=1, kmer_to_index=dic['features'], clf=dic['classifier'])
    test=learning.Test_streaming(batchsize=1, kmer_to_index=train.kmer_to_index, clf=train.cv_clf)
    test.run()

def train_test_model_batch():
    train=learning.Train_kmer_clf()
    train.run()


if __name__ == "__main__":
    if cfg.dtype=='df':
        create_genelimitsDB()
        train_test_model_batch()
    elif cfg.dtype=='sparse':
        create_sparseDB()
        train_test_model_stream()
        learning.Train_kmer_clf.run()


