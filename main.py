import pyscm
import os
import pickle
import config as cfg
import data
import learning
#import memory_profiler


def create_sparseDB():
    """
    Main function to create a sparse kmer-count matrix
    """
    datas = data.Kmercount_to_matrix()
    datas.run()
    print('***Sparse matrix created***')

def create_geneIDsDF():
    """
    Main function to create a gene ID dataframe
    """
    datas=data.plfam_to_matrix()
    datas.run()
    print('***Dataframe created***')


def train_test_model_stream():
    """
     Main function to train and test in streaming
    """
    train=learning.Train_kmer_clf()
    train.run()
    #with open(os.path.join(cfg.pathtoxp, cfg.xp_name, cfg.id, f'{cfg.model}_CVresults.pkl'), 'rb') as f:
    #    dic=pickle.load(f)
    #test=learning.Test_streaming(batchsize=1, kmer_to_index=dic['features'], clf=dic['classifier'])
    test=learning.Test_streaming(batchsize=1, kmer_to_index=train.kmer_to_index, clf=train.cv_clf)
    test.run()

def train_test_model_batch():
    """
    Main function to train and test in batch
    """
    train=learning.Train_kmer_clf()
    train.run()


if __name__ == "__main__":
    if cfg.dtype=='df':
        create_geneIDsDF()
        train_test_model_batch()
    elif cfg.dtype=='sparse':
        create_sparseDB()
        train_test_model_stream()



