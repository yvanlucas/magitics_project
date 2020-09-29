import argparse
# PATHs
id='8'

xp_name = 'pseud_levo_31'
#xp_name = 'esche_amox_31'
#xp_name='acineto_imip_20'

mode='local' #can be ['local', 'serv', 'laptop']
dtype='df' #can be ['df','sparse']

if mode == 'serv':
    pathtoxp = '/mnt/cbib/MAGITICS_Yvan/experiments_kmer_count/'
    pathtodata='/scratch/MAGITICS_data/'
    #testdir='Pseudomonas_aeruginosa/levofloxacin/test/test'
#    testdir='Escherichia_coli/test/test'
#    data='Escherichia_coli/traindata/'
    #data = 'Pseudomonas_aeruginosa/levofloxacin/traindata/'
    data='Acinetobacter_baumanii/traindata/'
    testdir='Acinetobacter_baumanii/test/test/'
elif mode == 'local':
    pathtoxp = '/home/ylucas/toydata_pseudomonas_levofloxacin/'
    pathtodata='/home/ylucas/toydata_pseudomonas_levofloxacin/'
    data='traindatabis'
    testdir='test/test'
elif mode == 'laptop':
    pathtoxp='/home/ylucas/Bureau/expe_postdoc/xp'
    pathtodata='/home/ylucas/Bureau/expe_postdoc/data_postdoc/'
    data= ''
    testdir='test'

# Kmer extraction parameters

min_abundance = 3 #not used atm
kmer_count=1 #1: kmer count, 0: presence/absence

len_kmers=20
# Learning parameters
model = 'gradient'  # can be ['rf','SCM', 'gradient', 'Ada']

rf_grid = {'max_features': ['sqrt', 'log2'],
           'max_depth': [4, 8]}

SCM_grid = {'p': [1, 10], 'max_rules': [1, 3 ,10], 'model_type':['conjunction','disjunction']}

gradient_grid = {'max_depth': [1, 5],
                 'n_estimators': [10, 1]}

ada_grid =  {'n_estimators': [ 5, 10, 20]}


pruning_tresh=0.9

def get_lenkmers():
    parser=argparse.ArgumentParser()
    parser.add_argument('--len_kmers', type=int, default=31)
    arg=parser.parse_args()
    print(arg.len_kmers)
    return arg.len_kmers

len_kmers= 20 #int(get_lenkmers())

