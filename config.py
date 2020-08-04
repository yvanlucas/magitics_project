# PATHs
id='4'
xp_name = 'pseud_levo_31'

data = ''

pathtoxp = '/home/ylucas/toydata_pseudomonas_levofloxacin/'
pathtodata='/home/ylucas/toydata_pseudomonas_levofloxacin/data/'

# Kmer extraction parameters
len_kmers = 31  # If len_kmers<=8, gerbil will set len_kmers to 8
min_abundance = 3

# Learning parameters
model = 'gradient'  # can be ['rf','SCM', 'gradient']

rf_grid = {'max_features': ['sqrt', 'log2'],
           'max_depth': [4, 8]}

SCM_grid = {'p': [0.5, 1, 10, 20], 'max_rules': [1, 3 ,10, 100], 'model_type':['conjunction','disjunction']}

gradient_grid = {'max_depth': [1, 5],
                 'n_estimators': [10, 40, 70]}
