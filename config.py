# PATHs
xp_name = 'pseud_levo_9'
data = 'pseud_levofloxacin/'

pathtoxp = '/home/ylucas/PycharmProjects/kmers_dummy/dummy_dataset/'
pathtodata='/home/ylucas/PycharmProjects/kmers_dummy/dummy_dataset/data'

# Kmer extraction parameters
len_kmers = 9  # If len_kmers<=8, gerbil will set len_kmers to 8
min_abundance = 3

# Learning parameters
model = 'gradient'  # can be ['rf','SCM', 'gradient']

rf_grid = {'max_features': ['sqrt', 'log2'],
           'max_depth': [4, 8]}

SCM_grid = {'model_type': ['disjunction'], 'max_rules': [10, 100]}

gradient_grid = {'learning_rate': [0.01, 0.1],
                 'n_estimators': [3, 5]}
