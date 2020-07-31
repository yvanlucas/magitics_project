# PATHs
id='3'
xp_name = 'pseud_levo_31'

data = ''

pathtoxp = '/home/ylucas/Bureau/Data/PATRIC/'
pathtodata='/home/ylucas/Bureau/Data/PATRIC/levofloxacin/'

# Kmer extraction parameters
len_kmers = 31  # If len_kmers<=8, gerbil will set len_kmers to 8
min_abundance = 3

# Learning parameters
model = 'SCM'  # can be ['rf','SCM', 'gradient']

rf_grid = {'max_features': ['sqrt', 'log2'],
           'max_depth': [4, 8]}

SCM_grid = {'model_type': ['disjunction', 'conjunction'], 'max_rules': [3 ,10, 100]}

gradient_grid = {'max_depth': [1, 5],
                 'n_estimators': [10, 40, 70]}
