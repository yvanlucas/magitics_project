# PATHs
id='8'
xp_name = 'pseud_levo_31'

data = 'Pseudomonas_aeruginosa/levofloxacin/'

pathtoxp = '/mnt/cbib/MAGITICS_Yvan/experiments_kmer_count/'
pathtodata='/scratch/MAGITICS_data/'

# Kmer extraction parameters
len_kmers = 31  # If len_kmers<=8, gerbil will set len_kmers to 8
min_abundance = 3 #not used atm

# Learning parameters
model = 'gradient'  # can be ['rf','SCM', 'gradient']

rf_grid = {'max_features': ['sqrt', 'log2'],
           'max_depth': [4, 8]}

SCM_grid = {'p': [1, 10], 'max_rules': [1, 3 ,10], 'model_type':['conjunction','disjunction']}

gradient_grid = {'max_depth': [1, 5],
                 'n_estimators': [10, 40, 70]}
