import Create_kmersDB as kmer
import resistance_prediction as pred


def create_DF():
    kmersDB = kmer.KmersCounts2Dataframe()
    kmersDB.iteratefastas()
    kmersDB.create_dataframe()  # Dataframe stored as a pickle
    print('***Dataframe created***')


def train_test_model():
    model = pred.Resistance_Prediction_kmers()
    model.preprocessing_dataframe()
    model.split_train_test()
    model.train_model()
    model.test_model()
    model.ROC_AUC_evaluation()


model = pred.Resistance_Prediction_kmers()
model.preprocessing_dataframe()
model.split_train_test()
model.train_model()
model.test_model()
model.ROC_AUC_evaluation()

# if __name__ == '__main__':
#     create_DF()
#     train_test_model()




