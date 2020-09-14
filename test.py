import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

X_all = np.random.randn(5000, 2)
y_all = ((X_all[:, 0]/X_all[:, 1]) > 1.5)*2 - 1
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.5, random_state=42)


clf = GradientBoostingClassifier(n_estimators=10, learning_rate=0.01, max_depth=3, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
print('Accuracy for a GBM: {}'.format(clf.score(X_test, y_test)))
print("Test logloss: {}".format(log_loss(y_test, y_pred)))

def compute_loss(y_true, scores_pred):
    return log_loss(y_true, sigmoid(scores_pred))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

cum_preds = np.array([x for x in clf.staged_decision_function(X_test)])[:, :, 0]

def prune_prediction(cum_pred, ls_index):
    preds_out=cum_preds[-1,:]
    for i in ls_index: #i can't be 0 but who would prune first tree of boosting
        preds_out=preds_out - (cum_preds[i-1,:]-cum_preds[i,:])
    return preds_out



