#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 22:11:17 2025

@author: s.moka,a.mathur
"""

import numpy as np
import _opt_log as olog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score



# Load the CSV file (assuming no headers)
train_path = "./data/n-200-p1000Replica1.csv"
test_path = "./data/n-200-p1000Test-Replica1.csv"
#train_path = "./data/Example_small_data_train.csv"
#test_path = "./data/Example_small_data_test.csv"
data = np.loadtxt(train_path, delimiter='\t', skiprows=0)  # skiprows=1 if header exists
data_test = np.loadtxt(test_path, delimiter='\t', skiprows=0)  # skiprows=1 if header exists
# Split into X and y
y = data[:, 0]  # First column is the response
X = data[:, 1:]  # All remaining columns are features
y_test = data_test[:, 0]  # First column is the response
X_test = data_test[:, 1:]  # All remaining columns are features
q = 10 # number of features to select



import time
# Start timing
start_wall = time.time()
start_cpu = time.process_time()

# sklearn
result_sklearn = olog.fw(
    X, y, q, solver='sklearn', r=1.5, m=10,
    delta_min=0.02, cg_tol=0.00001, verbose=False)
sklearn_models = result_sklearn.models

# End timing
end_wall = time.time()
end_cpu = time.process_time()

# Print like %%time
print(f"CPU times: total {end_cpu - start_cpu:.3f} s")
print(f"Wall time: {end_wall - start_wall:.3f} s")


misclass_errors = []
for i, model in enumerate(lbfgs_models):
    error = misclassification_error(model, X, y, X_test, y_test, threshold=0.5, return_error=True)
    misclass_errors.append(error)
    print(f"Model {i+1}: Features {model}, Misclassification Error: {error:.4f}")

# Find the model with minimum misclassification error
min_error_idx = np.argmax(misclass_errors)
min_error = misclass_errors[min_error_idx]
best_model = lbfgs_models[min_error_idx]

# ---------------------------------------------------------------
# helper:     compute in-sample logistic loss for a list of models
#             • accepts 1-based feature indices and converts to 0-based
#             • uses *no* regularisation   (penalty=None → 'none')
# ---------------------------------------------------------------
def model_log_losses(X, y, model_index_sets, *, penalty=None, **lr_kw):
    """
    Parameters
    ----------
    X  : ndarray, shape (n_samples, n_features)
    y  : ndarray, shape (n_samples,)   – binary labels {0,1}
    model_index_sets : iterable[sequence[int]]
        Each element is a list/array of **1-based** feature indices.
    penalty : None | {"l1","l2","elasticnet","none"}
        None  →   scikit-learn’s 'none'  (i.e. no regularisation).
        Otherwise passed through unchanged.
    **lr_kw : extra keyword args for LogisticRegression
    """
    if penalty is None:
        penalty_arg = None
    else:
        penalty_arg = penalty

    losses = []
    for idxs1 in model_index_sets:
        idxs0 = [i - 1 for i in idxs1]  
        print(idxs0)    # convert 1-based → 0-based
        Xsub  = X[:, idxs0]

        clf = LogisticRegression(
            penalty=penalty_arg,
            solver="lbfgs",     # supports 'none'
            max_iter=10_000,
            **lr_kw
        ).fit(Xsub, y)

        p_hat  = clf.predict_proba(Xsub)[:, 1]
        losses.append(log_loss(y, p_hat))

    return np.array(losses)

def misclassification_error(model_idx1, Xtrain, ytrain, Xtest, ytest,
                            threshold=0.5, return_error=False):

    # Remove first column before selecting
    Xtrain = Xtrain[:, 1:]
    Xtest = Xtest[:, 1:]

    # Convert from R 1-based indices to Python 0-based
    model_idx1 = np.asarray(model_idx1, dtype=int)
    cols = model_idx1 - 1

    Xtr = Xtrain[:, cols]
    Xte = Xtest[:, cols]

    # Ensure 2D for single-column selection
    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(-1, 1)
        Xte = Xte.reshape(-1, 1)

    # No regularization to match glm
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    clf.fit(Xtr, ytrain)

    probs = clf.predict_proba(Xte)[:, 1]
    ypred = (probs > 0.5).astype(int)

    # Accuracy as in R's table-based calculation
    acc = np.mean(ytest == ypred)
    return acc