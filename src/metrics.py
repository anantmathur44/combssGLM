"""
metrics.py

This private module contains logic for computing performance metrics for variable selection.
Metrics computed:
- Relative predition error
- Matthew's Correlation Coefficient 
- Accuracy
- Sensitivity
- Specificity
- F1 Score
- Precision
"""

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def classification_acc(model_idx1, Xtrain, ytrain, Xtest, ytest,
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



def performance_metrics(beta_true, beta_pred):
    

    """
    Computes MCC, Accuracy, Sensitivity, Specificity, and F1
    for COMBSS variable selection performance.
    """

    s_true = (beta_true != 0).astype(int).ravel()
    s_pred = (beta_pred != 0).astype(int).ravel()

    c_matrix = metrics.confusion_matrix(s_true, s_pred)
    TN, FP, FN, TP = c_matrix.ravel()

    acc = (TP + TN) / (TP + TN + FP + FN) if (TP+TN+FP+FN) > 0 else 0
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0

    # Handle degenerate cases safely
    if (TP + FP) == 0 or (TP + FN) == 0 or (TN + FP) == 0 or (TN + FN) == 0:
        mcc = 0.0
    else:
        mcc = (TP * TN - FP * FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    f1 = 2 * TP / (2 * TP + FP + FN) if TP > 0 else 0.0

    return {
        "mcc": mcc,
        "acc": acc,
        "sens": sens,
        "spec": spec,
        "f1": f1
    }

def find_best_model(models, X, y, X_test, y_test, threshold=0.5):
    misclass_errors = []
    
    for i, model in enumerate(models):
        error = classification_acc(
            model, X, y, X_test, y_test, threshold=threshold, return_error=True
        )
        misclass_errors.append(error)
        print(f"Model {i+1}: Features {model}, Classification Acc: {error:.4f}")

    # Get index of the model with minimum misclassification error
    max_error_idx = np.argmax(misclass_errors)
    max_error = misclass_errors[max_error_idx]
    best_model = lbfgs_models[max_error_idx]

    return len(best_model), max_error, best_model
