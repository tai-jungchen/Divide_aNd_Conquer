"""
Author: Alex (Tai-Jung) Chen

Implement the 2-level hierarchical method. This method utilizes a top layer which consist of a multi-class classifier
to determine which minority subclass the sample belongs to. After the top layer, the samples are assigned to the bottom
layer which is composed of several local models to take advantage of the benefit of the local models.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.metrics import specificity_score
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, balanced_accuracy_score, \
    precision_score, recall_score, cohen_kappa_score


def two_layer_hie(model: object, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
                  y_test: pd.DataFrame, verbose = False) -> pd.DataFrame:
    """
    Implement the two layer hierarchical method as described in the module comments above.

    :param model: classifier.
    :param X_train: training data.
    :param X_test: testing data.
    :param y_train: training label.
    :param y_test: testing label.

    :param verbose: whether to print out the confusion matrix or not.

    :return: the dataframe with the classification metrics.
    """

    record_metrics = ['model', 'method', 'acc', 'kappa', 'bacc', 'precision', 'recall', 'specificity', 'f1']
    metrics = {key: [] for key in record_metrics}

    # training
    # multi-class model for failure types
    multi_model = xgb.XGBClassifier().fit(X_train[y_train != 0], y_train[y_train != 0] - 1)

    # local models
    local_models = {}
    for sub in range(1, int(y_train.nunique())):
        local_model = clone(model)

        # select only majority and minority sub
        X_train_local = X_train[(y_train == sub) | (y_train == 0)]
        y_train_local = y_train[(y_train == sub) | (y_train == 0)]
        y_train_local[y_train_local != 0] = 1  # turn non-zero sub minority into 1
        local_model.fit(X_train_local, y_train_local)

        local_models[f"local_model_{sub}"] = model

    # testing
    y_pred_sub = multi_model.predict(X_test) + 1

    y_pred = np.full((len(y_test), ), -1)
    for sub in range(1, int(y_train.nunique())):
        mdl = f"local_model_{sub}"
        indices = np.where(y_pred_sub == sub)
        y_pred[indices[0]] = local_models[mdl].predict(X_test.iloc[indices[0], :])

    y_test[y_test != 0] = 1  # turn all sub minority into 1

    if verbose:
        print(f'Two Layer Hierarchical {model}')
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        print(classification_report(y_test, y_pred))

    # Store performance
    metrics['acc'].append(round(accuracy_score(y_test, y_pred), 4))
    metrics['kappa'].append(round(cohen_kappa_score(y_test, y_pred), 4))
    metrics['bacc'].append(round(balanced_accuracy_score(y_test, y_pred), 4))
    metrics['precision'].append(round(precision_score(y_test, y_pred), 4))
    metrics['recall'].append(round(recall_score(y_test, y_pred), 4))
    metrics['specificity'].append(round(specificity_score(y_test, y_pred), 4))
    metrics['f1'].append(round(f1_score(y_test, y_pred), 4))

    metrics['model'].append(model)
    metrics['method'].append("twoLayerHie")

    return pd.DataFrame(metrics)
