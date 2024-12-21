"""
Author: Alex (Tai-Jung) Chen

This code implements the proposed DNC method. DNC uses partial OvO and customized decision rules in voting to cope with
imbalance data classification with subclass information available in the minority class.
"""
import numpy as np
import pandas as pd
from imblearn.metrics import specificity_score
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, \
    balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone


def divide_n_conquer(model: object, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
                     y_test: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Carry out the DNC method on the Machine Predictive Maintenance Classification dataset. The classification results
    will be stored to a .csv file and the console information will be store to a .txt file.

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

    y_preds = []
    for sub in range(1, int(y_train.nunique())):
        local_model = clone(model)
        # select only majority and minority sub
        X_train_local = X_train[(y_train == sub) | (y_train == 0)]
        y_train_local = y_train[(y_train == sub) | (y_train == 0)]
        y_train_local[y_train_local != 0] = 1  # turn non-zero sub minority into 1

        local_model.fit(X_train_local, y_train_local)

        y_pred_sub = local_model.predict(X_test)
        y_preds.append(y_pred_sub)

    # voting
    y_preds = np.array(y_preds)
    y_pred = np.where(np.sum(y_preds, axis=0) > 0, 1, 0)
    y_test[y_test != 0] = 1  # turn all sub minority into 1

    if verbose:
        print(f'DNC {model}')
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
    metrics['method'].append("dnc")

    return pd.DataFrame(metrics)
