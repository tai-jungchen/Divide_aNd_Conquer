"""
Author: Alex (Tai-Jung) Chen

Test the performance of binary method.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, \
    precision_recall_fscore_support, cohen_kappa_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
from imblearn.metrics import specificity_score


def binary(model: object, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,
           verbose: bool=False) -> pd.DataFrame:
    """
    Carry out the binary method. The data will be partitioned and given from main.py to ensure identical train test
    split is used between different method.

    :param model: classifier.
    :param X_train: training data.
    :param X_test: testing data.
    :param y_train: training label.
    :param y_test: testing label.

    :param verbose: whether to print out the confusion matrix or not.

    :return: the dataframe with the classification metrics.
    """
    # metrics
    record_metrics = ['model', 'method', 'acc', 'kappa', 'bacc', 'precision', 'recall', 'specificity', 'f1', 'auc',
                      'apr']
    metrics = {key: [] for key in record_metrics}

    model.fit(X_train, y_train)

    # testing #
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    if verbose:
        print(f'Binary {model}')
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
    metrics['auc'].append(round(roc_auc_score(y_test, y_prob), 4))
    metrics['apr'].append(round(average_precision_score(y_test, y_prob), 4))

    metrics['model'].append(model)
    metrics['method'].append("binary")

    return pd.DataFrame(metrics)
