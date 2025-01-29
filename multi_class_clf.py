"""
Author: Alex (Tai-Jung) Chen

Implement the multi-class classification and collapse the classification to binary.
"""
import sys
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, \
    balanced_accuracy_score, precision_recall_fscore_support
from sklearn import tree, clone


def multi_clf(model: object, strat: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
              y_test: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Carry out the Direct or OvO method The classification results will be stored to a .csv file and the console
    information will be store to a .txt file.

    :param model: classifier.
    :param strat: multi-class classification strategy (OvO, OvR, or Direct)
    :param X_train: training data.
    :param X_test: testing data.
    :param y_train: training label. Note that these are the multi-class labels
    :param y_test: testing label. Note that these are the multi-class labels

    :param verbose: whether to print out the confusion matrix or not.

    :return: the dataframe with the classification metrics.
    """
    record_metrics = ['model', 'method', 'acc', 'kappa', 'bacc', 'precision', 'recall', 'specificity', 'f1']
    metrics = {key: [] for key in record_metrics}

    # Logistic Regression
    if isinstance(model, LogisticRegression):
        if strat == 'OvO':
            multi_model = OneVsOneClassifier(model)
        elif strat == 'OvR':
            # multi_model = clone(model).set_params(multi_class='ovr')
            multi_model = OneVsRestClassifier(model)
        elif strat == 'Direct':
            multi_model = clone(model).set_params(multi_class='multinomial')
    # RF / XGB / DT
    elif (isinstance(model, RandomForestClassifier) or isinstance(model, xgb.XGBClassifier)
          or isinstance(model, DecisionTreeClassifier)):
        if strat == 'OvO':
            multi_model = OneVsOneClassifier(model)
        elif strat == 'OvR':
            # multi_model = clone(model)
            multi_model = OneVsRestClassifier(model)
        elif strat == 'Direct':
            multi_model = clone(model)
    else:
        raise Exception(f"{model} not available in this package.")

    # training
    multi_model.fit(X_train, y_train)

    # testing
    y_pred_multi = multi_model.predict(X_test)
    y_pred = np.where(y_pred_multi > 0, 1, 0)
    y_test = np.where(y_test > 0, 1, 0)

    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    tn, fp, fn, tp = np.ravel(confusion_matrix(y_test, y_pred))
    spec = tn / (tn + fp)

    # Store performance
    metrics['acc'].append(round(acc, 4))
    metrics['bacc'].append(round(bacc, 4))
    metrics['precision'].append(round(precision[1], 4))
    metrics['recall'].append(round(recall[1], 4))
    metrics['specificity'].append(round(spec, 4))
    metrics['f1'].append(round(f1[1], 4))

    # print rep output
    print(f'Multi-class {model}')
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print(classification_report(y_test, y_pred))
