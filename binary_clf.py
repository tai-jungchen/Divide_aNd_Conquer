"""
Author: Alex (Tai-Jung) Chen

Test the performance of binary method.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, \
    precision_recall_fscore_support


def binary(model_type: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) \
        -> pd.DataFrame:
    """
    Carry out the binary method. The data will be partitioned and given from main.py to ensure identical train test
    split is used between different method.

    :param model_type: classifier.
    :param X_train: training data.
    :param X_test: testing data.
    :param y_train: training label.
    :param y_test: testing label.

    :return: the dataframe with the classification metrics.
    """
    # metrics
    df = pd.DataFrame()
    record_metrics = ['acc', 'bacc', 'f1', 'precision', 'recall', 'specificity']
    metrics = {key: [] for key in record_metrics}

    if model_type == 'XGBOOST':
        model = xgb.XGBClassifier()
    elif model_type == 'LR':
        model = LogisticRegression(max_iter=10000)
    elif model_type == 'DT':
        model = DecisionTreeClassifier()
    elif model_type == 'RF':
        model = RandomForestClassifier()
    else:
        raise Exception("Invalid model type.")

    model.fit(X_train, y_train)

    # testing #
    y_pred = model.predict(X_test)
    print(f'Binary {model_type}')
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print(classification_report(y_test, y_pred))

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

    return df
