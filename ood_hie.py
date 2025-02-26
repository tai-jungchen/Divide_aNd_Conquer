"""
Author: Alex (Tai-Jung) Chen

Implement the ood-aid hierarchical method. The first layer is a one-class SVM that is learned on all normal cases and
will be able to tell if the testing sample belongs to normal or failure. The second layer is a multi-class model that
learns on the minority types only. The final layer is consisted by all local models and the sample will be sent to
the corresponding local model to be classified into finer-grained types.
"""
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.metrics import specificity_score
from sklearn import clone
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, \
    balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsOneClassifier


def ood_2hie(model: object, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
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
    y_train_bin = y_train.copy()
    y_train_bin[y_train_bin != 0] = 1

    # ood_model = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.95).fit(X_train[y_train == 0], y_train[y_train == 0])
    ood_model = clone(model)
    ood_model.fit(X_train, y_train_bin)

    # testing
    y_pred_ood = ood_model.predict(X_test)

    y_preds = []
    for sub in range(1, int(y_train.nunique())):
        y_pred_sub = np.full((len(y_test), ), -1)
        local_model = clone(model)
        # select only majority and minority sub
        X_train_local = X_train[(y_train == sub) | (y_train == 0)]
        y_train_local = y_train[(y_train == sub) | (y_train == 0)]
        y_train_local[y_train_local != 0] = 1  # turn non-zero sub minority into 1

        local_model.fit(X_train_local, y_train_local)

        y_pred_sub[np.where(y_pred_ood == 1)] = local_model.predict(X_test.iloc[y_pred_ood == 1])
        y_pred_sub[np.where(y_pred_ood == 0)] = 0
        # y_pred_sub[np.where(y_pred_ood == -1)] = local_model.predict(X_test.iloc[y_pred_ood == -1])
        # y_pred_sub[np.where(y_pred_ood == 1)] = 0
        y_preds.append(y_pred_sub)

    # voting
    y_preds = np.array(y_preds)
    y_pred = np.where(np.sum(y_preds, axis=0) > 0, 1, 0)

    y_test[y_test != 0] = 1  # turn all sub minority into 1
    if verbose:
        print(f'ood 2 Layer Hierarchical {model}')
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
    metrics['method'].append("ood_2hie")

    return pd.DataFrame(metrics)


def ood_3hie(model: object, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
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
    # ood
    ood_model = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.95).fit(X_train[y_train == 0], y_train[y_train == 0])

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
    y_pred_ood = ood_model.predict(X_test)
    y_pred_sub = multi_model.predict(X_test) + 1

    y_pred = np.full((len(y_test), ), -1)
    for sub in range(1, int(y_train.nunique())):
        # failure cases
        mdl = f"local_model_{sub}"
        indices = np.where((y_pred_sub == sub) & (y_pred_ood == -1))
        y_pred[indices[0]] = local_models[mdl].predict(X_test.iloc[indices[0], :])
    y_pred[np.where(y_pred == -1)] = 0

    y_test[y_test != 0] = 1  # turn all sub minority into 1
    if verbose:
        print(f'ood 3 Layer Hierarchical {model}')
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
    metrics['method'].append("ood_3hie")

    return pd.DataFrame(metrics)

