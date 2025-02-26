"""
Author: Alex (Tai-Jung) Chen

This code implements the proposed DNC method. DNC uses partial OvO and customized decision rules in voting to cope with
imbalance data classification with subclass information available in the minority class.
"""
import itertools
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from imblearn.metrics import specificity_score
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, \
    balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def divide_n_conquer(model: object, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
                     y_test: pd.DataFrame, smote_param: dict = None, verbose: bool = False) -> pd.DataFrame:
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

    y_train_preds = []
    y_preds = []
    for sub in range(1, int(y_train.nunique())):
        local_model = clone(model)
        # select only majority and minority sub
        X_train_local = X_train[(y_train == sub) | (y_train == 0)]
        y_train_local = y_train[(y_train == sub) | (y_train == 0)]
        y_train_local[y_train_local != 0] = 1  # turn non-zero sub minority into 1

        # smote #
        if smote_param:
            opt_smote_param = smote_tuning(X_train_local, y_train_local, smote_param, clone(model))
            print(f"local model {sub}'s opt_smote_param: {opt_smote_param}")
            smote = SMOTE(sampling_strategy={1: opt_smote_param['sampling_strategy']}, k_neighbors=opt_smote_param[
                'k_neighbors'], random_state=21)
            X_final, y_final = smote.fit_resample(X_train_local, y_train_local)
            local_model.fit(X_final, y_final)

            # Visualize the new class distribution
            # pca = PCA()
            # transformed_data = pca.fit_transform(X_final)
            # plt.scatter(transformed_data[y_final == 0, 0], transformed_data[y_final == 0, 1], c='blue', label='Pass')
            # plt.scatter(transformed_data[y_final == 1, 0], transformed_data[y_final == 1, 1], c='red', label='Failure type 1')
            # plt.show()
        else:
            local_model.fit(X_train_local, y_train_local)

        # print("Local model performance:")
        # print(confusion_matrix(y_train_local, local_model.predict(X_train_local), labels=[0, 1]))
        # print(classification_report(y_train_local, local_model.predict(X_train_local)))

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
    if smote_param:
        metrics['method'].append(f"dnc_smote")
    else:
        metrics['method'].append(f"dnc")

    return pd.DataFrame(metrics)


def smote_tuning(X: pd.DataFrame, y: pd.Series, param_grid: dict, mdl: object, k_fold=5) -> dict:
    """
    Perform the grid search for smote.

    :param X:
    :param y:
    :param param_grid: Dictionary that stores the hyper-parameter for SMOTE
    :param mdl: model to be fit
    :param k_fold: number of fold

    :return: Optimal SMOTE hyper-parameter.
    """
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()

    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    cv_scores = {}
    for p in combinations:
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=21)
        zs = 0
        for train_index, test_index in skf.split(X, y):
            smote = SMOTE(sampling_strategy={1: p['sampling_strategy']}, k_neighbors=p['k_neighbors'], random_state=21)
            X_train, y_train = smote.fit_resample(X.iloc[train_index], y.iloc[train_index])
            mdl.fit(X_train, y_train)
            zs += f1_score(y.iloc[test_index], mdl.predict(X.iloc[test_index]))
        cv_scores[frozenset(p.items())] = zs / k_fold
    return dict(max(cv_scores, key=cv_scores.get))

