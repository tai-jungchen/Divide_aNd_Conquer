"""
Author: Alex (Tai-Jung) Chen

This code implements the lda aided DNC method. SMOTE is also applied before LDA dimension reduction.
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
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def divide_n_conquer_lda(model: object, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
                     y_test: pd.DataFrame, smote_param: dict = None, verbose: bool = False) -> pd.DataFrame:
    """
    Carry out the DNC method. The classification results.

    :param model: classifier.
    :param X_train: training data.
    :param X_test: testing data.
    :param y_train: training label.
    :param y_test: testing label.
    :param smote_param: SMOTE settings including SMOTE type, sampling strategy, and number of neighbors.
    :param verbose: whether to print out the confusion matrix or not.

    :return: the dataframe with the classification metrics.
    """
    record_metrics = ['model', 'method', 'f1', 'precision', 'recall', 'kappa', 'bacc', 'acc', 'specificity']
    metrics = {key: [] for key in record_metrics}

    ##### smote #####
    if smote_param:
        if smote_param['type'] == "SMOTE":
            smote = SMOTE(sampling_strategy=smote_param['sampling_strategy'], k_neighbors=smote_param['k_neighbors'],
                          random_state=521)
        elif smote_param['type'] == "Borderline":
            smote = BorderlineSMOTE(sampling_strategy=smote_param['sampling_strategy'], k_neighbors=smote_param[
                'k_neighbors'], kind='borderline-1', random_state=521)
        else:
            raise Exception("SMOTE type not exist!")
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
    ##### smote #####
    else:
        X_smote, y_smote = X_train.copy(), y_train.copy()

    # Fit LDA to training data and transform both train and test sets
    lda = LDA(n_components=y_smote.nunique() - 1)
    X_final = lda.fit_transform(X_smote, y_smote)
    X_test_lda = lda.transform(X_test)

    # Print explained variance ratio
    print("Explained variance ratio:", lda.explained_variance_ratio_)

    y_preds = []
    for sub in range(1, int(y_smote.nunique())):
        local_model = clone(model)

        # select only majority and minority sub
        X_train_local = X_final[(y_smote == sub) | (y_smote == 0)]
        y_train_local = y_smote[(y_smote == sub) | (y_smote == 0)]
        y_train_local[y_train_local != 0] = 1  # turn non-zero sub minority into 1

        local_model.fit(X_train_local, y_train_local)
        y_pred_sub = local_model.predict(X_test_lda)
        y_preds.append(y_pred_sub)

    # voting
    y_preds = np.array(y_preds)
    y_pred = np.where(np.sum(y_preds, axis=0) > 0, 1, 0)
    y_test[y_test != 0] = 1  # turn all sub minority into 1

    if verbose:
        print(f'DNC LDA with SMOTE {model}') if smote_param else print(f'DNC LDA {model}')
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
        if smote_param['type'] == "SMOTE":
            metrics['method'].append(f"dnc_lda_smote")
        elif smote_param['type'] == "Borderline":
            metrics['method'].append(f"dnc_lda_borderline")
        else:
            raise Exception("Invalid SMOTE type")
    else:
        metrics['method'].append(f"dnc_lda")
    return pd.DataFrame(metrics)

