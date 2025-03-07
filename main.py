"""
Author: Alex (Tai-Jung) Chen

Run through all the classification framework for comparison purpose.
"""
from tqdm import tqdm
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek

from binary_clf import binary
from divide_n_conquer import divide_n_conquer
from multi_class_clf import multi_clf
from two_layer_hie import two_layer_hie
from ood_hie import ood_2hie, ood_3hie
from lda_aided_divide_n_conquer import LDAAidedDNC


def main(dataset: str, models: List[object], smote_lst: List, n_rep: int, metric: str) -> pd.DataFrame:
    """
    Run through all the methods for comparison.

    :param dataset: dataset for testing.
    :param models: types of models used for classification.
    :param n_rep: number of replications.
    {:param metric: the metric for smote tuning.
    :return results stored in the DataFrame.
    """
    res_df = pd.DataFrame()
    scaler = StandardScaler()

    # read data
    for i in tqdm(range(n_rep)):
        if dataset == 'MPMC':
            df = pd.read_csv("datasets/preprocessed/mpmc.csv")
            df = df[df['failure.type'] != 5]
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['failure.type'], test_size=0.3,
                                                                stratify=df['failure.type'], random_state=i)
        elif dataset == 'FAULTS':
            df = pd.read_csv("datasets/preprocessed/faults.csv")
            df = df[(df['failure.type'] == 0) | (df['failure.type'] == 4) | (df['failure.type'] == 5) | (df['failure.type'] == 6)]

            # make the labels continuous
            label_mapping = {0: 0, 4: 1, 5: 2, 6: 3}
            df['failure.type'] = df['failure.type'].map(label_mapping)

            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['failure.type'], test_size=0.3,
                                                                stratify=df['failure.type'], random_state=i)

        elif dataset == 'GLASS':
            df = pd.read_csv("datasets/preprocessed/glass_data.csv")
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['Type'], test_size=0.3,
                                                                stratify=df['Type'], random_state=i)
        elif dataset == 'MNIST':
            df = pd.read_csv("datasets/preprocessed/imb_mnist.csv")
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df['label'], test_size=0.3,
                                                                stratify=df['label'], random_state=i)
        elif dataset == 'USPS':
            df = pd.read_csv("datasets/preprocessed/imb_digit.csv")
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.3,
                                                                stratify=df.iloc[:, 0], random_state=i)
        else:
            raise Exception("Invalid dataset.")
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # run models
        for model in models:
            lda_dnc = LDAAidedDNC(model, smote_lst, metric)

            res_bin = binary(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy())

            res_dnc_lda = lda_dnc.fit(X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(),
                                               y_test.copy(), "DNC")
            res_ovo_lda = lda_dnc.fit(X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(),
                                               y_test.copy(), "OvO")
            # res_dnc = divide_n_conquer(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy())
            # res_dnc_smote = (divide_n_conquer(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(), smote=smote_inst_1))
            # res_twoLH = two_layer_hie(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(), verbose=True)
            # res_ood3H = ood_3hie(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(), verbose=True)
            # res_ood2H = ood_2hie(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(), verbose=True)

            res_ovo = multi_clf(model, "OvO", X_train_scaled, X_test_scaled, y_train, y_test)
            res_ovr = multi_clf(model, "OvR", X_train_scaled, X_test_scaled, y_train, y_test)
            res_dir = multi_clf(model, "Direct", X_train_scaled, X_test_scaled, y_train, y_test)
            res_df = pd.concat([res_df, res_bin, res_dnc_lda, res_ovo_lda, res_ovo, res_ovr, res_dir], axis=0)

    # average the performance
    return res_df.groupby(by=["method", "model"], sort=False).mean()


def generate_borderline_smote_list(dataset: str) -> list:
    """
    Generate the SMOTE instances. ADASYN(), SMOTE(), SMOTETomek() to be added.

    :param dataset: which dataset to be input. This will affect the SMOTE hyper-parameter.

    :return: a list of SMOTE instances.
    """
    smote_list = []
    if dataset == "MPMC":
        for os_num in range(100, 600, 100):
            for k in range(1, 6, 2):
                for m in range(1, 11, 2):
                    smote_list.append(BorderlineSMOTE(sampling_strategy={1: os_num, 2: os_num, 3: os_num, 4: os_num},
                                                      k_neighbors=k, m_neighbors=m, random_state=42))
    elif dataset == "FAULTS":
        for os_num in range(120, 300, 10):
            for k in range(1, 6, 2):
                for m in range(1, 11, 2):
                    smote_list.append(BorderlineSMOTE(sampling_strategy={1: os_num, 2: os_num, 3: os_num},
                                                      k_neighbors=k, m_neighbors=m, random_state=42))
    else:
        raise Exception("Invalid smote setting. Check the input dataset.")
    return smote_list


if __name__ == "__main__":
    ##### MPMC #####
    DATASET = "MPMC"
    MODELS = [LogisticRegression(penalty='l1', solver='saga', max_iter=2000),
              GaussianNB(),
              LDA(),
              SVC(kernel='linear', C=0.01),
              SVC(kernel='rbf'),
              DecisionTreeClassifier(),
              RandomForestClassifier(),
              GradientBoostingClassifier(),
              xgb.XGBClassifier()]
    ##### MPMC #####

    ##### FAULTS #####
    # DATASET = "FAULTS"
    # MODELS = [LogisticRegression(penalty='l1', solver='saga', max_iter=5000),
    #           GaussianNB(),
    #           LDA(),
    #           SVC(kernel='linear', C=1),
    #           SVC(kernel='rbf', C=1),
    #           DecisionTreeClassifier(),
    #           RandomForestClassifier(),
    #           GradientBoostingClassifier(),
    #           xgb.XGBClassifier()]
    ##### FAULTS #####

    ##### USPS #####
    # DATASET = "USPS"
    # MODELS = [LogisticRegression(max_iter=10000), GaussianNB(), LDA(), SVC(kernel='linear', C=0.1),
    #           SVC(kernel='rbf', C=0.5), DecisionTreeClassifier(), RandomForestClassifier(),
    #           GradientBoostingClassifier(random_state=42), xgb.XGBClassifier()]
    # SMOTE_INST_1 = BorderlineSMOTE(kind='borderline-1')
    ##### USPS #####

    ##### MNIST #####
    # DATASET = "MNIST"
    # MODELS = [LogisticRegression(max_iter=300), GaussianNB(), LDA(), #SVC(kernel='linear', C=0.1),
    #           SVC(kernel='rbf', C=0.5), DecisionTreeClassifier(), RandomForestClassifier(),
    #           GradientBoostingClassifier(random_state=42), xgb.XGBClassifier()]
    # SMOTE_INST_1 = BorderlineSMOTE(kind='borderline-1')
    ##### MNIST #####

    ##### GLASS #####
    # DATASET = "GLASS"
    # MODELS = [LogisticRegression(penalty='l1', solver='saga'), GaussianNB(), LDA(),  SVC(kernel='linear', C=1),
    #           SVC(kernel='rbf', C=1), DecisionTreeClassifier(), RandomForestClassifier(),
    #           GradientBoostingClassifier(), xgb.XGBClassifier()]
    # SMOTE_INST_1 = BorderlineSMOTE(kind='borderline-1')
    ##### GLASS #####

    N_REP = 3
    METRIC = "bacc"
    SMOTES = generate_borderline_smote_list(DATASET)

    res = main(DATASET, MODELS, SMOTES, N_REP, METRIC)
    filename = f"results_0306_{DATASET}.csv"
    res.to_csv(filename)

