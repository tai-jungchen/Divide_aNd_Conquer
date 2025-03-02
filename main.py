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
from divide_n_conquer_lda import divide_n_conquer_lda


def main(dataset: str, models: List[str], n_rep: int, smote_inst_1: object) -> pd.DataFrame:
    """
    Run through all the methods for comparison.

    :param dataset: dataset for testing.
    :param models: types of models used for classification.
    :param n_rep: number of replications.
    :param smote_inst_1: SMOTE type 1
    :return results stored in the DataFrame.
    """
    res_df = pd.DataFrame()
    scaler = StandardScaler()

    # read data
    for i in tqdm(range(n_rep)):
        if dataset == 'MPMC':
            # df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")
            # df = df[df['failure.type'] != 5]
            # X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-4], df['failure.type'], test_size=0.3,
            #                                                     stratify=df['failure.type'], random_state=i)

            df = pd.read_csv("datasets/preprocessed/mpmc.csv")
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
            # res_oodH = ood_hie_test(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(), verbose=True)
            # newres_kmeans = divide_n_conquer_plus(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(),
            #                                "kmeans", verbose=True)
            # newres_gmm = divide_n_conquer_plus(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(),
            #                                "gmm", verbose=True)
            # newres_div = divide_n_conquer_plus(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(),
            #                                    "divisive", verbose=True)
            # newres_agg = divide_n_conquer_plus(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(),
            #                                    "agg", verbose=True)

            res_bin = binary(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy())
            res_dnc = divide_n_conquer(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy())
            res_dnc_smote = (divide_n_conquer(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(),
                                              y_test.copy(), smote=smote_inst_1))
            res_dnc_lda = divide_n_conquer_lda(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(),
                                               y_test.copy())
            res_dnc_lda_smote = divide_n_conquer_lda(model, X_train_scaled.copy(), X_test_scaled.copy(),
                                                     y_train.copy(), y_test.copy(), smote=smote_inst_1)
            # res_twoLH = two_layer_hie(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(), verbose=True)
            # res_ood3H = ood_3hie(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(), verbose=True)
            # res_ood2H = ood_2hie(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(), verbose=True)
            res_ovo = multi_clf(model, "OvO", X_train_scaled, X_test_scaled, y_train, y_test)
            res_ovr = multi_clf(model, "OvR", X_train_scaled, X_test_scaled, y_train, y_test)
            res_dir = multi_clf(model, "Direct", X_train_scaled, X_test_scaled, y_train, y_test)
            res_df = pd.concat([res_df, res_bin, res_dnc, res_dnc_smote, res_dnc_lda, res_dnc_lda_smote,
                                res_ovo, res_ovr, res_dir], axis=0)

    # average the performance
    return res_df.groupby(by=["method", "model"], sort=False).mean()


if __name__ == "__main__":
    N_REP = 30

    ##### MPMC #####
    # DATASET = "MPMC"
    # MODELS = [LogisticRegression(max_iter=10000), GaussianNB(), LDA(), SVC(kernel='linear', C=0.1),
    #           SVC(kernel='rbf', C=0.5), DecisionTreeClassifier(), RandomForestClassifier(),
    #           GradientBoostingClassifier(random_state=42), xgb.XGBClassifier()]
    # SMOTE_INST_1 = BorderlineSMOTE(kind='borderline-1', sampling_strategy={1: 100, 2: 100, 3: 100, 4: 100, 5: 100},
    #                                k_neighbors=1)
    # SMOTE_INST_2 = ADASYN()
    # SMOTE_INST_3 = SMOTE()
    # SMOTE_INST_4 = SMOTETomek()
    ##### MPMC #####

    ##### USPS #####
    # DATASET = "USPS"
    # MODELS = [LogisticRegression(max_iter=10000), GaussianNB(), LDA(), SVC(kernel='linear', C=0.1),
    #           SVC(kernel='rbf', C=0.5), DecisionTreeClassifier(), RandomForestClassifier(),
    #           GradientBoostingClassifier(random_state=42), xgb.XGBClassifier()]
    # SMOTE_INST_1 = BorderlineSMOTE(kind='borderline-1')
    ##### USPS #####

    ##### MNIST #####
    DATASET = "MNIST"
    MODELS = [LogisticRegression(max_iter=300), GaussianNB(), LDA(), #SVC(kernel='linear', C=0.1),
              SVC(kernel='rbf', C=0.5), DecisionTreeClassifier(), RandomForestClassifier(),
              GradientBoostingClassifier(random_state=42), xgb.XGBClassifier()]
    SMOTE_INST_1 = BorderlineSMOTE(kind='borderline-1')
    ##### MNIST #####

    ##### GLASS #####
    DATASET = "GLASS"
    MODELS = [LogisticRegression(max_iter=300), GaussianNB(), LDA(),  SVC(kernel='linear', C=1),
              SVC(kernel='rbf', C=1), DecisionTreeClassifier(), RandomForestClassifier(),
              GradientBoostingClassifier(), xgb.XGBClassifier()]
    SMOTE_INST_1 = BorderlineSMOTE(kind='borderline-1')
    ##### GLASS #####

    res = main(DATASET, MODELS, N_REP, SMOTE_INST_1)
    filename = f"results_0301_{DATASET}.csv"
    res.to_csv(filename)

