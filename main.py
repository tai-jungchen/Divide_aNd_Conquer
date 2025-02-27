"""
Author: Alex (Tai-Jung) Chen

Run through all the classification framework for comparison purpose.
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from binary_clf import binary
from divide_n_conquer import divide_n_conquer
from divide_n_conquer_with_unsupervised import divide_n_conquer_plus
from multi_class_clf import multi_clf
from typing import List
import xgboost as xgb
from two_layer_hie import two_layer_hie
from ood_hie import ood_2hie, ood_3hie
from divide_n_conquer_lda import divide_n_conquer_lda
from sklearn.preprocessing import StandardScaler


def main(dataset: str, models: List[str], n_rep: int, smote_param_1: dict, smote_param_2: dict) -> pd.DataFrame:
    """
    Run through all the methods for comparison.

    :param dataset: dataset for testing.
    :param models: types of models used for classification.
    :param n_rep: number of replications.
    :param smote_param_1: dictionary that stores the hyper-parameters for SMOTE type 1
    :param smote_param_2: dictionary that stores the hyper-parameters for SMOTE type 2
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

        elif dataset == 'nij':
            df = pd.read_csv("datasets/preprocessed/nij_data.csv")
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['Recidivism'], test_size=0.3,
                                                                stratify=df['Recidivism_Year'], random_state=i)
        elif dataset == 'mnist':
            df = pd.read_csv("datasets/preprocessed/mnist_imb.csv")
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['label_bin'], test_size=0.3,
                                                                stratify=df['label'], random_state=i)
        elif dataset == 'USPS':
            df = pd.read_csv("datasets/preprocessed/imb_digit_0.csv")

            # change label
            maj = 0
            label_mapping = {maj: 0, 1: 1, 6: 2, 9: 3}
            df.iloc[:, 0] = df.iloc[:, 0].map(label_mapping)
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
            res_dnc_smote = (divide_n_conquer(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(),
                                              smote_param=smote_param_1))
            res_dnc_borderline = (divide_n_conquer(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(),
                                                   y_test.copy(), smote_param=smote_param_2))
            res_dnc_lda = divide_n_conquer_lda(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(),
                                               y_test.copy())
            res_dnc_lda_smote = divide_n_conquer_lda(model, X_train_scaled.copy(), X_test_scaled.copy(),
                                                     y_train.copy(), y_test.copy(), smote_param=smote_param_1)
            res_dnc_lda_borderline = divide_n_conquer_lda(model, X_train_scaled.copy(), X_test_scaled.copy(),
                                                     y_train.copy(), y_test.copy(), smote_param=smote_param_2)

            # res_twoLH = two_layer_hie(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(), verbose=True)
            # res_ood3H = ood_3hie(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(), verbose=True)
            # res_ood2H = ood_2hie(model, X_train_scaled.copy(), X_test_scaled.copy(), y_train.copy(), y_test.copy(), verbose=True)
            res_ovo = multi_clf(model, "OvO", X_train_scaled, X_test_scaled, y_train, y_test)
            res_ovr = multi_clf(model, "OvR", X_train_scaled, X_test_scaled, y_train, y_test)
            res_dir = multi_clf(model, "Direct", X_train_scaled, X_test_scaled, y_train, y_test)
            res_df = pd.concat([res_df, res_bin, res_dnc, res_dnc_smote, res_dnc_borderline, res_dnc_lda,
                                res_dnc_lda_smote, res_dnc_lda_borderline, res_ovo, res_ovr, res_dir], axis=0)

    # average the performance
    return res_df.groupby(by=["method", "model"], sort=False).mean()


if __name__ == "__main__":
    N_REP = 10

    ##### MPMC #####
    DATASET = "MPMC"
    MODELS = [LogisticRegression(max_iter=10000), #SVC(kernel='linear', C=2),
              SVC(kernel='rbf', C=1), DecisionTreeClassifier(), RandomForestClassifier(),
              xgb.XGBClassifier()]
    SMOTE_PARAM_1 = {"type": "SMOTE",
                   "sampling_strategy": {1: 100, 2:100, 3: 100, 4: 100, 5: 100},
                   "k_neighbors": 1}
    SMOTE_PARAM_2 = {"type": "Borderline",
                   "sampling_strategy": {1: 100, 2: 100, 3: 100, 4: 100, 5: 100},
                   "k_neighbors": 1}
    ##### MPMC #####

    ##### USPS #####
    # DATASET = "USPS"
    # MODELS = [LogisticRegression(max_iter=10000), SVC(kernel='linear', C=1.0),
    #           SVC(kernel='rbf', C=1.0), DecisionTreeClassifier(max_depth=11), RandomForestClassifier(),
    #           xgb.XGBClassifier()]
    # SMOTE_PARAM = {"sampling_strategy": np.linspace(400, 400, num=1).astype(int),
    #                "k_neighbors": np.linspace(1, 5, num=2).astype(int)}
    ##### USPS #####

    res = main(DATASET, MODELS, N_REP, SMOTE_PARAM_1, SMOTE_PARAM_2)
    filename = f"sum_results_0226_{DATASET}.csv"
    res.to_csv(filename)

