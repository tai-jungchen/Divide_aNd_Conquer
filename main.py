"""
Author: Alex (Tai-Jung) Chen

Run through all the classification framework for comparison purpose.
"""
import numpy as np
import pandas as pd
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


def main(dataset: str, models: List[str], n_rep: int, smote_param: dict) -> pd.DataFrame:
    """
    Run through all the methods for comparison.

    :param dataset: dataset for testing.
    :param models: types of models used for classification.
    :param n_rep: number of replications.
    :param smote_param: dictionary that stores the hyper-parameters for SMOTE

    :return results stored in the DataFrame.
    """
    res_df = pd.DataFrame()

    # read data
    for i in range(n_rep):
        if dataset == 'MPMC':
            df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-4], df['failure.type'], test_size=0.3,
                                                                stratify=df['failure.type'], random_state=i)
        elif dataset == 'nij':
            df = pd.read_csv("datasets/preprocessed/nij_data.csv")
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['Recidivism'], test_size=0.3,
                                                                stratify=df['Recidivism_Year'], random_state=i)
        elif dataset == 'mnist':
            df = pd.read_csv("datasets/preprocessed/mnist_imb.csv")
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['label_bin'], test_size=0.3,
                                                                stratify=df['label'], random_state=i)
        else:
            raise Exception("Invalid dataset.")

        # run models
        for model in tqdm(models):
            # res_oodH = ood_hie_test(model, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), verbose=True)
            # newres_kmeans = divide_n_conquer_plus(model, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(),
            #                                "kmeans", verbose=True)
            # newres_gmm = divide_n_conquer_plus(model, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(),
            #                                "gmm", verbose=True)
            # newres_div = divide_n_conquer_plus(model, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(),
            #                                    "divisive", verbose=True)
            # newres_agg = divide_n_conquer_plus(model, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(),
            #                                    "agg", verbose=True)

            res_bin = binary(model, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
            res_dnc = divide_n_conquer(model, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
            res_dnc_smote = (divide_n_conquer(model, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(),
                                              smote_param=smote_param))
            # res_twoLH = two_layer_hie(model, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), verbose=True)
            # res_ood3H = ood_3hie(model, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), verbose=True)
            # res_ood2H = ood_2hie(model, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), verbose=True)
            # res_ovo = multi_clf(model, "OvO", X_train, X_test, y_train, y_test)
            # res_ovr = multi_clf(model, "OvR", X_train, X_test, y_train, y_test)
            # res_dir = multi_clf(model, "Direct", X_train, X_test, y_train, y_test)
            res_df = pd.concat([res_df, res_dnc, res_dnc_smote, res_bin], axis=0)
            # res_df = pd.concat([res_df, res_bin, res_dnc, res_dnc_smote, res_ovo, res_ovr, res_dir], axis=0)

    # average the performance
    return res_df.groupby(by=["method", "model"], sort=False).mean()


if __name__ == "__main__":
    N_REP = 10
    MODELS = [LogisticRegression(max_iter=1000), DecisionTreeClassifier(), RandomForestClassifier(),
              xgb.XGBClassifier()]
    DATASET = "MPMC"
    SMOTE_PARAM = {"sampling_strategy": np.linspace(80, 200, num=26).astype(int),
                   "k_neighbors": np.linspace(1, 5, num=5).astype(int)}

    res = main(DATASET, MODELS, N_REP, SMOTE_PARAM)

    # save the result...
    filename = f"results_0222_smote_{N_REP}.csv"
    res.to_csv(filename)

