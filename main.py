"""
Author: Alex (Tai-Jung) Chen

Run through all the classification framework for comparison purpose.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from binary_clf import binary
from divide_n_conquer import divide_n_conquer
from multi_class_clf import multi_clf, multi_analysis
from typing import List


def main(dataset: str, model_types: List[str], n_rep: int):
    """
    Run through all the methods for comparison.

    :param dataset: dataset for testing.
    :param model_types: types of models used for classification.
    :param n_rep: number of replications.
    """
    # read data
    for i in range(n_rep):
        if dataset == 'maintenance':
            df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-4], df['target'], test_size=0.3,
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
        for model in tqdm(model_types):
            binary(model, X_train, X_test, y_train, y_test)
            divide_n_conquer(model, X_train, X_test, y_train, y_test)
            multi_clf(model, STRATEGY_1, X_train, X_test, y_train, y_test)
            multi_clf(model, STRATEGY_3, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    N_REP = 10
    STRATEGY_1 = "OvO"
    STRATEGY_2 = "OvR"
    STRATEGY_3 = "Direct"
    model_types = ['LR', 'DT', 'RF', 'XGBOOST']
    dataset = "HMEQ"

    main(dataset, model_types, N_REP)

