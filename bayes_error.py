"""
Author: Alex (Tai-Jung) Chen

Monitor the bayes error.
"""
from scipy.stats import norm
from scipy.integrate import quad
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from binary_clf import binary
from divide_n_conquer import divide_n_conquer
from divide_n_conquer_with_unsupervised import divide_n_conquer_plus
from typing import List
import xgboost as xgb


def main(dataset: str, model: object, n_rep: int):
    """
    Get the Bayes error.

    :param dataset: dataset for testing.
    :param model: given classification model.
    :param n_rep: number of replications.
    """
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

        sum_bayes_error = 0
        for sub in range(1, int(y_train.nunique())):
            local_model = clone(model)
            # select only majority and minority sub
            X_train_local = X_train[(y_train == sub) | (y_train == 0)]
            y_train_local = y_train[(y_train == sub) | (y_train == 0)]
            y_train_local[y_train_local != 0] = 1  # turn non-zero sub minority into 1

            local_model.fit(X_train_local, y_train_local)
            y_pred_proba = local_model.predict_proba(X_test)
            min_probs = np.min(y_pred_proba, axis=1)
            bayes_error = np.mean(min_probs)
            sum_bayes_error += bayes_error
            # print(f"sub {sub} labels empirical Bayes Error Rate: {bayes_error:.4f}")
        print(f"sum of sub bayes errors: {sum_bayes_error:.4f}")

        # get binary labels
        y_train_bin = y_train.copy()
        y_train_bin[y_train_bin != 0] = 1
        y_test_bin = y_test.copy()
        y_test_bin[y_test_bin != 0] = 1

        model.fit(X_train, y_train_bin)
        probabilities = model.predict_proba(X_test)
        min_probs = np.min(probabilities, axis=1)
        bayes_error = np.mean(min_probs)

        print(f"Binary labels empirical Bayes Error Rate: {bayes_error:.4f}\n")


if __name__ == "__main__":
    N_REP = 10
    MODELS = [SVC(kernel='linear', C=1.0, probability=True), SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)]
    # MODELS = [LogisticRegression(max_iter=1000), SVC(kernel='linear', C=1.0, probability=True),
    # SVC(kernel='rbf', C=1.0, gamma='scale', probability=True), DecisionTreeClassifier(), RandomForestClassifier(),
    # xgb.XGBClassifier()]
    DATASET = "MPMC"
    # naive bayes
    # knn

    for MDL in MODELS:
        res = main(DATASET, MDL, N_REP)

