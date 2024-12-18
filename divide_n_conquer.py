"""
Author: Alex (Tai-Jung) Chen

This code implements the proposed DNC method. DNC uses partial OvO and customized decision rules in voting to cope with
imbalance data classification with subclass information available in the minority class.
"""
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, \
    balanced_accuracy_score, precision_recall_fscore_support
from sklearn import tree

SUBS = [1, 2, 3, 4, 5]
# SUBS = [1, 2, 3]
# SUBS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def divide_n_conquer(model_type: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
                     y_test: pd.DataFrame) -> pd.DataFrame:
    """
    Carry out the DNC method on the Machine Predictive Maintenance Classification dataset. The classification results
    will be stored to a .csv file and the console information will be store to a .txt file.

    :param dataset: the dataset given
    :type dataset: String
    :param model_type: model type (lr, svm, gnb, dt, rf, knn)
    :type model_type: String
    :param n_reps: number of replication for train-test split
    :type n_reps: int
    """
    log_file = 'log/' + 'dnc_' + dataset + '_' + model_type + '.txt'
    csv_file = 'results/' + 'dnc_' + dataset + '_' + model_type + '.csv'
    sys.stdout = open(log_file, "w")

    record_metrics = ['acc', 'bacc', 'f1', 'precision', 'recall', 'specificity']
    metrics = {key: [] for key in record_metrics}

    # read data
    if dataset == 'maintenance':
        df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")
    elif dataset == 'nij':
        df = pd.read_csv("datasets/preprocessed/nij_data.csv")
    elif dataset == 'mnist':
        df = pd.read_csv("datasets/preprocessed/mnist_imb.csv")
    else:
        df = None

    for i in range(n_reps):
        if dataset == 'maintenance':
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-4], df['failure.type'], test_size=0.3,
                                                                stratify=df['failure.type'], random_state=i)
        elif dataset == 'nij':
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['Recidivism_Year'], test_size=0.3,
                                                                stratify=df['Recidivism_Year'], random_state=i)
        elif dataset == 'mnist':
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['label'], test_size=0.3,
                                                                stratify=df['label'], random_state=i)
        else:
            X_train, X_test, y_train, y_test = None, None, None, None

        # training
        models = []
        for sub in SUBS:
            # training
            model = None
            if model_type == 'LR':
                model = LogisticRegression(max_iter=10000)
            elif model_type == 'DT':
                model = DecisionTreeClassifier()
            elif model_type == 'RF':
                model = RandomForestClassifier()
            elif model_type == 'XGBOOST':
                model = xgb.XGBClassifier()
            else:
                print("Error: model does not exist")

            # select only majority and minority sub
            X_train_local = X_train[(y_train == sub) | (y_train == 0)]
            y_train_local = y_train[(y_train == sub) | (y_train == 0)]
            y_train_local[y_train_local != 0] = 1  # turn non-zero sub minority into 1

            model.fit(X_train_local, y_train_local)
            models.append(model)

        # testing
        y_preds = []
        for md in models:
            y_pred = md.predict(X_test)
            y_preds.append(y_pred)

        # voting
        y_preds = np.array(y_preds)
        y_pred_agg = np.where(np.sum(y_preds, axis=0) > 0, 1, 0)
        y_test[y_test != 0] = 1  # turn all sub minority into 1

        print(f'DNC {model_type}')
        print(confusion_matrix(y_test, y_pred_agg, labels=[0, 1]))
        print(classification_report(y_test, y_pred_agg))

        acc = accuracy_score(y_test, y_pred_agg)
        bacc = balanced_accuracy_score(y_test, y_pred_agg)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_agg)
        tn, fp, fn, tp = np.ravel(confusion_matrix(y_test, y_pred_agg))
        spec = tn / (tn + fp)

        # Store performance
        metrics['acc'].append(round(acc, 4))
        metrics['bacc'].append(round(bacc, 4))
        metrics['precision'].append(round(precision[1], 4))
        metrics['recall'].append(round(recall[1], 4))
        metrics['specificity'].append(round(spec, 4))
        metrics['f1'].append(round(f1[1], 4))

    # save output to
    results_df = pd.DataFrame()
    print("\nAverage Performance:")
    for key, value in metrics.items():
        print(f'{key}: {round(float(np.mean(value)), 4)}, S.E.: {round(np.std(value) / len(value), 4)}')
        results_df[key] = value

    for key, value in metrics.items():
        results_df.at[len(value), key] = round(float(np.mean(value)), 4)
        results_df.at[len(value) + 1, key] = round(np.std(value) / len(value), 4)
    results_df.to_csv(csv_file, index=False)
    sys.stdout.close()
