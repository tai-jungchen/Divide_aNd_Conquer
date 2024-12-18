"""
Author: Alex (Tai-Jung) Chen

Implement the multi-class classification and collapse the classification to binary.
"""
import sys
from itertools import combinations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, \
    balanced_accuracy_score, precision_recall_fscore_support
from sklearn import tree
from collections import Counter


def multi_clf(model_type: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
                     y_test: pd.DataFrame) -> pd.DataFrame:
    """
    Carry out the Direct or OvO method The classification results will be stored to a .csv file and the console
    information will be store to a .txt file.

    :param strat: multiclass classification strategy
    :type strat: Sting
    :param dataset: the dataset given
    :type dataset: String
    :param model_type: model type (lr, svm, gnb, dt, rf, knn)
    :type model_type: String
    :param n_reps: number of replication for train-test split
    :type n_reps: int
    """
    log_file = 'log/' + 'multiClass_' + strat + "_" + dataset + '_' + model_type + '.txt'
    csv_file = 'results/' + 'multiClass_' + strat + "_" + dataset + '_' + model_type + '.csv'
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

        model = None
        if model_type == 'LR':
            if strat == 'OvO':
                log_reg = LogisticRegression(max_iter=1000)
                model = OneVsOneClassifier(log_reg)
            elif strat == 'OvR':
                model = LogisticRegression(multi_class='ovr', max_iter=1e3)
            elif strat == 'Direct':
                model = LogisticRegression(multi_class='multinomial', max_iter=10000)
        elif model_type == 'RF':
            if strat == 'OvO':
                rf = RandomForestClassifier()
                model = OneVsOneClassifier(rf)
            elif strat == 'OvR':
                model = RandomForestClassifier()
            elif strat == 'Direct':
                model = RandomForestClassifier()
        elif model_type == 'XGBOOST':
            if strat == 'OvO':
                xg = xgb.XGBClassifier()
                model = OneVsOneClassifier(xg)
            elif strat == 'Direct':
                model = LinearDiscriminantAnalysis()
        elif model_type == 'DT':
            if strat == 'OvO':
                dt = DecisionTreeClassifier()
                model = OneVsOneClassifier(dt)
            elif strat == 'Direct':
                model = DecisionTreeClassifier()
        else:
            print("Error: model does not exist")

        # training
        model.fit(X_train, y_train)

        # testing
        y_pred_multi = model.predict(X_test)
        y_pred = np.where(y_pred_multi > 0, 1, 0)
        y_test = np.where(y_test > 0, 1, 0)

        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
        tn, fp, fn, tp = np.ravel(confusion_matrix(y_test, y_pred))
        spec = tn / (tn + fp)

        # Store performance
        metrics['acc'].append(round(acc, 4))
        metrics['bacc'].append(round(bacc, 4))
        metrics['precision'].append(round(precision[1], 4))
        metrics['recall'].append(round(recall[1], 4))
        metrics['specificity'].append(round(spec, 4))
        metrics['f1'].append(round(f1[1], 4))

        # print rep output
        print(f'Multi-class {model_type}')
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        print(classification_report(y_test, y_pred))

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


def multi_analysis(model_type, n_reps, strat):
    subs = [0, 1, 2, 3, 4, 5]
    # read data
    df = pd.read_csv("predictive_maintenance.csv")
    df['type'] = df['type'].astype('category').cat.codes
    df['failure.type'] = df['failure.type'].astype('category')
    df['failure.type'].cat.set_categories(
        ['No Failure', 'Heat Dissipation Failure', 'Power Failure', 'Overstrain Failure', 'Tool Wear Failure',
         'Random Failures'], inplace=True)
    df['failure.type'] = df['failure.type'].astype('category').cat.codes

    for i in range(n_reps):
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='target'), df['failure.type'],
                                                            test_size=0.3,
                                                            stratify=df['failure.type'], random_state=i)

        model = None
        if model_type == 'LR':
            if strat == 'OvO':
                model = LogisticRegression(max_iter=1e4, solver='saga')
            elif strat == 'OvR':
                model = LogisticRegression(multi_class='ovr', max_iter=1e4, solver='saga')
            elif strat == 'direct':
                model = LogisticRegression(multi_class='multinomial', max_iter=1e4, solver='saga')
        elif model_type == 'RF':
            if strat == 'OvO':
                rf = RandomForestClassifier()
                model = OneVsOneClassifier(rf)
            elif strat == 'OvR':
                model = RandomForestClassifier()
        elif model_type == 'XGBOOST':
            if strat == 'OvO':
                model = xgb.XGBClassifier()
            elif strat == 'direct':
                model = LinearDiscriminantAnalysis()
        elif model_type == 'dt':
            if strat == 'OvO':
                dt = DecisionTreeClassifier()
                model = OneVsOneClassifier(dt)
            elif strat == 'direct':
                model = DecisionTreeClassifier()
        else:
            print("Error: model does not exist")

        # training / testing for OvO
        voting_table = []
        all_combinations = list(combinations(subs, 2))  # Get all combinations of length 2

        for j in range(len(all_combinations)):
            comb = all_combinations[j]
            c1, c2 = comb

            condition_train = (X_train['failure.type'] == c1) | (X_train['failure.type'] == c2)
            X_train_sub = X_train[condition_train].drop(columns='failure.type')
            y_train_sub = y_train[condition_train]
            model.fit(X_train_sub, y_train_sub)

            # testing aggregation
            y_test_pred = model.predict(X_test.drop(columns='failure.type'))
            voting_table.append(y_test_pred)

        # fit a binary model as well
        bin_model = LogisticRegression(max_iter=1e4, solver='saga')
        y_train_bin = np.where(y_train > 0, 1, y_train)
        y_test_bin = np.where(y_test > 0, 1, y_test)
        bin_model.fit(X_train.drop(columns='failure.type'), y_train_bin)

        # testing
        y_train_pred_bin = bin_model.predict(X_train.drop(columns='failure.type'))
        y_test_pred_bin = bin_model.predict(X_test.drop(columns='failure.type'))
        bin_predict_prob = bin_model.predict_proba(X_test.drop(columns='failure.type'))

        # comparison
        voting = np.array(voting_table).T
        frequency = np.apply_along_axis(lambda x: np.bincount(x, minlength=6), axis=1, arr=voting)
        ovo_predict_prob = frequency / len(all_combinations)
        y_test_pred_OvO = majority_vote_from_table(frequency)
        y_test_pred_collapse = np.where(y_test_pred_OvO > 0, 1, 0)

        tu = X_test, y_test_pred_bin, y_test_pred_collapse, y_test_bin
        data = dataframe_and_ndarrays_to_list_of_tuples(tu)

        diff_tuples = select_different_tuples(data)
        X = aggregate_first_items(diff_tuples)
        df_X = pd.concat(X, axis=1)

        pred_prob = bin_model.predict_proba(df_X.T.drop(columns='failure.type'))
        w = bin_model.coef_
        b = bin_model.intercept_

        z_values = np.dot(w, df_X.T.drop(columns='failure.type').T) + b
        # Calculate y values using the logistic function
        # y_values = logistic_function(x_values, w, b)

        # Plot the logistic function
        plt.plot(z_values.reshape(-1).tolist(), pred_prob.tolist(), label='Logistic Function', marker='o')
        plt.title('Logistic Regression Plot')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.plot(pred_prob[:, 1])
        plt.show()
        print()


def aggregate_first_items(list_of_tuples):
    first_items = [tpl[0] for tpl in list_of_tuples]
    return first_items


def select_different_tuples(list_of_tuples):
    selected_tuples = []
    for tpl in list_of_tuples:
        if tpl[1] != tpl[2]:
            selected_tuples.append(tpl)
    return selected_tuples


def logistic_function(z):
    return 1 / (1 + np.exp(-z))


def majority_vote_from_table(votes_table):
    max_indices = np.argmax(votes_table, axis=1)
    return max_indices


def dataframe_and_ndarrays_to_list_of_tuples(data_tuple):
    dataframe = data_tuple[0]
    arrays = data_tuple[1:]

    list_of_tuples = []
    for i in range(len(dataframe)):
        point_tuple = (dataframe.iloc[i],) + tuple(arr[i] for arr in arrays)
        list_of_tuples.append(point_tuple)
    return list_of_tuples
