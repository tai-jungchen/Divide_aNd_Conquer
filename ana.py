"""
Author: Alex (Tai-Jung) Chen

Offering some plots for analysis purpose.
"""
from symbol import pass_stmt

import pandas as pd
import numpy as np
from imblearn.metrics import specificity_score
from sklearn import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score, \
    balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier


def main(file_name):
    # Read the CSV file
    df = pd.read_csv(file_name)

    df['model'] = [md.split('(')[0] for md in df['model']]
    methods = df['method'].unique()

    # Extract unique metrics (excluding 'Method' and 'Model')
    metrics = [col for col in df.columns if col in ['acc', 'kappa', 'bacc',
                                                    'precision', 'recall',
                                                    'specificity', 'f1']]

    for metric in metrics:
        plt.figure(figsize=(15, 8))
        ax = sns.barplot(data=df, x="model", y=metric, hue="method", ci=None, dodge=True)

        # Annotate bars with their values
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3)  # Format to 2 decimal places

        # Labels and title
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison by Model and Method")
        plt.legend(title="Method")

        # Show plot
        plt.show()
        # plt.savefig(f"{metric}_comparison_0210.png")


def error_monitor(model, X_train, X_test, y_train, y_test, verbose=True):
    """
    Visualize the mis-classified points.
    """
    # go through train test process
    y_train_preds = []
    y_preds = []
    for sub in range(1, int(y_train.nunique())):
        local_model = clone(model)
        # select only majority and minority sub
        X_train_local = X_train[(y_train == sub) | (y_train == 0)]
        y_train_local = y_train[(y_train == sub) | (y_train == 0)]
        y_train_local[y_train_local != 0] = 1  # turn non-zero sub minority into 1

        local_model.fit(X_train_local, y_train_local)

        # training error
        y_train_preds.append(local_model.predict(X_train))

        # testing error
        y_preds.append(local_model.predict(X_test))

    # voting
    y_train_preds = np.array(y_train_preds)
    y_train_pred = np.where(np.sum(y_train_preds, axis=0) > 0, 1, 0)
    y_trains = y_train.copy()    # preserve the multi-class label
    y_train[y_train != 0] = 1  # turn all sub minority into 1

    y_preds = np.array(y_preds)
    y_pred = np.where(np.sum(y_preds, axis=0) > 0, 1, 0)
    y_tests = y_test.copy()  # preserve the multi-class label
    y_test[y_test != 0] = 1  # turn all sub minority into 1

    if verbose:
        print(f'DNC {model}')
        print(confusion_matrix(y_train, y_train_pred, labels=[0, 1]))
        print(classification_report(y_train, y_train_pred))

        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        print(classification_report(y_test, y_pred))

    # read data
    df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(df.iloc[:, -2:], df['failure.type'],
                                                                        test_size=0.3, stratify=df['failure.type'],
                                                                        random_state=0)

    # train
    pred__train_summary = pd.DataFrame(
        np.concatenate((y_train_preds.T, y_train_pred.reshape(-1, 1), y_trains.to_numpy().reshape(-1, 1), y_train.to_numpy().reshape(-1, 1), X_train_pca.to_numpy()), axis=1),
        columns=["local_model_1", "local_model_2", "local_model_3", "local_model_4", "local_model_5", "prediction", "multi-label", "bin-label", "PCA1", "PCA2"])
    pred__train_summary[["local_model_1", "local_model_2", "local_model_3", "local_model_4", "local_model_5", "prediction", "multi-label", "bin-label"]] = (
        pred__train_summary[["local_model_1", "local_model_2", "local_model_3", "local_model_4", "local_model_5", "prediction", "multi-label", "bin-label"]].astype(int))

    # test
    pred_summary = pd.DataFrame(np.concatenate((y_preds.T, y_pred.reshape(-1, 1), y_tests.to_numpy().reshape(-1, 1), y_test.to_numpy().reshape(-1, 1), X_test_pca.to_numpy()), axis=1),
                                columns=["local_model_1", "local_model_2", "local_model_3", "local_model_4", "local_model_5", "prediction", "multi-label", "bin-label", "PCA1", "PCA2"])
    pred_summary[["local_model_1", "local_model_2", "local_model_3", "local_model_4", "local_model_5", "prediction", "multi-label", "bin-label"]] = (
        pred_summary[["local_model_1", "local_model_2", "local_model_3", "local_model_4", "local_model_5", "prediction", "multi-label", "bin-label"]].astype(int))

    # visualization - scatter plot
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(pred__train_summary[(pred__train_summary['prediction'] == 0) & (pred__train_summary['bin-label'] == 0)]['PCA1'],
                pred__train_summary[(pred__train_summary['prediction'] == 0) & (pred__train_summary['bin-label'] == 0)]['PCA2'],
                c='orange', label='TN')
    plt.scatter(pred__train_summary[(pred__train_summary['prediction'] == 1) & (pred__train_summary['bin-label'] == 1)]['PCA1'],
                pred__train_summary[(pred__train_summary['prediction'] == 1) & (pred__train_summary['bin-label'] == 1)]['PCA2'],
                c='green', label='TP')
    plt.scatter(pred__train_summary[(pred__train_summary['prediction'] == 1) & (pred__train_summary['bin-label'] == 0)]['PCA1'],
                pred__train_summary[(pred__train_summary['prediction'] == 1) & (pred__train_summary['bin-label'] == 0)]['PCA2'],
                c='blue', label='FP')
    plt.scatter(pred__train_summary[(pred__train_summary['prediction'] == 0) & (pred__train_summary['bin-label'] == 1)]['PCA1'],
                pred__train_summary[(pred__train_summary['prediction'] == 0) & (pred__train_summary['bin-label'] == 1)]['PCA2'],
                c='red', label='FN')

    plt.legend(["TN", "TP", "FP", "FN"])
    plt.title(f"Training {str(model).split('(')[0]}")

    # test
    plt.subplot(1, 2, 2)
    plt.scatter(pred_summary[(pred_summary['prediction'] == 0) & (pred_summary['bin-label'] == 0)]['PCA1'],
                pred_summary[(pred_summary['prediction'] == 0) & (pred_summary['bin-label'] == 0)]['PCA2'],
                c='orange', label='TN')
    plt.scatter(pred_summary[(pred_summary['prediction'] == 1) & (pred_summary['bin-label'] == 1)]['PCA1'],
                pred_summary[(pred_summary['prediction'] == 1) & (pred_summary['bin-label'] == 1)]['PCA2'],
                c='green', label='TP')
    plt.scatter(pred_summary[(pred_summary['prediction'] == 1) & (pred_summary['bin-label'] == 0)]['PCA1'],
                pred_summary[(pred_summary['prediction'] == 1) & (pred_summary['bin-label'] == 0)]['PCA2'],
                c='blue', label='FP')
    plt.scatter(pred_summary[(pred_summary['prediction'] == 0) & (pred_summary['bin-label'] == 1)]['PCA1'],
                pred_summary[(pred_summary['prediction'] == 0) & (pred_summary['bin-label'] == 1)]['PCA2'],
                c='red', label='FN')

    plt.legend(["TN", "TP", "FP", "FN"])
    plt.title(f"Testing {str(model).split('(')[0]}")
    plt.show()
    # plt.savefig(f"Testing {str(model).split('(')[0]}.png")

    # visualization - scatter plot for individual local model
    mis = pred_summary[(pred_summary['prediction'] == 1) & (pred_summary['bin-label'] == 0)]
    correct = pred_summary[(pred_summary['prediction'] == 1) & (pred_summary['bin-label'] == 1)]
    plt.scatter(correct[correct['local_model_4'] == 1]['PCA1'], correct[correct['local_model_4'] == 1]['PCA2'], c='blue', label='correct')
    plt.scatter(mis[mis['local_model_4'] == 1]['PCA1'], mis[mis['local_model_4'] == 1]['PCA2'], c='red', label='mis')
    plt.legend(['correct', 'mis'])
    plt.show()

    # visualization - number of mis-classified for each local model
    mistake = pred__train_summary[((pred__train_summary['prediction'] == 1) & (pred__train_summary['bin-label'] == 0)) | (
            (pred__train_summary['prediction'] == 0) & (pred__train_summary['bin-label'] == 1))]

    ratios = []
    for i in range(len(mistake['multi-label'].value_counts())):
        l, v = mistake['multi-label'].value_counts().index[i], mistake['multi-label'].value_counts().values[i]
        t = pred__train_summary[pred__train_summary['multi-label'] == l].shape[0]
        ratios.append((l, v / t))

    # exact counts
    mistake['multi-label'].value_counts().plot(kind='bar')
    plt.show()

    # ratio
    plt.bar([lab for lab, ratio in ratios], [ratio for lab, ratio in ratios], color='skyblue', edgecolor='black')
    plt.show()

    # testing
    mistake = pred_summary[((pred_summary['prediction'] == 1) & (pred_summary['bin-label'] == 0)) | (
                (pred_summary['prediction'] == 0) & (pred_summary['bin-label'] == 1))]

    ratios = []
    for i in range(len(mistake['multi-label'].value_counts())):
        l, v = mistake['multi-label'].value_counts().index[i], mistake['multi-label'].value_counts().values[i]
        t = pred_summary[pred_summary['multi-label'] == l].shape[0]
        ratios.append((l, v/t))

    # exact counts
    mistake['multi-label'].value_counts().plot(kind='bar')
    plt.show()

    # ratio
    plt.bar([lab for lab, ratio in ratios], [ratio for lab, ratio in ratios], color='skyblue', edgecolor='black')
    plt.show()

    print()


def perfect(model, X_train, X_test, y_train, y_test, verbose=True):
    num_of_sub = int(y_train.nunique())
    y_train_preds = []
    y_preds = []
    for sub in range(1, num_of_sub):
        local_model = clone(model)
        # select only majority and minority sub
        X_train_local = X_train[(y_train == sub) | (y_train == 0)]
        y_train_local = y_train[(y_train == sub) | (y_train == 0)]
        y_train_local[y_train_local != 0] = 1  # turn non-zero sub minority into 1

        local_model.fit(X_train_local, y_train_local)

        # training error
        y_train_preds.append(local_model.predict(X_train))

        # testing error
        sub_data = X_test[(y_test == sub) | (y_test == 0)]
        y_preds.append((sub_data.index, local_model.predict(sub_data)))

    # voting
    y_train_preds = np.array(y_train_preds)
    y_train_pred = np.where(np.sum(y_train_preds, axis=0) > 0, 1, 0)
    y_trains = y_train.copy()  # preserve the multi-class label
    y_train[y_train != 0] = 1  # turn all sub minority into 1

    # testing
    y_pred = pd.concat([y_test] * (num_of_sub-1), axis=1)
    y_pred[:] = -1
    y_pred.columns = [f'local_model_{i + 1}' for i in range(num_of_sub-1)]  # Rename columns
    y_pred = pd.concat([y_pred, y_test], axis=1)
    for sub in range(num_of_sub-1):
        y_pred.loc[y_preds[sub][0], f'local_model_{sub+1}'] = y_preds[sub][1]

    tp = y_pred[(y_pred['failure.type'] > 0) & ((y_pred['local_model_1'] == 1) | (y_pred['local_model_2'] == 1) | (
            y_pred['local_model_3'] == 1) | (y_pred['local_model_4'] == 1) | (y_pred['local_model_5'] == 1))].shape[0]
    fn = y_pred[(y_pred['failure.type'] > 0) & ((y_pred['local_model_1'] == 0) | (y_pred['local_model_2'] == 0) | (
            y_pred['local_model_3'] == 0) | (y_pred['local_model_4'] == 0) | (y_pred['local_model_5'] == 0))].shape[0]

    ##### positive sensitive #####
    # tn = y_pred[(y_pred['failure.type'] == 0) & ((y_pred['local_model_1'] == 0) & (y_pred['local_model_2'] == 0) & (
    #         y_pred['local_model_3'] == 0) & (y_pred['local_model_4'] == 0) & (y_pred['local_model_5'] == 0))].shape[0]
    # fp = y_pred[(y_pred['failure.type'] == 0) & ((y_pred['local_model_1'] == 1) | (y_pred['local_model_2'] == 1) | (
    #         y_pred['local_model_3'] == 1) | (y_pred['local_model_4'] == 1) | (y_pred['local_model_5'] == 1))].shape[0]
    ##### positive sensitive #####

    ##### negative sensitive #####
    tn = y_pred[(y_pred['failure.type'] == 0) & ((y_pred['local_model_1'] == 0) | (y_pred['local_model_2'] == 0) | (
            y_pred['local_model_3'] == 0) | (y_pred['local_model_4'] == 0) | (y_pred['local_model_5'] == 0))].shape[0]
    fp = y_pred[(y_pred['failure.type'] == 0) & ((y_pred['local_model_1'] == 1) & (y_pred['local_model_2'] == 1) & (
            y_pred['local_model_3'] == 1) & (y_pred['local_model_4'] == 1) & (y_pred['local_model_5'] == 1))].shape[0]
    ##### negative sensitive #####

    if verbose:
        prec = tp / (tp+fp)
        rec = tp / (tp+fn)
        spec = tn / (tn+fp)
        print(f"DNC {model}")
        print(f"Confusion matrix:\n {[[tp, fn], [fp, tn]]}")
        print(f"acc: {(tp+tn) / (tp+tn+fp+fn)}")
        print(f"precision: {prec}")
        print(f"recall: {rec}")
        print(f"bacc: {0.5*rec + 0.5*spec}")
        print(f"f1: {2 / (1/prec + 1/rec)}")
        print()


if __name__ == '__main__':
    # FILENAME = "results_0212_k1.csv"
    # main(FILENAME)

    df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-4], df['failure.type'], test_size=0.3,
                                                        stratify=df['failure.type'], random_state=0)
    # models = [LogisticRegression(max_iter=1000), DecisionTreeClassifier(), RandomForestClassifier(), xgb.XGBClassifier()]
    models = [xgb.XGBClassifier()]
    for model in models:
        # perfect(model, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
        error_monitor(model, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
