"""
Author: Alex (Tai-Jung) Chen

Offering some plots for analysis purpose.
"""
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


def main():
    # Read the CSV file
    file_name = "results_0210_95.csv"
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
    y_train[y_train != 0] = 1  # turn all sub minority into 1

    y_preds = np.array(y_preds)
    y_pred = np.where(np.sum(y_preds, axis=0) > 0, 1, 0)
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
        np.concatenate((y_train_preds.T, y_train_pred.reshape(-1, 1), y_train.to_numpy().reshape(-1, 1), X_train_pca.to_numpy()), axis=1),
        columns=["local_model_1", "local_model_2", "local_model_3", "local_model_4", "local_model_5", "prediction", "label", "PCA1", "PCA2"])
    pred__train_summary[["local_model_1", "local_model_2", "local_model_3", "local_model_4", "local_model_5", "prediction", "label"]] = (
        pred__train_summary[["local_model_1", "local_model_2", "local_model_3", "local_model_4", "local_model_5", "prediction", "label"]].astype(int))

    # test
    pred_summary = pd.DataFrame(np.concatenate((y_preds.T, y_pred.reshape(-1, 1), y_test.to_numpy().reshape(-1, 1), X_test_pca.to_numpy()), axis=1),
                                columns=["local_model_1", "local_model_2", "local_model_3", "local_model_4", "local_model_5", "prediction", "label", "PCA1", "PCA2"])
    pred_summary[["local_model_1", "local_model_2", "local_model_3", "local_model_4", "local_model_5", "prediction", "label"]] = (
        pred_summary[["local_model_1", "local_model_2", "local_model_3", "local_model_4", "local_model_5", "prediction", "label"]].astype(int))

    # visualization
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(pred__train_summary[(pred__train_summary['prediction'] == 0) & (pred__train_summary['label'] == 0)]['PCA1'],
                pred__train_summary[(pred__train_summary['prediction'] == 0) & (pred__train_summary['label'] == 0)]['PCA2'],
                c='orange', label='TN')
    plt.scatter(pred__train_summary[(pred__train_summary['prediction'] == 1) & (pred__train_summary['label'] == 1)]['PCA1'],
                pred__train_summary[(pred__train_summary['prediction'] == 1) & (pred__train_summary['label'] == 1)]['PCA2'],
                c='green', label='TP')
    plt.scatter(pred__train_summary[(pred__train_summary['prediction'] == 1) & (pred__train_summary['label'] == 0)]['PCA1'],
                pred__train_summary[(pred__train_summary['prediction'] == 1) & (pred__train_summary['label'] == 0)]['PCA2'],
                c='blue', label='FP')
    plt.scatter(pred__train_summary[(pred__train_summary['prediction'] == 0) & (pred__train_summary['label'] == 1)]['PCA1'],
                pred__train_summary[(pred__train_summary['prediction'] == 0) & (pred__train_summary['label'] == 1)]['PCA2'],
                c='red', label='FN')

    plt.legend(["TN", "TP", "FP", "FN"])
    plt.title(f"Training {str(model).split('(')[0]}")

    # test
    plt.subplot(1, 2, 2)
    plt.scatter(pred_summary[(pred_summary['prediction'] == 0) & (pred_summary['label'] == 0)]['PCA1'],
                pred_summary[(pred_summary['prediction'] == 0) & (pred_summary['label'] == 0)]['PCA2'],
                c='orange', label='TN')
    plt.scatter(pred_summary[(pred_summary['prediction'] == 1) & (pred_summary['label'] == 1)]['PCA1'],
                pred_summary[(pred_summary['prediction'] == 1) & (pred_summary['label'] == 1)]['PCA2'],
                c='green', label='TP')
    plt.scatter(pred_summary[(pred_summary['prediction'] == 1) & (pred_summary['label'] == 0)]['PCA1'],
                pred_summary[(pred_summary['prediction'] == 1) & (pred_summary['label'] == 0)]['PCA2'],
                c='blue', label='FP')
    plt.scatter(pred_summary[(pred_summary['prediction'] == 0) & (pred_summary['label'] == 1)]['PCA1'],
                pred_summary[(pred_summary['prediction'] == 0) & (pred_summary['label'] == 1)]['PCA2'],
                c='red', label='FN')

    plt.legend(["TN", "TP", "FP", "FN"])
    plt.title(f"Testing {str(model).split('(')[0]}")
    # plt.show()
    plt.savefig(f"Testing {str(model).split('(')[0]}.png")


if __name__ == '__main__':
    # main()
    df = pd.read_csv("datasets/preprocessed/maintenance_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-4], df['failure.type'], test_size=0.3,
                                                        stratify=df['failure.type'], random_state=0)
    models = [LogisticRegression(max_iter=1000), DecisionTreeClassifier(), RandomForestClassifier(), xgb.XGBClassifier()]
    for model in models:
        error_monitor(model, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
