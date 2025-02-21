"""
Author: Alex (Tai-Jung) Chen

This code implements the proposed DNC method with an unsupervised learning technique. DNC uses partial OvO and
customized decision rules in voting to cope with imbalance data classification with subclass information available in
the minority class.
"""
import numpy as np
import pandas as pd
from imblearn.metrics import specificity_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score, \
    balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def divide_n_conquer_plus(model: object, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
                     y_test: pd.DataFrame, clus_algo, verbose: bool = False) -> pd.DataFrame:
    """
    Carry out the DNC plus method on the Machine Predictive Maintenance Classification dataset. The classification
    results will be stored to a .csv file and the console information will be store to a .txt file.

    :param model: classifier.
    :param X_train: training data.
    :param X_test: testing data.
    :param y_train: training label.
    :param y_test: testing label.

    :param verbose: whether to print out the confusion matrix or not.

    :return: the dataframe with the classification metrics.
    """
    # Only have binary information
    y_train[y_train != 0] = 1
    y_test[y_test != 0] = 1

    record_metrics = ['model', 'method', 'acc', 'kappa', 'bacc', 'precision', 'recall', 'specificity', 'f1']
    metrics = {key: [] for key in record_metrics}

    # get multi-class label
    y_train_multi = cluster(X_train, y_train, clus_algo)

    y_train_preds = []
    y_preds = []
    for sub in range(1, int(y_train_multi.nunique())):
        local_model = clone(model)
        # select only majority and minority sub
        X_train_local = X_train[(y_train_multi == sub) | (y_train_multi == 0)]
        y_train_local = y_train_multi[(y_train_multi == sub) | (y_train_multi == 0)]
        y_train_local[y_train_local != 0] = 1  # turn non-zero sub minority into 1

        local_model.fit(X_train_local, y_train_local)
        y_train_preds.append(local_model.predict(X_train))

        y_pred_sub = local_model.predict(X_test)
        y_preds.append(y_pred_sub)

    # voting
    y_preds = np.array(y_preds)
    y_pred = np.where(np.sum(y_preds, axis=0) > 0, 1, 0)

    if verbose:
        print(f'DNC Plus {model}')
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        print(classification_report(y_test, y_pred))

    # Store performance
    metrics['acc'].append(round(accuracy_score(y_test, y_pred), 4))
    metrics['kappa'].append(round(cohen_kappa_score(y_test, y_pred), 4))
    metrics['bacc'].append(round(balanced_accuracy_score(y_test, y_pred), 4))
    metrics['precision'].append(round(precision_score(y_test, y_pred), 4))
    metrics['recall'].append(round(recall_score(y_test, y_pred), 4))
    metrics['specificity'].append(round(specificity_score(y_test, y_pred), 4))
    metrics['f1'].append(round(f1_score(y_test, y_pred), 4))

    metrics['model'].append(model)
    metrics['method'].append(f"dncPlus_{clus_algo}")

    return pd.DataFrame(metrics)


def cluster(X, y, clus_algo):
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    y_multi = y.copy()

    if clus_algo == "kmeans":
        kmeans = KMeans(n_clusters=5, random_state=42)
        y_pred = kmeans.fit_predict(X_pos) + 1
    elif clus_algo == "divisive":
        distance_matrix = squareform(pdist(X))
        Z = linkage(X, method='ward')
        y_pred = fcluster(Z, t=5, criterion='maxclust')

        # plt.figure(figsize=(12, 6))
        # dendrogram(Z, truncate_mode='level', p=5, color_threshold=0.7 * max(Z[:, 2]))
        # plt.title('Dendrogram for Divisive Clustering (Ward Linkage)')
        # plt.xlabel('Data Point Index')
        # plt.ylabel('Distance')
        # plt.grid(True)
        # plt.show()
    elif clus_algo == "agg":
        agg_clustering = AgglomerativeClustering(n_clusters=5, linkage='ward')
        y_pred = agg_clustering.fit_predict(X)
    elif clus_algo == "dbscan":
        # pca = PCA(n_components=2)
        # X_pca = pca.fit_transform(X)

        dbscan = DBSCAN(eps=0.5, min_samples=5)
        y_pred = dbscan.fit_predict(X_pos)
    elif clus_algo == "gmm":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pos)

        # Apply GMM to cluster the data
        n_components = 5  # Number of clusters (components)
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(X_scaled)

        # Predict cluster labels
        y_pred = gmm.predict(X_scaled) + 1
    #
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)
    #
    # # Plot the clusters assigned by k-means
    # plt.figure(figsize=(10, 6))
    # scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    # plt.colorbar(scatter, label='Cluster Label')
    # plt.title('K-means Clustering on Synthetic Data (6 Classes)')
    # plt.xlabel('PCA Component 1')
    # plt.ylabel('PCA Component 2')
    # plt.grid(True)
    # plt.show()

    y_multi[y == 1] = y_pred
    return y_multi