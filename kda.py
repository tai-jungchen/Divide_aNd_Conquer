import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def rbf_kernel(X, Y=None, sigma=1.0):
    # Ensure inputs are 2D
    X = np.atleast_2d(X)
    if Y is None:
        Y = X
    else:
        Y = np.atleast_2d(Y)

    # Check feature dimension compatibility
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"Feature dimensions must match: X has {X.shape[1]}, Y has {Y.shape[1]}")

    # Compute pairwise distances
    if X is Y:  # Same data: use pdist for efficiency
        pairwise_dists = squareform(pdist(X, 'euclidean'))
    else:  # Different data: compute distances directly
        X_exp = X[:, np.newaxis, :]  # Shape: (n_test, 1, n_features)
        Y_exp = Y[np.newaxis, :, :]  # Shape: (1, n_train, n_features)
        pairwise_dists = np.sqrt(((X_exp - Y_exp) ** 2).sum(axis=2))  # Shape: (n_test, n_train)

    # Compute kernel
    K = np.exp(-pairwise_dists ** 2 / (2 * sigma ** 2))
    return K


# Kernel Discriminant Analysis implementation
class KDA:
    def __init__(self, sigma=1.0, n_components=None):
        self.sigma = sigma  # Kernel bandwidth
        self.n_components = n_components  # Number of discriminant components
        self.classes_ = None
        self.means_ = None
        self.W_ = None  # Projection matrix

    def fit(self, X, y):
        """
        Fit the KDA model.
        X: Training data (n_samples, n_features)
        y: Labels (n_samples,)
        """
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.classes_ = le.classes_
        n_classes = len(self.classes_)

        # Compute kernel matrix
        K = rbf_kernel(X, sigma=self.sigma)
        n_samples = X.shape[0]

        # Center the kernel matrix
        I = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - I @ K - K @ I + I @ K @ I

        # Compute within-class scatter in kernel space
        Sw = np.zeros((n_samples, n_samples))
        class_masks = [y == c for c in range(n_classes)]
        for mask in class_masks:
            K_class = K_centered[np.ix_(mask, mask)]
            n_c = np.sum(mask)
            if n_c > 1:  # Need at least 2 samples per class
                Sw[np.ix_(mask, mask)] += K_class - np.eye(n_c) * K_class.mean()

        # Compute between-class scatter in kernel space
        Sb = np.zeros((n_samples, n_samples))
        for c in range(n_classes):
            mask = y == c
            n_c = np.sum(mask)
            if n_c > 0:
                mean_c = K_centered[:, mask].mean(axis=1)
                Sb += n_c * np.outer(mean_c, mean_c)

        # Solve eigenvalue problem (Sb * w = lambda * Sw * w)
        # Add small regularization to Sw to avoid singularity
        Sw += np.eye(n_samples) * 1e-6
        eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(Sw) @ Sb)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Select number of components
        if self.n_components is None:
            self.n_components = n_classes - 1
        self.W_ = eigvecs[:, :self.n_components]

        # Store kernel means for prediction
        self.means_ = np.array([K[:, y == c].mean(axis=1) for c in range(n_classes)])
        self.X_train_ = X
        return self

    def transform(self, X):
        """
        Project data into the discriminant space.
        """
        K = rbf_kernel(X, self.X_train_, sigma=self.sigma)
        return K @ self.W_

    def predict(self, X):
        """
        Predict class labels for X.
        """
        K = rbf_kernel(X, self.X_train_, sigma=self.sigma)
        projections = K @ self.W_
        distances = np.array([np.linalg.norm(projections - (self.means_[c] @ self.W_), axis=1)
                              for c in range(len(self.classes_))]).T
        return self.classes_[np.argmin(distances, axis=1)]


# Example usage
# if __name__ == "__main__":
#     # Generate synthetic data
#     df = pd.read_csv("datasets/preprocessed/mpmc.csv")
#     df = df[df['failure.type'] != 5]
#     X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-2], df['failure.type'], test_size=0.3,
#                                                         stratify=df['failure.type'], random_state=0)
#
#     # Fit KDA
#     kda = KDA(sigma=1.0, n_components=y_train.nunique()-1)
#     kda.fit(X_train, y_train)
#
#     # Predict and evaluate
#     y_pred = kda.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy:.2f}")
#
#     # Transform data (for visualization or further use)
#     X_transformed = kda.transform(X_test)
#     print("Transformed data shape:", X_transformed.shape)
#
#     # Perform dimensionality reduction on training and test data
#     X_train_reduced = kda.transform(X_train)
#     X_test_reduced = kda.transform(X_test)
#
#     # Visualize the reduced data
#     plt.figure(figsize=(10, 6))
#
#     # Plot training data
#     for class_label in np.unique(y_train):
#         mask = y_train == class_label
#         plt.scatter(X_train_reduced[mask, 0], X_train_reduced[mask, 1],
#                     label=f"Train Class {class_label}", alpha=0.6, edgecolors='w')
#
#     # Plot test data with different marker
#     for class_label in np.unique(y_test):
#         mask = y_test == class_label
#         plt.scatter(X_test_reduced[mask, 0], X_test_reduced[mask, 1],
#                     label=f"Test Class {class_label}", marker='x', alpha=0.8)
#
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.title("KDA Dimensionality Reduction (2D Projection)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
