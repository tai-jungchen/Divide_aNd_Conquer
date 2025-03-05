import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array


class WeightedLDA(BaseEstimator, TransformerMixin):
    """
    Weighted Linear Discriminant Analysis with custom class weights.

    Parameters:
    - n_components: int or None, optional (default=None)
        Number of components for dimensionality reduction. If None, min(n_features, n_classes - 1).
    - solver: str, optional (default='eigen')
        Solver to use: 'eigen' for eigenvalue decomposition (only option here).
    - shrinkage: str or float, optional (default=None)
        Not implemented (placeholder for sklearn compatibility).
    - priors: array-like, shape (n_classes,), optional (default=None)
        Class priors (not used in fitting, included for compatibility).
    - class_weights: array-like, shape (n_classes,), optional (default=None)
        Custom weights for each class in scatter matrix computation. If None, uses sample counts.
    """

    def __init__(self, n_components=None, solver='eigen', shrinkage=None, priors=None, class_weights=None):
        self.n_components = n_components
        self.solver = solver
        self.shrinkage = shrinkage  # Placeholder, not implemented
        self.priors = priors  # Placeholder, not used in fitting
        self.class_weights = class_weights

    def fit(self, X, y):
        """Fit the Weighted LDA model."""
        # Input validation
        X, y = check_X_y(X, y, ensure_2d=True)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Determine number of components
        max_components = min(n_features, n_classes - 1)
        if self.n_components is None:
            self.n_components_ = max_components
        else:
            self.n_components_ = min(self.n_components, max_components)
            if self.n_components_ <= 0:
                raise ValueError("n_components must be positive.")

        # Compute class means and overall mean
        self.means_ = np.zeros((n_classes, n_features))
        class_counts = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            class_data = X[y == c]
            self.means_[i] = np.mean(class_data, axis=0)
            class_counts[i] = len(class_data)
        self.overall_mean_ = np.mean(X, axis=0)

        # Set class weights (default to sample counts if None)
        if self.class_weights is None:
            self.class_weights_ = class_counts
        else:
            if len(self.class_weights) != n_classes:
                raise ValueError(f"class_weights must have length {n_classes}, got {len(self.class_weights)}.")
            self.class_weights_ = np.asarray(self.class_weights)
            if np.any(self.class_weights_ < 0):
                raise ValueError("class_weights must be non-negative.")

        # Compute weighted within-class scatter (S_W)
        self.S_W_ = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            class_data = X[y == c]
            centered = class_data - self.means_[i]
            class_cov = np.dot(centered.T, centered) / (class_counts[i] - 1 if class_counts[i] > 1 else 1)
            self.S_W_ += self.class_weights_[i] * class_cov

        # Compute weighted between-class scatter (S_B)
        self.S_B_ = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            mean_diff = (self.means_[i] - self.overall_mean_).reshape(-1, 1)
            self.S_B_ += self.class_weights_[i] * np.dot(mean_diff, mean_diff.T)

        # Solve eigenvalue problem: S_W^-1 S_B w = lambda w
        eigenvals, eigenvecs = eigh(self.S_B_, self.S_W_)
        idx = np.argsort(eigenvals)[::-1]  # Sort in descending order
        self.eigenvals_ = eigenvals[idx]
        self.eigenvecs_ = eigenvecs[:, idx]

        # Select top n_components eigenvectors
        self.coef_ = self.eigenvecs_[:, :self.n_components_]

        return self

    def transform(self, X):
        """Transform X into the LDA subspace."""
        X = check_array(X, ensure_2d=True)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(f"X has {X.shape[1]} features, but WeightedLDA was fitted with {self.coef_.shape[0]}.")
        return np.dot(X, self.coef_)

    def fit_transform(self, X, y):
        """Fit the model and transform X."""
        self.fit(X, y)
        return self.transform(X)


def compute_fishers_criterion(X, y, w):
    """
    Compute Fisher's Criterion J(w) = (w^T S_B w) / (w^T S_W w) to evaluate LDA performance.

    Parameters:
    - X: numpy array of shape (n_samples, n_features), the input data
    - y: numpy array of shape (n_samples,), the class labels
    - w: numpy array of shape (n_features,), the projection vector from LDA

    Returns:
    - J: float, the value of Fisher's Criterion
    """
    # Ensure inputs are numpy arrays and w is a column vector
    X = np.asarray(X)
    y = np.asarray(y)
    w = np.asarray(w).reshape(-1, 1)  # Ensure w is (n_features, 1)

    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    # Compute class means and overall mean
    class_means = np.zeros((n_classes, n_features))
    class_counts = np.zeros(n_classes)
    for i, c in enumerate(classes):
        class_data = X[y == c]
        class_means[i] = np.mean(class_data, axis=0)
        class_counts[i] = len(class_data)
    overall_mean = np.mean(X, axis=0).reshape(-1, 1)

    # Compute within-class scatter matrix (S_W)
    S_W = np.zeros((n_features, n_features))
    for i, c in enumerate(classes):
        class_data = X[y == c]
        centered = class_data - class_means[i]
        S_W += np.dot(centered.T, centered)  # Sum of class covariances

    # Compute between-class scatter matrix (S_B)
    S_B = np.zeros((n_features, n_features))
    for i, c in enumerate(classes):
        mean_diff = (class_means[i] - overall_mean.T).reshape(-1, 1)
        S_B += class_counts[i] * np.dot(mean_diff, mean_diff.T)

    # Compute Fisher's Criterion: J(w) = (w^T S_B w) / (w^T S_W w)
    numerator = w.T @ S_B @ w  # w^T S_B w (scalar since w is a vector)
    denominator = w.T @ S_W @ w  # w^T S_W w (scalar)

    # Avoid division by zero
    if denominator == 0:
        raise ValueError("Within-class scatter (w^T S_W w) is zero, cannot compute J(w).")

    J = numerator / denominator

    return J.item()  # Return as scalar


# Example usage
if __name__ == "__main__":
    # Sample multi-class data
    X = np.array([[1, 2], [1.5, 1.8],  # Class 0
                  [5, 6], [5.2, 5.8]])  # Class 1
    y = np.array([0, 0, 1, 1])

    # Initialize with custom weights (equal weights for all classes)
    custom_weights = [0.66, 0.33]  # Equal weight regardless of sample size
    wlda = WeightedLDA(n_components=1, class_weights=custom_weights)

    # Fit and transform training data
    X_train_transformed = wlda.fit_transform(X, y)
    # print("Transformed training data:\n", X_train_transformed)
    print(f"compute_fishers_criterion: {compute_fishers_criterion(X, y, wlda.coef_)}")

    # Transform new/test data
    X_test = np.array([[1.2, 2.0], [5.1, 6.0], [2.1, 4.0]])
    X_test_transformed = wlda.transform(X_test)
    # print("Transformed test data:\n", X_test_transformed)

    # Compare with sklearn LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis(n_components=1, solver='eigen')
    X_lda_transformed = lda.fit_transform(X, y)
    # print("Sklearn LDA transformed data:\n", X_lda_transformed)
    print(f"compute_fishers_criterion: {compute_fishers_criterion(X, y, lda.coef_)}")