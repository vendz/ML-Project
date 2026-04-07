"""
Shared preprocessing pipeline

Usage
-----
from shared.preprocessing import load_data, StandardScaler, SMOTE, PCA, make_pipeline

X_train, X_test, y_train, y_test = load_data()
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split  # only for stratified split

from shared.config import DATA_RAW, RANDOM_SEED, TEST_SIZE, TARGET_COL, POSITIVE_CLASS


def load_data(csv_name: str = "dataset.csv"):
    """
    Load raw CSV, drop Dropout rows, encode target, return train/test splits.

    Returns
    -------
    X_train, X_test : float arrays
    y_train, y_test : int arrays  (0 = Enrolled, 1 = Graduate)
    feature_names   : list[str]
    """
    path = DATA_RAW / csv_name
    df = pd.read_csv(path, sep=";")

    # Keep only Enrolled and Graduate
    df = df[df[TARGET_COL].isin(["Enrolled", "Graduate"])].copy()
    df[TARGET_COL] = (df[TARGET_COL] == POSITIVE_CLASS).astype(int)

    X = df.drop(columns=[TARGET_COL]).values.astype(float)
    y = df[TARGET_COL].values.astype(int)
    feature_names = df.drop(columns=[TARGET_COL]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test, feature_names


class StandardScaler:
    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1  # avoid divide-by-zero for constant features
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class SMOTE:
    """Synthetic Minority Oversampling"""

    def __init__(self, k: int = 5, random_state: int = RANDOM_SEED):
        self.k = k
        self.rng = np.random.default_rng(random_state)

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        classes, counts = np.unique(y, return_counts=True)
        majority_count = counts.max()
        X_resampled, y_resampled = X.copy(), y.copy()

        for cls, count in zip(classes, counts):
            if count == majority_count:
                continue
            n_synthetic = majority_count - count
            X_cls = X[y == cls]
            synthetic = self._generate(X_cls, n_synthetic)
            X_resampled = np.vstack([X_resampled, synthetic])
            y_resampled = np.hstack([y_resampled, np.full(n_synthetic, cls)])

        return X_resampled, y_resampled

    def _generate(self, X_cls: np.ndarray, n: int) -> np.ndarray:
        synthetic = []
        for _ in range(n):
            idx = self.rng.integers(len(X_cls))
            sample = X_cls[idx]
            dists = np.linalg.norm(X_cls - sample, axis=1)
            dists[idx] = np.inf
            nn_indices = np.argsort(dists)[: self.k]
            neighbor = X_cls[self.rng.choice(nn_indices)]
            alpha = self.rng.random()
            synthetic.append(sample + alpha * (neighbor - sample))
        return np.array(synthetic)


def random_undersample(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_SEED):
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    minority_count = counts.min()
    indices = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        indices.append(rng.choice(cls_idx, size=minority_count, replace=False))
    idx = np.concatenate(indices)
    rng.shuffle(idx)
    return X[idx], y[idx]


class PCA:
    """PCA via eigendecomposition of the covariance matrix."""

    def __init__(self, variance_threshold: float = 0.95):
        self.variance_threshold = variance_threshold

    def fit(self, X: np.ndarray) -> "PCA":
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        cov = (X_c.T @ X_c) / (len(X) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        explained = np.cumsum(eigenvalues) / eigenvalues.sum()
        n_components = int(np.searchsorted(explained, self.variance_threshold) + 1)
        self.components_ = eigenvectors[:, :n_components]
        self.explained_variance_ratio_ = eigenvalues[:n_components] / eigenvalues.sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
