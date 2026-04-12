"""
Abstract base class that every model must implement.

This contract is what lets any cross-model comparison code call all three models identically.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """
        Train the model.

        Parameters
        ----------
        X : (n_samples, n_features) float array, already preprocessed
        y : (n_samples,) int array, 0 = Enrolled, 1 = Dropout

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Returns
        -------
        y_pred : (n_samples,) int array of 0s and 1s
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of the positive class (Dropout).

        Returns
        -------
        proba : (n_samples,) float array in [0, 1]
        """

    def get_params(self) -> dict:
        """Return hyperparameters as a dict (used by experiment logger)."""
        return {}
