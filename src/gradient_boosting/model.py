import numpy as np
from shared.base_model import BaseModel


class DecisionStump:
    """Single regression tree (shallow) used as a weak learner."""

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.tree_: dict | None = None

    def fit(self, X: np.ndarray, residuals: np.ndarray,
            p: np.ndarray) -> "DecisionStump":
        # TODO: implement regression tree that minimises MSE on residuals
        # Leaf values use Newton-Raphson: sum(r) / sum(p*(1-p))
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: traverse tree and return leaf values
        raise NotImplementedError


class GradientBoostedTrees(BaseModel):
    """
    Gradient Boosted Trees for binary classification (log-loss).

    Parameters
    ----------
    n_rounds        : number of boosting rounds
    learning_rate   : shrinkage applied to each tree's contribution
    max_depth       : max depth of each weak learner
    subsample       : fraction of training samples used per round (stochastic GBT)
    patience        : early-stopping rounds on validation loss (0 = disabled)
    class_weight    : True to upweight minority class via sample weights
    """

    def __init__(
        self,
        n_rounds: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        patience: int = 10,
        class_weight: bool = False,
        random_state: int = 42,
    ):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.patience = patience
        self.class_weight = class_weight
        self.random_state = random_state

        self.F0_: float | None = None
        self.trees_: list[DecisionStump] = []
        self.train_loss_: list[float] = []
        self.val_loss_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "GradientBoostedTrees":
        # TODO: implement boosting loop
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # TODO: F0 + sum(lr * tree.predict(X)) -> sigmoid
        raise NotImplementedError

    def feature_importance(self) -> np.ndarray:
        """Total gain per feature across all trees."""
        # TODO: implement
        raise NotImplementedError

    def get_params(self) -> dict:
        return {
            "n_rounds": self.n_rounds, "learning_rate": self.learning_rate,
            "max_depth": self.max_depth, "subsample": self.subsample,
            "patience": self.patience, "class_weight": self.class_weight,
        }

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _log_loss(self, y: np.ndarray, p: np.ndarray) -> float:
        eps = 1e-12
        return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    def _class_weights(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        classes, counts = np.unique(y, return_counts=True)
        w = np.ones(n)
        for cls, cnt in zip(classes, counts):
            w[y == cls] = n / (len(classes) * cnt)
        return w
