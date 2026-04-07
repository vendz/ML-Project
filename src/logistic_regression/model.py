import numpy as np
from shared.base_model import BaseModel


class LogisticRegression(BaseModel):
    """
    Binary logistic regression trained with gradient descent.

    Parameters
    ----------
    lr          : learning rate
    lambda_     : regularization strength
    reg         : 'l2' | 'l1' | 'elasticnet'
    l1_ratio    : mix ratio for elasticnet (0 = L2, 1 = L1)
    batch_size  : samples per update; None = full-batch
    max_epochs  : maximum training epochs
    patience    : early-stopping patience (epochs without val loss improvement)
    class_weight: True to upweight minority class
    """

    def __init__(
        self,
        lr: float = 0.01,
        lambda_: float = 0.001,
        reg: str = "l2",
        l1_ratio: float = 0.5,
        batch_size: int | None = 32,
        max_epochs: int = 1000,
        patience: int = 10,
        class_weight: bool = False,
        random_state: int = 42,
    ):
        self.lr = lr
        self.lambda_ = lambda_
        self.reg = reg
        self.l1_ratio = l1_ratio
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.class_weight = class_weight
        self.random_state = random_state

        self.w_: np.ndarray | None = None
        self.b_: float | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        # TODO: implement
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # TODO: implement sigmoid(X @ self.w_ + self.b_)
        raise NotImplementedError

    def get_params(self) -> dict:
        return {
            "lr": self.lr, "lambda_": self.lambda_, "reg": self.reg,
            "l1_ratio": self.l1_ratio, "batch_size": self.batch_size,
            "max_epochs": self.max_epochs, "patience": self.patience,
            "class_weight": self.class_weight,
        }

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _binary_cross_entropy(self, y: np.ndarray, p: np.ndarray,
                              sample_weight: np.ndarray | None = None) -> float:
        eps = 1e-12
        loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        if sample_weight is not None:
            loss = loss * sample_weight
        return float(loss.mean())

    def _class_weights(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        classes, counts = np.unique(y, return_counts=True)
        w = np.ones(n)
        for cls, cnt in zip(classes, counts):
            w[y == cls] = n / (len(classes) * cnt)
        return w
