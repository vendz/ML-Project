"""
Feedforward Neural Network — Binary (Enrolled vs. Graduate)
Owner: Vandit Vasa
"""
import numpy as np
from shared.base_model import BaseModel


class NeuralNetwork(BaseModel):
    """
    Fully-connected feedforward neural network.
    Architecture: Input -> [hidden layers with ReLU] -> Sigmoid output

    Parameters
    ----------
    hidden_dims  : list of ints, one entry per hidden layer (e.g. [64, 32])
    lr           : learning rate
    lambda_      : L2 weight-decay strength
    dropout_rate : fraction of hidden units to zero out during training
    batch_size   : mini-batch size; None = full-batch
    max_epochs   : maximum training epochs
    patience     : early-stopping patience
    momentum     : SGD momentum coefficient
    class_weight : True to upweight minority class
    """

    def __init__(
        self,
        hidden_dims: list[int] = [64],
        lr: float = 0.01,
        lambda_: float = 0.001,
        dropout_rate: float = 0.0,
        batch_size: int | None = 32,
        max_epochs: int = 500,
        patience: int = 10,
        momentum: float = 0.0,
        class_weight: bool = False,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.lambda_ = lambda_
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.momentum = momentum
        self.class_weight = class_weight
        self.random_state = random_state

        self.weights_: list[np.ndarray] = []   # W for each layer
        self.biases_:  list[np.ndarray] = []   # b for each layer
        self.train_loss_: list[float] = []
        self.val_loss_:   list[float] = []

    # ── public API ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "NeuralNetwork":
        # TODO: Xavier init, forward pass, backprop, weight decay, dropout, early stopping
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # TODO: forward pass (no dropout at inference)
        raise NotImplementedError

    def get_params(self) -> dict:
        return {
            "hidden_dims": self.hidden_dims, "lr": self.lr,
            "lambda_": self.lambda_, "dropout_rate": self.dropout_rate,
            "batch_size": self.batch_size, "max_epochs": self.max_epochs,
            "patience": self.patience, "momentum": self.momentum,
            "class_weight": self.class_weight,
        }

    # ── private helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    @staticmethod
    def _relu_grad(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _xavier_init(self, fan_in: int, fan_out: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return rng.normal(0, std, (fan_in, fan_out))

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
