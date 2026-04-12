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
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]

        # Xavier initialization
        std = np.sqrt(2.0 / (n_features + 1))
        self.w_ = rng.normal(0, std, n_features)
        self.b_ = 0.0
        self.loss_history_ = []

        # Internal stratified 85/15 split for early stopping
        val_frac = 0.15
        classes = np.unique(y)
        train_idx, val_idx = [], []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            n_val = max(1, int(len(cls_idx) * val_frac))
            val_idx.extend(cls_idx[:n_val].tolist())
            train_idx.extend(cls_idx[n_val:].tolist())
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Sample weights for training set
        sw = self._class_weights(y_tr) if self.class_weight else np.ones(len(y_tr))

        # l1_ratio effective per reg type
        if self.reg == "l2":
            l1_ratio_eff = 0.0
        elif self.reg == "l1":
            l1_ratio_eff = 1.0
        else:  # elasticnet
            l1_ratio_eff = self.l1_ratio

        batch_size = len(X_tr) if self.batch_size is None else self.batch_size

        best_val_loss = np.inf
        best_w = self.w_.copy()
        best_b = self.b_
        no_improve = 0

        for _ in range(self.max_epochs):
            # Shuffle training set each epoch
            perm = rng.permutation(len(X_tr))
            X_tr, y_tr, sw = X_tr[perm], y_tr[perm], sw[perm]

            # Mini-batch gradient descent
            for start in range(0, len(X_tr), batch_size):
                Xb = X_tr[start:start + batch_size]
                yb = y_tr[start:start + batch_size]
                swb = sw[start:start + batch_size]
                n = len(Xb)

                p = self.predict_proba(Xb)
                error = swb * (p - yb)

                grad_w = (Xb.T @ error) / n
                grad_b = error.mean()

                # L2 component of regularization (in-gradient)
                grad_w += self.lambda_ * (1.0 - l1_ratio_eff) * self.w_

                self.w_ -= self.lr * grad_w
                self.b_ -= self.lr * grad_b

                # L1 proximal operator (soft thresholding)
                if l1_ratio_eff > 0:
                    threshold = self.lr * self.lambda_ * l1_ratio_eff
                    self.w_ = np.sign(self.w_) * np.maximum(np.abs(self.w_) - threshold, 0.0)

            # Epoch-level tracking and early stopping
            train_p = self.predict_proba(X_tr)
            epoch_loss = self._binary_cross_entropy(y_tr, train_p, sw)
            self.loss_history_.append(epoch_loss)

            val_p = self.predict_proba(X_val)
            val_loss = self._binary_cross_entropy(y_val, val_p)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_w = self.w_.copy()
                best_b = self.b_
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        self.w_ = best_w
        self.b_ = best_b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.w_ + self.b_)

    def get_params(self) -> dict:
        if self.reg == "l2":
            l1_ratio_eff = 0.0
        elif self.reg == "l1":
            l1_ratio_eff = 1.0
        else:
            l1_ratio_eff = self.l1_ratio
        return {
            "lr": self.lr, "lambda_": self.lambda_, "reg": self.reg,
            "l1_ratio_eff": l1_ratio_eff, "batch_size": self.batch_size,
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
