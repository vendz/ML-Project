import numpy as np
from typing import Callable
from shared.base_model import BaseModel


class NeuralNetwork(BaseModel):
    def __init__(
        self,
        hidden_dims: list[int] = [64],
        activations: list[str] | str = "relu",
        optimizer: str = "adam",
        lr: float = 0.001,
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.001,
        leaky_alpha: float = 0.01,
        dropout_rate: float = 0.0,
        init_strategy: str = "xavier",
        lr_decay: str = "none",
        lr_decay_rate: float = 0.1,
        lr_decay_steps: int = 100,
        batch_size: int | None = 32,
        max_epochs: int = 500,
        patience: int = 10,
        class_weight: bool = False,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.leaky_alpha = leaky_alpha
        self.dropout_rate = dropout_rate
        self.init_strategy = init_strategy
        self.lr_decay = lr_decay
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.class_weight = class_weight
        self.random_state = random_state

        # Set after fit()
        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.activations_: list[str] = []
        self.train_loss_: list[float] = []
        self.val_loss_: list[float] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epoch_callback: Callable | None = None,
    ) -> "NeuralNetwork":
        self._rng = np.random.default_rng(self.random_state)
        self.activations_ = self._resolve_activations()
        self.train_loss_ = []
        self.val_loss_ = []

        dims = [X.shape[1]] + self.hidden_dims + [1]
        n_layers = len(dims) - 1

        self.weights_ = [self._init_weights(dims[i], dims[i + 1]) for i in range(n_layers)]
        self.biases_ = [np.zeros((1, dims[i + 1])) for i in range(n_layers)]
        self._init_optimizer()

        sample_weight = self._class_weights(y) if self.class_weight else None
        n = X.shape[0]
        batch_size = n if self.batch_size is None else self.batch_size

        best_loss = np.inf
        patience_counter = 0
        best_W = best_b = None
        t = 0

        for epoch in range(self.max_epochs):
            self._current_lr = self._get_lr(epoch)

            perm = self._rng.permutation(n)
            Xs, ys = X[perm], y[perm]
            sws = sample_weight[perm] if sample_weight is not None else None

            for start in range(0, n, batch_size):
                end = start + batch_size
                Xb, yb = Xs[start:end], ys[start:end]
                swb = sws[start:end] if sws is not None else None

                t += 1
                p, caches = self._forward(Xb, training=True)
                gW, gb = self._backward(caches, yb, p, swb)
                self._step(gW, gb, t)

            p_tr, _ = self._forward(X, training=False)
            train_loss = self._binary_cross_entropy(y, p_tr.ravel(), sample_weight)
            self.train_loss_.append(train_loss)

            val_loss = None
            if X_val is not None:
                p_va, _ = self._forward(X_val, training=False)
                val_loss = self._binary_cross_entropy(y_val, p_va.ravel())
                self.val_loss_.append(val_loss)

            if epoch_callback is not None:
                epoch_callback(epoch, train_loss, val_loss)

            monitor = val_loss if val_loss is not None else train_loss
            if monitor < best_loss - 1e-7:
                best_loss = monitor
                patience_counter = 0
                best_W = [w.copy() for w in self.weights_]
                best_b = [b.copy() for b in self.biases_]
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_W is not None:
            self.weights_, self.biases_ = best_W, best_b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p, _ = self._forward(X, training=False)
        return p.ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_params(self) -> dict:
        return {
            "hidden_dims": self.hidden_dims,
            "activations": self.activations,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "momentum": self.momentum,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "l1_lambda": self.l1_lambda,
            "l2_lambda": self.l2_lambda,
            "leaky_alpha": self.leaky_alpha,
            "dropout_rate": self.dropout_rate,
            "init_strategy": self.init_strategy,
            "lr_decay": self.lr_decay,
            "lr_decay_rate": self.lr_decay_rate,
            "lr_decay_steps": self.lr_decay_steps,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "class_weight": self.class_weight,
        }

    def _forward(self, X: np.ndarray, training: bool):
        caches = []
        A = X
        n_hidden = len(self.hidden_dims)

        for i in range(n_hidden):
            Z = A @ self.weights_[i] + self.biases_[i]
            A_act = self._activate(Z, self.activations_[i])
            mask = None
            if training and self.dropout_rate > 0.0:
                keep = 1.0 - self.dropout_rate
                mask = (self._rng.random(A_act.shape) < keep).astype(float) / keep
                A_act = A_act * mask
            caches.append((A, Z, mask))
            A = A_act

        Z_out = A @ self.weights_[-1] + self.biases_[-1]
        A_out = self._sigmoid(Z_out)
        caches.append((A, Z_out, None))
        return A_out, caches

    def _backward(
        self,
        caches: list,
        y: np.ndarray,
        p: np.ndarray,
        sample_weight: np.ndarray | None,
    ):
        n = len(y)
        n_layers = len(self.weights_)
        gW = [None] * n_layers
        gb = [None] * n_layers

        dZ = p - y.reshape(-1, 1)
        if sample_weight is not None:
            dZ = dZ * sample_weight.reshape(-1, 1)
        dZ /= n

        A_prev, _, _ = caches[-1]
        gW[-1] = A_prev.T @ dZ + self._reg_grad(self.weights_[-1])
        gb[-1] = dZ.sum(axis=0, keepdims=True)
        dA = dZ @ self.weights_[-1].T

        for i in range(n_layers - 2, -1, -1):
            A_prev, Z, mask = caches[i]
            if mask is not None:
                dA = dA * mask
            dZ = dA * self._activate_grad(Z, self.activations_[i])
            gW[i] = A_prev.T @ dZ + self._reg_grad(self.weights_[i])
            gb[i] = dZ.sum(axis=0, keepdims=True)
            dA = dZ @ self.weights_[i].T

        return gW, gb

    def _init_optimizer(self):
        if self.optimizer == "sgd":
            self._vel_W = [np.zeros_like(w) for w in self.weights_]
            self._vel_b = [np.zeros_like(b) for b in self.biases_]
        else:  # adam
            self._m_W = [np.zeros_like(w) for w in self.weights_]
            self._v_W = [np.zeros_like(w) for w in self.weights_]
            self._m_b = [np.zeros_like(b) for b in self.biases_]
            self._v_b = [np.zeros_like(b) for b in self.biases_]

    def _step(self, gW: list, gb: list, t: int):
        lr = self._current_lr
        if self.optimizer == "sgd":
            for i in range(len(self.weights_)):
                self._vel_W[i] = self.momentum * self._vel_W[i] - lr * gW[i]
                self._vel_b[i] = self.momentum * self._vel_b[i] - lr * gb[i]
                self.weights_[i] += self._vel_W[i]
                self.biases_[i] += self._vel_b[i]
        else:  # adam
            b1, b2, eps = self.beta1, self.beta2, self.epsilon
            b1_corr = 1 - b1 ** t
            b2_corr = 1 - b2 ** t
            for i in range(len(self.weights_)):
                self._m_W[i] = b1 * self._m_W[i] + (1 - b1) * gW[i]
                self._v_W[i] = b2 * self._v_W[i] + (1 - b2) * gW[i] ** 2
                self._m_b[i] = b1 * self._m_b[i] + (1 - b1) * gb[i]
                self._v_b[i] = b2 * self._v_b[i] + (1 - b2) * gb[i] ** 2

                self.weights_[i] -= lr * (self._m_W[i] / b1_corr) / (np.sqrt(self._v_W[i] / b2_corr) + eps)
                self.biases_[i]  -= lr * (self._m_b[i] / b1_corr) / (np.sqrt(self._v_b[i] / b2_corr) + eps)

    def _activate(self, Z: np.ndarray, name: str) -> np.ndarray:
        if name == "relu":
            return np.maximum(0.0, Z)
        if name == "leaky_relu":
            return np.where(Z > 0, Z, self.leaky_alpha * Z)
        if name == "tanh":
            return np.tanh(Z)
        if name == "sigmoid":
            return self._sigmoid(Z)
        raise ValueError(f"Unknown activation: {name!r}")

    def _activate_grad(self, Z: np.ndarray, name: str) -> np.ndarray:
        if name == "relu":
            return (Z > 0).astype(float)
        if name == "leaky_relu":
            return np.where(Z > 0, 1.0, self.leaky_alpha)
        if name == "tanh":
            return 1.0 - np.tanh(Z) ** 2
        if name == "sigmoid":
            s = self._sigmoid(Z)
            return s * (1.0 - s)
        raise ValueError(f"Unknown activation: {name!r}")

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _init_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        if self.init_strategy == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
        else:  # he
            std = np.sqrt(2.0 / fan_in)
        return self._rng.normal(0.0, std, (fan_in, fan_out))

    def _get_lr(self, epoch: int) -> float:
        if self.lr_decay == "step":
            return self.lr * (self.lr_decay_rate ** (epoch // self.lr_decay_steps))
        if self.lr_decay == "exponential":
            return self.lr * np.exp(-self.lr_decay_rate * epoch)
        return self.lr

    def _resolve_activations(self) -> list[str]:
        n = len(self.hidden_dims)
        if isinstance(self.activations, str):
            return [self.activations] * n
        if len(self.activations) != n:
            raise ValueError(
                f"activations length {len(self.activations)} must match "
                f"hidden_dims length {n}"
            )
        return list(self.activations)

    def _reg_grad(self, W: np.ndarray) -> np.ndarray:
        return self.l2_lambda * W + self.l1_lambda * np.sign(W)

    def _binary_cross_entropy(
        self,
        y: np.ndarray,
        p: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        eps = 1e-12
        loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        if sample_weight is not None:
            loss = loss * sample_weight
        return float(loss.mean())

    def get_layer_activations(self, X: np.ndarray) -> list[np.ndarray]:
        acts, A = [], X
        for i in range(len(self.hidden_dims)):
            Z = A @ self.weights_[i] + self.biases_[i]
            A = self._activate(Z, self.activations_[i])
            acts.append(A)
        return acts

    def _class_weights(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        classes, counts = np.unique(y, return_counts=True)
        w = np.ones(n)
        for cls, cnt in zip(classes, counts):
            w[y == cls] = n / (len(classes) * cnt)
        return w
