"""
Shared evaluation utilities.

All metric functions work on (y_true, y_pred) or (y_true, y_proba)
and return plain dicts/arrays so results are easy to log or plot.
"""
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from shared.config import EXPERIMENTS


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Returns per-class and macro precision, recall, F1, plus accuracy.
    Classes: 0 = Enrolled, 1 = Graduate
    """
    cm = confusion_matrix(y_true, y_pred)
    report = {}
    for cls in range(2):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        label = "enrolled" if cls == 0 else "dropout"
        report[label] = {"precision": precision, "recall": recall, "f1": f1,
                         "support": int(cm[cls, :].sum())}

    macro_f1 = np.mean([report[k]["f1"] for k in report])
    accuracy  = cm.diagonal().sum() / cm.sum()
    report["macro_f1"] = float(macro_f1)
    report["accuracy"]  = float(accuracy)
    return report


def roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    thresholds = np.sort(np.unique(y_proba))[::-1]
    fpr_list, tpr_list = [0.0], [0.0]
    neg = (y_true == 0).sum()
    pos = (y_true == 1).sum()
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fpr_list.append(fp / neg if neg > 0 else 0.0)
        tpr_list.append(tp / pos if pos > 0 else 0.0)
    fpr_list.append(1.0)
    tpr_list.append(1.0)
    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def stratified_kfold_indices(y: np.ndarray, k: int = 5, random_state: int = 42):
    """Yield (train_idx, val_idx) for k stratified folds."""
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    class_indices = {c: rng.permutation(np.where(y == c)[0]) for c in classes}
    folds = [[] for _ in range(k)]
    for indices in class_indices.values():
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)
    for fold_idx in range(k):
        val_idx   = np.array(folds[fold_idx])
        train_idx = np.array([i for j, f in enumerate(folds) if j != fold_idx for i in f])
        yield train_idx, val_idx


def cross_validate(model_cls, params: dict, X: np.ndarray, y: np.ndarray,
                   k: int = 5) -> dict:
    f1s, accs = [], []
    for train_idx, val_idx in stratified_kfold_indices(y, k):
        model = model_cls(**params).fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        report = classification_report(y[val_idx], y_pred)
        f1s.append(report["macro_f1"])
        accs.append(report["accuracy"])
    return {
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std":  float(np.std(f1s)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std":  float(np.std(accs)),
    }


def log_experiment(model_name: str, params: dict, metrics: dict,
                   extra: dict | None = None):
    out_dir = EXPERIMENTS / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "params":    params,
        "metrics":   metrics,
    }
    if extra:
        record["extra"] = extra
    with open(out_dir / "log.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
