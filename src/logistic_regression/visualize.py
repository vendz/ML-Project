"""
Logistic Regression visualizations.
Owner: Brandon Tran

Run with:  python -m logistic_regression.visualize
Outputs saved to:  results/figures/
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from shared.config import EXPERIMENTS, RESULTS


def plot_imbalance_comparison():
    """
    Grouped bar chart comparing the four class imbalance strategies on the test set.
    Uses the candidate_batch=32 results (empirically better config).

    Left subplot : macro F1, enrolled F1, dropout F1, accuracy
                   (accuracy included to show the accuracy paradox)
    Right subplot: dropout precision vs dropout recall
    """
    # Load and filter log
    log_path = EXPERIMENTS / "logistic_regression" / "log.jsonl"
    entries = [json.loads(line) for line in log_path.read_text().splitlines()]
    entries = [e for e in entries
               if e.get("extra", {}).get("split") == "test"
               and e.get("extra", {}).get("candidate_batch") == 32]

    order = ["none", "class_weight", "smote", "undersample"]
    by_strat = {e["extra"]["imbalance_strategy"]: e["metrics"] for e in entries}

    macro_f1     = [by_strat[s]["macro_f1"]             for s in order]
    enrolled_f1  = [by_strat[s]["enrolled"]["f1"]       for s in order]
    dropout_f1   = [by_strat[s]["dropout"]["f1"]        for s in order]
    accuracy     = [by_strat[s]["accuracy"]              for s in order]
    dropout_prec = [by_strat[s]["dropout"]["precision"]  for s in order]
    dropout_rec  = [by_strat[s]["dropout"]["recall"]     for s in order]

    x = np.arange(len(order))
    w = 0.18  # narrower to fit 4 bars per group

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Logistic Regression — Class Imbalance Strategy Comparison", fontsize=13)

    # Left: F1 scores + accuracy
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    bars_left = [
        ax1.bar(x + offsets[0], macro_f1,    w, label="Macro F1"),
        ax1.bar(x + offsets[1], enrolled_f1, w, label="Enrolled F1"),
        ax1.bar(x + offsets[2], dropout_f1,  w, label="Dropout F1"),
        ax1.bar(x + offsets[3], accuracy,    w, label="Accuracy"),
    ]
    ax1.set_title("F1 Scores & Accuracy by Strategy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=15)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Score")
    ax1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4)
    for bars in bars_left:
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    # Right: Dropout precision vs recall
    b4 = ax2.bar(x - w / 2, dropout_prec, w, label="Dropout Precision")
    b5 = ax2.bar(x + w / 2, dropout_rec,  w, label="Dropout Recall")
    ax2.set_title("Dropout Class — Precision vs Recall")
    ax2.set_xticks(x)
    ax2.set_xticklabels(order, rotation=15)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Score")
    ax2.legend()
    for bar in [*b4, *b5]:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = RESULTS / "figures" / "lr_imbalance_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_roc_curves():
    """
    ROC curves for each imbalance strategy on the test set.
    Re-fits one model per strategy using the confirmed best config (batch_size=32).
    All four curves plotted on a single axes for direct AUC comparison.
    """
    from shared.preprocessing import load_data, StandardScaler, SMOTE, random_undersample
    from shared.evaluation import roc_auc
    from shared.config import RANDOM_SEED
    from logistic_regression.model import LogisticRegression

    X_train, X_test, y_train, y_test, _ = load_data("raw/dataset.csv")
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    best = dict(lr=0.1, lambda_=0.001, reg="l1", batch_size=32,
                max_epochs=1000, patience=10)

    strategies = {
        "none":         (X_train_s, y_train, False),
        "class_weight": (X_train_s, y_train, True),
        "smote":        (*SMOTE(random_state=RANDOM_SEED).fit_resample(X_train_s, y_train), False),
        "undersample":  (*random_undersample(X_train_s, y_train, RANDOM_SEED), False),
    }

    _, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random (AUC = 0.500)")

    for name, (Xb, yb, use_cw) in strategies.items():
        model = LogisticRegression(**best, class_weight=use_cw).fit(Xb, yb)
        scores = model.predict_proba(X_test_s)
        fpr, tpr, auc = roc_auc(y_test, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Logistic Regression — ROC Curves by Imbalance Strategy")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = RESULTS / "figures" / "lr_roc_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray):
    """
    Compute precision-recall curve for the positive (dropout) class.
    Returns (recalls, precisions, area_under_curve).
    """
    thresholds = np.sort(np.unique(y_score))[::-1]
    pos = (y_true == 1).sum()
    precisions, recalls = [1.0], [0.0]  # anchor at recall=0, precision=1
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall    = tp / pos if pos > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
    area = float(np.trapezoid(precisions, recalls))
    return np.array(recalls), np.array(precisions), area


def plot_precision_recall_curves():
    """
    Precision-recall curves for each imbalance strategy on the test set (dropout class).
    PR curves are more informative than ROC for imbalanced classification — they focus
    entirely on the positive class and are not inflated by true negatives.
    Baseline is a horizontal line at the positive class prevalence (~0.235).
    """
    from shared.preprocessing import load_data, StandardScaler, SMOTE, random_undersample
    from shared.config import RANDOM_SEED
    from logistic_regression.model import LogisticRegression

    X_train, X_test, y_train, y_test, _ = load_data("raw/dataset.csv")
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    best = dict(lr=0.1, lambda_=0.001, reg="l1", batch_size=32,
                max_epochs=1000, patience=10)

    strategies = {
        "none":         (X_train_s, y_train, False),
        "class_weight": (X_train_s, y_train, True),
        "smote":        (*SMOTE(random_state=RANDOM_SEED).fit_resample(X_train_s, y_train), False),
        "undersample":  (*random_undersample(X_train_s, y_train, RANDOM_SEED), False),
    }

    _, ax = plt.subplots(figsize=(7, 6))
    prevalence = (y_test == 1).sum() / len(y_test)
    ax.axhline(prevalence, color="grey", linestyle="--",
               label=f"Random (AP = {prevalence:.3f})")

    for name, (Xb, yb, use_cw) in strategies.items():
        model = LogisticRegression(**best, class_weight=use_cw).fit(Xb, yb)
        scores = model.predict_proba(X_test_s)
        recall, precision, area = _precision_recall_curve(y_test, scores)
        ax.plot(recall, precision, label=f"{name} (AP = {area:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Logistic Regression — Precision-Recall Curves (Dropout Class)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = RESULTS / "figures" / "lr_pr_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_convergence_curves():
    """
    Epoch-level training loss convergence for each batch size configuration.
    Uses the same params as the batch sweep in experiments.py.
    Fits on the full training set (no CV split); reads model.loss_history_ after fit.

    Note: loss_history_ records one loss value per epoch (aggregated across all
    mini-batches), not per-update. One epoch contains vastly different numbers of
    parameter updates for batch_size=1 vs None — the x-axis shows epochs, not
    optimization effort.

    Two panels: left zoomed to first 100 epochs to show early differences among
    SGD/mini-batch configs; right shows the full range including full-batch.
    """
    from shared.preprocessing import load_data, StandardScaler
    from logistic_regression.model import LogisticRegression

    X_train, _, y_train, _, _ = load_data("raw/dataset.csv")
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    base = dict(lr=0.01, lambda_=0.01, reg="l2",
                max_epochs=1000, patience=10, class_weight=True)
    configs = [
        (1,    "SGD (batch=1)"),
        (32,   "Mini-batch (batch=32)"),
        (64,   "Mini-batch (batch=64)"),
        (None, "Full-batch"),
    ]

    histories = []
    for bs, label in configs:
        model = LogisticRegression(**base, batch_size=bs).fit(X_train_s, y_train)
        histories.append((label, model.loss_history_))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Logistic Regression — Training Loss by Batch Size", fontsize=13)

    for label, history in histories:
        ax1.plot(history, label=label, alpha=0.85)
        ax2.plot(history, alpha=0.85)

    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss (Cross-Entropy)")
    ax1.set_title("Early Epochs (0–100)")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.set_xlabel("Epoch")
    ax2.set_title("Full Training Range")

    plt.tight_layout()
    out = RESULTS / "figures" / "lr_convergence_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_regularization_path():
    """
    Shows how the learned weight vector changes as lambda_ increases for L1, L2,
    and Elastic Net regularization.

    Left panel : L2 norm of w_ vs lambda (log scale) — overall shrinkage rate
    Right panel: Fraction of near-zero coefficients (|w_i| < 1e-4) vs lambda —
                 sparsity; directly shows L1's tendency to zero out weights

    Elastic Net uses l1_ratio=0.5 to position it as a visible blend between L1 and L2.
    Lambda grid matches the regularization sweep in experiments.py.
    """
    from shared.preprocessing import load_data, StandardScaler
    from logistic_regression.model import LogisticRegression

    X_train, _, y_train, _, _ = load_data("raw/dataset.csv")
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    lambdas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    # lr=0.01 matches the regularization sweep in experiments.py exactly,
    # so this path directly visualizes the logged sweep results
    base = dict(lr=0.01, batch_size=32, max_epochs=1000,
                patience=10, class_weight=True)
    reg_configs = [
        ("l1",                          dict(reg="l1")),
        ("l2",                          dict(reg="l2")),
        ("elasticnet\n(l1_ratio=0.5)",  dict(reg="elasticnet", l1_ratio=0.5)),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Logistic Regression — Regularization Path", fontsize=13)

    for label, reg_kwargs in reg_configs:
        norms, sparsities = [], []
        for lam in lambdas:
            model = LogisticRegression(**base, lambda_=lam, **reg_kwargs).fit(
                X_train_s, y_train)
            w = model.w_
            norms.append(float(np.linalg.norm(w)))
            sparsities.append(float((np.abs(w) < 1e-4).mean()))
        ax1.plot(lambdas, norms,      marker="o", label=label)
        ax2.plot(lambdas, sparsities, marker="o", label=label)

    ax1.set_xscale("log")
    ax1.set_xlabel("lambda_ (log scale)")
    ax1.set_ylabel("||w||₂  (L2 norm of weights)")
    ax1.set_title("Weight Shrinkage")
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_xlabel("lambda_ (log scale)")
    ax2.set_ylabel("Fraction of |wᵢ| < 1e-4")
    ax2.set_title("Sparsity (Near-Zero Coefficients)")
    ax2.set_ylim(0, 1)
    ax2.legend()

    plt.tight_layout()
    out = RESULTS / "figures" / "lr_regularization_path.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_imbalance_comparison()
    plot_roc_curves()
    plot_precision_recall_curves()
    plot_convergence_curves()
    plot_regularization_path()
