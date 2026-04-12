import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_confusion_matrix(cm: np.ndarray,
                          labels: list[str] = ["Enrolled", "Graduate"]) -> plt.Figure:
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    return fig


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return fig


def plot_loss_curve(train_loss: list[float],
                    val_loss: list[float] | None = None) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(train_loss, label="Train loss")
    if val_loss:
        ax.plot(val_loss, label="Val loss")
    ax.set_xlabel("Epoch / Round")
    ax.set_ylabel("Loss")
    ax.set_title("Convergence")
    ax.legend()
    return fig


def plot_feature_importance(importances: np.ndarray,
                             feature_names: list[str],
                             top_n: int = 15) -> plt.Figure:
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(6, top_n * 0.4))
    ax.barh([feature_names[i] for i in idx], importances[idx])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features")
    fig.tight_layout()
    return fig


def plot_pr_curve(recalls: np.ndarray, precisions: np.ndarray,
                  auc_pr: float, baseline: float | None = None) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(recalls, precisions, label=f"AUC-PR = {auc_pr:.3f}")
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", color="gray",
                   label=f"Random = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_weight_distributions(weights: list[np.ndarray],
                               labels: list[str]) -> plt.Figure:
    n = len(weights)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, W, label in zip(axes, weights, labels):
        ax.hist(W.ravel(), bins=40, alpha=0.85, edgecolor="none")
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("Value", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
        ))
    fig.suptitle("Weight Distributions", fontsize=9)
    fig.tight_layout()
    return fig


def plot_dead_neurons(labels: list[str], pcts: list[float],
                      act_names: list[str]) -> plt.Figure:
    _legend = {
        "relu":       "dead (output = 0)",
        "leaky_relu": "negative activation",
        "tanh":       "saturated (|a| > 0.97)",
        "sigmoid":    "saturated (a < 0.05 or > 0.95)",
    }
    colors = ["#e74c3c" if p > 50 else "#f39c12" if p > 20 else "#2ecc71"
              for p in pcts]
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(labels) * 0.55 + 1)))
    visible_pcts = [max(p, 1.0) for p in pcts]
    bars = ax.barh(labels, visible_pcts, color=colors)
    for bar, p in zip(bars, pcts):
        ax.text(max(p + 1.5, 3.0), bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", fontsize=8)
    ax.set_xlim([0, 100])
    ax.set_title("Neuron Health per Layer")
    unique_acts = list(dict.fromkeys(act_names))
    note = "\n".join(f"{a}: {_legend.get(a, '')}" for a in unique_acts)
    ax.set_xlabel(f"% neurons flagged\n{note}", fontsize=7)
    fig.tight_layout()
    return fig


def plot_confidence_histogram(y_true: np.ndarray,
                               y_proba: np.ndarray) -> plt.Figure:
    bins = np.linspace(0, 1, 31)
    fig, ax = plt.subplots()
    ax.hist(y_proba[y_true == 0], bins=bins, alpha=0.6,
            label="Enrolled (true 0)", color="#3498db")
    ax.hist(y_proba[y_true == 1], bins=bins, alpha=0.6,
            label="Graduate (true 1)", color="#e74c3c")
    ax.axvline(0.5, linestyle="--", color="black",
               linewidth=0.9, label="Threshold = 0.5")
    ax.set_xlabel("P(Graduate)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence by True Class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_lr_schedule(lr_values: list[float]) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(lr_values)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule (actual epochs run)")
    _max_lr = max(lr_values) if lr_values else 1.0
    _min_lr = min(lr_values) if lr_values else 0.0
    if _max_lr == _min_lr:
        ax.set_ylim([0, _max_lr * 2 if _max_lr > 0 else 1.0])
    else:
        ax.set_ylim([0, _max_lr * 1.1])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
    fig.tight_layout()
    return fig
