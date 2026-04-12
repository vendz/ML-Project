import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm: np.ndarray,
                          labels: list[str] = ["Enrolled", "Dropout"]) -> plt.Figure:
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
            label="Dropout (true 1)", color="#e74c3c")
    ax.axvline(0.5, linestyle="--", color="black",
               linewidth=0.9, label="Threshold = 0.5")
    ax.set_xlabel("P(Dropout)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence by True Class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


