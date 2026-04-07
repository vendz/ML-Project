"""
Reusable plot helpers for the Streamlit dashboard.
Owner: Shivnarain Sarin

All functions return Matplotlib figures so they can be rendered
with st.pyplot(fig) in app.py.
"""
import numpy as np
import matplotlib.pyplot as plt


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
