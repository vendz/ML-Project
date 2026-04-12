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


if __name__ == "__main__":
    plot_imbalance_comparison()
