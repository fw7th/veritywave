import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = ["REAL", "FAKE"],
    accuracy: float = 0.8352,
    save_path: str = "confusion_matrix.png",
):
    """
    Plots and saves a styled confusion matrix.

    Args:
        cm         : 2x2 confusion matrix as numpy array [[TN, FP], [FN, TP]]
        class_names: list of class label strings
        accuracy   : overall accuracy to display in title
        save_path  : where to save the output PNG
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    cmap = LinearSegmentedColormap.from_list(
        "vw", ["#0d1117", "#0e4f4f", "#1a9e9e"], N=256
    )

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    # -- cell annotations --
    total = cm.sum()
    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = count / total * 100

            # label: TP / TN / FP / FN
            if i == j:
                cell_label = "True Positive" if i == 1 else "True Negative"
                color = "#00ffcc"
            else:
                cell_label = "False Negative" if j == 0 else "False Positive"
                color = "#ff6b6b"

            ax.text(
                j,
                i - 0.15,
                f"{count:,}",
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
                color=color,
            )

            ax.text(
                j,
                i + 0.15,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                color="#aaaaaa",
            )

            ax.text(
                j,
                i + 0.38,
                cell_label,
                ha="center",
                va="center",
                fontsize=8,
                color="#666666",
                style="italic",
            )

    # -- axes styling --
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=13, color="#cccccc", fontweight="bold")
    ax.set_yticklabels(class_names, fontsize=13, color="#cccccc", fontweight="bold")

    ax.set_xlabel("Predicted Label", fontsize=13, color="#cccccc", labelpad=12)
    ax.set_ylabel("Actual Label", fontsize=13, color="#cccccc", labelpad=12)

    ax.tick_params(colors="#cccccc", length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # -- title --
    ax.set_title(
        f"Confusion Matrix  —  Accuracy: {accuracy * 100:.2f}%",
        fontsize=14,
        color="#ffffff",
        fontweight="bold",
        pad=18,
    )

    # -- legend --
    tp_patch = mpatches.Patch(color="#00ffcc", label="Correct classification")
    fp_patch = mpatches.Patch(color="#ff6b6b", label="Misclassification")
    ax.legend(
        handles=[tp_patch, fp_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor="#aaaaaa",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved → {save_path}")
    plt.show()


if __name__ == "__main__":
    cm = np.array(
        [
            [10084, 1916],  # Actual REAL  → [predicted REAL, predicted FAKE]
            [2039, 9961],  # Actual FAKE  → [predicted REAL, predicted FAKE]
        ]
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=["REAL", "FAKE"],
        accuracy=0.8352,
        save_path="confusion_matrix.png",
    )
