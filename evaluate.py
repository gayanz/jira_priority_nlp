"""Evaluation: Macro F1, Precision-Recall curves, Confusion Matrix."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)
from pathlib import Path

from config import OUTPUT_DIR


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def classification_report_dict(y_true, y_pred, labels, target_names=None):
    return classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
    display_names: list | None = None,
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ticks = display_names if display_names is not None else labels
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=ticks, yticklabels=ticks, ax=ax, cmap="Blues")
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    if save_path is None:
        save_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: list,
    label_names: list,
    title: str = "Precision-Recall curves",
    save_path: Path | None = None,
):
    """One-vs-rest PR curves and macro AP."""
    n_classes = len(labels)
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=labels)
    if y_proba.ndim == 1:
        y_proba = y_proba.reshape(-1, 1)
    if y_proba.shape[1] != n_classes:
        # Assume proba order matches labels
        pass
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (lab, name) in enumerate(zip(labels, label_names)):
        if y_bin.shape[1] > i:
            ap = float(average_precision_score(y_bin[:, i], y_proba[:, i]))
        else:
            ap = 0.0
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(rec, prec, label=f"{name} (AP={ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    if save_path is None:
        save_path = OUTPUT_DIR / "precision_recall_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def full_evaluation(y_true, y_pred, y_proba, labels, label_names, model_name: str):
    """Macro F1, report, confusion matrix, PR curves."""
    out = {}
    out["macro_f1"] = macro_f1(y_true, y_pred, labels)
    out["report"] = classification_report_dict(y_true, y_pred, labels, label_names)
    plot_confusion_matrix(
        y_true, y_pred, labels,
        title=f"Confusion Matrix ({model_name})",
        save_path=OUTPUT_DIR / f"confusion_matrix_{model_name.replace(' ', '_')}.png",
        display_names=label_names,
    )
    if y_proba is not None and len(labels) == y_proba.shape[1]:
        plot_precision_recall_curves(
            y_true, y_proba, labels, label_names,
            title=f"Precision-Recall ({model_name})",
            save_path=OUTPUT_DIR / f"pr_curves_{model_name.replace(' ', '_')}.png",
        )
    return out
