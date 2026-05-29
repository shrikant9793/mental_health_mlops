from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Calculate all evaluation metrics."""
    metrics = {
        "accuracy"  : accuracy_score(y_true, y_pred),
        "f1_score"  : f1_score(y_true, y_pred),
        "precision" : precision_score(y_true, y_pred),
        "recall"    : recall_score(y_true, y_pred),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


def print_metrics(metrics: dict, model_name: str):
    """Pretty print evaluation metrics."""
    print(f"\n=== {model_name} Evaluation ===")
    for metric, value in metrics.items():
        print(f"{metric:<12} : {value:.4f}")


def save_confusion_matrix(y_true, y_pred, model_name: str, save_path: str):
    """Save confusion matrix as image."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Depression", "Depression"],
        yticklabels=["Non-Depression", "Depression"]
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")


def save_classification_report(y_true, y_pred, model_name: str, save_path: str):
    """Save classification report as text file."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Non-Depression", "Depression"]
    )
    with open(save_path, "w") as f:
        f.write(f"=== {model_name} Classification Report ===\n\n")
        f.write(report)
    print(f"✅ Classification report saved to {save_path}")