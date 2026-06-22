from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error for binary classification."""

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lower) & (y_prob < upper)

        if not np.any(mask):
            continue

        bin_confidence = np.mean(y_prob[mask])
        bin_prediction = (y_prob[mask] >= 0.5).astype(int)
        bin_accuracy = np.mean(bin_prediction == y_true[mask])

        ece += np.mean(mask) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def binary_classification_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute core binary metrics from raw model logits."""

    y_true = np.asarray(y_true).astype(int)
    logits = np.asarray(logits).astype(float)
    y_prob = sigmoid(logits)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {}

    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    metrics["ece"] = expected_calibration_error(y_true, y_prob)

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics["sensitivity"] = float(tp / max(1, tp + fn))
    metrics["specificity"] = float(tn / max(1, tn + fp))
    metrics["true_positives"] = float(tp)
    metrics["true_negatives"] = float(tn)
    metrics["false_positives"] = float(fp)
    metrics["false_negatives"] = float(fn)

    return metrics