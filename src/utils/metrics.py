from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve, precision_score, recall_score, roc_auc_score


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _precision_at_recall(labels: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, scores)
    valid = precision[recall >= target_recall]
    return float(valid.max()) if valid.size else float("nan")


def _recall_at_precision(labels: np.ndarray, scores: np.ndarray, target_precision: float) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, scores)
    valid = recall[precision >= target_precision]
    return float(valid.max()) if valid.size else float("nan")


def compute_binary_metrics(labels: list[int] | np.ndarray, fake_scores: list[float] | np.ndarray) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(fake_scores, dtype=np.float64)
    preds = (scores_array >= 0.5).astype(np.int64)
    result = {
        "accuracy": float(accuracy_score(labels_array, preds)),
        "roc_auc": _safe_auc(labels_array, scores_array),
        "pr_auc": _safe_average_precision(labels_array, scores_array),
        "precision_at_0_5": float(precision_score(labels_array, preds, zero_division=0)),
        "recall_at_0_5": float(recall_score(labels_array, preds, zero_division=0)),
        "precision_at_recall_0_8": _precision_at_recall(labels_array, scores_array, 0.8),
        "recall_at_precision_0_8": _recall_at_precision(labels_array, scores_array, 0.8),
    }
    if len(np.unique(labels_array)) >= 2:
        prec, rec, thresh = precision_recall_curve(labels_array, scores_array)
        tpr = rec[:-1]
        tnr = (np.sum(labels_array == 0) - np.searchsorted(np.sort(scores_array[labels_array == 0]), thresh, side="right")) / max(np.sum(labels_array == 0), 1)
        tnr = np.clip(tnr, 0, 1)
        bal_accs = (tpr + tnr) / 2.0
        if bal_accs.size:
            best_idx = int(np.argmax(bal_accs))
            idx_at_05 = int(np.clip(np.searchsorted(thresh, 0.5), 0, bal_accs.size - 1))
            result["balanced_accuracy_at_0_5"] = float(bal_accs[idx_at_05])
            result["balanced_accuracy_at_best"] = float(bal_accs[best_idx])
            result["best_threshold"] = float(thresh[best_idx])
        else:
            result["balanced_accuracy_at_0_5"] = float("nan")
            result["balanced_accuracy_at_best"] = float("nan")
            result["best_threshold"] = float("nan")
    return result
