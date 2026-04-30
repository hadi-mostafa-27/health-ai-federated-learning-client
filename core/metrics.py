from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - sklearn is expected, fallback keeps app importable.
    roc_auc_score = None


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    strategy: str
    metrics: dict


def compute_binary_metrics(
    y_true: Sequence[int] | Iterable[int],
    y_prob: Sequence[float] | Iterable[float],
    threshold: float = 0.5,
) -> dict:
    """Compute pneumonia-vs-normal metrics from probabilities.

    Class convention: 0 = NORMAL, 1 = PNEUMONIA. Sensitivity is pneumonia
    recall; specificity is true-negative rate for NORMAL cases.
    """
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_prob_arr = np.asarray(list(y_prob), dtype=np.float64)
    threshold = float(threshold)

    if y_true_arr.size == 0 or y_prob_arr.size == 0:
        return _empty_metrics(threshold)
    if y_true_arr.size != y_prob_arr.size:
        raise ValueError("y_true and y_prob must contain the same number of items.")

    y_pred = (y_prob_arr >= threshold).astype(np.int64)

    tp = int(np.sum((y_true_arr == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred == 0)))
    total = int(y_true_arr.size)

    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1_score = _safe_div(2.0 * precision * recall, precision + recall)

    auc = None
    if roc_auc_score is not None and len(set(y_true_arr.tolist())) == 2:
        try:
            auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        except ValueError:
            auc = None

    return {
        "threshold": threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "roc_auc": auc,
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "true_positives": tp,
        "support": total,
        "positive_support": int(np.sum(y_true_arr == 1)),
        "negative_support": int(np.sum(y_true_arr == 0)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def tune_threshold(
    y_true: Sequence[int] | Iterable[int],
    y_prob: Sequence[float] | Iterable[float],
    strategy: str = "best_f1",
    min_sensitivity: float = 0.95,
    thresholds: Sequence[float] | None = None,
) -> ThresholdSelection:
    """Select a validation threshold for binary medical classification."""
    strategy = (strategy or "best_f1").lower()
    y_true_list = list(y_true)
    y_prob_list = list(y_prob)

    if not y_true_list or not y_prob_list:
        metrics = compute_binary_metrics(y_true_list, y_prob_list, threshold=0.5)
        return ThresholdSelection(threshold=0.5, strategy=strategy, metrics=metrics)

    if strategy in {"fixed", "fixed_0_5", "0.5"}:
        metrics = compute_binary_metrics(y_true_list, y_prob_list, threshold=0.5)
        return ThresholdSelection(threshold=0.5, strategy="fixed_0_5", metrics=metrics)

    candidates = _candidate_thresholds(y_prob_list, thresholds)
    scored = [(thr, compute_binary_metrics(y_true_list, y_prob_list, threshold=thr)) for thr in candidates]

    if strategy in {"high_sensitivity", "sensitivity"}:
        eligible = [
            item for item in scored
            if item[1]["sensitivity"] >= float(min_sensitivity)
        ]
        if eligible:
            best = max(
                eligible,
                key=lambda item: (
                    item[1]["specificity"],
                    item[1]["f1_score"],
                    item[1]["accuracy"],
                    -item[0],
                ),
            )
        else:
            best = max(
                scored,
                key=lambda item: (
                    item[1]["sensitivity"],
                    item[1]["f1_score"],
                    item[1]["specificity"],
                    -item[0],
                ),
            )
    elif strategy in {"balanced", "balanced_sens_spec", "balanced_sensitivity_specificity"}:
        best = max(
            scored,
            key=lambda item: (
                (item[1]["sensitivity"] + item[1]["specificity"]) / 2.0,
                item[1]["f1_score"],
                item[1]["accuracy"],
                -abs(item[1]["sensitivity"] - item[1]["specificity"]),
            ),
        )
    else:
        best = max(
            scored,
            key=lambda item: (
                item[1]["f1_score"],
                item[1]["sensitivity"],
                item[1]["specificity"],
                item[1]["accuracy"],
            ),
        )
        strategy = "best_f1"

    threshold, metrics = best
    metrics["threshold_strategy"] = strategy
    return ThresholdSelection(threshold=float(threshold), strategy=strategy, metrics=metrics)


def _candidate_thresholds(
    probabilities: Sequence[float],
    thresholds: Sequence[float] | None,
) -> list[float]:
    if thresholds:
        values = [float(t) for t in thresholds]
    else:
        grid = np.linspace(0.0, 1.0, 101).tolist()
        observed = [float(p) for p in probabilities if np.isfinite(p)]
        values = grid + observed
    clipped = sorted({round(min(1.0, max(0.0, value)), 6) for value in values})
    return clipped or [0.5]


def _empty_metrics(threshold: float) -> dict:
    return {
        "threshold": float(threshold),
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "roc_auc": None,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "false_negatives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "true_positives": 0,
        "support": 0,
        "positive_support": 0,
        "negative_support": 0,
        "confusion_matrix": [[0, 0], [0, 0]],
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
