"""Classification metrics and image-clustered comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def hard_label_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None
) -> dict[str, float]:
    """Compute the hard-label metrics used in the manuscript comparison."""
    true = np.asarray(list(y_true), dtype=object)
    predicted = np.asarray(list(y_pred), dtype=object)
    metric_labels = list(labels) if labels is not None else sorted(set(true) | set(predicted))
    matrix = confusion_matrix(true, predicted, labels=metric_labels)
    total = float(matrix.sum())
    sensitivity = []
    specificity = []
    positive_predictive_value = []
    negative_predictive_value = []
    hard_label_auc = []

    for index in range(len(metric_labels)):
        true_positive = float(matrix[index, index])
        false_negative = float(matrix[index, :].sum() - true_positive)
        false_positive = float(matrix[:, index].sum() - true_positive)
        true_negative = total - true_positive - false_negative - false_positive
        class_sensitivity = _safe_divide(true_positive, true_positive + false_negative)
        class_specificity = _safe_divide(true_negative, true_negative + false_positive)
        sensitivity.append(class_sensitivity)
        specificity.append(class_specificity)
        positive_predictive_value.append(_safe_divide(true_positive, true_positive + false_positive))
        negative_predictive_value.append(_safe_divide(true_negative, true_negative + false_negative))
        hard_label_auc.append(
            0.5 * (class_sensitivity + class_specificity)
            if not (np.isnan(class_sensitivity) or np.isnan(class_specificity))
            else np.nan
        )

    return {
        "n": int(len(true)),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=metric_labels, average="macro", zero_division=0)),
        "macro_sensitivity": float(np.nanmean(sensitivity)),
        "macro_specificity": float(np.nanmean(specificity)),
        "macro_ppv": float(np.nanmean(positive_predictive_value)),
        "macro_npv": float(np.nanmean(negative_predictive_value)),
        "hard_prediction_macro_auroc": float(np.nanmean(hard_label_auc)),
        "mcc": float(matthews_corrcoef(true, predicted)),
    }


def clustered_bootstrap_accuracy_difference(
    frame,
    cluster_column: str,
    first_correct_column: str,
    second_correct_column: str,
    replicates: int = 5000,
    random_seed: int = 20260818,
) -> dict[str, float]:
    """Estimate a paired accuracy difference CI by resampling image clusters."""
    clusters = frame[cluster_column].drop_duplicates().to_numpy()
    rng = np.random.default_rng(random_seed)
    observed = float((frame[first_correct_column] - frame[second_correct_column]).mean())
    draws = np.empty(replicates, dtype=float)
    grouped = {key: value for key, value in frame.groupby(cluster_column, sort=False)}
    for replicate in range(replicates):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        values = [grouped[key][first_correct_column].sub(grouped[key][second_correct_column]).to_numpy() for key in sampled]
        draws[replicate] = np.concatenate(values).mean()
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"difference": observed, "lower_95": float(lower), "upper_95": float(upper), "replicates": replicates}


def clustered_sign_permutation_p_value(
    frame,
    cluster_column: str,
    first_correct_column: str,
    second_correct_column: str,
    replicates: int = 10000,
    random_seed: int = 20260818,
) -> float:
    """Two-sided paired sign-permutation test at the image-cluster level."""
    differences = frame[first_correct_column].sub(frame[second_correct_column])
    cluster_sums = differences.groupby(frame[cluster_column]).sum().to_numpy(dtype=float)
    observed = abs(float(cluster_sums.sum()))
    rng = np.random.default_rng(random_seed)
    extreme = 0
    for _ in range(replicates):
        signs = rng.choice((-1.0, 1.0), size=len(cluster_sums))
        extreme += abs(float((cluster_sums * signs).sum())) >= observed
    return float((extreme + 1) / (replicates + 1))
