#!/usr/bin/env python3
"""Compute manuscript-style metrics and clustered comparisons from predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score


REQUIRED_COLUMNS = {"sample_id", "image_id", "model", "ground_truth", "prediction"}


def hard_prediction_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    labels = sorted(set(frame["ground_truth"]) | set(frame["prediction"]))
    matrix = confusion_matrix(frame["ground_truth"], frame["prediction"], labels=labels)
    true_positive = np.diag(matrix).astype(float)
    false_negative = matrix.sum(axis=1) - true_positive
    false_positive = matrix.sum(axis=0) - true_positive
    true_negative = matrix.sum() - true_positive - false_negative - false_positive

    def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator != 0,
        )

    truth_codes = pd.Categorical(frame["ground_truth"], categories=labels).codes
    prediction_codes = pd.Categorical(frame["prediction"], categories=labels).codes
    truth_one_hot = np.eye(len(labels))[truth_codes]
    prediction_one_hot = np.eye(len(labels))[prediction_codes]
    return {
        "n": len(frame),
        "accuracy": float((frame["ground_truth"] == frame["prediction"]).mean()),
        "macro_f1": float(
            f1_score(
                frame["ground_truth"],
                frame["prediction"],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_sensitivity": float(safe_ratio(true_positive, true_positive + false_negative).mean()),
        "macro_specificity": float(safe_ratio(true_negative, true_negative + false_positive).mean()),
        "macro_ppv": float(safe_ratio(true_positive, true_positive + false_positive).mean()),
        "macro_npv": float(safe_ratio(true_negative, true_negative + false_negative).mean()),
        "hard_prediction_macro_auroc": float(
            roc_auc_score(truth_one_hot, prediction_one_hot, average="macro")
        ),
        "mcc": float(matthews_corrcoef(frame["ground_truth"], frame["prediction"])),
    }


def cluster_bootstrap(
    frame: pd.DataFrame, replicates: int, random_seed: int
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(random_seed)
    groups = {
        image_id: group.index.to_numpy()
        for image_id, group in frame.groupby("image_id", sort=False)
    }
    image_ids = np.asarray(list(groups), dtype=object)
    estimates = {"accuracy": [], "macro_f1": []}
    for _ in range(replicates):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        sample = frame.loc[np.concatenate([groups[image_id] for image_id in sampled])]
        estimates["accuracy"].append(
            float((sample["ground_truth"] == sample["prediction"]).mean())
        )
        labels = sorted(set(sample["ground_truth"]) | set(sample["prediction"]))
        estimates["macro_f1"].append(
            float(
                f1_score(
                    sample["ground_truth"],
                    sample["prediction"],
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )
        )
    return {
        metric: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "bootstrap_mean": float(np.mean(values)),
            "replicates": replicates,
        }
        for metric, values in estimates.items()
    }


def paired_cluster_comparison(
    first: pd.DataFrame,
    second: pd.DataFrame,
    bootstrap_replicates: int,
    permutation_replicates: int,
    random_seed: int,
) -> dict[str, float | int]:
    merged = first[["sample_id", "image_id", "ground_truth", "prediction"]].merge(
        second[["sample_id", "prediction"]],
        on="sample_id",
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    merged["first_correct"] = merged["ground_truth"].eq(merged["prediction_first"])
    merged["second_correct"] = merged["ground_truth"].eq(merged["prediction_second"])
    merged["difference"] = (
        merged["first_correct"].astype(int) - merged["second_correct"].astype(int)
    )
    clusters = merged.groupby("image_id", sort=False)["difference"].agg(["sum", "count"])
    differences = clusters["sum"].to_numpy(dtype=float)
    counts = clusters["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    bootstrap_values = []
    for start in range(0, bootstrap_replicates, 250):
        size = min(250, bootstrap_replicates - start)
        sampled = rng.integers(0, len(clusters), size=(size, len(clusters)))
        bootstrap_values.extend(
            (differences[sampled].sum(axis=1) / counts[sampled].sum(axis=1)).tolist()
        )
    bootstrap = np.asarray(bootstrap_values)
    observed_sum = differences.sum()
    extreme = 0
    for start in range(0, permutation_replicates, 250):
        size = min(250, permutation_replicates - start)
        signs = rng.choice([-1.0, 1.0], size=(size, len(clusters)))
        permuted = (signs * differences).sum(axis=1)
        extreme += int((np.abs(permuted) >= abs(observed_sum) - 1e-12).sum())
    return {
        "n_records": len(merged),
        "n_image_clusters": len(clusters),
        "first_accuracy": float(merged["first_correct"].mean()),
        "second_accuracy": float(merged["second_correct"].mean()),
        "accuracy_difference_first_minus_second": float(merged["difference"].mean()),
        "difference_lower_95": float(np.quantile(bootstrap, 0.025)),
        "difference_upper_95": float(np.quantile(bootstrap, 0.975)),
        "paired_cluster_permutation_p": float((extreme + 1) / (permutation_replicates + 1)),
        "bootstrap_replicates": bootstrap_replicates,
        "permutation_replicates": permutation_replicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-model")
    parser.add_argument("--second-model")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--permutation-replicates", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260818)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    output = {"models": {}}
    for model, frame in predictions.groupby("model", sort=True):
        output["models"][model] = {
            "metrics": hard_prediction_metrics(frame),
            "confidence_intervals": cluster_bootstrap(
                frame, args.bootstrap_replicates, args.random_seed
            ),
        }
    if bool(args.first_model) != bool(args.second_model):
        raise ValueError("Provide both --first-model and --second-model for a comparison")
    if args.first_model and args.second_model:
        output["paired_comparison"] = paired_cluster_comparison(
            predictions[predictions["model"].eq(args.first_model)],
            predictions[predictions["model"].eq(args.second_model)],
            args.bootstrap_replicates,
            args.permutation_replicates,
            args.random_seed,
        )
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
