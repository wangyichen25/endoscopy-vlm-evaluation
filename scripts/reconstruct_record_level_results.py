#!/usr/bin/env python3
"""Reconstruct manuscript metrics from released record-level predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import label_binarize


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )


def hard_label_metrics(frame: pd.DataFrame, labels: list[str] | None = None) -> dict[str, float | int]:
    frame = frame.dropna(subset=["prediction"])
    labels = labels or sorted(set(frame["ground_truth"]) | set(frame["prediction"]))
    matrix = confusion_matrix(frame["ground_truth"], frame["prediction"], labels=labels)
    true_positive = np.diag(matrix).astype(float)
    false_negative = matrix.sum(axis=1) - true_positive
    false_positive = matrix.sum(axis=0) - true_positive
    true_negative = matrix.sum() - true_positive - false_negative - false_positive
    sensitivity = safe_divide(true_positive, true_positive + false_negative)
    specificity = safe_divide(true_negative, true_negative + false_positive)
    precision = safe_divide(true_positive, true_positive + false_positive)
    negative_predictive_value = safe_divide(true_negative, true_negative + false_negative)
    return {
        "evaluable_count": len(frame),
        "accuracy": float((frame["ground_truth"] == frame["prediction"]).mean()),
        "macro_sensitivity": float(np.nanmean(sensitivity)),
        "macro_specificity": float(np.nanmean(specificity)),
        "macro_ppv": float(np.nanmean(precision)),
        "macro_npv": float(np.nanmean(negative_predictive_value)),
        "macro_auroc": float(np.nanmean((sensitivity + specificity) / 2)),
        "macro_f1": float(
            f1_score(
                frame["ground_truth"],
                frame["prediction"],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(frame["ground_truth"], frame["prediction"])),
    }


def probability_macro_auroc(frame: pd.DataFrame, labels: list[str]) -> float:
    probabilities = frame[[f"prob__{label}" for label in labels]].to_numpy(dtype=float)
    truth = label_binarize(frame["label_key"], classes=labels)
    return float(roc_auc_score(truth, probabilities, average="macro", multi_class="ovr"))


def cluster_bootstrap_accuracy(
    frame: pd.DataFrame, replicates: int, rng: np.random.Generator
) -> dict[str, float | int]:
    grouped = {
        image_id: group.index.to_numpy()
        for image_id, group in frame.groupby("image_id", sort=False)
    }
    image_ids = np.asarray(list(grouped), dtype=object)
    values = {"accuracy": [], "macro_f1": []}
    for _ in range(replicates):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        sample = frame.loc[np.concatenate([grouped[image_id] for image_id in sampled])]
        values["accuracy"].append(float((sample["ground_truth"] == sample["prediction"]).mean()))
        labels = sorted(set(sample["ground_truth"]) | set(sample["prediction"]))
        values["macro_f1"].append(
            float(f1_score(sample["ground_truth"], sample["prediction"], labels=labels, average="macro", zero_division=0))
        )
    return {
        metric: {
            "lower_95": float(np.quantile(estimates, 0.025)),
            "upper_95": float(np.quantile(estimates, 0.975)),
            "bootstrap_mean": float(np.mean(estimates)),
            "replicates": replicates,
        }
        for metric, estimates in values.items()
    }


def paired_cluster_accuracy(
    first: pd.DataFrame,
    second: pd.DataFrame,
    bootstrap_replicates: int,
    permutation_replicates: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    merged = first[["sample_id", "image_id", "ground_truth", "prediction"]].merge(
        second[["sample_id", "prediction"]],
        on="sample_id",
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    merged["first_correct"] = merged["ground_truth"].eq(merged["prediction_first"])
    merged["second_correct"] = merged["ground_truth"].eq(merged["prediction_second"])
    merged["difference"] = merged["first_correct"].astype(int) - merged["second_correct"].astype(int)
    clusters = merged.groupby("image_id", sort=False)["difference"].agg(["sum", "count"])
    differences = clusters["sum"].to_numpy(dtype=float)
    counts = clusters["count"].to_numpy(dtype=float)
    bootstrap_values = []
    for start in range(0, bootstrap_replicates, 250):
        size = min(250, bootstrap_replicates - start)
        sampled = rng.integers(0, len(clusters), size=(size, len(clusters)))
        bootstrap_values.extend(
            (differences[sampled].sum(axis=1) / counts[sampled].sum(axis=1)).tolist()
        )
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
        "difference_lower_95": float(np.quantile(bootstrap_values, 0.025)),
        "difference_upper_95": float(np.quantile(bootstrap_values, 0.975)),
        "paired_cluster_permutation_p": float((extreme + 1) / (permutation_replicates + 1)),
        "bootstrap_replicates": bootstrap_replicates,
        "permutation_replicates": permutation_replicates,
    }


def benchmark_table(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, frame in records.groupby("model", sort=False):
        metrics = hard_label_metrics(frame)
        metrics.update(
            {
                "model": model,
                "average_latency_seconds": frame.loc[frame.evaluable, "latency_seconds"].mean(),
                "average_cost_usd": frame.loc[frame.evaluable, "cost_usd"].mean(),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(["accuracy", "model"], ascending=[False, True])


def task_table(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task_id), frame in records.groupby(["model", "task_id"], sort=False):
        metrics = hard_label_metrics(frame, sorted(frame["ground_truth"].unique()))
        rows.append(
            {
                "model": model,
                "task_id": task_id,
                "n": len(frame),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def resnet_seed_tables(prompt_records: pd.DataFrame, image_records: pd.DataFrame, benchmark_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_rows = []
    prompt_rows = []
    labels = sorted(image_records["label_key"].unique())
    for seed, frame in image_records.groupby("seed", sort=True):
        metrics = hard_label_metrics(frame, sorted(frame["ground_truth"].unique()))
        metrics["probability_macro_auroc_ovr"] = probability_macro_auroc(frame, labels)
        image_rows.append({"seed": seed, **metrics})
    for seed, frame in prompt_records.groupby("seed", sort=True):
        for scope, subset in (("all_prompts", frame), ("benchmark", frame[frame.sample_id.isin(benchmark_ids)])):
            prompt_rows.append({"seed": seed, "scope": scope, **hard_label_metrics(subset)})
    return pd.DataFrame(image_rows), pd.DataFrame(prompt_rows)


def mcnemar(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, float | int]:
    merged = first[["sample_id", "ground_truth", "prediction"]].merge(
        second[["sample_id", "prediction"]], on="sample_id", suffixes=("_first", "_second"), validate="one_to_one"
    )
    first_correct = merged.ground_truth.eq(merged.prediction_first)
    second_correct = merged.ground_truth.eq(merged.prediction_second)
    first_only = int((first_correct & ~second_correct).sum())
    second_only = int((~first_correct & second_correct).sum())
    return {
        "first_correct_only": first_only,
        "second_correct_only": second_only,
        "exact_two_sided_p": float(binomtest(min(first_only, second_only), first_only + second_only, 0.5).pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/records"))
    parser.add_argument("--output-dir", type=Path, default=Path("tables/reconstructed"))
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--permutation-replicates", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260818)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = pd.read_csv(args.data_dir / "benchmark_predictions.csv")
    medgemma = pd.read_csv(args.data_dir / "medgemma_full_predictions.csv")
    resnet_prompt = pd.read_csv(args.data_dir / "resnet50_prompt_predictions.csv")
    resnet_image = pd.read_csv(args.data_dir / "resnet50_image_predictions.csv")

    benchmark_table(benchmark).to_csv(args.output_dir / "benchmark_metrics.csv", index=False)
    task_table(medgemma[medgemma.model.str.startswith("Fine-tuned")]).to_csv(
        args.output_dir / "finetuned_medgemma_per_task_metrics.csv", index=False
    )
    benchmark_ids = set(benchmark.sample_id)
    image_metrics, prompt_metrics = resnet_seed_tables(resnet_prompt, resnet_image, benchmark_ids)
    image_metrics.to_csv(args.output_dir / "resnet50_image_metrics_by_seed.csv", index=False)
    prompt_metrics.to_csv(args.output_dir / "resnet50_prompt_metrics_by_seed.csv", index=False)

    primary_resnet = resnet_prompt[resnet_prompt.seed.eq(42)]
    finetuned_27b = medgemma[medgemma.model.eq("Fine-tuned MedGemma-27B")]
    rng = np.random.default_rng(args.random_seed)
    statistics = {
        "resnet50_confidence_intervals": {
            "image_full": cluster_bootstrap_accuracy(
                resnet_image[resnet_image.seed.eq(42)], args.bootstrap_replicates, rng
            ),
            "prompt_full": cluster_bootstrap_accuracy(
                primary_resnet, args.bootstrap_replicates, rng
            ),
            "prompt_benchmark": cluster_bootstrap_accuracy(
                primary_resnet[primary_resnet.sample_id.isin(benchmark_ids)], args.bootstrap_replicates, rng
            ),
        },
        "paired_resnet50_minus_finetuned_medgemma_27b": {
            "image_only": paired_cluster_accuracy(
                primary_resnet[primary_resnet.task_id.eq("image_only")],
                finetuned_27b[finetuned_27b.task_id.eq("image_only")],
                args.bootstrap_replicates,
                args.permutation_replicates,
                rng,
            ),
            "all_prompts": paired_cluster_accuracy(
                primary_resnet,
                finetuned_27b,
                args.bootstrap_replicates,
                args.permutation_replicates,
                rng,
            ),
            "benchmark": paired_cluster_accuracy(
                primary_resnet[primary_resnet.sample_id.isin(benchmark_ids)],
                finetuned_27b[finetuned_27b.sample_id.isin(benchmark_ids)],
                args.bootstrap_replicates,
                args.permutation_replicates,
                rng,
            ),
        },
        "mcnemar_finetuned_medgemma_27b_vs_baseline": mcnemar(
            finetuned_27b,
            medgemma[medgemma.model.eq("MedGemma-27B baseline")],
        ),
    }
    (args.output_dir / "statistical_tests.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
