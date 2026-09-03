"""Validate aggregate release data against final manuscript values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    return parser.parse_args()


def require_close(actual: float, expected: float, label: str, tolerance: float = 5e-4) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=tolerance):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def main() -> None:
    args = parse_args()
    benchmark = pd.read_csv(args.data_dir / "benchmark_metrics.csv")
    if len(benchmark) != 25 or benchmark["model"].nunique() != 25:
        raise AssertionError("Expected 25 unique benchmark models")
    expected = {
        "Fine-tuned MedGemma-27B": (0.928125, 0.805),
        "Fine-tuned MedGemma-4B": (0.925, 0.839),
        "ResNet-50 (seed 42)": (0.840625, 0.6969699842167872),
    }
    for model, (accuracy, macro_f1) in expected.items():
        row = benchmark.loc[benchmark["model"] == model]
        if len(row) != 1:
            raise AssertionError(f"Missing unique benchmark row for {model}")
        require_close(float(row.iloc[0]["accuracy"]), accuracy, f"{model} accuracy")
        require_close(float(row.iloc[0]["macro_f1"]), macro_f1, f"{model} macro-F1")
        require_close(float(row.iloc[0]["evaluable_count"]), 320, f"{model} evaluable count", 0)

    counts = pd.read_csv(args.data_dir / "development_class_counts.csv")
    if len(counts) != 23 or int(counts["development_count"].sum()) != 8540:
        raise AssertionError("Development counts must contain 23 classes totaling 8,540 images")

    per_class = pd.read_csv(args.data_dir / "resnet50_seed42_per_class_metrics.csv")
    if len(per_class) != 23 or int(per_class["support"].sum()) != 2122:
        raise AssertionError("ResNet-50 per-class data must contain 23 classes totaling 2,122 images")
    if int((per_class["sensitivity"] == 0).sum()) != 7:
        raise AssertionError("Expected seven classes with zero sensitivity")

    seeds = pd.read_csv(args.data_dir / "resnet50_three_seed_metrics.csv")
    if sorted(seeds["seed"].astype(int).tolist()) != [17, 42, 123]:
        raise AssertionError("Expected ResNet-50 seeds 17, 42, and 123")
    primary = seeds.loc[seeds["seed"] == 42]
    if len(primary) != 1:
        raise AssertionError("Expected one prespecified seed-42 result")
    require_close(
        float(primary.iloc[0]["prompt_benchmark__accuracy"]),
        0.840625,
        "seed-42 benchmark accuracy",
    )
    require_close(
        float(primary.iloc[0]["prompt_benchmark__macro_f1"]),
        0.6969699842167872,
        "seed-42 benchmark macro-F1",
    )

    for stratum, expected_rows in (("high", 180), ("low", 420)):
        confusion = pd.read_csv(
            args.data_dir / f"finetuned_medgemma_{stratum}_sample_confusion.csv"
        )
        if len(confusion) != expected_rows:
            raise AssertionError(f"Unexpected {stratum}-sample confusion table size")
        row_totals = confusion.groupby(["model", "true_label"])["row_proportion"].sum()
        if not np.allclose(row_totals.to_numpy(), 1.0, rtol=0, atol=1e-12):
            raise AssertionError(f"{stratum}-sample confusion rows are not normalized")

    audit = json.loads((args.data_dir / "split_audit_summary.json").read_text(encoding="utf-8"))
    expected_counts = (8540, 2122, 10662, 0, 1)
    actual_counts = (
        audit["development_images"], audit["held_out_images"], audit["total_images"],
        audit["exact_cross_split_duplicates"], audit["near_pair_threshold_counts"]["both_phash_le_4"],
    )
    if actual_counts != expected_counts:
        raise AssertionError(f"Unexpected split-audit summary: {actual_counts}")
    print("PASS: aggregate release data match final manuscript values")


if __name__ == "__main__":
    main()
