#!/usr/bin/env python3
"""Generate manuscript-facing summary tables from released aggregate results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/aggregate"))
    parser.add_argument("--output-dir", type=Path, default=Path("tables"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = pd.read_csv(args.data_dir / "benchmark_metrics.csv")
    benchmark.sort_values(["accuracy", "model"], ascending=[False, True]).to_csv(
        args.output_dir / "benchmark_model_metrics.csv", index=False
    )

    seeds = pd.read_csv(args.data_dir / "resnet50_three_seed_metrics.csv")
    reported_columns = [
        "image_full__accuracy",
        "image_full__macro_f1",
        "image_full__probability_macro_auroc_ovr",
        "prompt_full__accuracy",
        "prompt_full__macro_f1",
        "prompt_benchmark__accuracy",
        "prompt_benchmark__macro_f1",
    ]
    seed_summary = pd.DataFrame(
        {
            "metric": reported_columns,
            "mean": [seeds[column].mean() for column in reported_columns],
            "standard_deviation": [seeds[column].std(ddof=1) for column in reported_columns],
            "minimum": [seeds[column].min() for column in reported_columns],
            "maximum": [seeds[column].max() for column in reported_columns],
        }
    )
    seed_summary.to_csv(args.output_dir / "resnet50_three_seed_summary.csv", index=False)

    statistics = json.loads((args.data_dir / "statistical_results.json").read_text())
    rows = []
    for analysis, metrics in statistics["confidence_intervals"].items():
        for metric, values in metrics.items():
            rows.append({"analysis": analysis, "metric": metric, **values})
    pd.DataFrame(rows).to_csv(args.output_dir / "resnet50_confidence_intervals.csv", index=False)

    paired_rows = [
        {"analysis": analysis, **values}
        for analysis, values in statistics["paired_comparisons"].items()
    ]
    pd.DataFrame(paired_rows).to_csv(args.output_dir / "paired_accuracy_comparisons.csv", index=False)


if __name__ == "__main__":
    main()
