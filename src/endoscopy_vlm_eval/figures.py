"""Generate final quantitative figures from aggregate manuscript data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COLORS = {
    "Fine-tuned MedGemma-27B": "#2166AC",
    "Fine-tuned MedGemma-4B": "#D89B25",
    "ResNet-50 (seed 42)": "#D97706",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def benchmark_figure(data_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(data_dir / "benchmark_metrics.csv").sort_values(["accuracy", "model"])
    labels = frame["model"].tolist()
    y_positions = np.arange(len(frame))
    colors = [COLORS.get(label, "#B8B8B8") for label in labels]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 10), sharey=True)
    for axis, values, title, xlabel in [
        (axes[0], frame["accuracy"].to_numpy() * 100, "A. Accuracy", "Accuracy (%)"),
        (axes[1], frame["macro_f1"].to_numpy() * 100, "B. Macro-F1", "Macro-F1 (%)"),
    ]:
        bars = axis.barh(y_positions, values, color=colors, edgecolor="#5D5D5D", linewidth=0.45)
        axis.set_xlim(0, 100)
        axis.set_xlabel(xlabel)
        axis.set_title(title, loc="left")
        axis.xaxis.grid(True, color="#D8D8D8", linewidth=0.65, linestyle="--")
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            axis.text(min(value + 0.8, 98.5), bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=8.5)
    axes[0].set_yticks(y_positions, labels=labels, fontsize=9)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    figure.suptitle("Accuracy and macro-F1 on the internal held-out prompt benchmark", fontsize=13, y=0.985)
    figure.text(0.18, 0.948, "320 randomly sampled prompt-answer records from 303 held-out images", fontsize=9.5, color="#444444")
    figure.text(0.18, 0.026, "Fine-tuned MedGemma models and the training-matched ResNet-50 are highlighted; generalist VLMs are gray.", fontsize=8.1, color="#444444")
    figure.subplots_adjust(left=0.25, right=0.975, top=0.90, bottom=0.105, wspace=0.16)
    figure.savefig(output_dir / "benchmark_accuracy_macro_f1.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def distribution_figure(data_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(data_dir / "development_class_counts.csv").sort_values("development_count")
    figure, axis = plt.subplots(figsize=(8.8, 9.2))
    bars = axis.barh(frame["source_class"], frame["development_count"], color="#4C78A8", edgecolor="#2B4C6F", linewidth=0.45)
    axis.set_xlim(0, 980)
    axis.set_xlabel("Development images (n)")
    axis.set_title("Development-set source-label distribution")
    axis.xaxis.grid(True, color="#DADADA", linestyle="--", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, frame["development_count"]):
        axis.text(value + 8, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", fontsize=8.5)
    figure.tight_layout()
    figure.savefig(output_dir / "development_class_distribution.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def fine_tuned_confusion_figure(data_dir: Path, output_dir: Path, stratum: str) -> None:
    frame = pd.read_csv(data_dir / f"finetuned_medgemma_{stratum}_sample_confusion.csv")
    models = ["Fine-tuned MedGemma-27B", "Fine-tuned MedGemma-4B"]
    labels = frame["true_label"].drop_duplicates().tolist()
    predicted_labels = labels + ["Other"]
    figure, axes = plt.subplots(1, 2, figsize=(17, 8.6 if stratum == "high" else 10.2), sharey=True)
    for axis, model in zip(axes, models):
        subset = frame[frame["model"] == model]
        matrix = subset.pivot(index="true_label", columns="predicted_label", values="row_proportion").reindex(index=labels, columns=predicted_labels)
        sns.heatmap(matrix, ax=axis, cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".2f", cbar=False)
        axis.set_title(model)
        axis.set_xlabel("Predicted source label")
        axis.set_ylabel("True source label")
        axis.tick_params(axis="x", labelrotation=52, labelsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / f"finetuned_medgemma_{stratum}_sample_confusion.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def resnet_confusion_figure(data_dir: Path, output_dir: Path) -> None:
    counts = pd.read_csv(data_dir / "resnet50_seed42_confusion_counts.csv", index_col=0)
    normalized = pd.read_csv(data_dir / "resnet50_seed42_confusion_row_normalized.csv", index_col=0)
    figure, axes = plt.subplots(1, 2, figsize=(18, 8.8), constrained_layout=True)
    sns.heatmap(counts, ax=axes[0], cmap="Blues", cbar_kws={"label": "Images"}, square=True)
    sns.heatmap(normalized, ax=axes[1], cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "Within-class proportion"}, square=True)
    axes[0].set_title("A. Counts")
    axes[1].set_title("B. Row-normalized proportions")
    for axis in axes:
        axis.set_xlabel("Predicted source class")
        axis.set_ylabel("True source class")
        axis.tick_params(axis="x", labelrotation=55, labelsize=7)
        axis.tick_params(axis="y", labelrotation=0, labelsize=7)
    figure.suptitle("ResNet-50 direct 23-class classification on 2,122 held-out images (seed 42)", fontsize=14)
    figure.savefig(output_dir / "resnet50_seed42_confusion.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def resnet_error_profile(data_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(data_dir / "resnet50_seed42_per_class_metrics.csv")
    frame["predicted_count"] = frame["tp"] + frame["fp"]
    frame["true_share"] = 100 * frame["support"] / frame["support"].sum()
    frame["predicted_share"] = 100 * frame["predicted_count"] / frame["predicted_count"].sum()
    frame["difference"] = frame["predicted_share"] - frame["true_share"]
    rho = frame["support"].rank().corr(frame["sensitivity"].rank())
    selected = frame.assign(abs_difference=frame["difference"].abs()).nlargest(8, "abs_difference").sort_values("difference")
    figure, axes = plt.subplots(1, 2, figsize=(16, 8.6))
    axes[0].scatter(frame["support"], 100 * frame["sensitivity"], color="#2B6CB0")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Held-out images in source class (log scale)")
    axes[0].set_ylabel("Sensitivity (%)")
    axes[0].set_title(f"A. Sensitivity versus support (Spearman rho = {rho:.2f})")
    positions = np.arange(len(selected))
    axes[1].barh(positions - 0.18, selected["true_share"], height=0.35, label="True", color="#2B6CB0")
    axes[1].barh(positions + 0.18, selected["predicted_share"], height=0.35, label="Predicted", color="#D97706")
    axes[1].set_yticks(positions, labels=selected["display_label"])
    axes[1].set_xlabel("Share of held-out images (%)")
    axes[1].set_title("B. Largest class-share discrepancies")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", color="#D9E0E7", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("ResNet-50 held-out error profile", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / "resnet50_seed42_error_profile.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_figure(args.data_dir, args.output_dir)
    distribution_figure(args.data_dir, args.output_dir)
    fine_tuned_confusion_figure(args.data_dir, args.output_dir, "high")
    fine_tuned_confusion_figure(args.data_dir, args.output_dir, "low")
    resnet_confusion_figure(args.data_dir, args.output_dir)
    resnet_error_profile(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
