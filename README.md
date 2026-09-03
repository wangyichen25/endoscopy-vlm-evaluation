# Endoscopy VLM Evaluation

This repository provides the data and downstream analysis and visualization code used to produce the final manuscript's quantitative tables, statistical results, and figures comparing fine-tuned MedGemma vision-language models, a training-matched ResNet-50 control, and 22 generalist vision-language models on HyperKvasir-derived tasks.

The scripts are publication copies of the analysis logic used for the final manuscript, with paths and labels normalized for public release.

## Contents

- Aggregate data underlying the final quantitative tables and figures.
- Record-level model responses paired with task questions and HyperKvasir image IDs.
- Three-seed ResNet-50 image and prompt predictions, including class probabilities.
- Validation code for the manuscript's reported sample counts, primary metrics, confidence intervals, and paired comparisons.
- Summary-table code for the released benchmark and ResNet-50 results.
- Visualization code that regenerates all quantitative manuscript figures.
- Dataset split-audit code for exact duplicates and close perceptual-hash matches.

The repository is self-contained for validating the released aggregate results and regenerating all quantitative tables and figures.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Reproduce Release

```bash
python scripts/reproduce_release.py
```

This reconstructs record-level results, validates reported aggregate results, and regenerates the quantitative tables and figures using the supplied data.

## Validate Reported Results

```bash
python scripts/validate_reported_results.py --data-dir data/aggregate
```

The validator checks the 8,540/2,122 development and held-out counts, the 25-model comparison table, the 320-record benchmark, fine-tuned MedGemma benchmark metrics, and the prespecified seed-42 ResNet-50 metrics.

## Regenerate Quantitative Figures

```bash
python scripts/generate_final_figures.py --data-dir data/aggregate --output-dir figures
```

This regenerates the benchmark, development-class distribution, fine-tuned-model confusion, ResNet-50 confusion, and ResNet-50 error-profile figures. Public labels use `Fine-tuned MedGemma-4B/27B` rather than branded checkpoint names.

## Regenerate Summary Tables

```bash
python scripts/generate_summary_tables.py --data-dir data/aggregate --output-dir tables
```

This writes the benchmark model table, three-seed ResNet-50 summary, bootstrap confidence intervals, and paired accuracy-comparison table from the released aggregate inputs.

## Analyze Record-Level Predictions

`scripts/analyze_record_predictions.py` is the sanitized publication version of the record-level metric, image-clustered bootstrap, and paired permutation analysis. It accepts a CSV containing `sample_id`, `image_id`, `model`, `ground_truth`, and `prediction`. For example:

```bash
python scripts/analyze_record_predictions.py \
  --predictions /path/to/private_predictions.csv \
  --first-model "ResNet-50 (seed 42)" \
  --second-model "Fine-tuned MedGemma-27B" \
  --output analysis.json
```

The released aggregate resampling results are available in `data/aggregate/statistical_results.json`.

## Reconstruct Record-Level Results

```bash
python scripts/reconstruct_record_level_results.py \
  --data-dir data/records \
  --output-dir tables/reconstructed
```

This reconstructs the 25-model benchmark metrics, fine-tuned MedGemma task-level metrics, three-seed ResNet-50 image and prompt metrics, image-clustered confidence intervals, paired comparisons, and the baseline-versus-fine-tuned McNemar test.

`benchmark_response_attempts.csv` preserves every saved benchmark attempt. `benchmark_predictions.csv` selects the final attempt for each model-question pair, including unevaluable responses as missing predictions, matching the manuscript analysis.

## Audit a Local HyperKvasir Split

```bash
python scripts/audit_dataset_split.py --development-root /path/to/development --held-out-root /path/to/held-out --output split_audit.json
```

The manuscript's aggregate split-audit results are provided in `data/aggregate/split_audit_summary.json`.

## Data Dictionary

- `benchmark_metrics.csv`: final 25-model benchmark metrics.
- `development_class_counts.csv`: 23-class development-set counts.
- `finetuned_medgemma_*_sample_confusion.csv`: aggregate fine-tuned-model confusion counts and row proportions.
- `resnet50_seed42_confusion_*.csv`: final seed-42 direct 23-class confusion matrices.
- `resnet50_seed42_per_class_metrics.csv`: final seed-42 class-level metrics.
- `resnet50_three_seed_metrics.csv`: aggregate results for seeds 17, 42, and 123.
- `statistical_results.json`: final image-bootstrap confidence intervals, paired image-clustered comparisons, and primary ResNet-50 results.
- `split_audit_summary.json`: sanitized split-audit counts.
- `data/records/benchmark_response_attempts.csv`: all saved benchmark LLM response attempts, including failed attempts.
- `data/records/benchmark_predictions.csv`: final responses for all 25 benchmark models, paired with questions and image IDs.
- `data/records/medgemma_full_predictions.csv`: complete held-out responses for the two fine-tuned and two baseline MedGemma models.
- `data/records/resnet50_prompt_predictions.csv`: projected prompt predictions for all three ResNet-50 seeds.
- `data/records/resnet50_image_predictions.csv`: direct image predictions and 23-class probabilities for all three ResNet-50 seeds.

## Reproducibility

The supplied record-level and aggregate data reproduce the reported benchmark metrics, task-level results, ResNet-50 results, confidence intervals, paired comparisons, and quantitative figures. Each response is linked to its evaluation question and HyperKvasir image identifier.

## Rights

All rights are reserved pending institutional intellectual-property review. HyperKvasir data and images remain governed by their original source license.
