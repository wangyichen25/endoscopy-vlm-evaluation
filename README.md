# Endoscopy VLM Evaluation

This repository provides the privacy-reviewed aggregate data and the complete downstream analysis and visualization release used to produce the final manuscript's quantitative tables, statistical results, and figures comparing fine-tuned MedGemma vision-language models, a training-matched ResNet-50 control, and 22 generalist vision-language models on HyperKvasir-derived tasks.

The scripts are publication copies of the analysis logic used for the final manuscript, with paths, branded checkpoint labels, and private infrastructure references normalized for public release. They are not model-development code.

## Scope

Included:

- Privacy-reviewed aggregate data underlying the final quantitative tables and figures.
- Validation code for the manuscript's reported sample counts, primary metrics, confidence intervals, and paired comparisons.
- Summary-table code for the released benchmark and ResNet-50 results.
- Visualization code that regenerates all quantitative manuscript figures.
- Dataset split-audit code for exact duplicates and close perceptual-hash matches.

Excluded intentionally for intellectual-property and data-governance reasons:

- Model training, fine-tuning, checkpoint construction, and inference code.
- Model weights and checkpoints.
- Raw endoscopy images, record-level predictions, prompts, API responses, and identifiers.
- Manuscript automation, submission files, logs, and internal QA artifacts.

The two composite manuscript figures containing representative HyperKvasir images are not redistributed and cannot be regenerated from this repository alone.

This is a privacy- and IP-reviewed publication release, not an unfiltered copy of the private research workspace. The analysis logic and final aggregate inputs are preserved, while local paths, private identifiers, intermediate files, and model-development implementation are omitted or normalized.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

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

The manuscript's record-level input is not published, so this script documents and supports the analysis but cannot reproduce the saved resampling results from the aggregate release alone.

## Audit a Local HyperKvasir Split

```bash
python scripts/audit_dataset_split.py --development-root /path/to/development --held-out-root /path/to/held-out --output split_audit.json
```

Detailed audit output contains local paths and file-level hashes and should be reviewed before sharing. The repository includes only the sanitized aggregate summary used for the manuscript.

## Data Dictionary

- `benchmark_metrics.csv`: final 25-model benchmark metrics.
- `development_class_counts.csv`: 23-class development-set counts.
- `finetuned_medgemma_*_sample_confusion.csv`: aggregate fine-tuned-model confusion counts and row proportions.
- `resnet50_seed42_confusion_*.csv`: final seed-42 direct 23-class confusion matrices.
- `resnet50_seed42_per_class_metrics.csv`: final seed-42 class-level metrics.
- `resnet50_three_seed_metrics.csv`: aggregate results for seeds 17, 42, and 123.
- `statistical_results.json`: final image-bootstrap confidence intervals, paired image-clustered comparisons, and primary ResNet-50 results.
- `split_audit_summary.json`: sanitized split-audit counts.

No released aggregate file contains image paths, image IDs, sample indices, patient identifiers, or record-level predictions.

## Reproducibility Boundary

Users can regenerate all released quantitative figures and summary tables and verify the aggregate values reported in the final manuscript. Because record-level predictions are intentionally excluded, users can verify—but cannot independently rerun—the record-level bootstrap and paired-resampling procedures. They also cannot rerun model training or inference or recreate the two image-containing composite figures from this repository.

## Rights

No software license is granted. All rights are reserved pending institutional intellectual-property review. HyperKvasir data and images remain governed by their original source license and are not included.
