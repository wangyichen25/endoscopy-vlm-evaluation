# Endoscopy VLM Evaluation

This repository contains the downstream evaluation and deterministic figure-generation code used for a manuscript comparing fine-tuned MedGemma vision-language models, a training-matched ResNet-50 control, and 22 generalist vision-language models on HyperKvasir-derived tasks.

## Contents

- `src/endoscopy_vlm_eval/metrics.py`: hard-label classification metrics and image-clustered bootstrap/permutation comparisons.
- `src/endoscopy_vlm_eval/split_audit.py`: exact-duplicate, perceptual-similarity, and image-artifact audit for user-supplied HyperKvasir split directories.
- `src/endoscopy_vlm_eval/figures.py`: deterministic generation of the final quantitative manuscript figures from aggregate data.
- `src/endoscopy_vlm_eval/validate_release.py`: consistency checks against the final reported counts and metrics.
- `data/aggregate/`: aggregate, non-image inputs used by the figure and validation workflows.

Model training, fine-tuning, checkpoint construction, prompts used for training, inference clients, model weights, raw images, and record-level predictions are intentionally excluded to protect intellectual property and data governance requirements.

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Validate Published Results

```bash
python -m endoscopy_vlm_eval.validate_release --data-dir data/aggregate
```

The validator checks the 8,540/2,122 development and held-out counts, the 25-model comparison table, the 320-record benchmark, fine-tuned MedGemma benchmark results, and the prespecified seed-42 ResNet-50 results.

## Regenerate Quantitative Figures

```bash
python -m endoscopy_vlm_eval.figures \
  --data-dir data/aggregate \
  --output-dir figures
```

This command generates:

- `benchmark_accuracy_macro_f1.png`
- `development_class_distribution.png`
- `finetuned_medgemma_high_sample_confusion.png`
- `finetuned_medgemma_low_sample_confusion.png`
- `resnet50_seed42_confusion.png`
- `resnet50_seed42_error_profile.png`

The manuscript's composite figures containing representative HyperKvasir images are not redistributed here.

## Audit a Local Dataset Split

Obtain HyperKvasir from its official source and arrange the development and held-out images in class subdirectories. Then run:

```bash
python -m endoscopy_vlm_eval.split_audit \
  --development-root /path/to/development \
  --held-out-root /path/to/held-out \
  --output split_audit.json
```

The detailed audit output contains local paths and file-level hashes and should be reviewed before sharing. The repository includes only a sanitized aggregate summary from the manuscript analysis.

## Reproducibility Boundary

The released aggregate files contain no raw endoscopy images, model checkpoints, prompts, API responses, or per-record predictions. Consequently, the repository reproduces reported aggregate checks and figures but cannot independently rerun model inference or all record-level paired analyses.

## Rights

No software license is granted. All rights are reserved pending institutional intellectual-property review. HyperKvasir data and images are governed by their original source license and are not included.
