import pandas as pd
import pytest

from endoscopy_vlm_eval.metrics import (
    clustered_bootstrap_accuracy_difference,
    clustered_sign_permutation_p_value,
    hard_label_metrics,
)


def test_hard_label_metrics_perfect_predictions():
    metrics = hard_label_metrics(["a", "b", "a", "b"], ["a", "b", "a", "b"])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_clustered_comparison_detects_difference():
    frame = pd.DataFrame({"image": [1, 1, 2, 3], "first": [1, 1, 1, 1], "second": [0, 0, 1, 0]})
    interval = clustered_bootstrap_accuracy_difference(frame, "image", "first", "second", replicates=200, random_seed=7)
    assert interval["difference"] == 0.75
    assert interval["lower_95"] >= 0
    p_value = clustered_sign_permutation_p_value(frame, "image", "first", "second", replicates=200, random_seed=7)
    assert 0 < p_value <= 1


def test_hard_label_metrics_handles_unpredicted_class():
    metrics = hard_label_metrics(["a", "b"], ["a", "a"], labels=["a", "b"])
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_sensitivity"] == 0.5
    assert metrics["hard_prediction_macro_auroc"] == pytest.approx(0.5)
