import numpy as np

from neurotrust_eeg.metrics import (
    binary_classification_metrics,
    expected_calibration_error,
    sigmoid,
)


def test_sigmoid_bounds():
    x = np.array([-100.0, 0.0, 100.0])
    y = sigmoid(x)

    assert np.all(y >= 0.0)
    assert np.all(y <= 1.0)
    assert np.isclose(y[1], 0.5)


def test_expected_calibration_error_is_non_negative():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    ece = expected_calibration_error(y_true, y_prob)

    assert ece >= 0.0


def test_binary_classification_metrics_keys():
    y_true = np.array([0, 0, 1, 1])
    logits = np.array([-3.0, -2.0, 2.0, 3.0])

    metrics = binary_classification_metrics(y_true, logits)

    expected_keys = {
        "f1",
        "brier",
        "ece",
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
    }

    assert expected_keys.issubset(metrics.keys())
    assert metrics["f1"] == 1.0