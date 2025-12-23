"""Tests for survival base score computation."""

import numpy as np
import pytest
from gbrs import GBRS


def test_survival_base_score_is_computed():
    """Test that survival models compute a meaningful base score (y0)."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 3)

    # Generate survival data
    beta = np.array([0.5, -0.3, 0.2])
    log_hazard = X @ beta
    time = np.random.exponential(np.exp(-log_hazard))
    event = np.random.binomial(1, 0.7, size=n).astype(float)

    # Fit model
    model = GBRS(n_iter=10, lr=0.1, n_quantiles=5)
    model.fit_survival(X, time, event)

    # Get params
    params = model._model.get_params()

    # y0 should be non-zero (log of event rate for low-risk group)
    assert params.y0 != 0.0, "Base score should be non-zero for survival"

    # y0 should be negative (log of a rate < 1 typically)
    # since events per time unit is usually < 1
    print(f"Base score (y0): {params.y0:.4f}")


def test_survival_base_score_interpretation():
    """Test that base score represents log(events/time) for low-risk group."""
    np.random.seed(123)
    n = 200
    X = np.random.randn(n, 2)

    # Simple survival data
    time = np.abs(np.random.randn(n)) + 0.1
    event = np.random.binomial(1, 0.5, size=n).astype(float)

    # Fit model
    model = GBRS(n_iter=20, lr=0.1, n_quantiles=5)
    model.fit_survival(X, time, event)

    params = model._model.get_params()

    # y0 should be a log-rate (typically negative since events/time < 1)
    # Verify it's in a reasonable range for log hazard rates
    print(f"Model y0: {params.y0:.4f}")

    # Compute overall event rate as sanity check
    overall_rate = event.sum() / time.sum()
    log_overall_rate = np.log(overall_rate)
    print(f"Log overall event rate: {log_overall_rate:.4f}")

    # Base score should be in a reasonable range relative to overall rate
    # (low-risk group rate should be lower than overall)
    assert (
        params.y0 < log_overall_rate + 1
    ), f"Base score {params.y0:.4f} should be <= overall log-rate {log_overall_rate:.4f} + 1"
    assert (
        params.y0 > log_overall_rate - 3
    ), f"Base score {params.y0:.4f} should be >= overall log-rate {log_overall_rate:.4f} - 3"


def test_survival_print_shows_base_score(capsys):
    """Test that printing survival model shows base score."""
    np.random.seed(456)
    n = 50
    X = np.random.randn(n, 2)
    time = np.random.exponential(1, size=n)
    event = np.ones(n)

    model = GBRS(n_iter=5, lr=0.1, n_quantiles=3)
    model.fit_survival(X, time, event)

    # Print model
    model.print({0: "Feature1", 1: "Feature2"})

    captured = capsys.readouterr()
    assert (
        "Base Score:" in captured.out
    ), "Print output should include 'Base Score:' for survival models"


if __name__ == "__main__":
    test_survival_base_score_is_computed()
    test_survival_base_score_interpretation()
    print("\nAll tests passed!")
