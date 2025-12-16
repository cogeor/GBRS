"""
Performance benchmarks for GBRS using pytest parametrization.

Tests training time across different dataset sizes with proper pytest integration.
"""

import pytest
import numpy as np
import time
from gbrs import GBRS


def generate_regression_data(n_samples, n_features, seed=42):
    """Generate synthetic regression data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    beta = np.random.randn(n_features)
    y = X @ beta + np.random.randn(n_samples) * 0.1
    return X, y


def generate_classification_data(n_samples, n_features, seed=42):
    """Generate synthetic classification data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    beta = np.random.randn(n_features)
    logits = X @ beta
    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)
    return X, y


def generate_survival_data(n_samples, n_features, seed=42):
    """Generate synthetic survival data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    beta = np.random.randn(n_features)
    log_hazard = X @ beta
    time = np.random.exponential(np.exp(-log_hazard))
    event = np.random.binomial(1, 0.7, size=n_samples)
    return X, time, event


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_samples,n_features,n_iter",
    [
        (1000, 10, 20),
        (10000, 10, 20),
    ],
)
def test_regression_performance(n_samples, n_features, n_iter):
    """Benchmark regression model performance."""
    X, y = generate_regression_data(n_samples, n_features)

    model = GBRS(n_iter=n_iter, lr=0.1, n_quantiles=10)

    start_time = time.time()
    model.fit(X, y)
    elapsed = time.time() - start_time

    # Verify predictions work
    preds = model.predict(X)
    mse = np.mean((preds - y) ** 2)

    print(f"\n  {n_samples:,} samples × {n_features} features")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  MSE: {mse:.4f}")

    # Performance should be reasonable (< 5s for 10K samples)
    if n_samples <= 10000:
        assert elapsed < 5.0, f"Training took {elapsed:.2f}s, expected < 5s"


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_samples,n_features,n_iter",
    [
        (1000, 10, 20),
        (10000, 10, 20),
    ],
)
def test_classification_performance(n_samples, n_features, n_iter):
    """Benchmark classification model performance."""
    X, y = generate_classification_data(n_samples, n_features)

    model = GBRS(n_iter=n_iter, lr=0.1, n_quantiles=10)

    start_time = time.time()
    model.fit_proba(X, y)
    elapsed = time.time() - start_time

    # Verify predictions work
    preds = model.predict_proba(X)
    acc = np.mean((preds > 0.5) == y)

    print(f"\n  {n_samples:,} samples × {n_features} features")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Accuracy: {acc:.2%}")

    # Performance should be reasonable (< 5s for 10K samples)
    if n_samples <= 10000:
        assert elapsed < 5.0, f"Training took {elapsed:.2f}s, expected < 5s"


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_samples,n_features,n_iter",
    [
        (1000, 10, 20),
    ],
)
def test_survival_performance(n_samples, n_features, n_iter):
    """Benchmark survival model performance."""
    X, survival_times, event = generate_survival_data(n_samples, n_features)

    model = GBRS(n_iter=n_iter, lr=0.1, n_quantiles=5)

    start_time = time.time()
    model._model.fit_survival(X, survival_times, event)
    elapsed = time.time() - start_time

    # Verify predictions work
    risk_scores = model.predict(X)
    risk_range = float(risk_scores.max() - risk_scores.min())

    print(f"\n  {n_samples:,} samples × {n_features} features")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Risk range: {risk_range:.2f}")

    # Performance should be reasonable
    assert elapsed < 10.0, f"Training took {elapsed:.2f}s, expected < 10s"
