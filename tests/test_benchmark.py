"""
Benchmark tests for GBRS performance optimization.

Tests training time across different dataset sizes:
- 1K samples x 10 features
- 10K samples x 10 features
- 100K samples x 10 features

For each size, tests:
- Regression (fit)
- Classification (fit_proba)
- Survival (fit_survival)
"""

import numpy as np
import time
import json
from gbrs import GBRS


def generate_regression_data(n_samples, n_features, seed=42):
    """Generate synthetic regression data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    # True coefficients
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
    event = np.random.binomial(1, 0.7, size=n_samples)  # 70% event rate
    return X, time, event


def benchmark_regression(n_samples, n_features, n_iter=20):
    """Benchmark regression model."""
    X, y = generate_regression_data(n_samples, n_features)

    model = GBRS(n_iter=n_iter, lr=0.1, n_quantiles=10)

    start_time = time.time()
    model.fit(X, y)
    elapsed = time.time() - start_time

    # Sanity check
    preds = model.predict(X)
    mse = np.mean((preds - y) ** 2)

    return {
        "time_seconds": elapsed,
        "mse": float(mse),
        "n_samples": n_samples,
        "n_features": n_features,
        "n_iter": n_iter,
    }


def benchmark_classification(n_samples, n_features, n_iter=20):
    """Benchmark classification model."""
    X, y = generate_classification_data(n_samples, n_features)

    model = GBRS(n_iter=n_iter, lr=0.1, n_quantiles=10)

    start_time = time.time()
    model.fit_proba(X, y)
    elapsed = time.time() - start_time

    # Sanity check
    preds = model.predict_proba(X)
    accuracy = np.mean((preds > 0.5) == y)

    return {
        "time_seconds": elapsed,
        "accuracy": float(accuracy),
        "n_samples": n_samples,
        "n_features": n_features,
        "n_iter": n_iter,
    }


def benchmark_survival(n_samples, n_features, n_iter=20):
    """Benchmark survival model."""
    X, time, event = generate_survival_data(n_samples, n_features)

    model = GBRS(n_iter=n_iter, lr=0.1, n_quantiles=10)

    start_time = time.time()
    model.fit_survival(X, time, event)
    elapsed = time.time() - start_time

    # Sanity check - just verify predictions work
    risk_scores = model.predict(X)
    risk_range = float(risk_scores.max() - risk_scores.min())

    return {
        "time_seconds": elapsed,
        "risk_range": risk_range,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_iter": n_iter,
    }


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 70)
    print("GBRS Performance Benchmarks")
    print("=" * 70)
    print()

    # Dataset sizes to test
    sizes = [
        (1000, 10, 20),  # 1K samples, 10 features, 20 iterations
        (10000, 10, 20),  # 10K samples, 10 features, 20 iterations
        (
            100000,
            10,
            10,
        ),  # 100K samples, 10 features, 10 iterations (fewer iters for speed)
    ]

    results = {"regression": [], "classification": [], "survival": []}

    for n_samples, n_features, n_iter in sizes:
        print(f"\n{'─' * 70}")
        print(
            f"Dataset: {n_samples:,} samples × {n_features} features ({n_iter} iterations)"
        )
        print(f"{'─' * 70}")

        # Regression
        print("\n  Regression...", end=" ", flush=True)
        result = benchmark_regression(n_samples, n_features, n_iter)
        results["regression"].append(result)
        print(f"✓ {result['time_seconds']:.3f}s (MSE: {result['mse']:.4f})")

        # Classification
        print("  Classification...", end=" ", flush=True)
        result = benchmark_classification(n_samples, n_features, n_iter)
        results["classification"].append(result)
        print(f"✓ {result['time_seconds']:.3f}s (Acc: {result['accuracy']:.2%})")

        # Survival
        print("  Survival...", end=" ", flush=True)
        result = benchmark_survival(n_samples, n_features, n_iter)
        results["survival"].append(result)
        print(f"✓ {result['time_seconds']:.3f}s (Range: {result['risk_range']:.2f})")

    print(f"\n{'=' * 70}")
    print("Summary Table")
    print(f"{'=' * 70}\n")

    # Print summary table
    print(f"{'Size':<12} {'Type':<15} {'Time (s)':<12} {'Metric':<20}")
    print(f"{'-' * 70}")

    for i, (n_samples, n_features, n_iter) in enumerate(sizes):
        size_str = f"{n_samples//1000}K×{n_features}"

        reg = results["regression"][i]
        print(
            f"{size_str:<12} {'Regression':<15} {reg['time_seconds']:<12.3f} MSE: {reg['mse']:.4f}"
        )

        clf = results["classification"][i]
        print(
            f"{size_str:<12} {'Classification':<15} {clf['time_seconds']:<12.3f} Acc: {clf['accuracy']:.2%}"
        )

        surv = results["survival"][i]
        print(
            f"{size_str:<12} {'Survival':<15} {surv['time_seconds']:<12.3f} Range: {surv['risk_range']:.2f}"
        )
        print()

    # Save results to JSON
    output_file = "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 70}\n")

    return results


if __name__ == "__main__":
    results = run_benchmarks()
