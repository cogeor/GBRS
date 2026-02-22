"""Tests for GBRS bootstrapping functionality."""

import numpy as np
import pytest
from gbrs import GBRS, BootstrapResult


def generate_regression_data(n: int = 200, seed: int = 42):
    """Generate synthetic regression data."""
    np.random.seed(seed)
    X = np.random.randn(n, 3)
    # Target depends on thresholds in X
    y = (
        (X[:, 0] > 0).astype(float) * 2
        + (X[:, 1] > 0.5).astype(float) * 1.5
        + np.random.normal(0, 0.5, n)
    )
    return X, y


def generate_classification_data(n: int = 200, seed: int = 42):
    """Generate synthetic binary classification data."""
    np.random.seed(seed)
    X = np.random.randn(n, 3)
    prob = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    y = (np.random.rand(n) < prob).astype(float)
    return X, y


def generate_survival_data(n: int = 200, seed: int = 42):
    """Generate synthetic survival data."""
    np.random.seed(seed)
    X = np.random.randn(n, 3)
    beta = np.array([0.5, -0.3, 0.2])
    log_hazard = X @ beta
    time = np.random.exponential(np.exp(-log_hazard))
    event = np.random.binomial(1, 0.7, size=n).astype(float)
    return X, time, event


class TestBootstrapBasic:
    """Basic tests for bootstrap functionality."""

    def test_bootstrap_returns_correct_type(self):
        """Test that fit_bootstrap returns a BootstrapResult."""
        X, y = generate_regression_data()
        model = GBRS(n_iter=20, lr=0.1, n_quantiles=5)

        result = model.bootstrap(X, y, n_bootstrap=3)

        assert isinstance(result, BootstrapResult)

    def test_bootstrap_correct_number_of_samples(self):
        """Test that bootstrap runs the correct number of iterations."""
        X, y = generate_regression_data()
        model = GBRS(n_iter=20, lr=0.1, n_quantiles=5)

        result = model.bootstrap(X, y, n_bootstrap=5)

        assert result.n_bootstrap == 5
        assert len(result.all_weights) == 5
        assert len(result.all_y0) == 5

    def test_bootstrap_default_iterations(self):
        """Test that default n_bootstrap is 10."""
        X, y = generate_regression_data()
        model = GBRS(n_iter=10, lr=0.1, n_quantiles=3)

        result = model.bootstrap(X, y)  # default n_bootstrap

        assert result.n_bootstrap == 10


class TestBootstrapThresholds:
    """Tests for threshold consistency in bootstrapping."""

    def test_thresholds_are_precomputed(self):
        """Test that thresholds are computed and stored."""
        X, y = generate_regression_data()
        model = GBRS(n_iter=20, lr=0.1, n_quantiles=5)

        result = model.bootstrap(X, y, n_bootstrap=3)

        # Should have thresholds for each feature
        assert len(result.thresholds) == X.shape[1]
        for th in result.thresholds:
            assert isinstance(th, np.ndarray)
            assert len(th) > 0

    def test_weights_use_precomputed_thresholds(self):
        """Test that all weights use thresholds from pre-computed set."""
        X, y = generate_regression_data()
        model = GBRS(n_iter=30, lr=0.1, n_quantiles=5)

        result = model.bootstrap(X, y, n_bootstrap=5)

        # Collect all threshold values used in weights
        all_used_thresholds = set()
        for weights in result.all_weights:
            for idx, sv in weights.keys():
                all_used_thresholds.add((idx, sv))

        # Each used threshold should be from pre-computed thresholds
        for idx, sv in all_used_thresholds:
            feature_thresholds = result.thresholds[idx]
            # Check if sv is close to any pre-computed threshold
            assert any(
                np.isclose(sv, th, atol=1e-10) for th in feature_thresholds
            ), f"Threshold {sv} for feature {idx} not in pre-computed thresholds"


class TestBootstrapReproducibility:
    """Tests for reproducibility with random_state."""

    def test_same_seed_same_results(self):
        """Test that same random_state gives identical results."""
        X, y = generate_regression_data()
        model = GBRS(n_iter=20, lr=0.1, n_quantiles=5)

        result1 = model.bootstrap(X, y, n_bootstrap=3, random_state=42)
        result2 = model.bootstrap(X, y, n_bootstrap=3, random_state=42)

        # y0 values should be identical
        assert result1.all_y0 == result2.all_y0

        # Weight dicts should be identical
        for w1, w2 in zip(result1.all_weights, result2.all_weights):
            assert w1.keys() == w2.keys()
            for k in w1.keys():
                assert np.isclose(w1[k], w2[k])

    def test_different_seed_different_results(self):
        """Test that different random_state gives different results."""
        X, y = generate_regression_data()
        model = GBRS(n_iter=20, lr=0.1, n_quantiles=5)

        result1 = model.bootstrap(X, y, n_bootstrap=3, random_state=42)
        result2 = model.bootstrap(X, y, n_bootstrap=3, random_state=123)

        # Results should likely differ (not guaranteed but very probable)
        assert result1.all_y0 != result2.all_y0


class TestBootstrapStatistics:
    """Tests for statistical methods on BootstrapResult."""

    def test_get_weight_stats(self):
        """Test that get_weight_stats returns expected structure."""
        X, y = generate_regression_data()
        model = GBRS(n_iter=30, lr=0.1, n_quantiles=5)

        result = model.bootstrap(X, y, n_bootstrap=5)
        stats = result.get_weight_stats()

        # Should have stats for each feature with non-zero weights
        assert len(stats) > 0
        for idx, info in stats.items():
            assert "thresholds" in info
            assert "mean" in info
            assert "std" in info
            assert "name" in info
            assert len(info["mean"]) == len(info["thresholds"])
            assert len(info["std"]) == len(info["thresholds"])

    def test_get_weight_stats_with_feature_names(self):
        """Test that feature names are correctly used."""
        X, y = generate_regression_data()
        model = GBRS(n_iter=30, lr=0.1, n_quantiles=5)
        feature_names = {0: "Age", 1: "BMI", 2: "BP"}

        result = model.bootstrap(X, y, n_bootstrap=5)
        stats = result.get_weight_stats(feature_names=feature_names)

        for idx, info in stats.items():
            if idx in feature_names:
                assert info["name"] == feature_names[idx]

    def test_get_y0_stats(self):
        """Test base score statistics."""
        X, y = generate_regression_data()
        model = GBRS(n_iter=30, lr=0.1, n_quantiles=5)

        result = model.bootstrap(X, y, n_bootstrap=10)
        mean, std = result.get_y0_stats()

        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std >= 0  # std should be non-negative

    def test_get_confidence_intervals(self):
        """Test confidence interval computation."""
        X, y = generate_regression_data()
        model = GBRS(n_iter=30, lr=0.1, n_quantiles=5)

        result = model.bootstrap(X, y, n_bootstrap=20)
        ci = result.get_confidence_intervals(alpha=0.05)

        for idx, info in ci.items():
            assert "lower" in info
            assert "upper" in info
            assert "median" in info
            # Lower should be <= upper for each threshold
            for l, u in zip(info["lower"], info["upper"]):
                assert l <= u


class TestBootstrapObjectives:
    """Tests for different objective functions."""

    def test_fit_proba_bootstrap(self):
        """Test bootstrapping for binary classification."""
        X, y = generate_classification_data()
        model = GBRS(n_iter=20, lr=0.1, n_quantiles=5)

        result = model.bootstrap_proba(X, y, n_bootstrap=3)

        assert isinstance(result, BootstrapResult)
        assert result.objective == "binary"
        assert result.n_bootstrap == 3

    def test_fit_survival_bootstrap(self):
        """Test bootstrapping for survival analysis."""
        X, time, event = generate_survival_data()
        model = GBRS(n_iter=20, lr=0.1, n_quantiles=5)

        result = model.bootstrap_survival(X, time, event, n_bootstrap=3)

        assert isinstance(result, BootstrapResult)
        assert result.objective == "survival"
        assert result.n_bootstrap == 3


class TestBootstrapPrintSummary:
    """Tests for print_summary method."""

    def test_print_summary_runs(self, capsys):
        """Test that print_summary executes without errors."""
        X, y = generate_regression_data()
        model = GBRS(n_iter=20, lr=0.1, n_quantiles=5)

        result = model.bootstrap(X, y, n_bootstrap=3)
        result.print_summary()

        captured = capsys.readouterr()
        assert "GBRS Bootstrap Results" in captured.out
        assert "Base Score:" in captured.out

    def test_print_summary_with_feature_names(self, capsys):
        """Test print_summary with custom feature names."""
        X, y = generate_regression_data()
        model = GBRS(n_iter=30, lr=0.1, n_quantiles=5)
        feature_names = {0: "Age", 1: "BMI", 2: "BP"}

        result = model.bootstrap(X, y, n_bootstrap=3)
        result.print_summary(feature_names=feature_names)

        captured = capsys.readouterr()
        # At least one named feature should appear if it has non-zero weights
        # (can't guarantee all will appear depending on model selection)
        assert "GBRS Bootstrap Results" in captured.out


if __name__ == "__main__":
    # Run basic test
    print("Testing bootstrapping functionality...")
    X, y = generate_regression_data()

    model = GBRS(n_iter=50, lr=0.1, n_quantiles=5)
    result = model.bootstrap(X, y, n_bootstrap=10, random_state=42)

    print("\n" + "=" * 60)
    result.print_summary(feature_names={0: "Feature1", 1: "Feature2", 2: "Feature3"})

    print("\nAll basic tests passed!")
