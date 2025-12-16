"""
Test GBRS model correctness against sklearn baseline models.

These tests verify that GBRS achieves reasonable performance compared to
standard baseline models on real datasets.
"""

import pytest
import numpy as np
from gbrs import GBRS
from tests.conftest import calculate_mse, calculate_accuracy, calculate_c_index


@pytest.mark.baseline
def test_regression_vs_linear_baseline(diabetes_data, linear_baseline):
    """
    Test that GBRS regression performs reasonably vs LinearRegression.

    Acceptance: GBRS MSE should be within 2x of LinearRegression MSE.
    """
    # Fit GBRS model
    model = GBRS(n_iter=50, lr=0.1, n_quantiles=10)
    model.fit(diabetes_data["X_train"], diabetes_data["y_train"])

    # Get predictions
    gbrs_pred = model.predict(diabetes_data["X_test"])
    baseline_pred = linear_baseline.predict(diabetes_data["X_test"])

    # Calculate MSE
    gbrs_mse = calculate_mse(diabetes_data["y_test"], gbrs_pred)
    baseline_mse = calculate_mse(diabetes_data["y_test"], baseline_pred)

    print(f"\n  GBRS MSE: {gbrs_mse:.2f}")
    print(f"  LinearRegression MSE: {baseline_mse:.2f}")
    print(f"  Ratio: {gbrs_mse/baseline_mse:.2f}x")

    # GBRS should be competitive (within 2x)
    assert (
        gbrs_mse < 2.0 * baseline_mse
    ), f"GBRS MSE ({gbrs_mse:.2f}) is more than 2x worse than baseline ({baseline_mse:.2f})"

    # GBRS should learn something (better than predicting mean)
    mean_pred_mse = calculate_mse(
        diabetes_data["y_test"],
        np.full_like(diabetes_data["y_test"], diabetes_data["y_train"].mean()),
    )
    assert (
        gbrs_mse < mean_pred_mse
    ), f"GBRS MSE ({gbrs_mse:.2f}) is worse than predicting mean ({mean_pred_mse:.2f})"


@pytest.mark.baseline
def test_classification_vs_logistic_baseline(breast_cancer_data, logistic_baseline):
    """
    Test that GBRS classification performs reasonably vs LogisticRegression.

    Acceptance: GBRS accuracy should be within 5% of LogisticRegression.
    """
    # Fit GBRS model
    model = GBRS(n_iter=50, lr=0.1, n_quantiles=10)
    model.fit_proba(breast_cancer_data["X_train"], breast_cancer_data["y_train"])

    # Get predictions
    gbrs_pred_proba = model.predict_proba(breast_cancer_data["X_test"])
    gbrs_pred = (gbrs_pred_proba > 0.5).astype(int)
    baseline_pred = logistic_baseline.predict(breast_cancer_data["X_test"])

    # Calculate accuracy
    gbrs_acc = calculate_accuracy(breast_cancer_data["y_test"], gbrs_pred)
    baseline_acc = calculate_accuracy(breast_cancer_data["y_test"], baseline_pred)

    print(f"\n  GBRS Accuracy: {gbrs_acc:.2%}")
    print(f"  LogisticRegression Accuracy: {baseline_acc:.2%}")
    print(f"  Difference: {abs(gbrs_acc - baseline_acc):.2%}")

    # GBRS should be competitive (within 5% accuracy)
    assert (
        abs(gbrs_acc - baseline_acc) < 0.05
    ), f"GBRS accuracy ({gbrs_acc:.2%}) differs from baseline ({baseline_acc:.2%}) by more than 5%"

    # GBRS should be better than random guessing
    majority_class_acc = np.max(np.bincount(breast_cancer_data["y_train"])) / len(
        breast_cancer_data["y_train"]
    )
    assert (
        gbrs_acc > majority_class_acc
    ), f"GBRS accuracy ({gbrs_acc:.2%}) is not better than majority class ({majority_class_acc:.2%})"


@pytest.mark.baseline
@pytest.mark.slow
def test_survival_vs_cox_baseline(veteran_data, cox_baseline):
    """
    Test that GBRS survival performs reasonably vs CoxPH.

    Acceptance: GBRS C-index should be > 0.6 (better than random).
    """
    # Fit GBRS survival model
    model = GBRS(n_iter=50, lr=0.1, n_quantiles=5)
    model._model.fit_survival(
        veteran_data["X_train"], veteran_data["time_train"], veteran_data["event_train"]
    )

    # Get risk scores
    gbrs_risk = model.predict(veteran_data["X_test"])

    # Calculate C-index
    gbrs_cindex = calculate_c_index(
        veteran_data["time_test"], veteran_data["event_test"], gbrs_risk
    )

    # Get CoxPH predictions
    import pandas as pd

    df_test = pd.DataFrame(veteran_data["X_test"])
    cox_risk = cox_baseline.predict_partial_hazard(df_test).values
    cox_cindex = calculate_c_index(
        veteran_data["time_test"], veteran_data["event_test"], cox_risk
    )

    print(f"\n  GBRS C-index: {gbrs_cindex:.3f}")
    print(f"  CoxPH C-index: {cox_cindex:.3f}")
    print(f"  Difference: {abs(gbrs_cindex - cox_cindex):.3f}")

    # GBRS should be better than random (0.5)
    assert (
        gbrs_cindex > 0.6
    ), f"GBRS C-index ({gbrs_cindex:.3f}) is not significantly better than random (0.5)"

    # GBRS should be somewhat competitive with CoxPH
    assert (
        gbrs_cindex > 0.5 * cox_cindex
    ), f"GBRS C-index ({gbrs_cindex:.3f}) is much worse than CoxPH ({cox_cindex:.3f})"


@pytest.mark.baseline
def test_regression_improves_over_iterations(diabetes_data):
    """Verify that GBRS regression improves with more iterations."""
    X_train, y_train = diabetes_data["X_train"], diabetes_data["y_train"]
    X_test, y_test = diabetes_data["X_test"], diabetes_data["y_test"]

    mse_values = []
    iterations = [1, 5, 10, 20, 50]

    for n_iter in iterations:
        model = GBRS(n_iter=n_iter, lr=0.1, n_quantiles=10)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = calculate_mse(y_test, pred)
        mse_values.append(mse)

    print("\n  MSE by iterations:")
    for n_iter, mse in zip(iterations, mse_values):
        print(f"    {n_iter:3d} iterations: MSE = {mse:.2f}")

    # MSE should generally decrease (allow some variance)
    # Check that final MSE is better than initial
    assert (
        mse_values[-1] < mse_values[0]
    ), f"MSE did not improve: {mse_values[0]:.2f} -> {mse_values[-1]:.2f}"
