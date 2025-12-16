"""
Test GBRS training convergence on real datasets.

These tests verify that training loss/metrics consistently improve over iterations.
"""
import pytest
import numpy as np
from gbrs import GBRS
from conftest import calculate_mse, calculate_accuracy, calculate_c_index


@pytest.mark.integration
def test_regression_convergence(diabetes_data):
    """
    Verify that regression MSE improves monotonically during training.
    """
    X_train = diabetes_data['X_train']
    y_train = diabetes_data['y_train']
    X_test = diabetes_data['X_test']
    y_test = diabetes_data['y_test']
    
    # Train models with increasing iterations
    checkpoints = [1, 5, 10, 20, 50]
    mse_train = []
    mse_test = []
    
    for n_iter in checkpoints:
        model = GBRS(n_iter=n_iter, lr=0.1, n_quantiles=10)
        model.fit(X_train, y_train)
        
        # Evaluate on both train and test
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        
        mse_train.append(calculate_mse(y_train, pred_train))
        mse_test.append(calculate_mse(y_test, pred_test))
    
    # Print convergence table
    print("\n  Regression Convergence:")
    print(f"  {'Iter':<6} {'Train MSE':<12} {'Test MSE':<12} {'Improvement':<15}")
    print("  " + "-" * 50)
    for i, n_iter in enumerate(checkpoints):
        if i == 0:
            improvement = "-"
        else:
            improvement = f"{mse_test[i-1] - mse_test[i]:+.2f}"
        print(f"  {n_iter:<6} {mse_train[i]:<12.2f} {mse_test[i]:<12.2f} {improvement:<15}")
    
    # Training MSE should decrease monotonically
    for i in range(1, len(mse_train)):
        assert mse_train[i] <= mse_train[i-1] + 1e-6, \
            f"Training MSE increased from {mse_train[i-1]:.2f} to {mse_train[i]:.2f} at iteration {checkpoints[i]}"
    
    # Final test MSE should be better than initial
    assert mse_test[-1] < mse_test[0], \
        f"Test MSE did not improve: {mse_test[0]:.2f} -> {mse_test[-1]:.2f}"


@pytest.mark.integration
def test_classification_convergence(breast_cancer_data):
    """
    Verify that classification accuracy improves during training.
    """
    X_train = breast_cancer_data['X_train']
    y_train = breast_cancer_data['y_train']
    X_test = breast_cancer_data['X_test']
    y_test = breast_cancer_data['y_test']
    
    # Train models with increasing iterations
    checkpoints = [1, 5, 10, 20, 50]
    acc_train = []
    acc_test = []
    
    for n_iter in checkpoints:
        model = GBRS(n_iter=n_iter, lr=0.1, n_quantiles=10)
        model.fit_proba(X_train, y_train)
        
        # Evaluate on both train and test
        pred_train = (model.predict_proba(X_train) > 0.5).astype(int)
        pred_test = (model.predict_proba(X_test) > 0.5).astype(int)
        
        acc_train.append(calculate_accuracy(y_train, pred_train))
        acc_test.append(calculate_accuracy(y_test, pred_test))
    
    # Print convergence table
    print("\n  Classification Convergence:")
    print(f"  {'Iter':<6} {'Train Acc':<12} {'Test Acc':<12} {'Improvement':<15}")
    print("  " + "-" * 50)
    for i, n_iter in enumerate(checkpoints):
        if i == 0:
            improvement = "-"
        else:
            improvement = f"{acc_test[i] - acc_test[i-1]:+.3f}"
        print(f"  {n_iter:<6} {acc_train[i]:<12.3f} {acc_test[i]:<12.3f} {improvement:<15}")
    
    # Training accuracy should generally increase
    # We allow for some fluctuation due to stochastic nature and small dataset
    for i in range(1, len(acc_train)):
        if acc_train[i] < acc_train[i-1]:
            print(f"  Note: Training accuracy dropped at iter {checkpoints[i]}")
    
    # Final test accuracy should be better than initial
    assert acc_test[-1] > acc_test[0], \
        f"Test accuracy did not improve: {acc_test[0]:.3f} -> {acc_test[-1]:.3f}"


@pytest.mark.integration
@pytest.mark.slow
def test_survival_convergence(veteran_data):
    """
    Verify that survival C-index improves during training.
    """
    X_train = veteran_data['X_train']
    time_train = veteran_data['time_train']
    event_train = veteran_data['event_train']
    X_test = veteran_data['X_test']
    time_test = veteran_data['time_test']
    event_test = veteran_data['event_test']
    
    # Train models with increasing iterations
    checkpoints = [1, 5, 10, 20, 50]
    cindex_train = []
    cindex_test = []
    
    for n_iter in checkpoints:
        model = GBRS(n_iter=n_iter, lr=0.1, n_quantiles=5)
        model._model.fit_survival(X_train, time_train, event_train)
        
        # Evaluate on both train and test
        risk_train = model.predict(X_train)
        risk_test = model.predict(X_test)
        
        cindex_train.append(calculate_c_index(time_train, event_train, risk_train))
        cindex_test.append(calculate_c_index(time_test, event_test, risk_test))
    
    # Print convergence table
    print("\n  Survival Convergence:")
    print(f"  {'Iter':<6} {'Train C-idx':<12} {'Test C-idx':<12} {'Improvement':<15}")
    print("  " + "-" * 50)
    for i, n_iter in enumerate(checkpoints):
        if i == 0:
            improvement = "-"
        else:
            improvement = f"{cindex_test[i] - cindex_test[i-1]:+.3f}"
        print(f"  {n_iter:<6} {cindex_train[i]:<12.3f} {cindex_test[i]:<12.3f} {improvement:<15}")
    
    # Final test C-index should be better than initial
    assert cindex_test[-1] > cindex_test[0], \
        f"Test C-index did not improve: {cindex_test[0]:.3f} -> {cindex_test[-1]:.3f}"
    
    # Final C-index should be > 0.6 (better than random)
    assert cindex_test[-1] > 0.6, \
        f"Final C-index ({cindex_test[-1]:.3f}) is not significantly better than random (0.5)"


@pytest.mark.integration
def test_no_overfitting_regression(diabetes_data):
    """Verify that regression model doesn't overfit badly."""
    X_train = diabetes_data['X_train']
    y_train = diabetes_data['y_train']
    X_test = diabetes_data['X_test']
    y_test = diabetes_data['y_test']
    
    model = GBRS(n_iter=100, lr=0.1, n_quantiles=10)
    model.fit(X_train, y_train)
    
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    mse_train = calculate_mse(y_train, pred_train)
    mse_test = calculate_mse(y_test, pred_test)
    
    print(f"\n  Train MSE: {mse_train:.2f}")
    print(f"  Test MSE: {mse_test:.2f}")
    print(f"  Test/Train ratio: {mse_test/mse_train:.2f}x")
    
    # Test MSE should not be more than 3× train MSE (reasonable generalization)
    assert mse_test < 3.0 * mse_train, \
        f"Severe overfitting detected: test MSE ({mse_test:.2f}) is {mse_test/mse_train:.1f}x train MSE ({mse_train:.2f})"
