
import numpy as np
import pytest
from gbrs import GBRS

def test_batch_subsampling_runs():
    # Synthetic data
    np.random.seed(42)
    N = 1000
    K = 10
    X = np.random.randn(N, K)
    # y = simple linear function + noise
    y = X[:, 0] + X[:, 1] + 0.5 * np.random.randn(N)
    # Binary target for proba
    y_bin = (y > 0).astype(float)
    # Survival target
    time = np.exp(y + np.random.randn(N))
    event = np.random.randint(0, 2, N)

    # 1. Regression (Continuous)
    print("Testing Regression with batch_size=50")
    model = GBRS(n_iter=10, batch_size=50)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (N,)
    assert not np.isnan(preds).any()

    # 2. Classification (Binary)
    print("Testing Classification with batch_size=50")
    model_bin = GBRS(n_iter=10, batch_size=50)
    model_bin.fit_proba(X, y_bin)
    preds_bin = model_bin.predict_proba(X) # predict_proba for probabilities
    assert preds_bin.shape == (N,)
    assert not np.isnan(preds_bin).any()

    # 3. Survival
    print("Testing Survival with batch_size=50")
    model_surv = GBRS(n_iter=10, batch_size=50)
    model_surv.fit_survival(X, time, event)
    preds_surv = model_surv.predict(X) # Survival returns risk scores via predict()
    assert preds_surv.shape == (N,)
    assert not np.isnan(preds_surv).any()

    print("All batch tests passed!")

if __name__ == "__main__":
    test_batch_subsampling_runs()
