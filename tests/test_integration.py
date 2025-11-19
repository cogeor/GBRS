import numpy as np
import gbrs
import sys

def test_integration():
    print("Running integration test...")
    
    # Generate dummy data
    np.random.seed(42)
    n = 100
    p = 5
    X = np.random.randn(n, p)
    beta = np.array([1, -1, 0.5, 0, 0])
    logits = X @ beta
    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)

    # Fit model
    print("Fitting model...")
    model = gbrs.Model(n_iter=10, lr=0.1, n_quantiles=10, ss_rate=0.5)
    model.fit_proba(X, y)
    
    # Check output
    params = model.get_params()
    print(f"Model fitted. y0: {params.y0}")
    
    preds = model.predict_proba(X)
    print(f"Predictions shape: {preds.shape}")
    
    if preds.shape[0] == n:
        print("Integration test passed!")
    else:
        print("Integration test failed!")
        sys.exit(1)

if __name__ == "__main__":
    test_integration()
