import numpy as np
from gbrs import GBRS
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
    model = GBRS(n_iter=10, lr=0.1, n_quantiles=10, ss_rate=0.5)
    model.fit_proba(X, y)
    
    # Print formatted output
    print("\n=== Python Model Output ===")
    feature_names = {i: f"X{i+1}" for i in range(p)}
    model.print(feature_names)
    
    # Verify predictions work
    preds = model.predict_proba(X)
    print(f"\nPredictions shape: {preds.shape}")
    
    if preds.shape[0] == n:
        print("\n✓ Integration test passed!")
    else:
        print("\n✗ Integration test failed!")
        sys.exit(1)

if __name__ == "__main__":
    test_integration()

