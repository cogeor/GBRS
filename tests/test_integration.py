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

def test_survival():
    print("\n\nRunning survival analysis test...")
    
    # Generate dummy survival data
    np.random.seed(123)
    n = 100
    p = 3
    X = np.random.randn(n, p)
    
    # Generate survival times and events
    beta = np.array([0.5, -0.3, 0.2])
    log_hazard = X @ beta
    time = np.random.exponential(np.exp(-log_hazard))
    event = np.random.binomial(1, 0.7, size=n)  # 70% event rate
    
    # Fit survival model
    print("Fitting survival model...")
    model = GBRS(n_iter=10, lr=0.1, n_quantiles=5, ss_rate=0.5)
    model._model.fit_survival(X, time, event)
    
    # Print formatted output
    print("\n=== Survival Model Output ===")
    feature_names = {i: f"Feature_{i+1}" for i in range(p)}
    model.print(feature_names)
    
    # Verify predictions work (risk scores)
    risk_scores = model.predict(X)
    print(f"\nRisk scores shape: {risk_scores.shape}")
    
    if risk_scores.shape[0] == n:
        print("\n✓ Survival test passed!")
    else:
        print("\n✗ Survival test failed!")
        sys.exit(1)

if __name__ == "__main__":
    test_integration()
    test_survival()
