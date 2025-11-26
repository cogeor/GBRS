import numpy as np
from gbrs import GBRS

def test_print_formats():
    print("\n=== Testing GBRS Print Formats ===")
    
    # Generate synthetic data
    np.random.seed(42)
    n = 100
    X = np.random.rand(n, 2)
    y = (X[:, 0] > 0.5).astype(float) + np.random.normal(0, 0.1, n)
    
    model = GBRS(n_iter=20, lr=0.1, n_quantiles=5)
    model.fit(X, y)
    
    feature_names = {0: "Feature_A", 1: "Feature_B"}
    
    print("\n--- Default (Text) ---")
    model.print(feature_names=feature_names)
    
    print("\n--- LaTeX Horizontal ---")
    model.print(feature_names=feature_names, format="latex_h")
    
    print("\n--- Markdown Horizontal ---")
    model.print(feature_names=feature_names, format="md_h")
    
    print("\n--- ASCII Horizontal ---")
    model.print(feature_names=feature_names, format="ascii_h")

if __name__ == "__main__":
    test_print_formats()
