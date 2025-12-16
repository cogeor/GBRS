import numpy as np

from gbrs import GBRS


def test_user_quantiles_python():
    print("\n=== Testing GBRS User-Defined Quantiles (Python) ===")

    # Generate synthetic data
    np.random.seed(42)
    n = 200
    X = np.random.rand(n, 3)
    # Feature 0: [0, 1]
    # Feature 1: [0, 1]
    # Feature 2: [0, 1]

    # Target depends on X[0] > 0.5 and X[1] > 0.7
    y = (
        (X[:, 0] > 0.5).astype(float)
        + (X[:, 1] > 0.7).astype(float)
        + np.random.normal(0, 0.1, n)
    )

    # Define custom quantiles
    # Must be a list of numpy arrays, one for each feature
    # If a feature should use auto-quantiles, pass None (or handle in wrapper if implemented)
    # Here we define for all
    user_quantiles = [
        np.array([0.2, 0.5, 0.8]),  # Feature 0
        np.array([0.3, 0.7]),  # Feature 1
        np.array([0.5]),  # Feature 2
    ]

    print("Fitting model with user_quantiles...")
    model = GBRS(n_iter=50, lr=0.1, n_quantiles=10)
    model.fit(X, y, user_quantiles=user_quantiles)

    print("Model fitted successfully.")

    # Verify splits used

    idxs = model._model.get_idxs()
    split_vals = model._model.get_split_val()

    # Check Feature 0 splits
    f0_splits = split_vals[idxs == 0]
    print(f"Splits for Feature 0: {np.unique(f0_splits)}")

    # We expect splits to be from our user_quantiles (approximate due to float precision)
    # Note: The model might not use ALL provided quantiles if they don't improve the loss

    expected_f0 = [0.2, 0.5, 0.8]
    found_f0 = False
    for s in f0_splits:
        if any(np.isclose(s, expected_f0, atol=1e-5)):
            found_f0 = True
            break

    if found_f0:
        print("✓ Found expected custom split for Feature 0")
    else:
        print(
            "⚠ No custom splits for Feature 0 were selected (possible if not informative)"
        )

    # Check Feature 1 splits
    f1_splits = split_vals[idxs == 1]
    print(f"Splits for Feature 1: {np.unique(f1_splits)}")

    expected_f1 = [0.3, 0.7]
    found_f1 = False
    for s in f1_splits:
        if any(np.isclose(s, expected_f1, atol=1e-5)):
            found_f1 = True
            break

    if found_f1:
        print("✓ Found expected custom split for Feature 1")

    # Test print
    print("\n=== Model Print ===")
    model.print()


if __name__ == "__main__":
    test_user_quantiles_python()
