"""
Test model import/export functionality
"""

import os
import numpy as np
import pytest
from gbrs import GBRS, save_predictions, load_predictions

DATA_DIR = os.path.join("tests", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def test_python_save_load():
    """Test saving and loading models in Python"""
    print("Testing Python model save/load...")

    # Create simple dataset
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1

    # Fit model
    model = GBRS(n_iter=10, lr=0.1, n_quantiles=5)
    model.fit(X, y)

    # Save model using method
    model.save_model(os.path.join(DATA_DIR, "test_model_python.json"))
    print("✓ Model saved to test_model_python.json")

    # Load model using class method
    loaded_model = GBRS.load_model(os.path.join(DATA_DIR, "test_model_python.json"))
    print("✓ Model loaded")

    # Verify loaded model predictions match original
    loaded_preds = loaded_model.predict(X)
    original_preds = model.predict(X)

    assert np.allclose(
        original_preds, loaded_preds
    ), "Loaded model predictions don't match original!"
    print("✓ Loaded model predictions match original!")

    # Make predictions and save them
    predictions = model.predict(X)
    save_predictions(predictions, os.path.join(DATA_DIR, "predictions_python.json"))
    print(f"✓ Predictions saved: {len(predictions)} values")

    # Load predictions
    loaded_preds_json = load_predictions(
        os.path.join(DATA_DIR, "predictions_python.json")
    )
    print(f"✓ Predictions loaded: {len(loaded_preds_json)} values")

    # Verify predictions match
    assert np.allclose(predictions, loaded_preds_json), "Predictions don't match!"
    print("✓ Predictions match!")

    print("\n✅ All Python save/load tests passed!\n")


if __name__ == "__main__":
    test_python_save_load()
