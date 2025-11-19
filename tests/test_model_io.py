"""
Test model import/export functionality
"""
import numpy as np
from gbrs import GBRS, save_model, load_model, save_predictions, load_predictions

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
    
    # Save model
    save_model(model._model, 'test_model_python.json', objective='continuous', formula='y ~ x1 + x2 + x3')
    print("✓ Model saved to test_model_python.json")
    
    # Load model
    loaded_data = load_model('test_model_python.json')
    print(f"✓ Model loaded: {len(loaded_data['rules'])} rules")
    
    # Make predictions and save them
    predictions = model.predict(X)
    save_predictions(predictions, 'predictions_python.json')
    print(f"✓ Predictions saved: {len(predictions)} values")
    
    # Load predictions
    loaded_preds = load_predictions('predictions_python.json')
    print(f"✓ Predictions loaded: {len(loaded_preds)} values")
    
    # Verify predictions match
    assert np.allclose(predictions, loaded_preds), "Predictions don't match!"
    print("✓ Predictions match!")
    
    print("\n✅ All Python save/load tests passed!\n")

if __name__ == "__main__":
    test_python_save_load()
