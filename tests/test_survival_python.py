import numpy as np
from gbrs import GBRS
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import pandas as pd

def calculate_c_index(risk_scores, time, event):
    """Calculate concordance index for survival analysis."""
    concordant = 0
    discordant = 0
    pairs = 0
    
    n = len(time)
    # Sample pairs for efficiency
    sample_size = min(n, 500)
    np.random.seed(42)
    indices = np.random.choice(n, size=sample_size, replace=False)
    
    for i in range(len(indices)):
        idx_i = indices[i]
        for j in range(i + 1, len(indices)):
            idx_j = indices[j]
            
            # Only consider pairs where at least one event occurred
            if event[idx_i] == 1 or event[idx_j] == 1:
                if time[idx_i] < time[idx_j] and event[idx_i] == 1:
                    pairs += 1
                    if risk_scores[idx_i] > risk_scores[idx_j]:
                        concordant += 1
                    else:
                        discordant += 1
                elif time[idx_j] < time[idx_i] and event[idx_j] == 1:
                    pairs += 1
                    if risk_scores[idx_j] > risk_scores[idx_i]:
                        concordant += 1
                    else:
                        discordant += 1
    
    c_index = concordant / pairs if pairs > 0 else 0.5
    return c_index, concordant, discordant, pairs

def test_survival_breast_cancer():
    """Test GBRS survival analysis using breast cancer dataset."""
    print("=== Testing GBRS Survival Analysis (Python) ===\n")
    
    # Load breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    print(f"Dataset: Breast Cancer Wisconsin")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}\n")
    
    # Create synthetic survival data from classification target
    # Use target as event indicator and create time from feature variance
    np.random.seed(42)
    event = y  # 1 = malignant (event), 0 = benign (censored)
    
    # Generate survival times based on features
    # Higher mean radius (feature 0) -> shorter survival time
    base_time = 100
    time_modifier = -X[:, 0] / X[:, 0].max() * 50  # Scale by mean radius
    time = base_time + time_modifier + np.random.exponential(20, size=len(y))
    time = np.maximum(time, 1)  # Ensure positive times
    
    print(f"Survival data summary:")
    print(f"Time range: {time.min():.2f} - {time.max():.2f}")
    print(f"Event rate: {event.mean():.2%}\n")
    
    # Split data
    X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
        X, time, event, test_size=0.3, random_state=42
    )
    
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}\n")
    
    # Fit GBRS survival model with optimized hyperparameters
    print("Fitting GBRS survival model...")
    model = GBRS(n_iter=50, lr=0.05, n_quantiles=10)
    model.fit_survival(X_train, time_train, event_train)
    print("Model fitted successfully!\n")
    
    # Get predictions on test set
    risk_scores = model.predict(X_test)
    
    print("Prediction statistics (test set):")
    print(f"Min: {risk_scores.min():.4f}")
    print(f"Max: {risk_scores.max():.4f}")
    print(f"Mean: {risk_scores.mean():.4f}")
    print(f"Std: {risk_scores.std():.4f}\n")
    
    # Calculate C-index
    print("Calculating C-index...")
    c_index, concordant, discordant, pairs = calculate_c_index(risk_scores, time_test, event_test)
    
    print("\n=== Results ===")
    print(f"C-index: {c_index:.4f}")
    print(f"Concordant pairs: {concordant}")
    print(f"Discordant pairs: {discordant}")
    print(f"Total pairs: {pairs}\n")
    
    # Interpretation
    if c_index > 0.6:
        print("✓ Model shows good discrimination (C-index > 0.6)")
    elif c_index > 0.55:
        print("⚠ Model shows moderate discrimination (C-index > 0.55)")
    else:
        print("✗ Model shows poor discrimination (C-index ≤ 0.55)")
    
    # Compare with Cox model
    print("\n=== Comparison with Cox Model ===")
    
    # Prepare data for Cox model
    train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    train_df['time'] = time_train
    train_df['event'] = event_train
    
    test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    test_df['time'] = time_test
    test_df['event'] = event_test
    
    # Fit Cox model
    print("Fitting Cox proportional hazards model...")
    cox_model = CoxPHFitter()
    cox_model.fit(train_df, duration_col='time', event_col='event')
    
    # Get Cox predictions (risk scores)
    cox_risk_scores = cox_model.predict_partial_hazard(test_df)
    
    # Calculate C-index for Cox model using lifelines built-in function
    cox_c_index = concordance_index(time_test, -cox_risk_scores, event_test)
    
    print(f"Cox model C-index: {cox_c_index:.4f}")
    print(f"GBRS C-index: {c_index:.4f}")
    print(f"Difference: {c_index - cox_c_index:+.4f}")
    
    if c_index > cox_c_index:
        print("\n✓ GBRS outperforms Cox model!")
    elif c_index > cox_c_index - 0.05:
        print("\n≈ GBRS performance is comparable to Cox model")
    else:
        print("\n⚠ Cox model outperforms GBRS")
    
    # Assert performance is reasonable
    assert c_index > 0.55, f"C-index {c_index:.4f} is too low (expected > 0.55)"

if __name__ == "__main__":
    c_index = test_survival_breast_cancer()
    exit(0 if c_index > 0.55 else 1)
