import risk_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np

# Load and prepare the dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# Convert to numpy arrays with correct dtype
X_train = np.asarray(X_train, dtype=np.float64)
y_train = np.asarray(y_train, dtype=np.float64)
X_test = np.asarray(X_test, dtype=np.float64)

model = risk_score.Model(
    n_iter=100,
    lr=0.005,
    n_quantiles=10,
    ss_rate=1
)

model.fit_proba(X_train, y_train)
probs = model.predict_proba(X_test)


probs = probs.clip(1e-6, 1 - 1e-6)  # avoid log(0)
print(probs)
# Evaluate
auc = roc_auc_score(y_test, probs)
print(f"AUC: {auc:.4f}")
