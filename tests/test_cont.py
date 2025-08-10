from gbrs import GBRS
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from benchmarks.datasets import load_dataset

# Load dataset
#X, y = load_diabetes(return_X_y=True)
X, y = load_dataset("housing")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Convert to float64 for Eigen/pybind compatibility
X_train = np.asarray(X_train, dtype=np.float64)
y_train = np.asarray(y_train, dtype=np.float64)
X_test = np.asarray(X_test, dtype=np.float64)
y_test = np.asarray(y_test, dtype=np.float64)


gbrs_model = GBRS(n_iter=500, lr=0.05, n_quantiles=4)
gbrs_model.fit(X_train, y_train)
gbrs_model.print()
preds_custom = gbrs_model.predict(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
preds_linear = linear_model.predict(X_test)
linear_model.get_params()

mse_linear = mean_squared_error(y_test, preds_linear)
mse_custom = mean_squared_error(y_test, preds_custom)

# --- Comparison ---
print(f"Custom Model MSE: {mse_custom:.4f}")
print(f"Linear Regression MSE: {mse_linear:.4f}")