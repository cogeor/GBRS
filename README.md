# GBRS: Gradient Boosted Risk Scoring

Risk scores are an interpretable, explainable, and actionable class of machine learning models used in clinical settings, insurance, and risk management. Unlike most computational methods, risk scores are designed to be computed by a human by attributing points to a data sample based on a limited set of criteria.

This library provides an algorithm based on gradient boosting that is capable of modeling nonlinear effects, along with a C++ implementation with Python and R bindings.

## Installation

### Prerequisites
- **C++ Compiler**: GCC/Clang (Linux/macOS) or MSVC (Windows).
- **Python**: 3.7+
- **R**: 4.0+ (for R package)
- **Eigen**: Included as a git submodule.

### Python Package

1.  Clone the repository with submodules:
    ```bash
    git clone --recursive https://gitlab.com/cgeo/GBRS.git
    cd GBRS
    ```
    If you already cloned without `--recursive`, run:
    ```bash
    git submodule update --init --recursive
    ```

2.  Install using pip:
    ```bash
    pip install .
    ```

### R Package

1.  Install dependencies (`Rcpp`, `RcppEigen`):
    ```R
    install.packages(c("Rcpp", "RcppEigen"))
    ```

2.  Install from source:
    ```bash
    R CMD INSTALL .
    ```

## Usage

### Python

The Python API provides a scikit-learn style interface.

```python
import numpy as np
from gbrs import GBRS

# Generate synthetic data
X_train = np.random.rand(100, 5)
y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(float)

# Initialize and fit model
# n_iter: Number of boosting iterations
# lr: Learning rate
# n_quantiles: Number of bins for feature discretization
model = GBRS(n_iter=300, lr=0.05, n_quantiles=5)

# For regression (continuous target)
# model.fit(X_train, y_train)

# For binary classification (probability)
model.fit_proba(X_train, y_train)

# Predict
preds = model.predict_proba(X_train)

# Print the interpretable score table
# You can provide feature names for better readability
feature_names = {i: f"Feature_{i}" for i in range(X_train.shape[1])}
model.print(feature_names)
```

### R

The R API provides a formula-based interface and supports survival analysis.

```R
library(gbrs)

# Standard regression/classification
model <- gbrs(y ~ x1 + x2, data = df, objective = "binary", 
              n_max = 300, lr = 0.05, n_quantiles = 5)

# Survival analysis
# Requires 'time' and 'status' columns in the response
surv_model <- gbrs(Surv(time, status) ~ ., data = survival_df, objective = "survival",
                   n_max = 300, lr = 0.05, n_quantiles = 5)

# Predict
preds <- predict(model, test_df)

# Print score table
print(model)
```

## Features

- **Non-linear effects**: Captures complex relationships through gradient boosting.
- **Interpretability**: Produces a simple points-based score card.
- **Multi-objective**: Supports regression, binary classification, and survival analysis (R only).
- **Cross-platform**: Works on Linux, macOS, and Windows.

