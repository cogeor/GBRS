# GBRS: Gradient Boosted Risk Scoring

[![pipeline status](https://gitlab.com/cgeo/GBRS/badges/dev/pipeline.svg)](https://gitlab.com/cgeo/GBRS/-/commits/dev)
![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![R](https://img.shields.io/badge/R-4.0%2B-blue)

Risk scores are an interpretable, explainable, and actionable class of machine learning models used in clinical settings, insurance, and risk management. Unlike most computational methods, risk scores are designed to be computed by a human by attributing points to a data sample based on a limited set of criteria.

This library provides an algorithm based on gradient boosting for generating risk scores, along with a C++ implementation with Python and R bindings.


## Features

- **Interpretability**: Produces a simple points-based score card (see below for example), optionally with user-defined thresholds.
- **Multi-objective**: Supports regression, binary classification, and survival analysis.
- **Non-linear effects**: Captures complex relationships through gradient boosting.
- **Cross-platform**: Works on Linux, macOS, and Windows.


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

# Optional: Define custom quantiles for specific features
# user_quantiles = [np.array([0.2, 0.8]), None, ...] 
user_quantiles = None

# Initialize and fit model
model = GBRS(n_iter=300, lr=0.05, n_quantiles=5)

# Fit (supports fit, fit_proba, fit_survival)
model.fit_proba(X_train, y_train, user_quantiles=user_quantiles)

# Predict
preds = model.predict_proba(X_train)

# Print the interpretable score table (defaults to horizontal format)
feature_names = {i: f"Feature_{i}" for i in range(X_train.shape[1])}
model.print(feature_names)
```

### R

The R API provides a formula-based interface

```R
library(gbrs)
library(survival)

# Optional: Define custom thresholds
custom_q <- list(age = c(50, 70))

# Fit model (supports continuous, binary, and survival objectives)
model <- gbrs(Surv(time, status) ~ trt + age + celltype, 
              data = veteran, 
              objective = "survival",
              n_max = 300, lr = 0.05, n_quantiles = 5,
              user_quantiles = custom_q)

# Print score table (defaults to horizontal format)
print(model)
```

## Example Output

GBRS models are printed in a horizontal format where thresholds are displayed above the scores, making them easy to read and include in documents.

### Markdown Horizontal (`format="md_h"`)

| Variable |  |  |  |
|:---|:---|:---|:---|
| **trt** | <1.0 | >=1.0 | |
| | 0.0 | 0.3 | |
| **celltype** | FALSE | TRUE | |
| | 0.0 | 0.8 | |
| **karno** | FALSE | TRUE | |
| | 0.0 | 0.7 | |
| **diagtime** | FALSE | TRUE | |
| | 0.0 | -0.4 | |
| **age** | <50.0 | [50.0,70.0) | >=70.0 |
| | 3.0 | 2.0 | 0.0 |
| **prior** | <5.0 | >=5.0 | |
| | 1.0 | 1.5 | |
