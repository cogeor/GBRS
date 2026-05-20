# GBRS: Gradient Boosted Risk Scoring

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![R](https://img.shields.io/badge/R-4.0%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![arXiv](https://img.shields.io/badge/arXiv-2605.02593-b31b1b.svg)](https://arxiv.org/abs/2605.02593)

Risk scores are an interpretable, explainable, and actionable class of machine learning models used in clinical settings, insurance, and risk management. Unlike most computational methods, risk scores are designed to be computed by a human by attributing points to a data sample based on a limited set of criteria.

This library provides an algorithm based on gradient boosting for generating risk scores, along with a C++ implementation with Python and R bindings. The method is described in [Georgantas and Richiardi (2026)](https://arxiv.org/abs/2605.02593), where it is benchmarked against AutoScore across twelve tabular datasets spanning regression, classification, and time-to-event tasks — producing 60% fewer rules for classification and 16% fewer for time-to-event problems, on average, at competitive predictive performance.


## Features

- **Interpretability**: Produces a simple points-based score card (see below for example), optionally with user-defined thresholds.
- **Multi-objective**: Supports regression, binary classification, and survival analysis.
- **Non-linear effects**: Captures complex relationships through gradient boosting.
- **Bootstrap**: Compute confidence intervals on weights via bootstrapping.
- **Cross-platform**: Works on Linux, macOS, and Windows.


## Installation

### Prerequisites
- **C++ Compiler**: GCC/Clang (Linux/macOS) or MSVC (Windows).
- **Python**: 3.7+
- **R**: 4.0+ (for R package)
- **Eigen**: Included as a git submodule.

### Python Package

Installs as `pygbrs` on PyPI; the import name is still `gbrs`:

```bash
pip install pygbrs
```

```python
import gbrs
```

For v0.1.0 only an sdist is published — `pip` will compile the C++ core
on your machine using your local toolchain. That keeps the first release
narrow in scope; binary wheels via `cibuildwheel` are tracked for a later
version. Build prerequisites: a C++17 compiler and Python headers
(`python-dev` / `python3-devel` on Linux).

To build from a clone instead:

```bash
git clone --recursive https://github.com/cogeor/GBRS.git
cd GBRS
pip install .
```

If you cloned without `--recursive`, run `git submodule update --init --recursive` first.

### R Package

1.  Install dependencies (`Rcpp`, `RcppEigen`):
    ```R
    install.packages(c("Rcpp", "RcppEigen"))
    ```

2.  Install from source (Rtools required on Windows):
    ```bash
    R CMD INSTALL .
    ```

    Once accepted on CRAN:
    ```R
    install.packages("gbrs")
    ```

#### Maximum performance

The CRAN-shipped binary is built with portable flags (`-O2`, baseline
ISA). For maximum throughput on your own machine, install from source
after adding the following to `~/.R/Makevars` (create the file if it
does not exist):

```
CXX17FLAGS = -O3 -march=native
```

Then `install.packages("gbrs", type = "source")` will compile with
CPU-specific vectorisation. Build on a representative host —
`-march=native` pins the binary to the build machine's
microarchitecture, so the resulting `.so` may crash with
`SIGILL` on older CPUs.

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

### Bootstrap

GBRS supports bootstrapping to compute confidence intervals on model weights. Thresholds are pre-computed from the full dataset so that weights are comparable across bootstrap samples.

#### Python

```python
model = GBRS(n_iter=300, lr=0.05, n_quantiles=5)
result = model.bootstrap(X_train, y_train, n_bootstrap=100, random_state=42)

# Print mean ± std for each weight
feature_names = {i: f"Feature_{i}" for i in range(X_train.shape[1])}
result.print_summary(feature_names)

# Programmatic access
stats = result.get_weight_stats()               # mean, std per weight
cis = result.get_confidence_intervals(alpha=0.05)  # 95% CIs
```

`bootstrap_proba` and `bootstrap_survival` are also available for classification and survival objectives.

#### R

```R
result <- gbrs_bootstrap(Surv(time, status) ~ trt + age + celltype,
                         data = veteran, objective = "survival",
                         n_bootstrap = 100, seed = 42)
print(result)
summary(result)
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

## Citation

If you use GBRS in academic work, please cite:

> Georgantas, C., & Richiardi, J. (2026). *Gradient Boosted Risk Scores.* arXiv:2605.02593. https://arxiv.org/abs/2605.02593

BibTeX:

```bibtex
@article{georgantas2026gbrs,
  title  = {Gradient Boosted Risk Scores},
  author = {Georgantas, Costa and Richiardi, Jonas},
  year   = {2026},
  eprint = {2605.02593},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi    = {10.48550/arXiv.2605.02593},
  url    = {https://arxiv.org/abs/2605.02593}
}
```

In R, `citation("gbrs")` returns the same reference programmatically.
