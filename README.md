# Abstract
Risk scores are an interpretable, explainable, and actionable class of machine learning models used in clinical settings, insurance, and risk management. Unlike most computational methods, risk scores are designed to be computed by a human by attributing points to a data sample based on a limited set of criteria. The most common approaches for generating risk scores use linear regressions to estimate the effect of selected variables. In this work, we take a principled approach towards building robust risk scores. We provide an algorithm based on gradient boosting that is capable of modeling nonlinear effects, along with a C++ implementation with Python and R bindings. We show that our method consistently performs well compared to other prediction models through extensive empirical evaluation on multiple tabular datasets in regression, classification tasks, and time-to-event tasks. Finally, we show that our approach yields scores that are significantly more compact than those of regression-based alternatives.

# Installation
Python: pip install .
R: R CMD INSTALL .

# Python Usage Example
from gbrs import GBRS

gbrs_model = GBRS(n_iter=500, lr=0.05, n_quantiles=4)
gbrs_model.fit(X_train, y_train) # fit_proba for binary classification
preds_custom = gbrs_model.predict(X_test)
gbrs_model.print()

# R Usage Example
library(gbrs)
gbrs_model <- gbrs(formula, train_val_set, objective = "survival", 
                  n_max = 500, lr = 0.05, n_quantiles = 5)
pred_score <- predict(gbrs_model, test_set)
print(gbrs_model)
