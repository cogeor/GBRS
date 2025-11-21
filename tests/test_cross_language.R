# Cross-language model compatibility test (R part)
#
# This script loads test data and model from Python,
# generates predictions in R, and saves them for comparison.

library(gbrs)
library(jsonlite)

test_cross_language_r_part <- function() {
  cat("=" , rep("=", 59), "\n", sep="")
  cat("Cross-Language Prediction Comparison Test (R Part)\n")
  cat("=", rep("=", 59), "\n", sep="")
  
  # Load test data from Python
  cat("\n1. Loading test data from Python...\n")
  test_data <- read_json("test_data.json", simplifyVector = TRUE)
  X <- test_data$X
  y <- test_data$y
  cat(sprintf("   Dataset: %d samples, %d features\n", nrow(X), ncol(X)))
  
  # Fit model in R with same data
  cat("\n2. Fitting model in R...\n")
  df <- data.frame(y = y, x1 = X[,1], x2 = X[,2])
  print(head(df))
  model_r <- gbrs(y ~ x1 + x2, data = df, n_max = 10, lr = 0.1, n_quantiles = 5)
  cat("   ✓ Model fitted\n")
  
  # Make predictions
  cat("\n3. Generating R predictions...\n")
  predictions_r <- predict(model_r, df)
  save_predictions(predictions_r, "predictions_r_test.json")
  cat(sprintf("   ✓ Predictions saved: %d values\n", length(predictions_r)))
  cat(sprintf("   Mean prediction: %.4f\n", mean(predictions_r)))
  cat(sprintf("   Std prediction: %.4f\n", sd(predictions_r)))
  
  # Load Python predictions for comparison
  cat("\n4. Loading Python predictions for comparison...\n")
  predictions_py <- load_predictions("predictions_python_test.json")
  
  # Compare predictions
  cat("\n5. Comparing predictions...\n")
  diff <- predictions_r - predictions_py
  max_diff <- max(abs(diff))
  cat(sprintf("   Max absolute difference: %.10f\n", max_diff))
  
  tolerance <- 1e-6
  if (max_diff < tolerance) {
    cat(sprintf("   ✓ Predictions match within tolerance (%.0e)!\n", tolerance))
  } else {
    cat(sprintf("   ✗ Predictions differ by more than tolerance (%.0e)\n", tolerance))
    cat("   First 5 differences:\n")
    print(head(diff, 5))
  }
  
  cat("\n", rep("=", 60), "\n", sep="")
  cat("R part complete!\n")
  cat(rep("=", 60), "\n", sep="")
}

# Run test
test_cross_language_r_part()
