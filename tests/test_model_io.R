# Test model import/export functionality for R

source("R/model_io.R")
source("R/utils.R")

test_r_save_load <- function() {
  cat("Testing R model save/load...\n")
  
  # Use mtcars dataset
  set.seed(42)
  
  # Fit model
  model <- gbrs(mpg ~ wt + hp, data = mtcars, n_max = 10, lr = 0.1)
  cat("✓ Model fitted\n")
  
  # Save model
  save_gbrs(model, "test_model_r.json")
  cat("✓ Model saved to test_model_r.json\n")
  
  # Load model
  loaded_model <- load_gbrs("test_model_r.json")
  cat(sprintf("✓ Model loaded: %d rules\n", nrow(loaded_model$weights)))
  
  # Make predictions and save them
  predictions <- predict(model, mtcars)
  save_predictions(predictions, "predictions_r.json")
  cat(sprintf("✓ Predictions saved: %d values\n", length(predictions)))
  
  # Load predictions
  loaded_preds <- load_predictions("predictions_r.json")
  cat(sprintf("✓ Predictions loaded: %d values\n", length(loaded_preds)))
  
  # Verify predictions match
  if (all.equal(predictions, loaded_preds)) {
    cat("✓ Predictions match!\n")
  } else {
    stop("Predictions don't match!")
  }
  
  cat("\n✅ All R save/load tests passed!\n\n")
}

# Run test
test_r_save_load()
