
# Install/Load required packages if not already available
if (!requireNamespace("mgcv", quietly = TRUE)) install.packages("mgcv")
if (!requireNamespace("randomForest", quietly = TRUE)) install.packages("randomForest")
devtools::load_all(".")

library(mgcv)
library(randomForest)

# 1. Load Wine Data
# Columns based on wine.names
col_names <- c("Class", "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium", 
               "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins", 
               "Color_intensity", "Hue", "OD280_OD315_of_diluted_wines", "Proline")

wine_data <- read.csv("wine/wine.data", header = FALSE, col.names = col_names)

# Target: Class 1 vs Rest (Binary Classification)
target <- "is_class_1"
wine_data$is_class_1 <- as.numeric(wine_data$Class == 1)
model_data <- wine_data[, !names(wine_data) %in% c("Class")]

formula_str <- paste(target, "~ .")
formula <- as.formula(formula_str)

# Split data
set.seed(42)
train_idx <- sample(1:nrow(model_data), 0.7 * nrow(model_data))
train_data <- model_data[train_idx, ]
test_data <- model_data[-train_idx, ]

cat("Dataset: Wine (Predicting Class 1 vs Rest)\n")
cat("Train size:", nrow(train_data), "Test size:", nrow(test_data), "\n\n")

# --- Benchmarks ---

# 1. Linear Model (LM)
start_time <- Sys.time()
lm_model <- lm(formula, data = train_data)
lm_time <- Sys.time() - start_time
lm_pred <- predict(lm_model, test_data)
lm_mse <- mean((test_data[[target]] - lm_pred)^2)
cat(sprintf("LM - Time: %.4f s, MSE: %.4f\n", as.numeric(lm_time, units="secs"), lm_mse))

# 2. GAM
start_time <- Sys.time()
gam_model <- gam(formula, data = train_data) 
gam_time <- Sys.time() - start_time
gam_pred <- predict(gam_model, test_data)
gam_mse <- mean((test_data[[target]] - gam_pred)^2)
cat(sprintf("GAM - Time: %.4f s, MSE: %.4f\n", as.numeric(gam_time, units="secs"), gam_mse))

# 3. Random Forest
start_time <- Sys.time()
rf_model <- randomForest(formula, data = train_data)
rf_time <- Sys.time() - start_time
rf_pred <- predict(rf_model, test_data)
rf_mse <- mean((test_data[[target]] - rf_pred)^2)
cat(sprintf("RF  - Time: %.4f s, MSE: %.4f\n", as.numeric(rf_time, units="secs"), rf_mse))

# 4. GBRS
# Test a couple of settings to ensure convergence
settings <- list(
  list(lr = 0.01, n_iter = 100),
  list(lr = 0.01, n_iter = 500),
  list(lr = 0.05, n_iter = 100),
  list(lr = 0.05, n_iter = 500)
)

best_gbrs_model <- NULL
best_gbrs_mse <- Inf

cat("\n--- GBRS Tuning ---\n")
for (s in settings) {
  start_time <- Sys.time()
  gbrs_model <- gbrs(formula, data = train_data, lr = s$lr, n_iter = s$n_iter)
  gbrs_time <- Sys.time() - start_time
  gbrs_pred <- predict(gbrs_model, test_data)
  gbrs_mse <- mean((test_data[[target]] - gbrs_pred)^2)
  
  cat(sprintf("GBRS (lr=%.2f, n_iter=%d) - Time: %.4f s, MSE: %.4f\n", 
              s$lr, s$n_iter, as.numeric(gbrs_time, units="secs"), gbrs_mse))
  
  if (gbrs_mse < best_gbrs_mse) {
    best_gbrs_mse <- gbrs_mse
    best_gbrs_model <- gbrs_model
  }
}

cat("\nBest GBRS MSE:", best_gbrs_mse, "\n")

# Print scores to file
output_file <- "wine_scores.md"
cat("Printing best GBRS model scores to", output_file, "...\n")
sink(output_file)
cat("## GBRS Model Scores (Wine Dataset)\n\n")
cat("### Vertical Format\n\n")
print(best_gbrs_model, format="md")
cat("\n\n### Horizontal Format\n\n")
print(best_gbrs_model, format="md_h")
sink()
cat("Done.\n")
