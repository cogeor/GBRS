library(gbrs)
library(survival)

# Load a real survival dataset
data(veteran)

cat("=== Testing GBRS Survival Analysis ===\n\n")
cat("Dataset: veteran (lung cancer survival)\n")
cat("Samples:", nrow(veteran), "\n\n")

# Fit GBRS survival model using formula interface
cat("Fitting GBRS survival model...\n")
model <- gbrs(
  Surv(time, status) ~ trt + celltype + karno + diagtime + age + prior,
  data = veteran,
  n_max = 50,
  lr = 0.05,
  n_quantiles = 10,

  objective = "survival"
)

cat("Model fitted successfully!\n\n")

# Get predictions
scores <- predict(model, veteran)

cat("Prediction statistics:\n")
cat("Min:", min(scores), "\n")
cat("Max:", max(scores), "\n")
cat("Mean:", mean(scores), "\n")
cat("Std:", sd(scores), "\n\n")

# Calculate C-index manually
cat("Calculating C-index...\n")
concordant <- 0
discordant <- 0
pairs <- 0

n <- nrow(veteran)
time <- veteran$time
status <- veteran$status

# Sample pairs for efficiency
sample_size <- min(n, 500)
set.seed(42)
indices <- sample(1:n, sample_size)

for (i in 1:(length(indices)-1)) {
  idx_i <- indices[i]
  for (j in (i+1):length(indices)) {
    idx_j <- indices[j]
    
    # Only consider pairs where at least one event occurred
    if (status[idx_i] == 1 || status[idx_j] == 1) {
      if (time[idx_i] < time[idx_j] && status[idx_i] == 1) {
        pairs <- pairs + 1
        if (scores[idx_i] > scores[idx_j]) {
          concordant <- concordant + 1
        } else {
          discordant <- discordant + 1
        }
      } else if (time[idx_j] < time[idx_i] && status[idx_j] == 1) {
        pairs <- pairs + 1
        if (scores[idx_j] > scores[idx_i]) {
          concordant <- concordant + 1
        } else {
          discordant <- discordant + 1
        }
      }
    }
  }
}

c_index <- concordant / pairs
cat("\n=== Results ===\n")
cat("C-index:", c_index, "\n")
cat("Concordant pairs:", concordant, "\n")
cat("Discordant pairs:", discordant, "\n")
cat("Total pairs:", pairs, "\n\n")

# Interpretation
if (c_index > 0.6) {
  cat("✓ Model shows good discrimination (C-index > 0.6)\n")
} else if (c_index > 0.55) {
  cat("⚠ Model shows moderate discrimination (C-index > 0.55)\n")
} else {
  cat("✗ Model shows poor discrimination (C-index ≤ 0.55)\n")
}

# Compare with Cox model
cat("\n=== Comparison with Cox Model ===\n")
cox_model <- coxph(Surv(time, status) ~ ., data = veteran)
cat("Cox model C-index:", summary(cox_model)$concordance[1], "\n")
