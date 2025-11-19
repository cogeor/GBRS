# Simple integration test for GBRS
library(gbrs)

# Generate dummy data
set.seed(42)
n <- 100
p <- 5
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
colnames(X) <- paste0("X", 1:p)
beta <- c(1, -1, 0.5, 0, 0)
logits <- X %*% beta
probs <- 1 / (1 + exp(-logits))
y <- rbinom(n, 1, probs)

# Create data frame
df <- as.data.frame(cbind(y = y, X))

# Fit model using formula interface
cat("Fitting model...\n")
model <- gbrs(y ~ X1 + X2 + X3 + X4 + X5, df = df, n_max = 10, lr = 0.1, n_quantiles = 10, ss_rate = 0.5, objective = "binary")

# Check output
if (!is.null(model)) {
  cat("Model fitted successfully.\n")
  cat("Model class:", class(model), "\n")
  cat("Model weights (first few rows):\n")
  print(head(model$weights))
} else {
  stop("Model fitting failed.")
}

cat("Integration test passed!\n")

