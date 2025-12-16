library(gbrs)
library(survival)

# Load a real survival dataset
data(veteran)

cat("=== Testing GBRS User-Defined Quantiles (R) ===\n\n")

# Define custom quantiles for selected features
# trt is binary (1, 2)
# age is continuous
custom_q <- list(
  trt = c(1.5),             # Split at 1.5
  age = c(40, 60, 80)       # Custom age bins
)

# Fit GBRS survival model with user_quantiles
cat("Fitting GBRS survival model with custom quantiles...\n")
model <- gbrs(
  Surv(time, status) ~ trt + age + celltype,
  data = veteran,
  n_max = 50,
  lr = 0.05,
  n_quantiles = 10,

  objective = "survival",
  user_quantiles = custom_q
)

cat("Model fitted successfully!\n\n")

# Verify that the model used the custom quantiles
# We check the split values in the model weights
weights <- model$weights
print(head(weights))

# Check trt splits
trt_idx <- which(attr(terms(model$formula), "term.labels") == "trt") - 1
trt_splits <- weights$split_val[weights$idx == trt_idx]
cat("\nSplits for 'trt' (expected ~1.5):", paste(sort(unique(trt_splits)), collapse=", "), "\n")

if (any(abs(trt_splits - 1.5) < 1e-5)) {
    cat("✓ Custom split for 'trt' found.\n")
} else {
    cat("✗ Custom split for 'trt' NOT found.\n")
}

# Check age splits
age_idx <- which(attr(terms(model$formula), "term.labels") == "age") - 1
age_splits <- weights$split_val[weights$idx == age_idx]
cat("Splits for 'age' (expected subset of 40, 60, 80):", paste(sort(unique(age_splits)), collapse=", "), "\n")

expected_age_splits <- c(40, 60, 80)
found_splits <- intersect(round(age_splits), expected_age_splits)
if (length(found_splits) > 0) {
    cat("✓ Found custom splits for 'age':", paste(found_splits, collapse=", "), "\n")
} else {
    cat("⚠ No custom splits for 'age' were selected by the model (this can happen if they are not informative).\n")
}

# Test standard print (horizontal default)
cat("\n=== Standard Print (Horizontal) ===\n")
print(model)

# Test legacy vertical print
cat("\n=== Legacy Vertical Print ===\n")
print.vertical(model)

# Test LaTeX print (vertical)
cat("\n=== LaTeX Print (Vertical) ===\n")
print(model, format="latex")

# Test Markdown print (vertical)
cat("\n=== Markdown Print (Vertical) ===\n")
print(model, format="md")

cat("\nTest completed.\n")
