# Test bootstrapping functionality for GBRS R package

library(gbrs)

test_bootstrap_basic <- function() {
  cat("\n=== Testing GBRS Bootstrapping (R) ===\n\n")
  
  # Test 1: Basic regression bootstrap
  cat("1. Testing basic regression bootstrap...\n")
  set.seed(42)
  n <- 100
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  y <- 2 * (x1 > 0) + 1.5 * (x2 > 0.5) + rnorm(n, sd = 0.3)
  df <- data.frame(y = y, x1 = x1, x2 = x2)
  
  result <- gbrs_bootstrap(y ~ x1 + x2, data = df, n_bootstrap = 5, 
                           n_max = 30, lr = 0.1, n_quantiles = 5, seed = 42)
  
  stopifnot(inherits(result, "gbrs_bootstrap"))
  stopifnot(result$n_bootstrap == 5)
  stopifnot(length(result$results) == 5)
  cat("   [OK] Basic regression bootstrap works\n")
  
  # Test 2: Print method
  cat("\n2. Testing print method...\n")
  print(result)
  cat("   [OK] Print method works\n")
  
  # Test 3: Summary method
  cat("\n3. Testing summary method...\n")
  stats <- summary(result)
  stopifnot(is.data.frame(stats))
  stopifnot("mean" %in% colnames(stats))
  stopifnot("std" %in% colnames(stats))
  cat("   [OK] Summary method works\n")
  print(stats)
  
  # Test 4: Reproducibility with seed
  cat("\n4. Testing reproducibility with seed...\n")
  result1 <- gbrs_bootstrap(y ~ x1 + x2, data = df, n_bootstrap = 3, 
                            n_max = 20, seed = 123)
  result2 <- gbrs_bootstrap(y ~ x1 + x2, data = df, n_bootstrap = 3, 
                            n_max = 20, seed = 123)
  
  # Check that results are identical
  y0_1 <- sapply(result1$results, function(r) r$cst[1])
  y0_2 <- sapply(result2$results, function(r) r$cst[1])
  stopifnot(all.equal(y0_1, y0_2))
  cat("   [OK] Results are reproducible with same seed\n")
  
  # Test 5: Different seeds give different results
  cat("\n5. Testing different seeds give different results...\n")
  result3 <- gbrs_bootstrap(y ~ x1 + x2, data = df, n_bootstrap = 3, 
                            n_max = 20, seed = 456)
  y0_3 <- sapply(result3$results, function(r) r$cst[1])
  stopifnot(!all.equal(y0_1, y0_3) || !is.logical(all.equal(y0_1, y0_3)))
  cat("   [OK] Different seeds produce different results\n")
  
  cat("\n=== All R bootstrap tests passed! ===\n\n")
}

# Test with mtcars dataset
test_bootstrap_mtcars <- function() {
  cat("\n=== Testing Bootstrap with mtcars ===\n\n")
  
  result <- gbrs_bootstrap(mpg ~ wt + hp + cyl, data = mtcars, 
                           n_bootstrap = 10, n_max = 50, seed = 42)
  
  cat("Bootstrap results for mtcars:\n")
  print(result)
  
  cat("\nSummary statistics:\n")
  print(summary(result))
  
  cat("\n=== mtcars test passed! ===\n")
}

# Run tests
if (interactive() || !exists("skip_tests")) {
  test_bootstrap_basic()
  test_bootstrap_mtcars()
}
