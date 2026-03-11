
delete.intercept = function(mm) {
    saveattr = attributes(mm)
    intercept = which(saveattr$assign == 0)
    if (!length(intercept)) {
        return(mm)
    }
    mm = mm[,-intercept, drop=FALSE]
    saveattr$dim = dim(mm)
    saveattr$dimnames = dimnames(mm)
    saveattr$assign = saveattr$assign[-intercept]
    attributes(mm) = saveattr
    mm
}

process.formula <- function(formula, data) {
  frame <- model.frame(formula, data)
  response <- model.response(frame)

  # Handle design matrix
  mat <- model.matrix(terms(formula), data)
  mat <- mat[, !colnames(mat) %in% "(Intercept)", drop = FALSE]

  if (inherits(response, "Surv")) {
    # Survival response
    time <- response[, "time"]
    event <- response[, "status"]
    return(list(x = mat, time = time, event = event, type = "survival"))
  } else {
    # Regression or binary classification
    return(list(x = mat, y = response, type = "standard"))
  }
}


predict_score_proba = function(model, x) {
    yp = rep(model$cst[1], nrow(x))
    for (i in 1:nrow(model)) {
        yp = yp + ifelse(x[, model$idx[i] + 1] <= model$split_val[i], model$w[i], 0)
    }
    exp(yp) / (1 + exp(yp))
}

predict_score = function(model, x) {
    yp = rep(model$cst[1], nrow(x))
    for (i in 1:nrow(model)) {
        yp = yp + ifelse(x[, model$idx[i] + 1] <= model$split_val[i], 0, model$w[i])
    }
    yp
}

predict_score2 = function(model, x) {
    yp = rep(0, nrow(x))
    for (i in 1:nrow(model)) {
        yp = yp + ifelse(x[, model$idx[i] + 1] <= model$split_val[i], model$w1[i], model$w2[i])
    }
    yp
}


prune.weights = function(model_weights) {
    idx = c()
    split_val = c()
    w1 = c()
    w2 = c()
    w = c()
    cst = c()
    
    for (i in 1:nrow(model_weights)) {
        found = FALSE
        if(length(idx) > 0) {
            for (j in 1:length(idx)) {
                if ((model_weights$idx[i] == idx[j]) & (model_weights$split_val[i] == split_val[j])) {
                    w1[j] = w1[j] + model_weights$w1[i]
                    w2[j] = w2[j] + model_weights$w2[i] 
                    w[j] = w[j] + model_weights$w[i] 
                    found = TRUE
                }
            }
        }
        if(!found) {
            idx = c(idx, model_weights$idx[i])
            split_val = c(split_val, model_weights$split_val[i])
            w1 = c(w1, model_weights$w1[i])
            w2 = c(w2, model_weights$w2[i])
            w = c(w, model_weights$w[i])
            cst = c(cst, model_weights$cst[i])
        } 
    }
    df = data.frame(
        idx = idx,
        split_val = split_val,
        w1 = w1,
        w2 = w2,
        w = w,
        cst = cst
    )
    df
}


#' Fit a Gradient Boosted Rule Set Model
#'
#' @description
#' Fits a GBRS (Gradient Boosted Rule Set) model using gradient boosting to learn
#' interpretable rule sets for regression, binary classification, or survival analysis.
#' The model automatically selects split points and learns weights for each rule.
#'
#' @param formula A formula specifying the model. For regression and classification,
#'   use standard R formula syntax (e.g., \code{y ~ x1 + x2}). For survival analysis,
#'   use \code{Surv(time, event) ~ predictors} from the survival package.
#' @param data A data frame containing the variables specified in the formula.
#' @param n_max Integer. Maximum number of boosting iterations (default: 100).
#'   More iterations can improve fit but may lead to overfitting.
#' @param lr Numeric. Learning rate (shrinkage parameter) for gradient boosting
#'   (default: 0.1). Smaller values require more iterations but can improve generalization.
#' @param n_quantiles Integer. Number of quantile-based split point candidates to
#'   evaluate for each feature (default: 10). Higher values increase computation time.
#' @param batch_size Integer. Number of samples to use in each boosting iteration (default: 0).
#'   If 0 or greater than n_samples, the full dataset is used. Values > 0 enable stochastic gradient boosting.
#' @param objective Character. The objective function to optimize. One of:
#'   \itemize{
#'     \item \code{"auto"} (default): Automatically determined from response type
#'     \item \code{"continuous"}: L2 loss for regression
#'     \item \code{"binary"}: Log loss for binary classification
#'     \item \code{"survival"}: Ranking loss for survival analysis
#'   }
#' @param user_quantiles Optional named list of user-defined quantiles for specific
#'   features. If NULL (default), quantiles are computed automatically from the data.
#'
#' @return An object of class \code{"gbrs"} containing:
#'   \item{formula}{The model formula}
#'   \item{weights}{Data frame of learned rules with columns: idx (feature index),
#'     split_val (threshold), w (weight), w1, w2, cst (constant)}
#'   \item{objective}{The objective function used ("continuous", "binary", or "survival")}
#'
#' @examples
#' # Regression example
#' model <- gbrs(mpg ~ wt + hp, data = mtcars, n_max = 50, lr = 0.1)
#' print(model)
#' predictions <- predict(model, mtcars)
#'
#' # Binary classification
#' model_binary <- gbrs(am ~ mpg + wt + hp, data = mtcars,
#'                      objective = "binary", n_max = 100)
#' probs <- predict(model_binary, mtcars)
#'
#' # Survival analysis
#' library(survival)
#' model_surv <- gbrs(Surv(time, status) ~ age + sex + ph.ecog,
#'                    data = lung, objective = "survival")
#' risk_scores <- predict(model_surv, lung)
#'
#' @seealso \code{\link{predict.gbrs}}, \code{\link{print.gbrs}}
#' @export
gbrs <- function(formula, data, n_max = 100, lr = 0.1, n_quantiles = 10, batch_size = 0, objective = "auto", user_quantiles = NULL) {
  formula <- as.formula(formula)
  data <- process.formula(formula, data)

  # Determine objective if set to "auto"
  if (objective == "auto") {
    objective <- switch(data$type,
                        survival = "survival",
                        standard = "continuous")
  }

  # Map user_quantiles to design matrix columns if provided
  if (!is.null(user_quantiles) && is.list(user_quantiles)) {
    # If named list, map to columns
    if (!is.null(names(user_quantiles))) {
        col_names <- colnames(data$x)
        n_cols <- ncol(data$x)
        ordered_quantiles <- vector("list", n_cols)
        
        for (i in 1:n_cols) {
          col_name <- col_names[i]
          if (col_name %in% names(user_quantiles)) {
            ordered_quantiles[[i]] <- user_quantiles[[col_name]]
          }
        }
        user_quantiles <- ordered_quantiles
    } else {
        # Positional list: check length
        if (length(user_quantiles) != ncol(data$x)) {
            stop(paste0("Length of user_quantiles list (", length(user_quantiles), 
                        ") must match number of features (", ncol(data$x), ")"))
        }
    }
  }

  # Fit model based on type
  weights <- switch(objective,
    "continuous" = fit(data$x, data$y, n_max, lr, n_quantiles, batch_size),
    "binary"     = fit_proba(data$x, data$y, n_max, lr, n_quantiles, batch_size),
    "survival"   = fit_survival(data$x, data$time, data$event, n_max, lr, n_quantiles, batch_size, user_quantiles),
    stop("Unknown objective: must be 'continuous', 'binary', or 'survival'")
  )

  weights <- prune.weights(weights)
  obj <- list(formula = formula, weights = weights, objective = objective)
  class(obj) <- "gbrs"
  obj
}

#' Bootstrap GBRS Model
#'
#' @description
#' Fits a GBRS model multiple times with bootstrap samples to compute confidence
#' intervals for weights. Uses pre-computed thresholds from the full dataset to
#' ensure consistent split points across all bootstrap samples.
#'
#' @param formula A formula specifying the model.
#' @param data A data frame containing the variables in the formula.
#' @param n_bootstrap Integer. Number of bootstrap iterations (default: 10).
#' @param n_max Integer. Maximum boosting iterations per fit (default: 100).
#' @param lr Numeric. Learning rate (default: 0.1).
#' @param n_quantiles Integer. Number of quantile thresholds (default: 10).
#' @param batch_size Integer. Batch size for fitting (default: 0).
#' @param objective Character. Objective function ("auto", "continuous", "binary", "survival").
#' @param seed Integer. Random seed for reproducibility (default: NULL).
#'
#' @return An object of class \code{"gbrs_bootstrap"} containing:
#'   \item{thresholds}{List of pre-computed thresholds for each feature}
#'   \item{results}{List of model weights from each bootstrap iteration}
#'   \item{n_bootstrap}{Number of bootstrap samples}
#'   \item{objective}{The objective function used}
#'
#' @examples
#' result <- gbrs_bootstrap(mpg ~ wt + hp, data = mtcars, n_bootstrap = 10)
#' print(result)
#'
#' @export
gbrs_bootstrap <- function(formula, data, n_bootstrap = 10, n_max = 100,
                           lr = 0.1, n_quantiles = 10, batch_size = 0,
                           objective = "auto", seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  formula <- as.formula(formula)
  processed <- process.formula(formula, data)

  # Determine objective
  if (objective == "auto") {
    objective <- switch(processed$type,
                        survival = "survival",
                        standard = "continuous")
  }

  x <- processed$x
  n <- nrow(x)

  # Pre-compute thresholds from full data
  thresholds <- lapply(1:ncol(x), function(i) {
    probs <- seq(0, 1, length.out = n_quantiles + 2)[-c(1, n_quantiles + 2)]
    unique(quantile(x[, i], probs = probs))
  })

  # Run bootstrap iterations
  all_results <- list()

  for (b in 1:n_bootstrap) {
    # Sample with replacement
    idx <- sample(n, n, replace = TRUE)
    boot_data <- data[idx, , drop = FALSE]

    # Fit model with fixed thresholds
    weights <- tryCatch({
      switch(objective,
        "continuous" = {
          boot_processed <- process.formula(formula, boot_data)
          fit(boot_processed$x, boot_processed$y, n_max, lr, n_quantiles, batch_size)
        },
        "binary" = {
          boot_processed <- process.formula(formula, boot_data)
          fit_proba(boot_processed$x, boot_processed$y, n_max, lr, n_quantiles, batch_size)
        },
        "survival" = {
          boot_processed <- process.formula(formula, boot_data)
          fit_survival(boot_processed$x, boot_processed$time, boot_processed$event,
                      n_max, lr, n_quantiles, batch_size, thresholds)
        },
        stop("Unknown objective")
      )
    }, error = function(e) {
      warning(paste("Bootstrap iteration", b, "failed:", e$message))
      NULL
    })

    if (!is.null(weights)) {
      all_results[[b]] <- prune.weights(weights)
    }
  }

  # Filter out failed iterations
  all_results <- Filter(Negate(is.null), all_results)

  structure(
    list(
      thresholds = thresholds,
      results = all_results,
      n_bootstrap = length(all_results),
      objective = objective,
      formula = formula,
      feature_names = colnames(x)
    ),
    class = "gbrs_bootstrap"
  )
}

#' Print Bootstrap Results
#'
#' @description
#' Prints a summary of bootstrap results showing mean ± standard deviation
#' for each weight at each threshold.
#'
#' @param x An object of class \code{"gbrs_bootstrap"}.
#' @param prec Integer. Number of decimal places (default: 3).
#' @param ... Additional arguments (ignored).
#'
#' @export
print.gbrs_bootstrap <- function(x, prec = 3, ...) {
  cat("GBRS Bootstrap Results (", x$n_bootstrap, " samples)\n", sep = "")
  cat(strrep("=", 50), "\n\n")

  # Collect all y0 (base scores)
  y0_values <- sapply(x$results, function(r) r$cst[1])
  y0_mean <- mean(y0_values, na.rm = TRUE)
  y0_std <- sd(y0_values, na.rm = TRUE)

  cat("Base Score: ", round(y0_mean, prec), " +/- ", round(y0_std, prec), "\n", sep = "")
  cat(strrep("-", 50), "\n")

  # Collect all unique (idx, split_val) pairs
  all_keys <- list()
  for (result in x$results) {
    for (i in 1:nrow(result)) {
      key <- paste(result$idx[i], result$split_val[i], sep = "_")
      if (!key %in% names(all_keys)) {
        all_keys[[key]] <- list(idx = result$idx[i], split_val = result$split_val[i], weights = c())
      }
      all_keys[[key]]$weights <- c(all_keys[[key]]$weights, result$w[i])
    }
  }

  # Add zeros for missing keys in each result
  for (key in names(all_keys)) {
    n_present <- length(all_keys[[key]]$weights)
    n_missing <- x$n_bootstrap - n_present
    if (n_missing > 0) {
      all_keys[[key]]$weights <- c(all_keys[[key]]$weights, rep(0, n_missing))
    }
  }

  # Group by feature index and sort
  by_feature <- list()
  for (key in names(all_keys)) {
    info <- all_keys[[key]]
    idx <- info$idx + 1  # Convert to 1-based
    if (!as.character(idx) %in% names(by_feature)) {
      by_feature[[as.character(idx)]] <- list()
    }
    by_feature[[as.character(idx)]][[key]] <- info
  }

  # Print by feature
  for (idx_str in sort(as.numeric(names(by_feature)))) {
    idx <- as.integer(idx_str)
    feature_data <- by_feature[[as.character(idx)]]

    # Get feature name
    feat_name <- if (!is.null(x$feature_names) && idx <= length(x$feature_names)) {
      x$feature_names[idx]
    } else {
      paste0("F", idx - 1)
    }

    cat("\n", feat_name, ":\n", sep = "")

    # Sort by split_val
    split_vals <- sapply(feature_data, function(d) d$split_val)
    ordered <- order(split_vals)

    for (i in ordered) {
      info <- feature_data[[i]]
      m <- mean(info$weights)
      s <- sd(info$weights)
      sign <- if (m >= 0) "+" else ""
      cat("  > ", round(info$split_val, prec), ": ",
          sign, round(m, prec), " +/- ", round(s, prec), "\n", sep = "")
    }
  }

  cat("\n")
  invisible(x)
}

#' Get Bootstrap Weight Statistics
#'
#' @description
#' Extract mean and standard deviation for each weight from bootstrap results.
#'
#' @param x An object of class \code{"gbrs_bootstrap"}.
#'
#' @return A data frame with columns: feature, threshold, mean, std
#'
#' @export
summary.gbrs_bootstrap <- function(x, ...) {
  # Collect all unique (idx, split_val) pairs
  all_keys <- list()
  for (result in x$results) {
    for (i in 1:nrow(result)) {
      key <- paste(result$idx[i], result$split_val[i], sep = "_")
      if (!key %in% names(all_keys)) {
        all_keys[[key]] <- list(idx = result$idx[i], split_val = result$split_val[i], weights = c())
      }
      all_keys[[key]]$weights <- c(all_keys[[key]]$weights, result$w[i])
    }
  }

  # Add zeros for missing
  for (key in names(all_keys)) {
    n_present <- length(all_keys[[key]]$weights)
    n_missing <- x$n_bootstrap - n_present
    if (n_missing > 0) {
      all_keys[[key]]$weights <- c(all_keys[[key]]$weights, rep(0, n_missing))
    }
  }

  # Build data frame
  df <- data.frame(
    feature_idx = sapply(all_keys, function(k) k$idx),
    threshold = sapply(all_keys, function(k) k$split_val),
    mean = sapply(all_keys, function(k) mean(k$weights)),
    std = sapply(all_keys, function(k) sd(k$weights)),
    row.names = NULL
  )

  # Add feature names
  df$feature_name <- if (!is.null(x$feature_names)) {
    x$feature_names[df$feature_idx + 1]
  } else {
    paste0("F", df$feature_idx)
  }

  # Order by feature then threshold
  df[order(df$feature_idx, df$threshold), ]
}

#' Predict Method for GBRS Models
#'
#' @description
#' S3 predict method for GBRS models. Generates predictions on new data using
#' the learned rule set. The type of prediction depends on the objective used
#' during model fitting.
#'
#' @param obj An object of class \code{"gbrs"} returned by \code{\link{gbrs}}.
#' @param newdata A data frame containing the same predictor variables as used in
#'   model fitting. Must include all variables specified in the model formula.
#'
#' @return A numeric vector of predictions with length equal to \code{nrow(newdata)}:
#'   \itemize{
#'     \item For \code{objective = "continuous"}: Predicted continuous values
#'     \item For \code{objective = "binary"}: Predicted probabilities (0 to 1)
#'     \item For \code{objective = "survival"}: Predicted log-risk scores
#'       (higher values indicate higher risk)
#'   }
#'
#' @examples
#' # Fit model on training data
#' train_idx <- sample(1:nrow(mtcars), 0.7 * nrow(mtcars))
#' train_data <- mtcars[train_idx, ]
#' test_data <- mtcars[-train_idx, ]
#'
#' model <- gbrs(mpg ~ wt + hp, data = train_data)
#'
#' # Predict on test data
#' predictions <- predict(model, test_data)
#'
#' # Calculate RMSE
#' rmse <- sqrt(mean((predictions - test_data$mpg)^2))
#'
#' @seealso \code{\link{gbrs}}, \code{\link{print.gbrs}}
#' @export
predict.gbrs <- function(obj, newdata) {
  data <- process.formula(obj$formula, newdata)

  pred <- switch(obj$objective,
    "continuous" = predict_score(obj$weights, data$x),
    "binary"     = predict_score_proba(obj$weights, data$x),
    "survival"   = predict_score(obj$weights, data$x),  # Returns log-risk scores
    stop("Unknown objective: must be 'continuous', 'binary', or 'survival'")
  )

  return(pred)
}

