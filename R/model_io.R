# Model serialization functions for GBRS models
#
# This file provides functions to save and load GBRS models in JSON format,
# enabling cross-language compatibility between Python and R implementations.

#' Save a GBRS Model to JSON File
#'
#' @param model A gbrs model object
#' @param filepath Path to save the model file
#' @export
save_gbrs <- function(model, filepath) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
  
  # Extract model components
  weights <- model$weights
  
  # Build rules list
  rules <- list()
  for (i in 1:nrow(weights)) {
    rules[[i]] <- list(
      idx = as.integer(weights$idx[i]),
      split_val = as.numeric(weights$split_val[i]),
      w = as.numeric(weights$w[i]),
      cst = as.numeric(weights$cst[i])
    )
  }
  
  # Create model structure
  model_data <- list(
    version = "1.0",
    objective = model$objective,
    formula = deparse(model$formula),
    rules = rules
  )
  
  # Write to JSON file
  jsonlite::write_json(model_data, filepath, pretty = TRUE, auto_unbox = TRUE)
}

#' Load a GBRS Model from JSON File
#'
#' @param filepath Path to the model file
#' @return A gbrs model object
#' @export
load_gbrs <- function(filepath) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
  
  # Read JSON file
  model_data <- jsonlite::read_json(filepath)
  
  # Validate version
  if (model_data$version != "1.0") {
    stop(paste("Unsupported model version:", model_data$version))
  }
  
  # Reconstruct weights data frame
  n_rules <- length(model_data$rules)
  weights <- data.frame(
    idx = integer(n_rules),
    split_val = numeric(n_rules),
    w = numeric(n_rules),
    w1 = numeric(n_rules),
    w2 = numeric(n_rules),
    cst = numeric(n_rules)
  )
  
  for (i in 1:n_rules) {
    rule <- model_data$rules[[i]]
    weights$idx[i] <- rule$idx
    weights$split_val[i] <- rule$split_val
    weights$w[i] <- rule$w
    weights$cst[i] <- rule$cst
    # Note: w1 and w2 are not stored in JSON, set to 0
    weights$w1[i] <- 0
    weights$w2[i] <- 0
  }
  
  # Reconstruct gbrs object
  obj <- list(
    formula = as.formula(model_data$formula),
    weights = weights,
    objective = model_data$objective
  )
  class(obj) <- "gbrs"
  
  return(obj)
}

#' Save Predictions to JSON File
#'
#' @param predictions Numeric vector of predictions
#' @param filepath Path to save predictions
#' @export
save_predictions <- function(predictions, filepath) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
  
  pred_data <- list(predictions = as.numeric(predictions))
  jsonlite::write_json(pred_data, filepath, pretty = TRUE, auto_unbox = FALSE)
}

#' Load Predictions from JSON File
#'
#' @param filepath Path to predictions file
#' @return Numeric vector of predictions
#' @export
load_predictions <- function(filepath) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
  
  pred_data <- jsonlite::read_json(filepath)
  return(as.numeric(pred_data$predictions))
}
