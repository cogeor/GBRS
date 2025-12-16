
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

cross.entropy <- function(p, phat){
  x <- 0
  for (i in 1:length(p)){
    x <- x + (p[i] * log(phat[i]))
  }
  return(-x)
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

prune.weights.old = function(w) {
    idxs = c()
    vals = c()
    w1 = c()
    w2 = c()

    for (i in 1:ncol(w)) {
        found = FALSE
        if(length(idxs) > 0) {
            for (j in 1:length(idxs)) {
                if ((w[1, i] == idxs[j]) & (w[2, i] == vals[j])) {
                    w1[j] = w1[j] + w[3, i] 
                    w2[j] = w2[j] + w[4, i] 
                    found = TRUE
                }
            }
        }
        if(!found) {
            idxs = c(idxs, w[1, i])
            vals = c(vals, w[2, i])
            w1 = c(w1, w[3, i])
            w2 = c(w2, w[4, i])
        } 
    }
    n_unique = length(idxs)
    new_w = matrix(ncol=n_unique, nrow=4)
    new_w[1,] = idxs
    new_w[2,] = vals
    new_w[3,] = w1 
    new_w[4,] = w2 
    new_w
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

convert.weights.score = function(w) {
    cst = 0
    score_mat = matrix(ncol=ncol(w), nrow=3)
    for (i in 1:ncol(w)) {
        cst = cst + w[4, i]
        score_mat[1, i] = w[1, i]
        score_mat[2, i] = w[2, i]
        score_mat[3, i] = w[3, i] - w[4, i]
    }
    list("score"=score_mat, "cst"=cst)
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

predict_round = function(coeffs, formula, data) {
    formula = as.formula(formula)
    frame = model.frame(terms(formula), data)
    resp =  model.extract(frame, "response")
    mat = model.matrix(terms(formula), data)
    r_coeffs = round(coeffs / min(abs(coeffs))) * min(abs(coeffs))
    yp_r = mat %*% r_coeffs # yp == yp_lm
    yp_r
}
