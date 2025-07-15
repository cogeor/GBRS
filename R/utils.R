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

load.ukb = function(data_path) {
    load(data_path)
    df = df %>% drop_na()
    df["event_time_CIR007"]  = df["event_time_CIR007"] - df$time_to_visit_2
    df[df$event_time_CIR007 < 0, "hist_CIR007"] = 1
    df = df %>% filter(hist_CIR007 == 0)
    df["label"] = (df$event_time_CIR007 < 10 ) * df$event_CIR007
    df["Sex_0"] = ifelse(df$Sex_0 == "Male", 1, 0)
    df["label_status"] = df$label
    df["label_time"] = df$event_time_CIR019
    df
}

#process.formula = function(formula, data) {
#
#    frame = model.frame(terms(formula), data)
#    resp =  model.extract(frame, "response")
#    mat = model.matrix(terms(formula), data)
#    mat = delete.intercept(mat)
#    return(list("x" = mat, "y" = resp))
#}

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

predict.score.proba = function(model, x) {
    yp = rep(model$cst[1], nrow(x))
    for (i in 1:nrow(model)) {
        yp = yp + ifelse(x[, model$idx[i] + 1] <= model$split_val[i], model$w[i], 0)
    }
    exp(yp) / (1 + exp(yp))
}

predict.score = function(model, x) {
    yp = rep(model$cst[1], nrow(x))
    for (i in 1:nrow(model)) {
        yp = yp + ifelse(x[, model$idx[i] + 1] <= model$split_val[i], 0, model$w[i])
    }
    yp
}

predict.score2 = function(model, x) {
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

get.score.breaks = function(scores, idx) {
    prec=1
    vals = scores[scores$idx==idx-1, ] 
    if (length(vals) == 0) {
        return(list()) 
    }
    if (is.null(nrow(vals))) {
        return(list("index" = idx, "breaks" = c(paste0("<", vals$split_val)), "weights"=c(sprintf(paste0("%.", prec, "f"), vals$w))))
    }
    sorted_idxs = order(vals$split_val)
    vals = vals[sorted_idxs, ]
    weights = double(nrow(vals) + 1)
    for(i in 1:(length(weights)-1)) {
        weights[(i+1):length(weights)] = weights[(i+1):length(weights)] + vals$w[i]
    }
    weights = sprintf(paste0("%.", prec, "f"), weights)
    sorted_vals = vals$split_val
    sorted_vals = sprintf(paste0("%.", prec, "f"), sorted_vals)

    # binary case
    if (nrow(vals) == 1 && vals$split_val == 0) {
        return(list("index" = idx, "weights" = weights, "breaks" = c("FALSE", "TRUE")))
    }
    out = character(length(sorted_vals) + 1)
    out[1] = paste0("<", sorted_vals[1])
    for (i in 2:length(sorted_vals)) {
        out[i] = paste0("[", sorted_vals[i-1], ",", sorted_vals[i], ")")
    }
    out[length(sorted_vals)+1] = paste0(">=", sorted_vals[length(sorted_vals)])
    list("index" = idx, "weights" = weights, "breaks" = out)
}

score.line = function(score_breaks, names) {
    da = data.frame(breaks=c(names[score_breaks$index], score_breaks$breaks), weights=c("", score_breaks$weights))
    t(da)
}

print.model.score.old = function(scores, formula) {
    sorted_vals = sort(scores[2,scores[1,]==2])
    out = character(length(sorted_vals) + 1)
    out[1] = paste0("<", sorted_vals[1])
    for (i in 2:length(sorted_vals)) {
        out[i] = paste0(sorted_vals[i-1], "-", sorted_vals[i])
    }
    out[length(sorted_vals)+1] = paste0(sorted_vals[length(sorted_vals)],"<")

    a = scores[,scores[1,]==2]
    b = order(a[2,])
    c = a[, b]
    d = double(ncol(c)+1)
    for(i in 1:(length(d)-1)) {
        d[(i+1):length(d)] = d[(i+1):length(d)] + c[3, i]
    }

    terms_obj = terms(formula)
    independent_vars <- attr(terms_obj, "term.labels")
    a = data.frame()
    for (i in 1:length(independent_vars)) {
        score_breaks = get.score.breaks(scores, i)
        if (length(score_breaks) > 0) {
            line = score.line(score_breaks, independent_vars)
            print(ascii(line))
            a = bind_rows(a, as.data.frame(line))
        }
    }
    #ascii(a, include.rownames=FALSE, include.colnames=FALSE, header=FALSE)
}

print.model.score = function(scores, formula) {
    formula = as.formula(formula)
    terms_obj = terms(formula)
    independent_vars <- attr(terms_obj, "term.labels")
    a = data.frame()
    for (i in 1:length(independent_vars)) {
        if ((i-1) %in% scores$idx){
            score_breaks = get.score.breaks(scores, i)
            if (length(score_breaks) > 0) {
                line = score.line(score_breaks, independent_vars)
                output <- capture.output(print(ascii(line)))
                cleaned <- output[2:3]
                line_width <- max(nchar(cleaned))
                separator <- strrep("=", line_width)
                if(i==1) {
                    cat("", separator, "\n") # "" looks useless but isnt
                }
                cat(paste(cleaned, collapse = "\n"), "\n", separator, "\n")
                #print(ascii(line), frame="none")
                #a = bind_rows(a, as.data.frame(line))
            }

        }
    }
    #ascii(a, include.rownames=FALSE, include.colnames=FALSE, header=FALSE)
}

#sm = function(formula, df, n_max = 100, lr = 0.1, n_quantiles = 10, ss_rate=1, objective = "continuous") {
#    formula = as.formula(formula)
#    data = process.formula(formula, df)
#    if (objective == "continuous") {
#        weights = fit(data$x, data$y, n_max, lr, n_quantiles, ss_rate)
#    } else {
#        weights = fit_proba(data$x, data$y, n_max, lr, n_quantiles, ss_rate)
#    }
#    weights = prune.weights(weights)
#    obj = list(formula = formula, data = data, weights = weights, objective = objective)
#    class(obj) = "sm"
#    obj
#}

sm <- function(formula, df, n_max = 100, lr = 0.1, n_quantiles = 10, ss_rate = 1, objective = "auto", user_quantiles = NULL) {
  formula <- as.formula(formula)
  data <- process.formula(formula, df)

  # Determine objective if set to "auto"
  if (objective == "auto") {
    objective <- switch(data$type,
                        survival = "survival",
                        standard = "continuous")
  }

  # Fit model based on type
  weights <- switch(objective,
    "continuous" = fit(data$x, data$y, n_max, lr, n_quantiles, ss_rate, user_quantiles),
    "binary"     = fit_proba(data$x, data$y, n_max, lr, n_quantiles, ss_rate),
    "survival"   = fit_survival(data$x, data$time, data$event, n_max, lr, n_quantiles, ss_rate, user_quantiles),
    stop("Unknown objective: must be 'continuous', 'binary', or 'survival'")
  )

  weights <- prune.weights(weights)
  obj <- list(formula = formula, data = data, weights = weights, objective = objective)
  class(obj) <- "sm"
  obj
}

predict.sm <- function(obj, df) {
  data <- process.formula(obj$formula, df)

  pred <- switch(obj$objective,
    "continuous" = predict.score(obj$weights, data$x),
    "binary"     = predict.score.proba(obj$weights, data$x),
    "survival"   = predict.score(obj$weights, data$x),  # Returns log-risk scores
    stop("Unknown objective: must be 'continuous', 'binary', or 'survival'")
  )

  return(pred)
}

#predict.sm = function(obj, df) {
#    data = process.formula(obj$formula, df)
#    if(obj$objective == "continuous") {
#        predict.score(obj$weights, data$x)
#    } else {
#        predict.score.proba(obj$weights, data$x)
#    }
#}

predict.round = function(coeffs, formula, data) {
    formula = as.formula(formula)
    frame = model.frame(terms(formula), data)
    resp =  model.extract(frame, "response")
    mat = model.matrix(terms(formula), data)
    r_coeffs = round(coeffs / min(abs(coeffs))) * min(abs(coeffs))
    yp_r = mat %*% r_coeffs # yp == yp_lm
    yp_r
}

