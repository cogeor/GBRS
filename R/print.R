
#' @importFrom stats as.formula terms
NULL

get.score.breaks = function(scores, idx) {
    prec=1
    vals = scores[scores$idx==idx-1, ] 
    if (length(vals) == 0) {
        return(list()) 
    }
    if (is.null(nrow(vals))) {
        return(list("index" = idx, "breaks" = c(paste0("<", vals$split_val)), "weights"=c(sprintf(paste0("%.", prec, "f"), vals$w))))
    }
    if (nrow(vals) == 1 && vals$split_val == 0) {
        weights <- c(sprintf(paste0("%.", prec, "f"), 0), sprintf(paste0("%.", prec, "f"), vals$w))
        return(list("index" = idx, "weights" = weights, "breaks" = c("FALSE", "TRUE")))
    }
    sorted_idxs = order(vals$split_val)
    vals = vals[sorted_idxs, ]
    weights = double(nrow(vals) + 1)
    for(i in 1:(length(weights)-1)) {
        if (vals$w[i] > 0) {
            weights[(i+1):length(weights)] = weights[(i+1):length(weights)] + vals$w[i]
        } else {
            weights[1:i] = weights[1:i] - vals$w[i]
            #weights[(i+1):length(weights)] = weights[(i+1):length(weights)] + vals$w[i]
        }
    }
    weights = sprintf(paste0("%.", prec, "f"), weights)
    sorted_vals = vals$split_val
    sorted_vals = sprintf(paste0("%.", prec, "f"), sorted_vals)

    out = character(length(sorted_vals) + 1)
    out[1] = paste0("<", sorted_vals[1])
    for (i in 2:length(sorted_vals)) {
        out[i] = paste0("[", sorted_vals[i-1], ",", sorted_vals[i], ")")
    }
    out[length(sorted_vals)+1] = paste0(">=", sorted_vals[length(sorted_vals)])
    list("index" = idx, "weights" = weights, "breaks" = out)
}

score_line = function(score_breaks, feature_names) {
    da = data.frame(breaks=c(feature_names[score_breaks$index], score_breaks$breaks), weights=c("", score_breaks$weights))
    t(da)
}

print_model_score = function(scores, formula) {
    formula = as.formula(formula)
    terms_obj = terms(formula)
    independent_vars <- attr(terms_obj, "term.labels")
    a = data.frame()
    for (i in 1:length(independent_vars)) {
        if ((i-1) %in% scores$idx){
            score_breaks = get.score.breaks(scores, i)
            if (length(score_breaks) > 0) {
                line = score_line(score_breaks, independent_vars)
                print(line)
                cat("\n")
            }
        }
    }
}

#' Print a GBRS Model
#'
#' @description
#' S3 print method for GBRS models. Displays the learned rules in a human-readable
#' format, showing feature names, split thresholds, and associated weights.
#' Can also generate LaTeX or Markdown tables.
#'
#' @param x An object of class \code{"gbrs"} returned by \code{\link{gbrs}}.
#' @param format Character string specifying the output format: "text" (default), "latex", or "md".
#' @param ... Additional arguments passed to the specific print method.
#'
#' @return Invisibly returns the input object or the formatted string.
#'
#' @examples
#' model <- gbrs(mpg ~ wt + hp, data = mtcars)
#' print(model)          # Text output
#' print(model, "latex") # LaTeX output
#' print(model, "md")    # Markdown output
#'
#' @seealso \code{\link{gbrs}}, \code{\link{predict.gbrs}}
#' @export
print.gbrs <- function(x, format = "text", ...) {
    if (format == "latex") {
        print.latex(x, ...)
    } else if (format == "md") {
        print.md(x, ...)
    } else {
        print_model_score(x$weights, x$formula)
    }
}

#' Print GBRS Model as LaTeX Table
#'
#' @description
#' S3 method to generate a LaTeX table representation of the GBRS model score,
#' formatted like clinical scoring systems for publication.
#'
#' @param x An object of class \code{"gbrs"} returned by \code{\link{gbrs}}.
#' @param caption Optional caption for the LaTeX table.
#' @param label Optional label for cross-referencing in LaTeX.
#' @param ... Additional arguments (unused).
#'
#' @return Invisibly returns the LaTeX code as a character string.
#'
#' @examples
#' model <- gbrs(mpg ~ wt + hp, data = mtcars)
#' print(model, "latex", caption = "GBRS Risk Score")
#'
#' @export
print.latex <- function(x, caption = "GBRS Clinical Score", label = "tab:gbrs", ...) {
  if (!inherits(x, "gbrs")) {
    stop("Object must be of class 'gbrs'")
  }
  
  scores <- x$weights
  formula <- as.formula(x$formula)
  terms_obj <- terms(formula)
  independent_vars <- attr(terms_obj, "term.labels")
  
  # Start LaTeX table
  latex_lines <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    paste0("\\caption{", caption, "}"),
    paste0("\\label{", label, "}"),
    "\\begin{tabular}{lcc}",
    "\\hline",
    "\\textbf{Variable} & \\textbf{Category} & \\textbf{Points} \\\\",
    "\\hline"
  )
  
  # Process each feature
  for (i in 1:length(independent_vars)) {
    if ((i-1) %in% scores$idx) {
      score_breaks <- get.score.breaks(scores, i)
      if (length(score_breaks) > 0) {
        feature_name <- independent_vars[score_breaks$index]
        breaks <- score_breaks$breaks
        weights <- score_breaks$weights
        
        # Escape LaTeX special characters in breaks
        breaks <- gsub("<", "$<$", breaks)
        breaks <- gsub(">=", "$\\\\ge$", breaks)
        breaks <- gsub(">", "$>$", breaks)
        
        # First row with feature name
        latex_lines <- c(latex_lines, 
                        paste0("\\textbf{", feature_name, "}"))
        
        # Add each category and its points
        for (j in 1:length(breaks)) {
          if (j == 1) {
            latex_lines[length(latex_lines)] <- paste0(
              latex_lines[length(latex_lines)],
              " & ", breaks[j], " & ", weights[j], " \\\\"
            )
          } else {
            latex_lines <- c(latex_lines,
                           paste0(" & ", breaks[j], " & ", weights[j], " \\\\"))
          }
        }
        
        # Add separator between features
        latex_lines <- c(latex_lines, "\\hline")
      }
    }
  }
  
  # Close table
  latex_lines <- c(latex_lines,
                  "\\end{tabular}",
                  "\\end{table}")
  
  # Print the LaTeX code
  cat(paste(latex_lines, collapse = "\n"))
  cat("\n")
  
  invisible(paste(latex_lines, collapse = "\n"))
}

#' Print GBRS Model as Markdown Table
#'
#' @description
#' S3 method to generate a Markdown table representation of the GBRS model score.
#'
#' @param x An object of class \code{"gbrs"} returned by \code{\link{gbrs}}.
#' @param ... Additional arguments (unused).
#'
#' @return Invisibly returns the Markdown code as a character string.
#'
#' @examples
#' model <- gbrs(mpg ~ wt + hp, data = mtcars)
#' print(model, "md")
#'
#' @export
print.md <- function(x, ...) {
  if (!inherits(x, "gbrs")) {
    stop("Object must be of class 'gbrs'")
  }
  
  scores <- x$weights
  formula <- as.formula(x$formula)
  terms_obj <- terms(formula)
  independent_vars <- attr(terms_obj, "term.labels")
  
  # Start Markdown table
  md_lines <- c(
    "| Variable | Category | Points |",
    "|:---|:---|:---|"
  )
  
  # Process each feature
  for (i in 1:length(independent_vars)) {
    if ((i-1) %in% scores$idx) {
      score_breaks <- get.score.breaks(scores, i)
      if (length(score_breaks) > 0) {
        feature_name <- independent_vars[score_breaks$index]
        breaks <- score_breaks$breaks
        weights <- score_breaks$weights
        
        # First row with feature name
        md_lines <- c(md_lines, 
                      paste0("| **", feature_name, "** | ", breaks[1], " | ", weights[1], " |"))
        
        # Add remaining categories and points
        if (length(breaks) > 1) {
          for (j in 2:length(breaks)) {
            md_lines <- c(md_lines,
                           paste0("| | ", breaks[j], " | ", weights[j], " |"))
          }
        }
      }
    }
  }
  
  # Print the Markdown code
  cat(paste(md_lines, collapse = "\n"))
  cat("\n")
  
  invisible(paste(md_lines, collapse = "\n"))
}
