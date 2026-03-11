
#' @importFrom stats as.formula terms
NULL

get.score.breaks = function(scores, idx, prec=1) {
    vals = scores[scores$idx==idx-1, ]
    if (nrow(vals) == 0 || is.null(vals)) {
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
        }
    }
    weights = sprintf(paste0("%.", prec, "f"), weights)
    sorted_vals = vals$split_val
    sorted_vals_fmt = sprintf(paste0("%.", prec, "f"), sorted_vals)

    out = character(length(sorted_vals_fmt) + 1)
    out[1] = paste0("<", sorted_vals_fmt[1])
    for (i in 2:length(sorted_vals_fmt)) {
        out[i] = paste0("[", sorted_vals_fmt[i-1], ",", sorted_vals_fmt[i], ")")
    }
    out[length(sorted_vals_fmt)+1] = paste0(">=", sorted_vals_fmt[length(sorted_vals_fmt)])

    # Collapse adjacent bins with identical formatted weights
    c_breaks = out[1]
    c_weights = weights[1]
    for (i in 2:length(weights)) {
        if (weights[i] == c_weights[length(c_weights)]) {
            # Merge: extend previous bin
            prev = c_breaks[length(c_breaks)]
            if (i < length(weights)) {
                # Middle bin absorbed
                if (startsWith(prev, "<")) {
                    c_breaks[length(c_breaks)] = paste0("<", sorted_vals_fmt[i])
                } else if (startsWith(prev, "[")) {
                    lower = sub(",.*", "", sub("^\\[", "", prev))
                    c_breaks[length(c_breaks)] = paste0("[", lower, ",", sorted_vals_fmt[i], ")")
                }
            } else {
                # Last bin absorbed
                if (startsWith(prev, "<")) {
                    c_breaks[length(c_breaks)] = "all"
                } else if (startsWith(prev, "[")) {
                    lower = sub(",.*", "", sub("^\\[", "", prev))
                    c_breaks[length(c_breaks)] = paste0(">=", lower)
                }
            }
        } else {
            c_breaks = c(c_breaks, out[i])
            c_weights = c(c_weights, weights[i])
        }
    }

    list("index" = idx, "weights" = c_weights, "breaks" = c_breaks)
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
#' @param format Character string specifying the output format: "text" (default), "latex", "md", "latex_h", "md_h", or "ascii_h".
#' @param ... Additional arguments passed to the specific print method.
#'
#' @return Invisibly returns the input object or the formatted string.
#'
#' @examples
#' model <- gbrs(mpg ~ wt + hp, data = mtcars)
#' print(model)          # Text output
#' print(model, "latex") # LaTeX output
#' print(model, "md")    # Markdown output
#' print(model, "latex_h") # Horizontal LaTeX output
#'
#' @seealso \code{\link{gbrs}}, \code{\link{predict.gbrs}}
#' @export
print.gbrs <- function(x, format = "ascii_h", ...) {
    if (format == "latex") {
        print.latex(x, ...)
    } else if (format == "md") {
        print.md(x, ...)
    } else if (format == "latex_h") {
        print.latex.horizontal(x, ...)
    } else if (format == "md_h") {
        print.md.horizontal(x, ...)
    } else if (format == "ascii_h") {
        print.ascii.horizontal(x, ...)
    } else {
        print_model_score(x$weights, x$formula)
    }
}

#' Print GBRS Model in Legacy Vertical Format
#'
#' @description
#' Prints the GBRS model using the legacy vertical format.
#'
#' @param x An object of class \code{"gbrs"} returned by \code{\link{gbrs}}.
#' @param ... Additional arguments passed to print.
#'
#' @export
print.vertical <- function(x, ...) {
    print(x, format = "text", ...)
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
    "|:---|:---:|---:|"
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

#' Print GBRS Model as Horizontal LaTeX Table
#'
#' @export
print.latex.horizontal <- function(x, ...) {
    scores <- x$weights
    formula <- as.formula(x$formula)
    terms_obj <- terms(formula)
    independent_vars <- attr(terms_obj, "term.labels")

    # Calculate max columns needed
    max_cols <- 0
    for (i in 1:length(independent_vars)) {
        if ((i-1) %in% scores$idx) {
            score_breaks <- get.score.breaks(scores, i)
            max_cols <- max(max_cols, length(score_breaks$breaks))
        }
    }

    # Build column spec: l for variable name, then l for each bin
    col_spec <- paste0("l", paste(rep("l", max_cols), collapse=""))

    latex_lines <- c(
        "\\begin{tabular}{"  , col_spec, "}",
        "\\hline"
    )

    for (i in 1:length(independent_vars)) {
        if ((i-1) %in% scores$idx) {
            score_breaks <- get.score.breaks(scores, i)
            if (length(score_breaks) > 0) {
                feature_name <- independent_vars[score_breaks$index]
                breaks <- score_breaks$breaks
                weights <- score_breaks$weights

                # Escape LaTeX
                breaks <- gsub("<", "\\\\ensuremath{<} ", breaks)
                breaks <- gsub(">=", "\\\\ensuremath{\\\\ge} ", breaks)
                breaks <- gsub(">", "\\\\ensuremath{>} ", breaks)
                
                # Pad with empty strings if fewer than max_cols
                n_pad <- max_cols - length(breaks)
                breaks_padded <- c(breaks, rep("", n_pad))
                weights_padded <- c(weights, rep("", n_pad))

                # Row 1: Variable Name | Thresholds
                row1 <- paste0(feature_name, " & ", paste(breaks_padded, collapse = " & "), " \\\\")
                
                # Row 2: Empty | Scores
                row2 <- paste0(" & ", paste(weights_padded, collapse = " & "), " \\\\")

                latex_lines <- c(latex_lines, row1, row2)
            }
        }
    }
    latex_lines <- c(latex_lines, "\\hline", "\\end{tabular}")
    
    cat(paste(latex_lines, collapse = "\n"))
    cat("\n")
    invisible(paste(latex_lines, collapse = "\n"))
}

#' Print GBRS Model as Horizontal Markdown Table
#'
#' @export
print.md.horizontal <- function(x, ...) {
    scores <- x$weights
    formula <- as.formula(x$formula)
    terms_obj <- terms(formula)
    independent_vars <- attr(terms_obj, "term.labels")

    # Calculate max columns needed
    max_cols <- 0
    for (i in 1:length(independent_vars)) {
        if ((i-1) %in% scores$idx) {
            score_breaks <- get.score.breaks(scores, i)
            max_cols <- max(max_cols, length(score_breaks$breaks))
        }
    }

    # Header
    header <- paste0("| Variable | ", paste(rep(" | ", max_cols), collapse=""), "|")
    separator <- paste0("|:---|", paste(rep(":---|", max_cols), collapse=""), "|")
    
    md_lines <- c(header, separator)

    for (i in 1:length(independent_vars)) {
        if ((i-1) %in% scores$idx) {
            score_breaks <- get.score.breaks(scores, i)
            if (length(score_breaks) > 0) {
                feature_name <- independent_vars[score_breaks$index]
                breaks <- score_breaks$breaks
                weights <- score_breaks$weights

                # Pad
                n_pad <- max_cols - length(breaks)
                breaks_padded <- c(breaks, rep("", n_pad))
                weights_padded <- c(weights, rep("", n_pad))

                # Row 1: Variable Name | Thresholds
                row1 <- paste0("| **", feature_name, "** | ", paste(breaks_padded, collapse = " | "), " |")
                
                # Row 2: Empty | Scores
                row2 <- paste0("| | ", paste(weights_padded, collapse = " | "), " |")

                md_lines <- c(md_lines, row1, row2)
            }
        }
    }
    
    cat(paste(md_lines, collapse = "\n"))
    cat("\n")
    invisible(paste(md_lines, collapse = "\n"))
}

#' Print GBRS Model as Horizontal ASCII Table
#'
#' @export
print.ascii.horizontal <- function(x, ...) {
    scores <- x$weights
    formula <- as.formula(x$formula)
    terms_obj <- terms(formula)
    independent_vars <- attr(terms_obj, "term.labels")

    # Add Base Score
    if ("cst" %in% names(scores)) {
        base_score <- sum(scores$cst) 
        cat(sprintf("Base Score: %.4f\n", base_score))
        cat(paste(rep("-", 20), collapse=""), "\n\n")
    }

    # We need to calculate column widths dynamically
    # First pass: collect all data
    rows <- list()
    for (i in 1:length(independent_vars)) {
        if ((i-1) %in% scores$idx) {
            score_breaks <- get.score.breaks(scores, i)
            if (length(score_breaks) > 0) {
                rows[[length(rows)+1]] = list(
                    name = independent_vars[score_breaks$index],
                    breaks = score_breaks$breaks,
                    weights = score_breaks$weights
                )
            }
        }
    }

    if (length(rows) == 0) return()

    max_bins <- max(sapply(rows, function(r) length(r$breaks)))
    
    # Calculate width for each column (0 is variable name, 1..max_bins are bins)
    col_widths <- integer(max_bins + 1)
    
    # Variable name width
    col_widths[1] <- max(sapply(rows, function(r) nchar(r$name)))
    
    # Bin widths
    for (j in 1:max_bins) {
        w <- 0
        for (r in rows) {
            if (length(r$breaks) >= j) {
                w <- max(w, nchar(r$breaks[j]), nchar(r$weights[j]))
            }
        }
        col_widths[j+1] <- w
    }
    
    # Add padding
    col_widths <- col_widths + 2 # 1 space padding on each side

    # Print function helper
    print_row <- function(cols) {
        line <- ""
        for (j in 1:length(cols)) {
            val <- cols[j]
            width <- col_widths[j]
            # Left align
            padded <- paste0(val, paste(rep(" ", width - nchar(val)), collapse=""))
            
            if (j == 1) {
                line <- paste0(line, padded)
            } else {
                line <- paste0(line, "| ", padded)
            }
        }
        cat(line, "\n")
    }

    # Print rows
    for (i in 1:length(rows)) {
        r <- rows[[i]]
        # Row 1: Name + Breaks
        cols1 <- c(r$name, r$breaks)
        # Pad with empty strings if needed
        if (length(cols1) < length(col_widths)) {
            cols1 <- c(cols1, rep("", length(col_widths) - length(cols1)))
        }
        print_row(cols1)
        
        # Row 2: Empty + Weights
        cols2 <- c("", r$weights)
        if (length(cols2) < length(col_widths)) {
            cols2 <- c(cols2, rep("", length(col_widths) - length(cols2)))
        }
        print_row(cols2)
        
        # Separation
        if (i < length(rows)) {
             total_len <- sum(col_widths) + (length(col_widths)-1)*2
             cat(paste(rep("-", total_len), collapse=""), "\n")
        } else {
             cat("\n")
        }
    }
}
