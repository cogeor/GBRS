library(gbrs)
library(survival)

# Load veteran dataset
data(veteran)

# Fit GBRS survival model
model <- gbrs(
  Surv(time, status) ~ trt + celltype + karno + diagtime + age + prior,
  data = veteran,
  n_max = 50,
  lr = 0.05,
  n_quantiles = 10,

  objective = "survival"
)

# Generate LaTeX table and save to file
sink("model_latex_table.tex")
print.latex(model, caption = "GBRS Survival Risk Score for Veteran Lung Cancer Dataset", label = "tab:veteran_score")
sink()

sink("model_md_table.md")
print(model, "md")
sink()
