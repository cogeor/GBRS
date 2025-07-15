library(Rcpp)
library(dplyr)
library(tidyr)
library(xtable)
library(ascii)
library(datasets)
library(survival)
library(survcomp)
library(ggplot2)
library(ggfortify)
library(ggsurvfit)
library(survminer)
library(pROC)
library(boot)
library(mice)
library(glue)
library(caret) 
library(randomForestSRC)
#library(RiskScore)
Sys.setenv("PKG_LIBS"="-fopenmp")
Sys.setenv("PKG_CXXFLAGS" = paste(
  "-fopenmp -O3",
  paste0("-I", normalizePath("inst/include"))
))
sourceCpp("src/R_bindings.cpp", verbose=TRUE, rebuild=TRUE)
source("R/utils.R")

TMAX = 5
MAX_TIME = 5

print("load dataframe")
load("/home/cgeor/src/trees/medscore/data/data_ukb_tp2_ht.rda")

print("preprocessing")

covariates = c("Age_2" , "Sex_0" , "BMI_2" , "SBP_2" , "lv_ef_2")

df <- df %>%
  mutate(across(where(is.logical), as.integer))

code = "CIR019"

df$event = df[[glue("event_{code}")]] 
df$event_time = df[[glue("event_time_{code}")]] - df[["time_to_visit_2"]]
df_sub = df %>% filter(df$event_time > 0)

df_sub = na.omit(df_sub) 


# remove ind. with history of disease
df_sub = df_sub %>% filter(.data[[glue("hist_{code}")]] == 0) # filter history
cox_formula = glue("surv_o ~ ", paste(covariates, collapse=" + "))

cox_formula = as.formula(cox_formula)
                
### Perf. benchmark
df = df_sub
a = df$event_time
b = df$event
TMAX = MAX_TIME
b2 = (a < TMAX) * b
c = a * (a < TMAX) + TMAX * (a > TMAX)
surv_o = Surv(c, b2)
df$tp = c
df$ep = b2
df$Sex_0 = ifelse(df$Sex_0 == "Male", 1, 0)

cox_cindices <- numeric(10)
rsf_cindices <- numeric(10)
sm_cindices <- numeric(10)

for (i in 1:2) {
  train_index <- createDataPartition(df$event, p = 0.7, list = FALSE)
  train_data <- df[train_index, ]
  test_data <- df[-train_index, ]

  formula <- as.formula("Surv(tp, ep) ~ Age_2 + Sex_0 + BMI_2 + SBP_2 + lv_ef_2 + is_alcoholic_2 + is_smoker_2")

  # Cox
  cox_model <- coxph(formula, data = train_data)
  cox_pred <- predict(cox_model, newdata = test_data, type = "risk")

  # RSF
  rsf_model <- rfsrc(formula, data = train_data, ntree = 300)
  rsf_pred <- predict(rsf_model, newdata = test_data)$predicted

q_list <- list(
  c(50, 60, 70),  # for feature 1
  NULL,         # use default for feature 2
  c(20, 25, 30),
  NULL,
  c(50, 60),
  NULL,
  NULL
)

  # Custom SM model
  model_surv <- sm(formula, train_data,
                   n_max = 100,
                   lr = 0.01,
                   n_quantiles = 3,
                   ss_rate = 1,
                   objective = "survival", user_quantiles = q_list)
  risk_scores <- predict(model_surv, test_data)
  print.model.score(model_surv$weights, formula)

  # Compute C-indexes
  cox_cindices[i] <- concordance.index(x = cox_pred,
                                       surv.time = test_data$tp,
                                       surv.event = test_data$ep)$c.index

  rsf_cindices[i] <- concordance.index(x = rsf_pred,
                                       surv.time = test_data$tp,
                                       surv.event = test_data$ep)$c.index

  sm_cindices[i] <- concordance.index(x = risk_scores,
                                      surv.time = test_data$tp,
                                      surv.event = test_data$ep)$c.index
}

results_df <- data.frame(
  Method = rep(c("Cox", "RSF", "SM"), each = 10),
  CIndex = c(cox_cindices, rsf_cindices, sm_cindices)
)

# Summarize: mean and std
summary_df <- results_df %>%
  group_by(Method) %>%
  summarise(mean_cindex = mean(CIndex),
            sd_cindex = sd(CIndex))

ggplot(summary_df, aes(x = Method, y = mean_cindex, fill = Method)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_errorbar(aes(ymin = mean_cindex - sd_cindex, ymax = mean_cindex + sd_cindex),
                width = 0.2) +
  ylab("C-index (mean ± SD)") +
  xlab("Model") +
  theme_minimal() +
  ggtitle("Comparison of C-index Across Models") +
  theme(text = element_text(size = 14))

ggsave("res1.png")


#TODO: add distribution of scores in pop. eg
# Score   1   2   3   4   5
# Pop.%   10  20  40  60  70
# to both training and test pop.
# so some sort of score distribution method 

#https://www.soa.org/globalassets/assets/Files/Research/research-2016-risk-scoring-health-insurance.pdf