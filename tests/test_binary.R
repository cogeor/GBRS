library(dplyr)
library(tidyr)
library(AutoScore)
library(pROC)

library(Rcpp)
library(RiskScore)
Sys.setenv("PKG_LIBS"="-fopenmp")
Sys.setenv("PKG_CXXFLAGS" = paste(
  "-fopenmp ",
  paste0("-I", normalizePath("inst/include"))
))
sourceCpp("src/R_bindings.cpp", verbose=TRUE, rebuild=TRUE)
source("R/utils.R")
source("tests/datasets.R")

df = load_dataset("cardio")
formula = as.formula("target ~ age + gender + height + weight + ap_hi + ap_lo + cholesterol + smoke + alco + active")

#df = load.ukb("/home/cgeor/src/trees/medscore/data/data_ukb_tp2_ht.rda")
#formula = as.formula("label ~ Age_2 + Sex_0 + BMI_2 + lv_ef_2 + is_alcoholic_2 + is_smoker_2")
df = load_dataset("abalone")

smp_size = floor(0.75 * nrow(df))
n_max = 500
lr = 0.1
n_quantiles = 5
ss_rate = 0.8


train_ind = sample(seq_len(nrow(df)), size = smp_size)
train_df = df[train_ind,]
test_df = df[-train_ind,]
score_model = sm(formula, train_df, objective="binary", n_max=n_max, lr=lr, n_quantiles=n_quantiles, ss_rate=ss_rate)
pred_score = predict(score_model, test_df)
print(roc(test_df$target, pred_score)$auc)

#print(cross.entropy(test_df$label, yp))
#print.model.score(score_model$weights, formula)

linear_model <- glm(formula, data = train_df, family = binomial(link = "logit"))
pred_lin <- predict(linear_model, newdata = test_df, type = "response")
print(roc(test_df$target, pred_lin)$auc)
#print(summary(linear_model))


#print(roc(test_df$label, predicted_probs)$auc)

#print(roc(train_df$label, probs))