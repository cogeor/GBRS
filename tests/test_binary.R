library(dplyr)
library(tidyr)
library(AutoScore)
library(pROC)

library(Rcpp)
library(RiskScore)
source("R/utils.R")

#data("mtcars")
#df = mtcars
#df = data.frame(x=c(2, 2, 0.5, 1), y=c(1, 1, 0, 1))
df = load.ukb("/home/cgeor/src/trees/medscore/data/data_ukb_tp2_ht.rda")

smp_size = floor(0.75 * nrow(df))
train_ind = sample(seq_len(nrow(df)), size = smp_size)
train_df = df[train_ind,]
test_df = df[-train_ind,]

formula = as.formula("label ~ Age_2 + Sex_0 + BMI_2 + lv_ef_2 + is_alcoholic_2 + is_smoker_2")
#formula = as.formula("label ~ lv_ef_2")
#formula = as.formula("vs ~ mpg + cyl + disp + am + drat")
#formula = as.formula("vs ~ cyl + disp + cyl + am + drat")
#formula = as.formula("y ~ x")

n_max = 100
lr = 0.1
n_quantiles = 10
ss_rate = 0.8

score_model = sm(formula, train_df, objective="binary", n_max=n_max, lr=lr, n_quantiles=n_quantiles, ss_rate=ss_rate)
pred_score = predict(score_model, test_df)

#print(cross.entropy(test_df$label, yp))
print(roc(test_df$label, yp)$auc)

formula = as.formula("label ~ Age_2 + Sex_0 + BMI_2 + lv_ef_2 + is_alcoholic_2 + is_smoker_2")

linear_model <- glm(formula, data = train_df, family = binomial(link = "logit"))
pred_lin <- predict(linear_model, newdata = test_df, type = "response")
print(summary(linear_model))

score_model = sm(formula, train_df, objective="binary")
pred_score = predict(score_model, test_df)
print(score_model)

#print(roc(test_df$label, predicted_probs)$auc)

#print(roc(train_df$label, probs))