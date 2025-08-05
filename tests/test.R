library(Rcpp)
library(dplyr)
library(tidyr)
library(xtable)
library(ascii)
library(datasets)
Sys.setenv("PKG_LIBS"="-fopenmp")
Sys.setenv("PKG_CXXFLAGS" = paste(
  "-fopenmp ",
  paste0("-I", normalizePath("inst/include"))
))
sourceCpp("src/R_bindings.cpp", verbose=TRUE, rebuild=TRUE)
source("R/utils.R")

#data("mtcars")
#df = mtcars
#df = data.frame(x=c(2, 2, 0.5, 1), y=c(1, 1, 0, 1))


#formula = as.formula("Median_Myo ~ Age_2 + Sex_0 + BMI_2 + lv_ef_2 + SBP_2 + FEV1_2 + is_alcoholic_2 + is_smoker_2")
#formula = as.formula("wt ~ mpg + hp + drat + cyl")

n_max = 300
lr = 0.05
n_quantiles = 5
ss_rate = 1

#df = mtcars
#formula = "wt ~ mpg + cyl + disp + drat + hp + qsec"


#df = swiss
#formula = "Fertility ~ Agriculture + Examination + Education + Catholic + Infant.Mortality"

data_path = "/home/cgeor/src/pattern_mining/data/"
#df = read.csv(paste0(data_path, "processed_cleveland.csv"))
#formula = "num ~ age + sex + cp + trestbps + chol + fbs + restecg + thalach + exang + slope"

#df = read.csv(paste0(data_path, "insurance.csv"))
#formula = "charges ~ age + bmi + sex + region + children + smoker"

#df = read.csv(paste0(data_path, "housing.csv"))
#formula = "median_house_value ~ longitude + latitude + housing_median_age + total_rooms + total_bedrooms + population + households + median_income"

#df = load.ukb("data/data_ukb_tp2_ht.rda")
#formula = "Median_Myo ~ Age_2 + Sex_0 + BMI_2 + lv_ef_2 + is_alcoholic_2 + is_smoker_2 + is_alcoholic_2:is_smoker_2"

df = drop_na(df)
n_runs = 5
lm_res = numeric(n_runs)
lm_r_res = numeric(n_runs)
score_res = numeric(n_runs)
f = as.formula(formula)
resp_var = all.vars(f)[1]

for (i in 1:n_runs) {
    smp_size = floor(0.75 * nrow(df))
    train_ind = sample(seq_len(nrow(df)), size = smp_size)
    train_df = df[train_ind,]
    test_df = df[-train_ind,]

    lin_model = lm(formula, train_df)
    yp_lm = predict(lin_model, test_df)

    score_model = sm(formula, train_df, n_max = n_max, lr = lr, n_quantiles=n_quantiles, ss_rate=ss_rate, objective = "continuous")
    yp = predict(score_model, test_df)

    yp_r = predict.round(lin_model$coefficients, formula, test_df)

    y = test_df[resp_var]
    print(norm(y - yp, "2"))
    print(norm(y - yp_lm, "2"))
    print(norm(y - yp_r, "2"))
    print(norm(y - yp, "2") / norm(y - yp_r, "2") * 100)
    lm_res[i] = norm(y - yp_lm, "2") 
    lm_r_res[i] = norm(y - yp_r, "2") 
    score_res[i] = norm(y - yp, "2") 
}
lm_res = sqrt(lm_res)
lm_r_res = sqrt(lm_r_res)
score_res = sqrt(score_res)

sd1 = sd(lm_res)
sd11 = sd(lm_r_res)
sd2 = sd(score_res)
m1 = mean(lm_res)
m11 = mean(lm_r_res)
m2 = mean(score_res)
df = data.frame(method = c("Lin. model", "Lin. model (rounded)", "Ours"), score = c(m1, m11, m2),
                sd = c(sd1, sd11, sd2))

library(ggplot2)
ggplot(df, aes(x=method, y=score, fill=method))  + theme_light() +
            scale_fill_manual(values=c("#a7a7a7",
                                       "#a7a7a7",
                                     "#2a61ca")) + 
            geom_col() + 
            scale_y_continuous("MSE", limits=c(m1 - m1/10, m2 + m2 / 10), oob = scales::squish) + 
            geom_errorbar(aes(x=method, ymin=score-sd, ymax=score+sd, width=.2, linewidth=2.5)) + 
            theme(text=element_text(size=30), axis.title = element_text(size=30),
                                            axis.text.x=element_blank(),
                                            axis.ticks.x=element_blank(),
                                            axis.title.x=element_blank(),
                                            strip.text.x=element_text(size=25, color="black"),
                                            strip.background=element_rect(fill="#E4E4E4"))
ggsave("bar_results.png", width = 10, height = 15, dpi = 300)
test = t.test(lm_r_res, score_res, paired=TRUE)
#print.model.score(score_model$weights, formula)

#ggplot(df2, aes(x=a* 100)) + geom_histogram() + theme_minimal() +
#        theme(axis.text=element_text(size=20), axis.title=element_blank())