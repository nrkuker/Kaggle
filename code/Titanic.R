# Kaggle Competition - Titanic
# Nicole Kuker
# 8 Aug 2021


# PREP #########################################################################
# Packages
pacman::p_load(
  pacman, rio, tidyverse, magrittr, janitor,  # general stuff
  psych,        # EDA
  visdat,       # missingness
  mice,         # missing value imputation
  rsample,      # data partition
  caret,        # general ML
  ROCR,         # ROC/AUC
  corrplot      # correlation plot
)


# Data
data <- import("data/titanic_train.csv")
str(data)



# EDA ##########################################################################

# viz missings
visdat::vis_miss(data, cluster = T, sort_miss = T)


sapply(data, function(x) {sum(is.na(x))})
  # 177 missing in Age
sapply(data, function(x) {sum(x=="")})
  # 687 empty in Cabin, 2 empty in Embarked


prop.table(table(data$Survived))
  # ~38% survived




# DATA CLEANING / PRE-PROCESSING ###############################################

# make empty into missing
data$Embarked[data$Embarked==""] <- NA


# derive new vars
data %<>%
  mutate(
    Relations = SibSp + Parch,
    CabinKnown = as_factor(ifelse(Cabin=="", "No", "Yes")),
    Pclass = recode(Pclass, `1` = "1st", `2` = "2nd", `3` = "3rd"),
    Survived = recode(Survived, `0` = "No", `1` = "Yes"),
    Title = sapply(Name, function(x) {
      str_sub(x,
              start = (str_locate(x, ",")[, 1] + 2),
              end = (str_locate(x, "\\.")[, 1] - 1))
    })
  )
names(data$Title) <- NULL

data %<>%
  mutate(
    Title_lump = fct_lump_prop(data$Title, prop = 0.1, other_level = "Other")
  )

# make factors
data %<>% 
  mutate(across(c("Survived", "Pclass", "Sex", "Embarked", "Title"), as_factor))


str(data)






# DIVIDE TRAIN/TEST ############################################################

# Data partition
set.seed(42)
titanic_index <- 
  initial_split(subset(data, select = -c(PassengerId, Name, Ticket, Cabin,
                                         Relations, Title)),
                prop = 0.7, strata = "Survived")
train <- training(titanic_index)
test <- testing(titanic_index)

str(train)

prop.table(table(data$Survived))
prop.table(table(train$Survived))
prop.table(table(test$Survived))





# Missing Data - impute using mice package
impute_train <- mice(train, seed = 2021)
train <- complete(impute_train)

impute_test <- mice(test, seed = 2021)
test <- complete(impute_test)






# LOGISTIC REGRESSION ##########################################################


# TAKE 1 
options(scipen = 999)

model1 <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + 
                CabinKnown + Title_lump
                , family = binomial(link = "logit"), data = train)
summary(model1)




pred1_test <- predict(model1, newdata = (test), type = "response") %>% tibble(prob = .) %>% 
  mutate(response = factor(ifelse(prob > 0.5, "Yes", "No")))
cm1_test <- caret::confusionMatrix(data = pred1_test$response, reference = test$Survived, positive = "Yes")
cm1_test
  # bal acc 0.7901
  # auc 0.871

roc_pred <- prediction(predictions = pred1_test$prob, labels = test$Survived)
roc_perf <- performance(roc_pred , "tpr" , "fpr")

plot(roc_perf,
     main = "ROC Curve for Attrition Prediction Model",
     ylab = "True Positive Rate",
     xlab = "False Positive Rate",
     col = "steelblue", lwd = 2.5,
     print.cutoffs.at = seq(0.1,0.9,0.1),
     text.adj = c(-0.2,1.35),
     text.cex = 0.9)
lines(x = c(0,1), y = c(0,1),
      lwd = 0.8, col = "grey")
text(x = 0.7, y = 0.3, font = 2, col = "steelblue4", cex = 1.2,
     labels = paste("AUC =", round(as.numeric(performance(roc_pred, "auc")@y.values),3)))
legend("topleft", pch = 1, legend = c("Probability Thresholds"), cex = 0.8)










# TAKE 2

ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  search = "grid"
)

set.seed(1863)
model2 <- train(Survived ~ .,
               data = train,
               method = "glm",
               trControl = ctrl,
               preProc = c("BoxCox", "center", "scale")
)


fitted <- predict(model2, test, type = "prob")
pred <- ROCR::prediction(fitted[,2], test$Survived)
perf <- ROCR::performance(pred, "tpr", "fpr")
auc <- performance(pred, "auc")@y.values %>% as.numeric()

cm <- caret::confusionMatrix(data = predict(model2, test), 
                             reference = test$Survived,
                             positive = "Yes")
cm
auc
