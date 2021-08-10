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
    Deck = ifelse(Cabin == "", "Unknown", str_extract(Cabin, "[:alpha:]")),   # knowingly simpliyfing here
    # Pclass = recode(Pclass, `1` = "1st", `2` = "2nd", `3` = "3rd"),
    Survived = recode(Survived, `0` = "No", `1` = "Yes"),
    Age_Class = Age*Pclass,
    Fare_pp = Fare / (Relations+1),
    Title = sapply(Name, function(x) {
      str_sub(x,
              start = (str_locate(x, ",")[, 1] + 2),
              end = (str_locate(x, "\\.")[, 1] - 1))
    })
  )
names(data$Title) <- NULL

# group up low-freq titles
data %<>%
  mutate(Title_group =
           ifelse(
             Title %in% c("Don", "Major", "Capt", "Jonkheer", "Rev", "Col", "Sir"),
             "Mr",
             ifelse(
               Title %in% c("the Countess", "Mme", "Lady"),
               "Mrs",
               ifelse(Title %in% c("Ms", "Mlle"), 
                      "Miss",
                      ifelse((Title == "Dr" & Sex == "male"), 
                             "Mr",
                             ifelse((Title == "Dr" & Sex == "female"), 
                                    "Mrs",
                                    Title)
                             )
                      )
               )
             )
         )


str(data)







# make factors
data %<>% 
  mutate(across(c("Survived", "Pclass", "Sex", "Embarked", "Title_group", "Deck"), as_factor))



str(data)






# DIVIDE TRAIN/TEST ############################################################

# Data partition
set.seed(4242)
titanic_index <- 
  initial_split(subset(data, select = -c(PassengerId, Name, Ticket, Cabin,
                                         Relations, Title, CabinKnown)),
                prop = 0.7, strata = "Survived")
train <- training(titanic_index)
test <- testing(titanic_index)

# train %>% filter(Deck =="T")
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

model1 <- glm(Survived ~ .
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
summary(model2)


fitted <- predict(model2, test, type = "prob")
pred <- ROCR::prediction(fitted[,2], test$Survived)
perf <- ROCR::performance(pred, "tpr", "fpr")
auc <- performance(pred, "auc")@y.values %>% as.numeric()

cm <- caret::confusionMatrix(data = predict(model2, test), 
                             reference = test$Survived,
                             positive = "Yes")
cm
auc





# TAKE 3

ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  search = "grid"
)

set.seed(1863)
model3 <- train(Survived ~ .,
                data = train,
                method = "glmStepAIC",
                trControl = ctrl,
                preProc = c("BoxCox", "center", "scale")
)
summary(model3)


fitted3 <- predict(model3, test, type = "prob")
pred3 <- ROCR::prediction(fitted3[,2], test$Survived)
perf3 <- ROCR::performance(pred3, "tpr", "fpr")
auc3 <- performance(pred3, "auc")@y.values %>% as.numeric()

cm3 <- caret::confusionMatrix(data = predict(model3, test), 
                             reference = test$Survived,
                             positive = "Yes")
cm3
auc3
# same





# TREE-BASED MODELS ############################################################

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  search = "grid"
)

search_grid <- expand.grid(
  iter = c(10, 100, 500),
  maxdepth = c(2, 3, 5),
  nu = 10^seq(-6, 0, by = 2)
)  

set.seed(1863)
model4 <- train(Survived ~ .,
                data = train,
                method = "ada",
                trControl = ctrl,
                tuneGrid = search_grid,
                preProc = c("BoxCox", "center", "scale")
)
model4
plot(model4)
# nu        maxdepth  iter  ROC        Sens       Spec     
# 0.000001  5         500   0.8901591  0.8881066  0.7406915




set.seed(1863)
model5 <- train(Survived ~ .,
                data = train,
                method = "ada",
                trControl = ctrl,
                tuneGrid = expand.grid(
                  iter = c(100, 500, 1000),
                  maxdepth = c(3, 5, 7),
                  nu = seq(10 ^ -6, 10 ^ -1, length.out = 5)
                ), 
                preProc = c("BoxCox", "center", "scale")
)
model5
plot(model5)
# nu          maxdepth  iter  ROC        Sens       Spec     
# 0.000001    7         100   0.8923767  0.9063568  0.7237589
# 0.02500075  7          100  0.8897037  0.8958988  0.7364362

fitted5 <- predict(model5, test, type = "prob")
pred5 <- ROCR::prediction(fitted5[,2], test$Survived)
perf5 <- ROCR::performance(pred5, "tpr", "fpr")
auc5 <- performance(pred5, "auc")@y.values %>% as.numeric()

cm5 <- caret::confusionMatrix(data = predict(model5, test), 
                              reference = test$Survived,
                              positive = "Yes")
cm5
auc5
# balacc 0.7877   auc 0.847








set.seed(1863)
model6 <- train(Survived ~ .,
                data = train,
                method = "ada",
                trControl = ctrl,
                tuneGrid = expand.grid(
                  iter = c(75, 100, 200),
                  maxdepth = c(5, 7, 9),
                  nu = seq(10 ^ -6, 10 ^ -2, length.out = 5)
                ), 
                preProc = c("BoxCox", "center", "scale")
)
model6
plot(model6)
# nu        maxdepth  iter  ROC        Sens       Spec     
# 0.010000  9         75    0.8900153  0.9193438  0.7070922

caret::confusionMatrix(data = predict(model6, test), 
                       reference = test$Survived,
                       positive = "Yes")






set.seed(1863)
model7 <- train(Survived ~ .,
                data = train,
                method = "ada",
                trControl = ctrl,
                tuneGrid = expand.grid(
                  iter = c(100, 200, 500, 1000),
                  maxdepth = c(5, 7, 10),
                  nu = 0.01
                ), 
                preProc = c("BoxCox", "center", "scale")
)
model7
plot(model7)
# nu        maxdepth  iter  ROC        Sens       Spec     
# 0.010000  7         100   0.8878048  0.9220437  0.7071809

fitted7 <- predict(model7, test, type = "prob")
pred7 <- ROCR::prediction(fitted7[,2], test$Survived)
perf7 <- ROCR::performance(pred7, "tpr", "fpr")
auc7 <- performance(pred7, "auc")@y.values %>% as.numeric()

cm7 <- caret::confusionMatrix(data = predict(model7, test), 
                              reference = test$Survived,
                              positive = "Yes")
cm7
auc7
# balacc 0.7925   auc 0.8466



set.seed(1863)
model9 <- train(Survived ~ .,
                data = train,
                method = "treebag",
                trControl = ctrl,
                preProc = c("BoxCox", "center", "scale")
)
model9
plot(model9)
# ROC        Sens       Spec     
# 0.8706349  0.872488  0.7282801

caret::confusionMatrix(data = predict(model9, test), 
                       reference = test$Survived,
                       positive = "Yes")




set.seed(1863)
model10 <- train(Survived ~ .,
                data = train,
                method = "ranger",
                trControl = ctrl,
                tuneGrid = expand.grid(
                  mtry = seq(2, 10, by=2),
                  splitrule = c("gini", "extratrees", "hellinger"),
                  min.node.size = c(1, 3, 5, 7)
                ), 
                preProc = c("BoxCox", "center", "scale")
)
model10
plot(model10)
# mtry  min.node.size  ROC        Sens       Spec     
# 6      5             0.8800839  0.8934381  0.7197695

fitted10 <- predict(model10, test, type = "prob")
pred10 <- ROCR::prediction(fitted10[,2], test$Survived)
perf10 <- ROCR::performance(pred10, "tpr", "fpr")
auc10 <- performance(pred10, "auc")@y.values %>% as.numeric()

cm10 <- caret::confusionMatrix(data = predict(model10, test), 
                              reference = test$Survived,
                              positive = "Yes")
cm10
auc10
# balacc 0.8029   auc 0.846












# SUPPORT VECTOR MACHINES ######################################################

# Define parameters
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  search = "grid"
)

# svmGrid <- expand.grid(C = 10^seq(-6, 2, by = 1),
#                        sigma = 10^seq(-6, 2, by = 1))
svmGrid <- expand.grid(C = 10^seq(-6, 2, by = 1),
                       sigma = seq(10^-6, 0.5, length.out = 10))


# Define and save model
set.seed(1863)
model8 <- train(
  Survived ~ .,
  data = train,
  method = "svmRadial", 
  trControl = ctrl,
  tuneGrid = svmGrid,
  preProcess = c("center", "scale")
)
model8
plot(model8)
# C     sigma       ROC        Sens       Spec     
# 1.00  0.05555644  0.8713406  0.8986671  0.7238475

confusionMatrix(data = (predict(model8, newdata = test)), 
                reference = test$Survived, 
                positive = "Yes")
# Acc 0.8209  Sens 0.6990  Kappa 0.6116






# BOOSTING #####################################################################

gbm_ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  search = "grid"
)

gbm_grid <- expand.grid(
  n.trees = (1:50)*100,
  interaction.depth = 2:5, 
  shrinkage = c(0.1, 0.01),
  n.minobsinnode = c(1, 3, 5)
)
  # Tuning parameters:
  # 
  # n.trees (# Boosting Iterations)
  #   interaction.depth (Max Tree Depth)
  #   shrinkage (Shrinkage)
  #   n.minobsinnode (Min. Terminal Node Size)
    
    
set.seed(1863)
model11 <- train(Survived ~ .,
                 data = train,
                 method = "gbm",
                 trControl = gbm_ctrl,
                 tuneGrid = gbm_grid, 
                 preProc = c("BoxCox", "center", "scale"),
                 verbose = F
)
model11
plot(model11)
# shrinkage  interaction.depth  n.minobsinnode  n.trees  ROC        Sens       Spec      
# 0.1                  5         1                  600  0.8770054 0.8542379 0.7408688
# 0.1                  4         1                 1000  0.8783283 0.8541695 0.7365248
# 0.01                 5         1                 4800  0.8777582 0.8438141 0.7407801
# 0.01                 5         3                  800  0.895163  0.890704  0.7491135 

fitted11 <- predict(model11, test, type = "prob")
pred11 <- ROCR::prediction(fitted11[,2], test$Survived)
perf11 <- ROCR::performance(pred11, "tpr", "fpr")
auc11 <- performance(pred11, "auc")@y.values %>% as.numeric()

cm11 <- caret::confusionMatrix(data = predict(model11, test), 
                               reference = test$Survived,
                               positive = "Yes")
cm11
auc11
# balacc 0.7926   auc 0.855

