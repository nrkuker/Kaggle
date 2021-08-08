# Kaggle Competition - Titanic
# Nicole Kuker
# 8 Aug 2021


# PREP #########################################################################
# Packages
pacman::p_load(
  pacman, rio, tidyverse, magrittr, janitor,  # general stuff
  psych,        # EDA
  visdat,       # missingness
  data.table,   # working with data.tables
  corrplot     # correlation plot
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

# make empty into missing
data$Embarked[data$Embarked==""] <- NA


# derive new vars
data %<>%
  mutate(
    CabinKnown = as_factor(ifelse(Cabin=="", "No", "Yes")),
    Survived = recode(Survived, `0` = "No", `1` = "Yes"),
    title = sapply(name, function(x) {
      str_sub(x,
              start = (str_locate(x, ",")[, 1] + 2),
              end = (str_locate(x, "\\.")[, 1] - 1))
    })
  )
names(data$title) <- NULL

str(data)



# make factors
data %<>% 
  mutate(across(c("Survived", "Pclass", "Sex", "Embarked"), as_factor))
str(data)

table(data$Name)


# working with name
name <- data$Name

str_locate(name, ",")[,1]+2
str_sub(name[1], 9)

no_sur <- 
  sapply(name, function(x) {
    str_sub(x, (str_locate(x, ",")[, 1] + 2))
  })

names(no_sur) <- NULL
no_sur

str_locate(no_sur, "\\.")[,1]
head(no_sur)
sapply(no_sur, function(x) {
  str_sub(x, end = (str_locate(x, "\\.")[, 1]-1))
})

title <- 
  sapply(name, function(x) {
    str_sub(x, 
            start = (str_locate(x, ",")[, 1] + 2),
            end = (str_locate(x, "\\.")[, 1] - 1))
  })
names(title) <- NULL
title

table(title)
