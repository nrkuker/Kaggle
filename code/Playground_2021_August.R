# Kaggle Competition - August Playground
# Nicole Kuker
# 17 Aug 2021


# PREP #########################################################################
# Packages
pacman::p_load(
  pacman, rio, tidyverse, magrittr, janitor,  # general stuff
  psych,        # EDA
  visdat,       # missingness
  data.table,   # working with data.tables
  mice,         # missing value imputation
  rsample,      # data partition
  caret,        # general ML
  corrplot,     # correlation plot
  FactoMineR,   # EDA, PCA, MFA
  factoextra,   # extract and visualize PCA/MFA
  nFactors,     # how many factors/components to retain
  cluster,      # clustering algorithms
  NbClust,      # number of clusters
  clValid       # ?
)


# Data
data <- import("data/august_playground_train.csv")
glimpse(data)





# EDA / CLEANING ###############################################################

sum(is.na(data))
# no missing

# make features & response vectors
Y <- data.frame(loss = data$loss)
X <- data %>% dplyr::select(f0:f99) %>% as.data.frame()

options(scipen = 999)
psych::describe(X)



# corr plot
c <- cor(X)

corrplot(c, method = "color",
         # addCoef.col = T, number.digits = 2, number.cex = 0.5,
         type = "full",
         diag = F,
         addgrid.col = "darkgrey",
         # mar = c(0,0,2,0),
         # title = "Variable Correlation Plot",
         # order  = "FPC",
         # hclust.method = "ward.D2",
         addrect = 6,
         tl.col = "black",
         tl.cex = 0.8,
         tl.srt = 45
)
text(x = 12.5, y = 30, font = 2, col = "black", cex = 1.2,
     labels = "Variable Correlation Plot")



# making a list of correlations
cor_df <- tibble(var1 = rep(NA_character_, nrow(c)),
                 var2 = rep(NA_character_, ncol(c)),
                 cor = rep(NA_real_, ncol(c)))

count <- 1
for (i in 1:(ncol(c)-1)) {
  for (j in (i+1):ncol(c)) {
    cor_df[count, 1] <- rownames(c)[i]
    cor_df[count, 2] <- colnames(c)[j]
    cor_df[count, 3] <- c[i, j]
    count <- count+1
  }
}

cor_df %>% 
  # filter(cor > 0.4) %>%
  arrange(-abs(cor))




# Summary of EDA findings:
  # no missings
  # essentially orthogonal




# scale vars
X <- X %>% scale()
X[1:10, 1:10]
psych::describe(X)




# PCA ##########################################################################

# RUN THE PCA

pca <- prcomp(X, 
              center = TRUE,
              scale. = TRUE)
summary(pca)



# add labels to PCA results
pca_label <- cbind(X, as.data.frame(pca$x))
head(pca_label)




# ANALYZING THE RESULTS

# get eigenvalues
eig.val <- get_eigenvalue(pca)
eig.val %>% round(., 2)

# screeplot
fviz_eig(pca, ncp=50,
         choice = "variance", 
         addlabels = T) + 
  labs(title = "Scree Plot of Variance Explained by PCA Dimension") +
  theme(plot.title = element_text(hjust=0.5, size=14, face="bold"))
fviz_eig(pca,  ncp=50,
         choice = "eigenvalue",
         addlabels = T) + 
  labs(title = "Scree Plot of Eigenvalues by PCA Dimension") +
  theme(plot.title = element_text(hjust=0.5, size=14, face="bold")) # 590x590

# estiamte number of factors
nfactors(X, n = 50)
  # between 25-30?



#compute variance
pr_var <- (pca$sdev)^2

#check variance of first 10 components
pr_var

#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
cumulvarex <- data.frame(PCs = seq(1,length(prop_varex),1),
                         cumulvar = cumsum(prop_varex)) 

ggplot(cumulvarex, aes(PCs, cumulvar)) + theme_bw() + 
  geom_hline(yintercept = 0.7, linetype = "longdash", color = "grey40", lwd = 1) + 
  scale_x_continuous(breaks = seq(0, 100, 2)) + 
  scale_y_continuous(breaks = seq(0, 1, 0.1)) + 
  geom_line(color = "#85200C", lwd = 1.25) + #geom_point(color = "#85200C", size = 2) + 
  labs(title = "Cumulative Variance Explained", 
       subtitle = "by Number of Principal Components Retained",
       x = "Principal Components", 
       y = "Cumulative Proportion of Variance Explained") + 
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, face = "bold", size = 14))


# PCA findings:
  # features were already fairly orthogonal, reducing dimensions probably won't help?




######