setwd("C:/Users/Bryan/Desktop/PortoSeguro")

#-------------------------------------------------------------------
# --------------------------IMPORT PACKAGES-------------------------
# ------------------------------------------------------------------

library(mlbench)
library(mboost)
library(plyr)
library(import)
library(robustbase)
library(Amelia)
library(mice)
library(DMwR)
library(gridExtra)
library(rpart)
library(randomForest)
library(caret)
library(corrplot)
library(dplyr)
library(ggplot2)

#-------------------------------------------------------------------
# --------------------------GINI FUNCTION---------------------------
# ------------------------------------------------------------------
# normalized gini function taked from:
# https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703
normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}


#-------------------------------------------------------------------
# -------------------IMPORT DONNEES ET FUSION-----------------------
# ------------------------------------------------------------------

train <- read.csv(file= "train.csv", header=T, stringsAsFactors = F)
test <- read.csv(file = "test.csv", header=T, stringsAsFactors = F)
sample <- read.csv(file = "sample_submission.csv", header=T, stringsAsFactors = F)


glimpse(train)
glimpse(test)
glimpse(sample)

#Merging datasets
test$IsTargeted <- F
train$IsTargeted <- T

#Add same feature as titanic.train
test$target <- NA

#fusion
full <- rbind(train, test)

qplot(ps_ind_03, data=train, bins=10)
qplot(ps_ind_15, data=train, bins=14)
qplot(ps_reg_03, data=train, bins=20)

#Plot la matrice de corrélation sur les features type numeric
WhichNums <- sapply(train, is.numeric)
Nums <- cor(train[WhichNums], use="pairwise.complete.obs") 
#pairwise.complete.obs : cet argument permet de faire le calcul seulement si il y a des données dans les deux variables
correlations <- corrplot(Nums, method="square", type = 'upper')
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(Nums, cutoff=0.7)
# print indexes of highly correlated attributes
print(highlyCorrelated)

#PAS DE NA, il faudra chercher les -1


#On resplit 
test <- full[full$IsTargeted == F,]
train <- full[full$IsTargeted == T,]

train$target <- as.factor(train$target)

# collect names of all categorical variables
cat_vars <- names(train)[grepl('_cat$', names(train))]

# turn categorical features into factors
train[, (cat_vars) := lapply(.SD, factor), .SDcols = cat_vars]
test[, (cat_vars) := lapply(.SD, factor), .SDcols = cat_vars]

# one hot encode the factor levels
dtrain <- model.matrix(~. - 1, data = train)
dtest <- model.matrix(~ . - 1, data = test)

# create index for train/test split
train_index <- sample(c(TRUE, FALSE), size = nrow(dtrain), replace = TRUE, prob = c(0.8, 0.2))

# perform x/y ,train/test split.
x_train <- dtrain[train_index, 3:ncol(dtrain)]
y_train <- as.factor(dtrain[train_index, 'target'])

x_test <- dtrain[!train_index, 3:ncol(dtrain)]
y_test <- as.factor(dtrain[!train_index, 'target'])

# Convert target factor levels to 0 = "No" and 1 = "Yes" to avoid this error when predicting class probs:
# https://stackoverflow.com/questions/18402016/error-when-i-try-to-predict-class-probabilities-in-r-caret
levels(y_train) <- c("No", "Yes")
levels(y_test) <- c("No", "Yes")


giniSummary <- function (data, lev = "Yes", model = NULL) {
  levels(data$obs) <- c('0', '1')
  out <- normalizedGini(as.numeric(levels(data$obs))[data$obs], data[, lev[2]])  
  names(out) <- "NormalizedGini"
  out
}

train <- setdiff(names(train), "ps_ind_14")
test <- setdiff(names(test), "ps_ind_14")

# create the training control object. Two-fold CV to keep the execution time under the kaggle
# limit. You can up this as your compute resources allow. 
trControl = trainControl(
  method = 'cv',
  number = 2,
  summaryFunction = giniSummary,
  classProbs = TRUE,
  verboseIter = TRUE,
  allowParallel = TRUE)

# create the tuning grid. Again keeping this small to avoid exceeding kernel memory limits.
# You can expand as your compute resources allow. 
tuneGridXGB <- expand.grid(
  nrounds=c(150),
  max_depth = c(4, 6),
  eta = c(0.01, 0.05, 0.1, 0.2),
  gamma = c(0.01),
  colsample_bytree = c(0.75),
  subsample = c(0.50),
  min_child_weight = c(0))

start <- Sys.time()

# train the xgboost learner
xgbmod <- train(
  x = x_train,
  y = y_train,
  method = 'xgbTree',
  metric = 'NormalizedGini',
  trControl = trControl,
  tuneGrid = tuneGridXGB)


print(Sys.time() - start)

# make predictions
preds <- predict(xgbmod, newdata = x_test, type = "prob")
preds_final <- predict(xgbmod, newdata = dtest, type = "prob")


# convert test target values back to numeric for gini and roc.plot functions
levels(y_test) <- c("0", "1")
y_test_raw <- as.numeric(levels(y_test))[y_test]

# Diagnostics
print(xgbmod$results)
print(xgbmod$resample)

# plot results (useful for larger tuning grids)
plot(xgbmod)

# score the predictions against test data
normalizedGini(y_test_raw, preds$Yes)

# plot the ROC curve
roc.plot(y_test_raw, preds$Yes, plot.thres = c(0.02, 0.03, 0.04, 0.05))

# prep the predictions for submissions
sub <- data.frame(id = as.integer(dtest[, 'id']), target = preds_final$Yes)

# write to csv
write.csv(sub, 'xgb_submission.csv', row.names = FALSE)
