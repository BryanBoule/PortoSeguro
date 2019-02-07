# PortoSeguro
Predict if a driver will file an insurance claim next year in R using XGBoost.

Link : https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

## Description
build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year.

In the train and test data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder.

## Data


## Software
Rstudio.

## Usage
I did this competition to challenge myself on using more complex algorithms with specific constraints on datas. All the more as these ones are explicitely described. This is a regression problem. The normalized gini score is used to predict whether a client will file an insurance claim next year.

