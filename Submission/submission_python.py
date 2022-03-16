# === What is this Project? ===

# This is my approach on the "Titanic - Machine Learning from Disaster" machine learning competition.
# The objective here is to try and predict if a passenger survived or not the Titanic disaster (more on that on the next section).

# This script will generate a .csv file that can be submitted on the competition, here: https://www.kaggle.com/c/titanic/submit

# If you want to check out the notebook with EDA and Feature Engineering explained, go here: https://github.com/augusto-gontijo/ml-titanic/tree/main/Notebook

# Author: Augusto Gontijo
# LinkedIn: https://www.linkedin.com/in/augusto-gontijo/?locale=en_US
# GitHub: https://github.com/augusto-gontijo


# === Data import ===

# IMPORTANT: Don't forget to change to your computer path:

train_raw = pd.read_csv("C:/... YOUR COMPUTER PATH .../Data/train.csv")
test_raw = pd.read_csv("C:/... YOUR COMPUTER PATH .../Data/test.csv")

test_id = test_raw['PassengerId']

# === Lib Imports ===

# Data manipulation libs
import pandas as pd
import numpy as np

# Data visualization libs
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white")
sns.set_style('whitegrid')

# Machine Learining classification libs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Metrics libs
from sklearn.metrics import accuracy_score

from sklearn.dummy import DummyClassifier

# Dataset split lib
from sklearn.model_selection import train_test_split

# Cross validation libs:
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# Encoding lib
from sklearn.preprocessing import OneHotEncoder

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



# === Feature Engineering ===

# Creating a copy of the datatsets:
train = train_raw.copy()
test = test_raw.copy()

# List containing both datasets:
list_DS = [train, test]

# === Dealing with missing data === 

# Creating a new binary column named 'cabin_bin':
train["cabin_bin"] = (train["Cabin"].notnull().astype('int'))
test["cabin_bin"] = (test["Cabin"].notnull().astype('int'))

# Using the "S" category to fill "Embarked" NA's and using 28 (median) to fill "Age" NA's
na_values = {"Embarked": "S",
             "Age": 28.0}

train.fillna(value = na_values, inplace = True)

# Using the "S" category to fill "Embarked" NA's and using 27 (median) to fill "Age" NA's
na_values = {"Fare": 35.0,
             "Age": 27.0}

test.fillna(value = na_values, inplace = True)

# === Transforming Categorical variables to binary ===

# Adding the 'child' column
for dataset in list_DS:

    # If the passenger's 'Age' were < 18, then child = 1
    dataset["child"] = 0

    # If the passenger had 0 siblings/spouses and 0 parent/children aboard, then alone = 1
    for i in range(len(dataset["PassengerId"])):
        if dataset["Age"][i] < 18:
            dataset["child"][i] = 1        
        else:
            pass


# Adding the 'alone' variable
for dataset in list_DS:

    # Creating a new column 'alone' with only 0's on both datasets:
    dataset["alone"] = 0

    # If the passenger had 0 siblings/spouses and 0 parent/children aboard, then alone = 1
    for i in range(len(dataset["PassengerId"])):
        if dataset["SibSp"][i] == 0 and dataset["Parch"][i] == 0:
            dataset["alone"][i] = 1        
        else:
            pass


# Adding the 'relatives' variable
for dataset in list_DS:

    # Creating a new column 'relatives' with only 0's on both datasets:
    dataset["relatives"] = 0

    # The 'relatives' value will be the sum of siblings/spouses and parent/children:
    for i in range(len(dataset["PassengerId"])):
        dataset["relatives"][i] = dataset["SibSp"][i] + dataset["Parch"][i]  


# Converting 'sex' to binary

# Creating a function that returns 0 for 'female' and 1 for 'male':
def sex_binary(value):
    if value == 'female':
        return 0
    else:
        return 1

# Applying the function on "train" dataset:
# Using the 'map()' function do apply the function above to all values in the 'Sex' column:
train['sex_bin'] = train['Sex'].map(sex_binary)

# Applying the function on "test" dataset:
# Using the 'map()' function do apply the function above to all values in the 'Sex' column:
test['sex_bin'] = test['Sex'].map(sex_binary)


# Converting 'embarked' to binary

# Creating a function that returns 0 for 'C', 1 for 'Q' and 2 for 'S':
def embarked_binary(value):
    if value == 'C':
        return 0
    elif value == 'Q':
        return 1
    else:
        return 2

# Applying the function on "train" dataset:
# Using the 'map()' function do apply the function above to all values in the 'Sex' column:
train['embarked_bin'] = train['Embarked'].map(embarked_binary)

# Applying the function on "test" dataset:
# Using the 'map()' function do apply the function above to all values in the 'Sex' column:
test['embarked_bin'] = test['Embarked'].map(embarked_binary)


# Creating new encoded columns on the "train" dataset:
# Creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

# Passing "embarked_bin" column (label encoded values of "Embarked")
enc_df = pd.DataFrame(enc.fit_transform(train[['embarked_bin']]).toarray())

# merge with main df bridge_df on key values
train = train.join(enc_df)

# Renaming the new columns:
new_names = {0: "embarked_c", 1: "embarked_q", 2: "embarked_s"}
train.rename(columns = new_names, inplace = True)


# Creating new encoded columns on the "test" dataset:
# Creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

# Passing "embarked_bin" column (label encoded values of "Embarked")
enc_df = pd.DataFrame(enc.fit_transform(test[['embarked_bin']]).toarray())

# merge with main df bridge_df on key values
test = test.join(enc_df)

# Renaming the new columns:
new_names = {0: "embarked_c", 1: "embarked_q", 2: "embarked_s"}
test.rename(columns = new_names, inplace = True)


# Feature Engineering the 'Fare' variable:

# Creating the 'fare_cat' variable on the train dataset:
train['fare_cat'] = pd.qcut(train['Fare'], 5, labels = [1, 2, 3, 4, 5])
train['fare_cat'] = train['fare_cat'].astype(int)

# Creating the 'fare_cat' variable on the test dataset:
test['fare_cat'] = pd.qcut(test['Fare'], 5, labels = [1, 2, 3, 4, 5])
test['fare_cat'] = test['fare_cat'].astype(int)



# Feature Engineering the 'Name' variable:

list_DS = [train, test]

# Creating the 'title' column on both datasets, based on the title extraction from 'Name'
for dataset in list_DS:
    dataset['title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = True)


for dataset in list_DS:

    # Creating 5 new columns based on custom title groups:
    dataset["navy"] = 0
    dataset["scholar"] = 0
    dataset["vip"] = 0
    dataset["women"] = 0
    dataset["men"] = 0

    # Looping and assigning each title to its group:
    for i in range(len(dataset["PassengerId"])):
        if dataset["title"][i] in ['Col', 'Capt', 'Major']:
            dataset["navy"][i] = 1        
        
        elif dataset["title"][i] in ['Dr', 'Master']:
            dataset["scholar"][i] = 1

        elif dataset["title"][i] in ['Countess', 'Don', 'Dona', 'Jonkheer', 'Lady', 'Sir']:
            dataset["vip"][i] = 1

        elif dataset["title"][i] in ['Miss', 'Mlle', 'Mme', 'Mrs', 'Ms']:
            dataset["women"][i] = 1

        elif dataset["title"][i] in ['Mr', 'Rev']:
            dataset["men"][i] = 1
        
        else:
            pass



# Feature Engineering the 'Ticket' variable:
for dataset in list_DS:

    # Creating 8 new columns based on custom ticket groups (both datasets):
    dataset["ticket_A"] = 0
    dataset["ticket_C"] = 0
    dataset["ticket_F"] = 0
    dataset["ticket_P"] = 0
    dataset["ticket_S"] = 0
    dataset["ticket_W"] = 0
    dataset["ticket_N"] = 0


    # Looping and assigning each title to its group:
    for i in range(len(dataset["PassengerId"])):
        if dataset["Ticket"][i].startswith('A'):
            dataset["ticket_A"][i] = 1   

        elif dataset["Ticket"][i].startswith('C'):
            dataset["ticket_C"][i] = 1

        elif dataset["Ticket"][i].startswith('F'):
            dataset["ticket_F"][i] = 1

        elif dataset["Ticket"][i].startswith('P'):
            dataset["ticket_P"][i] = 1

        elif dataset["Ticket"][i].startswith('S'):
            dataset["ticket_S"][i] = 1

        elif dataset["Ticket"][i].startswith('W'):
            dataset["ticket_W"][i] = 1
        
        else:
            dataset["ticket_N"][i] = 1 



# Removing Useless Columns:

# Setting the column names that will be removed:
columns_to_remove = ["PassengerId", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked", "embarked_bin", "title"]

# Removing the useless columns from both datasets:
train.drop(columns = columns_to_remove, inplace = True)
test.drop(columns = columns_to_remove, inplace = True)




# === Building the Model === 

# Setting the seed:
np.random.seed(0)

# Setting the features:
features = ['Pclass', 'cabin_bin', 'child', 'alone', 'relatives',
            'sex_bin', 'embarked_c', 'embarked_q', 'embarked_s', 'fare_cat',
            'navy', 'scholar', 'vip', 'women', 'men',
            'ticket_A', 'ticket_C', 'ticket_F', 'ticket_P', 
            'ticket_S', 'ticket_W', 'ticket_N']


# Dataset that contains only the features:
x_train = train[features]

# Dataset that contains the targets:
y_train = train['Survived']

# RANDOM FOREST
# Creating Random Forest instance 
RFC = RandomForestClassifier()

# Setting the cross validation method
kfold = StratifiedKFold(n_splits = 10)

# Creating a dict with some parameters
rf_param_grid = {"max_depth": [None],
                 "max_features": [1, 3, 10],
                 "min_samples_split": [2, 3, 10],
                 "min_samples_leaf": [1, 3, 10],
                 "bootstrap": [False],
                 "n_estimators": [100, 300],
                 "criterion": ["gini"]
                 }

# Creating the Grid Search instance (this will iterate through the parameters above and find the optimal combination)
gsRFC = GridSearchCV(RFC,
                     param_grid = rf_param_grid, 
                     cv = kfold, 
                     scoring = "accuracy", 
                     n_jobs = 4, 
                     verbose = 1)

# Fitting the Grid Search with our data
gsRFC.fit(x_train, y_train)

# Getting the best estimator (with the optimal parameter combination)
RFC_best = gsRFC.best_estimator_

# Getting the best accuracy score
RFC_score = (gsRFC.best_score_ * 100).round(2)


# DECISION TREE
# Creating Decision Tree instance 
DTC = DecisionTreeClassifier()

# Setting the cross validation method
kfold = StratifiedKFold(n_splits = 10)

# Creating a dict with some parameters
dtc_param_grid = {"max_depth": [2, 4, 6, 8, 10, 12],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],              
              "criterion": ['gini', 'entropy']}

# Creating the Grid Search instance (this will iterate through the parameters above and find the optimal combination)
gsDTC = GridSearchCV(DTC,
                     param_grid = dtc_param_grid, 
                     cv = kfold, 
                     scoring = "accuracy", 
                     n_jobs = 4, 
                     verbose = 1)

# Fitting the Grid Search with our data
gsDTC.fit(x_train, y_train)

# Getting the best estimator (with the optimal parameter combination)
DTC_best = gsDTC.best_estimator_

# Getting the best accuracy score
DTC_score = (gsDTC.best_score_ * 100).round(2)


# XGBOOST
# Creating XGB instance 
XGB = XGBClassifier()

# Setting the cross validation method
kfold = StratifiedKFold(n_splits=10)

# Creating a dict with some parameters
xgb_params = {'min_child_weight': [1, 5, 10],
              'gamma': [0.5, 1, 1.5, 2, 5],
              'subsample': [0.6, 0.8, 1.0],
              'colsample_bytree': [0.6, 0.8, 1.0],
              'max_depth': [3, 4, 5]
             }

# Creating the Grid Search instance (this will iterate through the parameters above and find the optimal combination)
gsXGB = GridSearchCV(XGB,
                     param_grid = xgb_params, 
                     cv = kfold, 
                     scoring = "accuracy", 
                     n_jobs = 4, 
                     verbose = 1)

# Fitting the Grid Search with our data
gsXGB.fit(x_train, y_train)

# Getting the best estimator (with the optimal parameter combination)
XGB_best = gsXGB.best_estimator_

# Getting the best accuracy score
XGB_score = (gsXGB.best_score_ * 100).round(2)


# LOGISTIC REGRESSION
# Creating Logistic Regression instance 
LRC = LogisticRegression()

# Setting the cross validation method
kfold = StratifiedKFold(n_splits = 10)

# Creating a dict with some parameters
lrc_params = {'penalty': ['l1', 'l2'],
              'C': np.logspace(-4, 4, 50),
              'max_iter': [100, 200, 300],
              'fit_intercept': [True, False]
             }

# Creating the Grid Search instance (this will iterate through the parameters above and find the optimal combination)
gsLRC = GridSearchCV(LRC,
                     param_grid = lrc_params, 
                     cv = kfold, 
                     scoring = "accuracy", 
                     n_jobs = 4, 
                     verbose = 1)

# Fitting the Grid Search with our data
gsLRC.fit(x_train, y_train)

# Getting the best estimator (with the optimal parameter combination)
LRC_best = gsLRC.best_estimator_

# Getting the best accuracy score
LRC_score = (gsLRC.best_score_ * 100).round(2)



# GRADIENT BOOSTING CLASSIFIER
# Creating GBC instance 
GBC = GradientBoostingClassifier()

# Setting the cross validation method
kfold = StratifiedKFold(n_splits=10)

# Creating a dict with some parameters
gb_param_grid = {'loss': ["deviance"],
                 'n_estimators': [100, 200, 300],
                 'learning_rate': [0.1, 0.05, 0.01],
                 'max_depth': [4, 8],
                 'min_samples_leaf': [100, 150],
                 'max_features': [0.3, 0.1] 
                 }

# Creating the Grid Search instance (this will iterate through the parameters above and find the optimal combination)
gsGBC = GridSearchCV(GBC,
                     param_grid = gb_param_grid, 
                     cv = kfold, 
                     scoring = "accuracy", 
                     n_jobs = 4, 
                     verbose = 1
                     )

# Fitting the Grid Search with our data
gsGBC.fit(x_train, y_train)

# Getting the best estimator (with the optimal parameter combination)
GBC_best = gsGBC.best_estimator_

# Getting the best accuracy score
GBC_score = (gsGBC.best_score_ * 100).round(2)



# === Ensemble Modeling ===
voting_classifier = VotingClassifier(estimators=[('rfc', RFC_best), 
                                                 ('dtc', DTC_best),
                                                 ('xgb', XGB_best), 
                                                 ('lrc', LRC_best),
                                                 ('gbc', GBC_best)], 
                           voting = 'soft', 
                           n_jobs = 4)

votingC = voting_classifier.fit(x_train, y_train)



## Predicting and creating the submission file
test_predict = pd.Series(voting_classifier.predict(test), name = "Survived")

results = pd.concat([test_id, test_predict],
                    axis = 1)

results.to_csv("titanic_submission.csv", index = False)