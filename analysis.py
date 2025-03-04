# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:21:35 2025

@author: Paul
"""

# %% [1] Import Packages

import os
import pandas as pd  # data, plotting, etc...
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

import statsmodels.api as sm  # stats models for analysis

# import random  # for setting seed
from scipy.stats import randint, uniform


# import lazypredict  # ml packages
from sklearn.utils import shuffle
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.decomposition import KernelPCA
# from sklearn.svm import SVC
# import xgboost as xgb
from sklearn.naive_bayes import BernoulliNB  # new models to use
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Random Seed
# seed = random.randint(1000, 9999)
# print(seed)

seed = 3921

# Show all rows and columns
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 1000)  # Adjust width for long outputs

# %% [2] Background

'''
- We are one of the fastest growing startups in the logistics and delivery
domain. We work with several partners and make on-demand delivery to our
customers. From operational standpoint we have been facing several different
challenges and everyday we are trying to address these challenges.

- We thrive on making our customers happy. As a growing startup, with a
global expansion strategy we know that we need to make our customers
happy and the only way to do that is to measure how happy each customer
is. If we can predict what makes our customers happy or unhappy, we can then
take necessary actions.

- Getting feedback from customers is not easy either, but we do our best to
get constant feedback from our customers. This is a crucial function to
improve our operations across all levels.

- We recently did a survey to a select customer cohort. You are presented with
a subset of this data. We will be using the remaining data as a private test
set.

'''

# %% [3] Setup

'''
Y = target attribute (Y) with values indicating 0 (unhappy) and 1 (happy)
customers)
X1 = my order was delivered on time
X2 = contents of my order was as I expected
X3 = I ordered everything I wanted to order
X4 = I paid a good price for my order
X5 = I am satisfied with my courier
X6 = the app makes ordering easy for me
'''

# Set Dir
path_dir = r'E:\My Stuff\Projects\Apziva\pG736HzU7DLB8fEa'
os.chdir(path_dir)

# data
df = pd.read_csv("ACME-HappinessSurvey2020.csv")

# X, y
X, y = shuffle(df.iloc[:, 1:], df['Y'], random_state=seed)

# ml scaler
scaler = StandardScaler().set_output(transform="pandas")

# Seeds
seeds_to_run = ['1110', '1314', '1319', '1385', '1654', '2041', '2043', '2229',
                '2495', '2534', '2553', '2743', '2921', '2995', '3128', '3155',
                '3175', '3461', '3560', '3700', '3921', '4198', '4266', '4473',
                '4840', '5106', '5640', '5666', '6019', '6058', '6465', '7143',
                '7191', '7622', '7795', '7828', '8287', '8712', '8758', '8777',
                '8854', '8998', '9085', '9125', '9160', '9198', '9328', '9410',
                '9444', '9829'
                ]

# %% [4] Data Exploration

# Describe; mean, std, etc..
df.describe()

# Correlation
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Violin Plots
fig_violin, axes = plt.subplots(3, 2, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    sns.violinplot(x=df['Y'], y=df.iloc[:, i+1], ax=ax)
    ax.set_title(f"{df.columns[i+1]}")

    # ax.set_title(f"{df.columns[i+1]}", color='white')  # Set title color
    # ax.set_xlabel(df['Y'].name, color='white')  # Set x-label color
    # ax.set_ylabel(df.columns[i+1], color='white')  # Set y-label color
    # ax.tick_params(colors='white')  # Set tick label color

plt.tight_layout()

fig_violin.suptitle("Violin Plots")
plt.show()

# fig_violin.suptitle("Violin Plots", color='white')
# plt.savefig("violin_chart.png", transparent=True, dpi=300)


# %% [5] Logistic Regressions, Confusion Matrix, Etc..
'''
- run single regressions
- review results
- plot confusion matrix
'''


def run_single_regressions(y_df, x_df):
    results = []
    pred = []
    models = []

    for col in x_df.columns:
        X = df[col]
        X = sm.add_constant(X)

        # Fit the regression model
        model = sm.Logit(y_df, X).fit()
        models.append(model)

        # Predict
        pred.append(model.predict(X))

        # Collect the results
        results.append({
            'Variable': col,
            'Coefficient': model.params[1],
            'Intercept': model.params[0],
            'P-Value': model.pvalues[1],
            # 'R-Squared': model.rsquared
        })
    return results, pred, models


# Return results and setup for cm
results, pred, models = run_single_regressions(df['Y'], df[df.columns[1:]])

# Standardizing did nothing to improve accuracy
# results, pred, models = run_single_regressions(
#     df['Y'],
#     scaler.fit_transform(df[df.columns[1:]])
# )

results = pd.DataFrame(results)
pred = [[int(y >= 0.5) for y in x] for x in pred]
results['accuracy'] = list(map(lambda x: accuracy_score(df['Y'], x), pred))
results['description'] = ('''X1 = my order was delivered on time
X2 = contents of my order was as I expected
X3 = I ordered everything I wanted to order
X4 = I paid a good price for my order
X5 = I am satisfied with my courier
X6 = the app makes ordering easy for me ''').split('\n')

# Plot Confusion Matrix
fig_cm, axes = plt.subplots(3, 2, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    cm = confusion_matrix(df['Y'], pred[i])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred: 0', 'Pred: 1'],
                yticklabels=['True: 0', 'True: 1'], ax=ax
                )
    ax.set_title(f'{df.columns[i+1]}')

fig_cm.suptitle("Confusion Matrix Heatmap", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# %% [6] Initial Machine Learning Tests

# =============================================================================
# Lazypredict
# =============================================================================


def log_lz_results(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        try:
            with open(".log-lz.json", "r") as log_file:
                data = json.load(log_file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(e)
            data = {}

        data[seed] = {x: result[x].to_json() for x in result}

        df = args[0]

        data[seed]['.data_by_split'] = {x: df[x == df.loc[:, 'Split']].to_json(
            orient='index') for x in set(df['Split'])
            }

        # write to json file
        with open(".log-lz.json", "w") as log_lz:
            json.dump(data, log_lz, indent=4)

        return result

    return wrapper


@log_lz_results
def measure_lz_results(lzresults):
    # group by model, find mean and std
    lz_mean_by_level = lzresults.groupby(level=0).mean().drop(
        columns=['Split', 'Time Taken']
        )

    lz_std_by_level = lzresults.groupby(level=0).std().drop(
        columns=['Split', 'Time Taken']
        )

    # group by split, find mean and std
    lz_mean_by_split = lzresults.groupby('Split').mean().drop(
        columns=['Time Taken']
        )

    lz_std_by_split = lzresults.groupby('Split').std().drop(
        columns=['Time Taken']
        )

    dictionary = {'mean_by_level': lz_mean_by_level,
                  'std_by_level': lz_std_by_level,
                  'mean_by_split': lz_mean_by_split,
                  'std_by_split': lz_std_by_split
                  }

    return dictionary


# for seed in seeds_to_run:
    # seed = int(seed)
# lzresults = pd.DataFrame()
# for i in np.round(np.arange(0.2, 0.85, 0.05), 3):
#     # for i in np.round(np.arange(0.45, 0.65, 0.05), 3):

#     X, y = shuffle(df.iloc[:, 1:], df['Y'], random_state=seed)

#     X_train, X_test, y_train, y_test = train_test_split(X,
#                                                         y,
#                                                         test_size=i,
#                                                         random_state=seed
#                                                         )
#     clf = LazyClassifier(verbose=0, ignore_warnings=True,
#                          custom_metric=None
#                          )
#     lazymodels, predictions = clf.fit(X_train, X_test, y_train, y_test)
#     lazymodels = pd.DataFrame(lazymodels)
#     lazymodels["Split"] = i
#     lzresults = pd.concat([lzresults, lazymodels], ignore_index=False)

# lz_results_grouped = measure_lz_results(lzresults)

# %% [7A] Review Results - Apply Measures
'''
Averages and STD accross models and seeds, etc..
'''

# read log file
with open(".log-lz  (Splits 0.2 to 0.8).json", "r") as log_file:
    lz_log = json.load(log_file)

len(lz_log.keys())

# setup empty dict
measures = ['mean_by_level', 'std_by_level',
            'mean_by_split', 'std_by_split'
            ]

seed_results = {key: [] for key in measures}

#  consolidate each measure by seed -------------------------------------------
for measure in measures:
    for key in lz_log.keys():  # each seed
        json_data = json.loads(lz_log[key][measure])
        seed_results[measure].append(pd.DataFrame(json_data))

for key in seed_results.keys():
    seed_results[key] = pd.concat(seed_results[key], ignore_index=False)
    seed_results[key] = seed_results[key].groupby(level=0).mean()
# -----------------------------------------------------------------------------

# %% [7B] Review Results - Plot
'''
Plot results by split and seed.
Plot results by split accross seeds.
'''


# Graph Model Performance Accross Splits for a seed ---------------------------
# def plot_model_performance_by_split_seed(model, seed):
#     performance = {}

#     for key, value in lz_log[seed]['.data_by_split'].items():
#         performance[key] = json.loads(value)[model]['Accuracy']

#     plt.figure(figsize=(10, 6))

#     plt.plot(
#         performance.keys(),
#         performance.values(),
#         label="Data 1",
#         marker='o'
#         )

#     # Fix the y-axis scale to be between 0 and 1
#     plt.ylim(0, 1)

#     plt.title(f'{model} Accuracy by Split for Seed: {seed}')
#     plt.show()


# for key in seeds_to_run:
#     plot_model_performance_by_split_seed('DecisionTreeClassifier', key)
# -----------------------------------------------------------------------------

# Graph Model Performance Accross Splits Accross Seeds ------------------------
# model = 'RandomForestClassifier'

# model_list = [
#     'AdaBoostClassifier', 'BaggingClassifier', 'BernoulliNB',
#     'CalibratedClassifierCV', 'DecisionTreeClassifier', 'DummyClassifier',
#     'ExtraTreeClassifier', 'ExtraTreesClassifier', 'GaussianNB',
#     'KNeighborsClassifier', 'LGBMClassifier', 'LabelPropagation',
#     'LabelSpreading', 'LinearDiscriminantAnalysis', 'LinearSVC',
#     'LogisticRegression', 'NearestCentroid', 'NuSVC',
#     'PassiveAggressiveClassifier', 'Perceptron', 'QuadraticDiscriminantAnalysis',
#     'RandomForestClassifier', 'RidgeClassifier', 'RidgeClassifierCV',
#     'SGDClassifier', 'SVC', 'XGBClassifier'
#     ]


# for model in model_list:
#     df_performance = pd.DataFrame(columns=['split', 'accuracy'])
#     for key in seeds_to_run:
#         for split, value in lz_log[key]['.data_by_split'].items():
#             get_accuracy = {
#                 'split': split,
#                 'accuracy': json.loads(value)[model]['Accuracy']
#                 }
#             get_accuracy = pd.Series(get_accuracy)

#             df_performance = df_performance.append(
#                 get_accuracy,
#                 ignore_index=True
#                 )

#     plt.figure(figsize=(10, 6))

#     sns.violinplot(
#         x=df_performance['split'],
#         y=df_performance['accuracy']
#         )

#     # Fix the y-axis scale to be between 0 and 1
#     plt.ylim(0, 1)

#     plt.title(f'{model} Accuracy by Split Accross Seeds')
#     plt.show()
# # -----------------------------------------------------------------------------

# %% [8] Machine Learning Model Implementation

# =============================================================================
# Models
# =============================================================================
'''
Run Top Models
--------------
BernoulliNB - This seems to just take a naive strategy approach.
LGBMClassifier
XGBClassifier
RandomForestClassifier
'''

# X = X.loc[:,['X1','X5','X6']]

# Split at 0.55
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.55,
                                                    random_state=seed
                                                    )

y_train.mean()
y_test.mean()

# Variables
Fitted_Models = {}
n_cv = 10
num_iter = 200

# =============================================================================
# Models with GridSearch
# =============================================================================

# # BernoulliNB -----------------------------------------------------------------
# param_grid = {'alpha': np.round(np.arange(0.5, 1.05, 0.05), 3)}

# model = GridSearchCV(
#     estimator=BernoulliNB(),
#     param_grid=param_grid,
#     cv=n_cv,
#     n_jobs=-1,
#     return_train_score=True,
#     random_state=seed
#     )

# model.fit(X_train, y_train)
# Fitted_Models["BernoulliNB"] = model
# # -----------------------------------------------------------------------------

# # LGBMClassifier --------------------------------------------------------------
# param_grid = {
#     'num_leaves': [3],
#     'max_depth': [3],
#     'learning_rate': [0.001],
#     'n_estimators': [50],
#     # 'min_child_samples': [10, 20, 50],
#     # 'subsample': [0.6, 0.8, 1.0],
#     # 'colsample_bytree': [0.6, 0.8, 1.0],
#     # 'reg_alpha': [0, 0.1, 1.0],
#     # 'reg_lambda': [0, 0.1, 1.0]
#     'random_state': [seed]
# }

# model = GridSearchCV(
#     estimator=LGBMClassifier(),
#     param_grid=param_grid,
#     cv=n_cv,
#     n_jobs=-1,
#     return_train_score=True,
#     random_state=seed
#     )

# model.fit(X_train, y_train)
# Fitted_Models["LGBMClassifier"] = model
# # -----------------------------------------------------------------------------

# # XGBClassifier ---------------------------------------------------------------
# param_grid = {
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [300],
#     # 'colsample_bytree': [0.8, 1.0],
#     # 'gamma': [0, 0.1, 0.2],
#     # 'reg_alpha': [0, 0.1, 1],
#     # 'reg_lambda': [0.1, 1, 10],
# }


# model = GridSearchCV(
#     estimator=XGBClassifier(),
#     param_grid=param_grid,
#     cv=n_cv,
#     n_jobs=-1,
#     return_train_score=True,
#     random_state=seed
#     )

# model.fit(X_train, y_train)
# Fitted_Models["XGBClassifier"] = model
# Fitted_Models["XGBClassifier"].best_params_
# # -----------------------------------------------------------------------------

# # RandomForestClassifier ------------------------------------------------------
# param_grid = {
#     'n_estimators': [150],
#     'max_depth': [3],
#     'min_samples_split': [5],
#     'min_samples_leaf': [5],
# }

# model = GridSearchCV(
#     estimator=RandomForestClassifier(),
#     param_grid=param_grid,
#     cv=n_cv,
#     n_jobs=-1,
#     return_train_score=True,
#     random_state=seed
#     )

# model.fit(X_train, y_train)
# Fitted_Models["RandomForestClassifier"] = model
# Fitted_Models["RandomForestClassifier"].best_params_
# # -----------------------------------------------------------------------------

# # NearestCentroid -------------------------------------------------------------
# param_grid = {
#     'metric': ['euclidean', 'manhattan', 'cosine'],
#     'shrink_threshold': [None, 0.0, 0.1, 0.5, 1.0],
# }

# model = GridSearchCV(
#     estimator=NearestCentroid(),
#     param_grid=param_grid,
#     cv=n_cv,
#     n_jobs=-1,
#     return_train_score=True,
#     random_state=seed
# )

# model.fit(X_train, y_train)
# Fitted_Models["NearestCentroid"] = model
# # -----------------------------------------------------------------------------

# =============================================================================
# Models with RandomizedSearchCV
# =============================================================================

# BernoulliNB -----------------------------------------------------------------
param_dist = {'alpha': uniform(0.1, 1.0)}

model = RandomizedSearchCV(
    estimator=BernoulliNB(),
    param_distributions=param_dist,
    n_iter=num_iter,
    cv=n_cv,
    n_jobs=-1,
    return_train_score=True,
    random_state=seed
)

model.fit(X_train, y_train)
Fitted_Models["BernoulliNB"] = model
# -----------------------------------------------------------------------------

# LGBMClassifier --------------------------------------------------------------
param_dist = {
    'num_leaves': randint(20, 100),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.001, 0.1),
    'n_estimators': randint(50, 200),
    'random_state': [seed]
}

model = RandomizedSearchCV(
    estimator=LGBMClassifier(),
    param_distributions=param_dist,
    n_iter=num_iter,
    cv=n_cv,
    n_jobs=-1,
    return_train_score=True,
    random_state=seed
)

model.fit(X_train, y_train)
Fitted_Models["LGBMClassifier"] = model
# -----------------------------------------------------------------------------

# XGBClassifier ---------------------------------------------------------------
param_dist = {
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(100, 500),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.1, 1),
    'gamma': uniform(0, 0.3),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

model = RandomizedSearchCV(
    estimator=XGBClassifier(),
    param_distributions=param_dist,
    n_iter=num_iter,
    cv=n_cv,
    n_jobs=-1,
    return_train_score=True,
    random_state=seed,
)

model.fit(X_train, y_train)
Fitted_Models["XGBClassifier"] = model
# -----------------------------------------------------------------------------

# RandomForestClassifier ------------------------------------------------------
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 15),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy', 'log_loss']
}


model = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=num_iter,
    cv=n_cv,
    n_jobs=-1,
    return_train_score=True,
    random_state=seed
)

model.fit(X_train, y_train)
Fitted_Models["RandomForestClassifier"] = model
# -----------------------------------------------------------------------------

# NearestCentroid -------------------------------------------------------------
param_dist = {
    'metric': ['euclidean', 'manhattan', 'cosine'],
    'shrink_threshold': uniform(0.0, 1.0)
}

model = RandomizedSearchCV(
    estimator=NearestCentroid(),
    param_distributions=param_dist,
    n_iter=num_iter,
    cv=n_cv,
    n_jobs=-1,
    return_train_score=True,
    random_state=seed
)

model.fit(X_train, y_train)
Fitted_Models["NearestCentroid"] = model
# -----------------------------------------------------------------------------

# ExtraTreesClassifier ------------------------------------------------------
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(2, 20),
    'min_samples_split': randint(2, 15),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

model = RandomizedSearchCV(
    estimator=ExtraTreesClassifier(),
    param_distributions=param_dist,
    n_iter=num_iter,
    cv=n_cv,
    n_jobs=-1,
    return_train_score=True
)

model.fit(X_train, y_train)
Fitted_Models["ExtraTreesClassifier"] = model
# -----------------------------------------------------------------------------

# %% [9] ML Results

# =============================================================================
# Results and Analysis
# =============================================================================


# Gather Accuracy Scores ------------------------------------------------------
def get_accuracy_score(x):
    model = Fitted_Models[x]
    y_pred = model.predict(X_test)
    y_pred = (y_pred >= 0.5).astype(int)
    result = accuracy_score(y_test, y_pred)
    return x, result


def get_confusion_matricies(x):
    model = Fitted_Models[x]
    y_pred = model.predict(X_test)
    y_pred = (y_pred >= 0.5).astype(int)
    result = confusion_matrix(y_test, y_pred)
    return x, result


def get_split_results(x):
    result = Fitted_Models.cv_results_
    return x, result


accuracy = dict(
    map(lambda x: get_accuracy_score(x), Fitted_Models)
    )

confusion_matricies = dict(
    map(lambda x: get_confusion_matricies(x), Fitted_Models)
    )

cv_results = {
    x: Fitted_Models[x].cv_results_ for x in Fitted_Models
    }

best_parameters = {
    x: Fitted_Models[x].best_params_ for x in Fitted_Models
    }
# -----------------------------------------------------------------------------

# Confusion Plots -------------------------------------------------------------
fig_cm2, axes = plt.subplots(3, 2, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    try:
        cm = confusion_matricies[list(confusion_matricies.keys())[i]]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Pred: 0', 'Pred: 1'],
                    yticklabels=['True: 0', 'True: 1'],
                    ax=ax
                    )
        ax.set_title(f'{list(confusion_matricies.keys())[i]}')
    except Exception:
        ax.axis('off')

fig_cm2.suptitle("Confusion Matrix Heatmap: ML", fontsize=16, y=1.00)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------

# Feature Importance ----------------------------------------------------------
importance = {}

importance['RandomForestClassifier'] = (
    np.abs(
        Fitted_Models[
            'RandomForestClassifier'].best_estimator_.feature_importances_
        )
    )

importance['RandomForestClassifier'] = Fitted_Models[
    'RandomForestClassifier'].feature_importances_

importance['XGBClassifier'] = Fitted_Models[
    'XGBClassifier'].feature_importances_

# -----------------------------------------------------------------------------

# Bar Charts ------------------------------------------------------------------
fig_bar, axes = plt.subplots(2, 1, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    model_name = ['RandomForestClassifier','XGBClassifier'][i]
    sorted_idx = np.argsort(importance[model_name])
    ax.barh(results['description'][sorted_idx], importance[model_name][sorted_idx])
    ax.set_title(f'{model_name}')

fig_bar.suptitle("Feature Importance", fontsize=16, y=1.00)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------

# =============================================================================
# Plots
# =============================================================================

for i in range(len(list(cv_results.keys()))):
    array_to_plot = cv_results[list(cv_results.keys())[i]]['std_train_score']

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(array_to_plot, label='Standard Deviation of Train Score')
    plt.title(f'Standard Deviation of Train Scores - {list(cv_results.keys())[i]}')
    plt.xlabel('Index')
    plt.ylabel('Standard Deviation')
    plt.ylim(min(array_to_plot)*0.9, max(array_to_plot)*1.1)
    plt.legend()
    plt.grid(True)
    plt.show()