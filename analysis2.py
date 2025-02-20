# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:37:37 2025

@author: Paul
"""

# %% [1] Import Packages

import os
import pandas as pd  # data, plotting, etc...
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import json

# import statsmodels.api as sm  # stats models for analysis
# import random  # for setting seed
# from scipy.stats import randint, uniform
# import lazypredict  # ml packages, models, tools, etc..
from sklearn.utils import shuffle
# from lazypredict.Supervised import LazyClassifier

from sklearn.model_selection import train_test_split  # ml transformations
from sklearn.preprocessing import Binarizer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV

from sklearn.naive_bayes import BernoulliNB  # new models to use
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score  # accuracy score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.utils.class_weight import compute_class_weight  # weights

from sklearn.ensemble import VotingClassifier  # ensemble techniques
from sklearn.ensemble import StackingClassifier


from hyperopt import fmin, tpe, hp, STATUS_OK  # hyper parameter tuning
from hyperopt.pyll.base import scope
from hyperopt import space_eval
from sklearn.metrics import matthews_corrcoef

from sklearn.feature_selection import RFE  # feature selection

# from sklearn.model_selection import cross_val_score

# Random Seed
# seed = random.randint(1000, 9999)
# print(seed)

seed = 3921
np.random.seed(seed)

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
fitted_models = {}

# X, y
X, y = shuffle(df.iloc[:, 1:], df['Y'], random_state=seed)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.55,
                                                    random_state=seed
                                                    )

binarizer = Binarizer(threshold=3)
X_train_binary = binarizer.fit_transform(X_train)
X_test_binary = binarizer.fit_transform(X_test)

# Assuming y_train contains the labels
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))
sample_weights = np.array([class_weight_dict[label] for label in y_train])

# ml scaler
# scaler = StandardScaler().set_output(transform="pandas")

# %% [4] Run Models
'''
- bernoulli
- extratress
- xgb
- voting
- stacking
'''

# BernoulliNB -----------------------------------------------------------------
model = BernoulliNB(
    alpha=0.5,
    force_alpha=False,
    binarize=3,
    fit_prior=True,
    class_prior=[0.55, 0.45]
    )
# model = model.fit(X_train, y_train)
model = model.fit(X_train, y_train, sample_weight=sample_weights)
fitted_models['BernoulliNB'] = model
# -----------------------------------------------------------------------------

# ExtraTreesClassifier --------------------------------------------------------
model = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=2,
    min_samples_split=3,
    min_samples_leaf=1,
    bootstrap=False,
    random_state=seed,
    class_weight={0: 1.025, 1: 0.975}
    )
# model = model.fit(X_train, y_train)
model = model.fit(X_train, y_train, sample_weight=sample_weights)
fitted_models['ExtraTreesClassifier'] = model
# -----------------------------------------------------------------------------

# # KNeighborsClassifier --------------------------------------------------------
# model = KNeighborsClassifier(
#     n_neighbors=2,  # Number of neighbors
#     # weights='uniform',  # Use uniform weights for neighbors
#     algorithm='auto',  # Let the model choose the best algorithm
#     n_jobs=-1  # Use all cores
# )
# model = model.fit(X_train, y_train)
# fitted_models['KNeighborsClassifier'] = model
# # -----------------------------------------------------------------------------

# # LogisticRegression ----------------------------------------------------------
# model = LogisticRegression(
#     penalty='none',
#     C=1,
#     solver='lbfgs',
#     class_weight={0: 1.05, 1: 1},
#     random_state=seed
# )
# model = model.fit(X_train, y_train, sample_weight=sample_weights)
# fitted_models['LogisticRegression'] = model
# # -----------------------------------------------------------------------------

# XGBClassifier ---------------------------------------------------------------
model = XGBClassifier(
    n_estimators=50,
    learning_rate=0.030,
    max_depth=3,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.9,
    # gamma=0.1,
    # objective='binary:logistic',
    # eval_metric='logloss',
    random_state=seed,
)
# model = model.fit(X_train, y_train)
model = model.fit(X_train, y_train, sample_weight=sample_weights)
fitted_models['XGBClassifier'] = model
# -----------------------------------------------------------------------------

# %% [6] Ensemble Techniques
'''
- voting
- stacking
- ...
'''

# Copy Models for Ensemble ----------------------------------------------------
voting_models = fitted_models.copy()
voting_models = list(voting_models.items())
# -----------------------------------------------------------------------------

# Voting ----------------------------------------------------------------------
model = VotingClassifier(
    estimators=voting_models,
    voting='soft',
    # weights=[1, 1, .75]
    # weights=[1, 5, .5, .5]
    )
# model = model.fit(X_train, y_train)
model = model.fit(X_train, y_train, sample_weight=sample_weights)
fitted_models['Voting'] = model
# -----------------------------------------------------------------------------

# Stacking --------------------------------------------------------------------
meta_model = LogisticRegression()
# meta_model = XGBClassifier(
#     n_estimators=10,
#     learning_rate=0.01,
#     max_depth=5,
#     objective='binary:logistic',
#     random_state=seed,
#     )
model = StackingClassifier(
    estimators=voting_models, final_estimator=meta_model)
# model = model.fit(X_train, y_train)
model = model.fit(X_train, y_train, sample_weight=sample_weights)
fitted_models['Stacking'] = model
# -----------------------------------------------------------------------------

# %% [7] Performance Results

# Functions & Results ---------------------------------------------------------
'''
- recall score
- confusion
- classification
'''


def get_recall_score(x):
    model = fitted_models[x]

    y_pred = model.predict(X_test)

    result = recall_score(y_test, y_pred, pos_label=0)
    return x, result


def get_confusion_matricies(x):
    try:
        model = fitted_models[x]

        y_pred = model.predict(X_test)

        result = confusion_matrix(y_test, y_pred)
        return x, result
    except Exception:
        return x, None


def get_classification_report(x):
    model = fitted_models[x]

    y_pred = model.predict(X_test)

    result = classification_report(y_test, y_pred)
    return x, result


recall_results = dict(map(get_recall_score, fitted_models))

confusion_matricies = dict(map(get_confusion_matricies, fitted_models))

classification_report_results = dict(
    map(get_classification_report, fitted_models))
# -----------------------------------------------------------------------------

# Benchmarks ------------------------------------------------------------------
'''
- naive
- mean
'''


def calculate_random_predictions(y_train, y_test):
    unique, counts = np.unique(y_train, return_counts=True)
    class_probs = counts / len(y_train)  # Probabilities for y=0 and y=1
    random_y = np.random.choice(unique, size=len(y_test), p=class_probs)
    return random_y


# based on train class distributions - keeps being random each time I run?
baseline_random = recall_score(
    y_test,
    calculate_random_predictions(y_train, y_test),
    pos_label=0
    )

baseline_mean = recall_score(
    y_test,
    [0] * len(y_test),
    pos_label=0
    )
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

# Bar Plot for Accuracy -------------------------------------------------------
plt.figure(figsize=(8, 6))  # Set the figure size
bars = plt.bar(
        recall_results.keys(),
        recall_results.values(),
        color='lightblue',
        edgecolor='black'
        )

# titles
plt.title('Recall for Class 0', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Score', fontsize=14)

# grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# benchmarks
plt.axhline(y=baseline_mean, color='red',
            linestyle='--', linewidth=1.5, label='Naive')

plt.axhline(y=baseline_random, color='blue',
            linestyle='--', linewidth=1.5, label='Random')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01,
             f'{yval:.3f}', ha='center', va='bottom', fontsize=12)

# rotate axis on x
plt.xticks(rotation=45)

# legend
plt.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------

# Feature Importance ----------------------------------------------------------
importance = {}

_nb_model = fitted_models['BernoulliNB']
importance['BernoulliNB'] = np.abs(
    _nb_model.feature_log_prob_[1] - _nb_model.feature_log_prob_[0])

importance['ExtraTreesClassifier'] = fitted_models[
    'ExtraTreesClassifier'].feature_importances_

importance['XGBClassifier'] = fitted_models[
    'XGBClassifier'].feature_importances_

# bar plot
fig_bar, axes = plt.subplots(3, 1, figsize=(8, 8))

_feature_names = list(df.columns)[1:]

for i, ax in enumerate(axes.flat):
    model_name = list(importance.keys())[i]
    sorted_idx = np.argsort(importance[model_name])

    _sorted_feature_names = [_feature_names[i] for i in sorted_idx]
    ax.barh(_sorted_feature_names, importance[model_name][sorted_idx])
    ax.set_title(f'{model_name}')

fig_bar.suptitle("Feature Importance", fontsize=16, y=1.00)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------

# %% [8] Tuning the Model

# =============================================================================
# Hyper Opt Class, Run Objectives/Optimization, Return Results
# =============================================================================

# variables

final_scores = {}

# Create Hyperopt Class


class HyperOptClassifier:
    def __init__(
            self, X_train, X_test, y_train, y_test, sample_weights):

        self.space = space
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.sample_weights = sample_weights

    def objective(self, params):
        model = ExtraTreesClassifier(**params)

        model = model.fit(
            self.X_train,
            self.y_train,
            sample_weight=self.sample_weights
            )

        y_pred = model.predict(self.X_test)

        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # TNR
        # score = tn / (tn + fp)  # TNR (overfits...)
        # score = tn / (tn + fn)  # NPV
        # score = matthews_corrcoef(y_test, y_pred)  # mcc
        score = model.score(self.X_test, self.y_test)

        # return negative of the score
        return {'loss': -score, 'status': STATUS_OK}

    def run_optimization(self, space, evals=50):
        best = fmin(
            fn=self.objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=evals
            )

        print("Best hyperparameters:", best)

        return space_eval(self.space, best)


# Define the hyperparameter search space
space = {
    'n_estimators':  scope.int(hp.quniform('n_estimators', 10, 200, 10)),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 15, 1)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 15, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 15, 1)),

    'min_weight_fraction_leaf': hp.uniform(
        'min_weight_fraction_leaf', 0.0, 0.5),

    'max_leaf_nodes': scope.int(hp.quniform(
        'max_leaf_nodes', 2, 15, 1)),

    'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
    'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 0.2),
    'ccp_alpha': hp.uniform('ccp_alpha', 0.0, 0.1),
    'max_samples': scope.int(hp.quniform('max_samples', 1, 55, 1)),
    'random_state': seed,  # Fixed Parameters...
    'bootstrap': True,
    'n_jobs': -1,
    'class_weight': {0: 1.025, 1: 0.975}
    }

# Hyperopt:ExtraTreesClassifier -----------------------------------------------
hpyer_opt1 = HyperOptClassifier(  # set new object
    X_train,
    X_test,
    y_train,
    y_test,
    sample_weights
    )
best = hpyer_opt1.run_optimization(space)

# create model and fit
model = ExtraTreesClassifier(
    **best,
    )
model = model.fit(X_train, y_train, sample_weight=sample_weights)

# return results
y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)

final_scores['Hyperopt:ExtraTreesClassifier'] = recall_score(
    y_test,
    pd.Series(y_pred),
    pos_label=0
    )

fitted_models['HyperOpt:ExtraTreesClassifier'] = model
# -----------------------------------------------------------------------------

# Hyperopt:ExtraTreesClassifier:LessFeatures ----------------------------------
hpyer_opt2 = HyperOptClassifier(  # set new object, less features
    X_train[['X5', 'X4', 'X1']],
    X_test[['X5', 'X4', 'X1']],
    y_train,
    y_test,
    sample_weights
    )

best = hpyer_opt2.run_optimization(space)

# create model and fit
model = ExtraTreesClassifier(
    **best,
    )

model = model.fit(
    X_train[['X5', 'X4', 'X1']], y_train, sample_weight=sample_weights
    )

# return results
y_pred = model.predict(X_test[['X5', 'X4', 'X1']])

final_scores['Hyperopt:ExtraTreesClassifier:LessFeatures'] = recall_score(
    y_test,
    pd.Series(y_pred),
    pos_label=0
    )

fitted_models['HyperOpt:ExtraTreesClassifier:LessFeatures'] = model
# -----------------------------------------------------------------------------

# HyperOpt:ExtraTreesClassifier:RTE -------------------------------------------
rfe = RFE(estimator=ExtraTreesClassifier(
    random_state=seed,
    class_weight={0: 1.025, 1: 0.975},
    n_jobs=-1),
    n_features_to_select=3
    )

rfe.fit(X_train, y_train)
for i in range(X_train.shape[1]):
    print(
        f"Column: {X_train.columns[i]}, "
        f"Selected: {rfe.support_[i]}, "
        f"Rank: {rfe.ranking_[i]:.3f}"
    )

hpyer_opt3 = HyperOptClassifier(  # set new object, less features
    X_train[['X5', 'X3', 'X2']],
    X_test[['X5', 'X3', 'X2']],
    y_train,
    y_test,
    sample_weights
    )

best = hpyer_opt3.run_optimization(space)

# create model and fit
model = ExtraTreesClassifier(
    **best,
    )

model = model.fit(
    X_train[['X5', 'X3', 'X2']], y_train, sample_weight=sample_weights
    )

# return results
y_pred = model.predict(X_test[['X5', 'X3', 'X2']])

final_scores['Hyperopt:ExtraTreesClassifier:RTE'] = recall_score(
    y_test,
    pd.Series(y_pred),
    pos_label=0
    )

fitted_models['HyperOpt:ExtraTreesClassifier:RTE'] = model
# -----------------------------------------------------------------------------

# %% [9] Performance Results

# =============================================================================
# Graph the Results
# =============================================================================

confusion_matricies = dict(map(get_confusion_matricies, fitted_models))

# manual addition of less features model for ETC.
model = fitted_models['HyperOpt:ExtraTreesClassifier:LessFeatures']

y_pred = model.predict(X_test[['X5', 'X4', 'X1']])
result = confusion_matrix(y_test, y_pred)
confusion_matricies['HyperOpt:ExtraTreesClassifier:LessFeatures'] = result

model = fitted_models['HyperOpt:ExtraTreesClassifier:RTE']

y_pred = model.predict(X_test[['X5', 'X3', 'X2']])
result = confusion_matrix(y_test, y_pred)
confusion_matricies['HyperOpt:ExtraTreesClassifier:RTE'] = result

# Confusion Matricies ---------------------------------------------------------
# Set up the figure and axes
fig_cm2, axes = plt.subplots(2, 2, figsize=(8, 8))
# List of model names and corresponding axes
models = [
    'ExtraTreesClassifier',
    'HyperOpt:ExtraTreesClassifier',
    'HyperOpt:ExtraTreesClassifier:LessFeatures',
    'HyperOpt:ExtraTreesClassifier:RTE'
]

titles = [
    'ExtraTreesClassifier',
    'HyperOpt:ExtraTreesClassifier',
    'HyperOpt:ExtraTreesClassifier:LessFeatures',
    'HyperOpt:ExtraTreesClassifier:RTE'
]

# Loop through axes, models, and titles using enumerate with zip
for i, (ax, model, title) in enumerate(zip(axes.flat, models, titles)):
    sns.heatmap(confusion_matricies[model], annot=True, fmt='d', cmap='Blues',
                cbar=False, xticklabels=['Pred: 0', 'Pred: 1'],
                yticklabels=['True: 0', 'True: 1'], ax=ax)
    ax.set_title(title)

# Adjust layout and show the figure
fig_cm2.suptitle("ExtraTreesClassifier Models", fontsize=16, y=1.00)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------

# Bar Chart -------------------------------------------------------------------

final_scores['ExtraTreesClassifier']['recall_for_zero'] = (
    recall_results['ExtraTreesClassifier']
    )

plt.figure(figsize=(8, 8))  # Set the figure size
bars = plt.bar(
        final_scores.keys(),
        final_scores.values(),
        color='lightblue',
        edgecolor='black'
        )

# titles
plt.title('Recall for Class 0', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Score', fontsize=14)

# grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# benchmarks
plt.axhline(y=baseline_mean, color='red',
            linestyle='--', linewidth=1.5, label='Naive')

plt.axhline(y=baseline_random, color='blue',
            linestyle='--', linewidth=1.5, label='Random')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01,
             f'{yval:.3f}', ha='center', va='bottom', fontsize=12)

# rotate axis on x
plt.xticks(rotation=45)

# legend
plt.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------
