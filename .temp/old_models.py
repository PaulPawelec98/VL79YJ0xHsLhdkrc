# Logistic --------------------------------------------------------------------
param_grid = {
    'C': [0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
    }

model = GridSearchCV(
    LogisticRegression(),
    param_grid=param_grid,
    cv=n_cv, scoring='accuracy'
    )

model.fit(X, y)
Fitted_Models["Logistic"] = model
# -----------------------------------------------------------------------------

# Random Forest ---------------------------------------------------------------
param_grid = {
    'max_depth': [10],
    'n_estimators': [150],
    "min_samples_split": [2],
    "min_samples_leaf": [2]
    }

model = GridSearchCV(
    estimator=RandomForestRegressor(
        max_features='sqrt',
        random_state=1,
        n_jobs=-1
        ),
    param_grid=param_grid, cv=n_cv
    )

model.fit(X, y)
Fitted_Models["Random Forest"] = model
# -----------------------------------------------------------------------------

# SVR -------------------------------------------------------------------------
param_grid = {
    'C': [0.05, 0.1],
    'kernel': ['sigmoid'],
    'degree': [3],
}

model = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=35)
model.fit(X, y)
Fitted_Models["SVR"] = model
# -----------------------------------------------------------------------------

# XGB -------------------------------------------------------------------------
param_grid = {
    "n_estimators": [5, 100, 250],
    "max_depth": [2, 3],
    "learning_rate": [0.01, 0.1, 0.2, 0.5, 0.75]
}

model = GridSearchCV(
    estimator=xgb.XGBRegressor(
        objective='reg:squarederror'
        ),
    param_grid=param_grid, cv=n_cv
    )

model.fit(X, y)
Fitted_Models["XGB"] = model
# ------------------------------------------------------------------------------

# PCA + Log -------------------------------------------------------------------
pipe = Pipeline([
    ("kpca", KernelPCA(kernel="poly")),
    ("log", LogisticRegression())
])

param_grid = [{"kpca__n_components": range(1, X.shape[1],
                                            int(np.ceil(X.shape[1] / 10)))}]

model = GridSearchCV(pipe, param_grid=param_grid, cv=5)
model.fit(X, y)
Fitted_Models["PCA + Log"] = model
# ------------------------------------------------------------------------------


# PCA + Random Forest ---------------------------------------------------------
pipe = Pipeline([
    ("kpca", KernelPCA(kernel="poly")),
    ("rf", RandomForestRegressor(
        max_features='sqrt',
        random_state=1,
        n_jobs=-1
    ))
])

param_grid = {
    "kpca__n_components": range(1, X.shape[1], int(np.ceil(X.shape[1] / 10))),
    "rf__max_depth": [10],
    "rf__n_estimators": [150],
    "rf__min_samples_split": [2],
    "rf__min_samples_leaf": [2]
}

model = GridSearchCV(pipe, param_grid=param_grid, cv=5)
model.fit(X, y)

Fitted_Models["PCA + RF"] = model
# -----------------------------------------------------------------------------
