# 05
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    cross_validate,
    ParameterGrid,
)
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.utils import resample
import numpy as np
from evaluation_metrics import compute_scores


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


def feature_selection(X_train, X_test, y_train):
    lasso = Lasso(alpha=0.1, random_state=0).fit(X_train, y_train)
    selector = SelectFromModel(lasso, prefit=True)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    return X_train, X_test


def cross_validation(X, y, clf):
    scores = cross_validate(clf, X, y, cv=5, scoring=["accuracy", "roc_auc"])


def bootstrap(X, y):
    # It is important to sample with:
    # (1) the same size as the initial df (df_size)
    # (2) with replacement (replace=True)
    # for the bootstrap samples to be representative.
    df_size = len(X)
    B = 100

    # Generate B bootstrap samples of the dataset with replacement.
    samples = [resample(X, y, replace=True, n_samples=df_size) for b in range(B)]
    # Train a random forest classifier for each bootstrap set
    clfs = [
        RandomForestClassifier(random_state=42).fit(X_b, y_b) for X_b, y_b in samples
    ]
    # Calculate the predictions for each bootstrap sample (b in range(B)).
    # Compare predictions against the ground truth (y.loc[[user]]).
    # Take the mean of predictions for each student (over on the number of times they were predicted).
    # Takes ~2 mins
    accuracies = [
        np.mean(
            [
                clfs[b].predict(X.loc[[user]]) == y.loc[[user]]
                for b in range(B)
                if user not in samples[b][0].index
            ]
        )
        for user in X.index
    ]

    # Take the mean of predictions across all students.
    bootstrap_err = np.mean(accuracies)
    print("Bootstrap error: ", bootstrap_err)

    # Calculate the training error for each bootstrapped model, then average across bootstraps.
    training_err_bootstrap = [
        clfs[b].score(samples[b][0], samples[b][1]) for b in range(B)
    ]
    training_err = np.mean(training_err_bootstrap)
    print("Training error: ", training_err)

    accuracy_632 = 0.632 * bootstrap_err + 0.368 * training_err
    print(f"Mean accuracy with .632 leave-one-out bootstrapping: {accuracy_632:.3f}")


def parameter_tuning(X_train, y_train, X_val, y_val, X_test, y_test):
    # We compute a grid search across the following parameter space
    parameters = {
        "n_estimators": [20, 50, 100],
        "criterion": ["entropy", "gini"],
        "max_depth": np.arange(3, 9),
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 3, 5],
    }

    params_grid = ParameterGrid(parameters)

    # For each combination of candidate parameters, fit a classifier on the training set and evaluate it on the validation set
    results = [
        [
            params,
            compute_scores(
                RandomForestClassifier(random_state=42, **params),
                X_train,
                y_train,
                X_val,
                y_val,
            ),
        ]
        for params in params_grid
    ]

    # Sort candidate parameters according to their accuracy
    results = sorted(results, key=lambda x: x[1][0], reverse=True)

    # Obtain the best parameters
    best_params = results[0][0]
    print("Best parameters:", best_params)

    # Train and evaluate a model based on the best parameter settings
    clf = RandomForestClassifier(random_state=42, **best_params)

    X_1, y_1 = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    accuracy, AUC = compute_scores(clf, X_1, y_1, X_test, y_test)

    print(f"Accuracy for train-validation-test setting: {accuracy}")
    print(f"AUC for train-validation-test setting: {AUC}")


def parameter_tuning_with_cross_validation(X, y):
    # We compute a grid search across the following parameter space
    parameters = {
        "n_estimators": [20, 50, 100],
        "criterion": ["entropy", "gini"],
        "max_depth": np.arange(3, 7),
        "min_samples_split": [2],
        "min_samples_leaf": [1],
    }

    # Inner cross validation loop
    clf = GridSearchCV(RandomForestClassifier(random_state=42), parameters, cv=10)

    # Outer cross validation loop
    scores_nested_cv = cross_validate(clf, X, y, cv=3, scoring=["accuracy", "roc_auc"])

    accuracies_nested_cv = scores_nested_cv["test_accuracy"]
    AUC_nested_cv = scores_nested_cv["test_roc_auc"]

    print(
        f"Mean accuracy with nested cross-validation: {accuracies_nested_cv.mean():.3f}"
    )
    print(f"Mean AUC with nested cross-validation: {AUC_nested_cv.mean():.3f}")

    # report the uncertainty of the prediction using the standard deviation
    # Compute standard deviation of Accuracy and AUC
    print(
        f"Accuracy standard deviation with nested cross-validation: {accuracies_nested_cv.std():.3f}"
    )
    print(
        f"AUC standard deviation with nested cross-validation: {AUC_nested_cv.std():.3f}"
    )


def bootstrap_with_cross_validation(X, y, df_lq):
    parameters = {
        "n_estimators": [20, 50],
        "criterion": ["gini"],
        "max_depth": np.arange(3, 5),
        "min_samples_split": [2],
        "min_samples_leaf": [1],
    }
    df_size = len(df_lq)
    B = 100

    # Generate B samples with replacement
    samples = [resample(X, y, replace=True, n_samples=df_size) for b in range(B)]
    # Train a random forest classifier for each sample, cross-validating to find the best parameters
    clfs = [
        GridSearchCV(RandomForestClassifier(random_state=42), parameters, cv=3).fit(
            X_b, y_b
        )
        for X_b, y_b in samples
    ]

    # Calculate the predictions for each bootstrap sample (b in range(B)).
    # Compare predictions against the ground truth (y.loc[[user]]).
    # Take the mean of predictions for each student (over on the number of times they were predicted).
    # Takes ~2 mins
    accuracies_bootstrap = [
        np.mean(
            [
                clfs[b].predict(X.loc[[user]]) == y.loc[[user]]
                for b in range(B)
                if user not in samples[b][0].index
            ]
        )
        for user in df_lq.index
    ]

    # Take the mean of predictions across all students.
    bootstrap_err = np.mean(accuracies_bootstrap)
    print(f"Bootstrap error: {bootstrap_err:.3f}")

    # Calculate the training error for each bootstrapped model, then average across bootstraps.
    training_err_bootstrap = [
        clfs[b].score(samples[b][0], samples[b][1]) for b in range(B)
    ]
    training_err = np.mean(training_err_bootstrap)
    print(f"Training error: {training_err:.3f}")

    accuracy_632 = 0.632 * bootstrap_err + 0.368 * training_err
    print(f"Mean accuracy with .632 leave-one-out bootstrapping: {accuracy_632:.3f}")
    print(
        f"Accuracy standard deviation with .632 bootstrapping: {accuracies_632.std():.3f}"
    )


def pipeline_example(X, y):
    scalers = [StandardScaler(), "passthrough"]  # none

    feature_selectors = [
        SelectFromModel(Lasso(alpha=0.1, random_state=0)),
        "passthrough",
    ]

    steps = [
        ("scaler", StandardScaler()),  # preprocessing steps
        (
            "feature_selector",
            SelectFromModel(Lasso(alpha=0.1, random_state=0)),
        ),  # Feature selection
        ("clf", RandomForestClassifier(random_state=0)),
    ]  # Model

    param_grid = {
        "scaler": scalers,
        "feature_selector": feature_selectors,
        "clf__n_estimators": [10, 1000],
        "clf__max_depth": [1, None],
    }

    pipeline = Pipeline(steps)

    search = GridSearchCV(
        pipeline, param_grid, n_jobs=-1, cv=5, scoring="balanced_accuracy"
    )
    search.fit(X, y)
    print("Best parameter (CV score=%0.2f):" % search.best_score_)
    print(search.best_params_)

    results = pd.DataFrame(search.cv_results_)
    results.sort_values("rank_test_score")[
        [
            "param_clf__max_depth",
            "param_clf__n_estimators",
            "param_feature_selector",
            "param_scaler",
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ]

    results[
        [
            "split0_test_score",
            "split1_test_score",
            "split2_test_score",
            "split3_test_score",
            "split4_test_score",
            "mean_test_score",
            "std_test_score",
        ]
    ].sort_values("std_test_score")
