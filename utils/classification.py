# 04, 05
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
)
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder  # , normalize
from sklearn.linear_model import ElasticNet, LogisticRegression

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.special import softmax

from evaluation_metrics import compute_scores

from preprocessing import split_by_all_weeks


def decision_tree(X_train, y_train, X_val, y_val, depth=2):
    clf = tree.DecisionTreeClassifier(
        max_depth=depth, random_state=0, criterion="entropy"
    )
    accuracy, auc = compute_scores(clf, X_train, y_train, X_val, y_val)
    print("Decision tree. Balanced Accuracy = {}, AUC = {}".format(accuracy, auc))

    # Visualization
    plt.figure(figsize=(20, 10))
    tree.plot_tree(clf, feature_names=X_train.columns)
    plt.show()


def compare_decision_tree_over_many_depths(X_train, y_train, X_val, y_val):
    # We can change the max depth
    accuracy_list = []
    auc_list = []
    for depth in range(1, len(X_train.columns)):
        clf = tree.DecisionTreeClassifier(
            max_depth=depth, random_state=0, criterion="entropy"
        )
        accuracy, auc = compute_scores(clf, X_train, y_train, X_val, y_val)
        accuracy_list.append(accuracy)
        auc_list.append(auc)
        # print("Decision tree. Depth = {}, Balanced Accuracy = {}, AUC = {}".format(depth, accuracy, auc))
    x = list(range(1, len(X_train.columns)))
    plt.plot(x, accuracy_list, "r", label="accuray")
    plt.plot(x, auc_list, "b", label="auc")
    plt.xlabel("Decision tree Depth")
    plt.ylabel("EvaluationMetrics")
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


def random_forest(X_train, y_train, X_val, y_val):
    """
    For a single tree, in fact, keeping a low depth is necessary to avoid overfitting and to reduce the variance.
    Random forests, instead, can have a higher depth, and consequently a lower bias,
      since the variance is reduced in the aggregation step."""
    rf = RandomForestClassifier(
        n_estimators=100, random_state=0, criterion="entropy"
    )  # create a Random Forest

    # The following is done in compute_scores
    # rf.fit(X_train, y_train)
    # preds = rf.predict(X_val)

    accuracy, auc = compute_scores(rf, X_train, y_train, X_val, y_val)
    print("Random Forest. Balanced Accuracy = {}, AUC = {}".format(accuracy, auc))


def normalize(distance_matrix):
    """normalize the pairwise distance matrices such that we can sum them up"""
    range_matrix = np.max(distance_matrix) - np.min(distance_matrix)
    normalized = (distance_matrix - np.min(distance_matrix)) / range_matrix
    return normalized


def knn(X_train, y_train, X_val, y_val, feature, k=5):
    # Compute the pairwise distance matrix for all the elements of the training set
    X_train_dist = squareform(
        pdist(X_train[feature].to_numpy().reshape(-1, 1), metric="euclidean")
    )

    # Compute the distance between all elements of the training set and of the validation set
    X_val_dist = cdist(
        X_val[feature].to_numpy().reshape(-1, 1),
        X_train[feature].to_numpy().reshape(-1, 1),
        metric="euclidean",
    )

    # If you want to use multiple features
    X_train_vectors = X_train.groupby("user")[feature].agg(list)
    X_val_vectors = X_val.groupby("user")[feature].agg(list)
    # Compute the pairwise distance matrix for all the elements of the training set, by computing
    # the distances between the vectors, for each of the features selected, and summing up
    # the resulting matrices
    X_train_dist = sum(
        map(
            lambda x: normalize(squareform(pdist(x[1].tolist(), metric="euclidean"))),
            X_train_vectors.iteritems(),
        )
    )
    # Same thing but between all elements of the training set and of the validation set
    X_val_dist = sum(
        map(
            lambda x: normalize(
                cdist(x[0][1].tolist(), x[1][1].tolist(), metric="euclidean")
            ),
            zip(X_val_vectors.iteritems(), X_train_vectors.iteritems()),
        )
    )

    knn = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
    accuracy, auc = compute_scores(knn, X_train_dist, y_train, X_val_dist, y_val)
    print("k-nearest neighbors. Balanced Accuracy = {}, AUC = {}".format(accuracy, auc))


def logistic_function(x, beta0, beta1):
    """assign probability to every point"""
    return 1 / (1 + np.exp(-(beta0 + beta1 * x)))


def multiclass_logistic_function(x):
    # one regression per class
    return softmax(x)
    # TODO check if this is correct
    # softmax = np.exp(beta.dot(x)) / np.sum(np.exp(beta.dot(x)))
    # return softmax[k]


def logistic_regression(X_train, y_train, X_val, y_val):
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = LogisticRegression(random_state=0)
    accuracy, auc = compute_scores(clf, X_train_scaled, y_train, X_val_scaled, y_val)
    print("Logistic Regression. Balanced Accuracy = {}, AUC = {}".format(accuracy, auc))


def elastic_net(X, y, X_train, y_train, X_test, y_test):
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                ["group"],
            ),
            ("numerical", MinMaxScaler(), ["studying_hours"]),
        ]
    )

    preprocessor.fit_transform(X_train)

    # Fit a pipeline with transformers and an estimator to the training data
    pipe = Pipeline([("preprocessor", preprocessor), ("model", ElasticNet())])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    error = round(mean_squared_error(y_test, y_pred), 3)
    print(f"Mean Squared Error = {error}")

    print(
        "CV Score",
        (-1)
        * np.mean(cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error")),
    )


def time_series_classifcation_over_all_weeks(df, pipe, X, y):
    time_splits = split_by_all_weeks(df)

    errors = (-1) * cross_val_score(
        pipe, X, y, cv=time_splits, scoring="neg_mean_squared_error"
    )
    print("Errors: np.mean(errors)")

    sns.lineplot(y=errors, x=list(range(4, 27)))
    plt.show()


def grid_search_with_cv(pipe, X, y):
    param_grid = {"model__alpha": [0.1, 1], "model__l1_ratio": [0.1, 0.5, 1]}

    search = GridSearchCV(
        pipe,
        param_grid,
        n_jobs=-1,
        cv=KFold(n_splits=4, shuffle=True, random_state=123),
        scoring="neg_mean_squared_error",
    )
    search.fit(X, y)

    print("Best score", (-1) * search.best_score_)


def nested_cv(pipe, X, y):
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=123)
    outer_cv = KFold(n_splits=3, shuffle=True, random_state=123)

    param_grid = {"model__alpha": [0.1, 1], "model__l1_ratio": [0.1, 0.5, 1]}

    search = GridSearchCV(
        pipe, param_grid, n_jobs=-1, cv=inner_cv, scoring="neg_mean_squared_error"
    )
    errors = (-1) * cross_val_score(search, X=X, y=y, cv=outer_cv)

    print(np.mean(errors))


if __name__ == "__main__":
    pass
