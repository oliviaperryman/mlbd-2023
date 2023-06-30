from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, Normalizer
import numpy as np


def split(df):
    train, test = train_test_split(
        df, test_size=0.2, random_state=0, stratify=df["passed"]
    )


def split_with_validation(X, y):
    # Select the test set as 20% of the initial data set
    X_1, X_test, y_1, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Select the training set as 70% of the initial dataset
    # Select the validation set at 10% of the initial dataset (we use 1/8 here because we've already split the set once)
    X_train, X_val, y_train, y_val = train_test_split(
        X_1, y_1, test_size=1 / 8, random_state=42, stratify=y_1
    )


def split_by_user(df, y):
    # Split by user_id
    users = df.user.unique()
    users_train, users_val, y_train, y_val = train_test_split(
        users, y, test_size=0.2, random_state=0, stratify=y
    )
    X_train = df[df.user.isin(users_train)]
    X_val = df[df.user.isin(users_val)]
    # Sort indexes to make label arrays consistent with the data
    y_train = y_train.sort_index()
    y_val = y_val.sort_index()


def split_by_user2(df):
    users = df["user_id"].unique()
    users_train = list(np.random.choice(users, int(len(users) * 0.8), replace=False))
    users_test = list(set(users) - set(users_train))

    X_train, X_test = (
        df[df["user_id"].isin(users_train)],
        df[df["user_id"].isin(users_test)],
    )


def split_by_week(df, week_number):
    df_train = df.query("week < @week_number")
    df_test = df.query("week == @week_number")
    X_train = df_train[["studying_hours", "group", "week"]]
    y_train = df_train["quiz_grade"]

    X_test = df_test[["studying_hours", "group", "week"]]
    y_test = df_test["quiz_grade"]


def split_by_all_weeks(df):
    time_splits = [
        tuple([list(df.query("week < @i").index), list(df.query("week == @i").index)])
        for i in range(4, 27)
    ]
    return time_splits


def standardize(df):
    # Standardization helps correctly compare multiple variables (in different units) and
    # reduce multicollinearity.
    StandardScaler().fit_transform(df)


def aggregate_mean_std(X_train):
    # Aggregate by mean and std,  for example for a decision tree
    groups_train = X_train.drop("week", axis=1).groupby("user", as_index=False)
    standard_train = groups_train.std()
    averages_train = groups_train.mean()
    X_train_aggregate = pd.concat([standard_train, averages_train], axis=1)


def aggregate_to_list(X_train):
    # Aggregate to list, for example for knn
    features = []
    X_train_vectors = X_train.groupby("user")[features].agg(list)


def preprocess_cat_and_numerical(df):
    # Preprocess categorical and numerical features
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                ["group"],
            ),
            ("numerical", Normalizer(norm="l1"), ["studying_hours"]),
            # ("numerical", MinMaxScaler(), ["studying_hours"]),
        ]
    )


def normalize(df):
    normalized_df = (df - df.mean()) / df.std()


def create_iterator(data):
    """
    Create an iterator to split interactions in data into train and test,
    with the same student not appearing in two diverse folds.
    We will use a train-test setting (20% of students in the test set).
    The create_iterator function creates an iterator object able to
    split student's interactions included in data in 10 folds such that
    the same student does not appear in two different folds. To do so,
    we appropriately initialize a scikit-learn's GroupShuffleSplit
    iterator with 80% training set size and non-overlapping groups,
    then return the iterator.
    :param data:        Dataframe with student's interactions.
    :return:            An iterator.
    """
    # Both passing a matrix with the raw data or just an array of indexes works
    X = np.arange(len(data.index))
    # Groups of interactions are identified by the user id
    # (we do not want the same user appearing in two folds)
    groups = data["user_id"].values
    return model_selection.GroupShuffleSplit(
        n_splits=1, train_size=0.8, test_size=0.2, random_state=0
    ).split(X, groups=groups)


def create_iterator_groupkfold(data):
    """
    Create an iterator to split interactions in data in 10 folds,
    with the same student not appearing in two diverse folds.

    How to use it:
    for iteration, (train_index, test_index) in enumerate(create_iterator(data)):

    :param data:        Dataframe with student's interactions.
    :return:            An iterator.
    """
    # Both passing a matrix with the raw data or just an array of indexes works
    X = np.arange(len(data.index))
    # Groups of interactions are identified by the user id (we do not want the same user appearing in two folds)
    groups = data["user_id"].values
    return model_selection.GroupKFold(n_splits=10).split(X, groups=groups)


def bin_values(df):
    learn_maps = {
        0: "less than 10s",
        1: "less than 20s",
        2: "less than 30s",
        3: "less than 40s",
        4: "less than 50s",
    }
    df["bin_s_first_response"] = (
        (df["ms_first_response"] // (10 * 1000)).map(learn_maps).fillna("other")
    )
